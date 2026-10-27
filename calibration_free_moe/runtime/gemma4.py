"""Bind sparse routed-expert operators into Gemma 4 served by vLLM (text only).

Gemma 4 runs a dense MLP beside its routed experts in every layer. The native dense
MLP, attention and KV cache stay intact; the layer's feed-forward section is
rescheduled through one opaque custom op so that, without tensor parallelism, the
dense MLP overlaps the router and the routed reader on separate CUDA streams.
"""

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor, nn
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models import gemma4 as native
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    aux_stream,
    direct_register_custom_op,
)

from calibration_free_moe.runtime.experts import (
    PreparedModel,
    ProgressiveExperts,
    bind_experts,
    prepare_processed_weights,
    reader_stream,
    validate_binding,
)


class ProgressiveGemmaMoE(native.Gemma4MoE):
    """Construct the routed runner only once, with native router semantics."""

    def __init__(self, config: Any, quant_config: object = None, prefix: str = ""):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

        def routing_function(
            hidden_states: Tensor, gating_output: Tensor, topk: int, renormalize: bool
        ) -> tuple[Tensor, Tensor]:
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                return native.gemma4_fused_routing_kernel_triton(
                    gating_output, topk, self.per_expert_scale
                )
            return native.gemma4_routing_function_torch(
                gating_output, topk, self.per_expert_scale
            )

        def recorded_routing_function(
            hidden_states: Tensor, gating_output: Tensor, topk: int, renormalize: bool
        ) -> tuple[Tensor, Tensor]:
            routed = routing_function(hidden_states, gating_output, topk, renormalize)
            self.routed.record()
            return routed

        # Recorded after routing on the routed side's stream, so the dense down
        # projection becomes ready together with the screen.
        self.routed = torch.cuda.Event()
        self.routing_function = recorded_routing_function
        bind_experts(self, get_current_vllm_config(), prefix, gemma=True)


def _feedforward_branches(
    dense_input: Tensor,
    routed_input: Tensor,
    router_input: Tensor,
    layer_name: LayerNameType,
) -> tuple[Tensor, Tensor]:
    """Run a decoder layer's dense and routed branches inside one opaque op."""
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    return cast(ProgressiveGemma4DecoderLayer, layer).branches(
        dense_input, routed_input, router_input
    )


def _feedforward_branches_fake(
    dense_input: Tensor,
    routed_input: Tensor,
    router_input: Tensor,
    layer_name: LayerNameType,
) -> tuple[Tensor, Tensor]:
    """Return outputs of the right shapes for tracing (fake implementation)."""
    return torch.empty_like(dense_input), torch.empty_like(routed_input)


direct_register_custom_op(
    op_name="tokenwise_gemma4_feedforward_branches",
    op_func=_feedforward_branches,
    fake_impl=_feedforward_branches_fake,
)


class OpaqueActivation(nn.Module):
    """Run a disabled custom op's native form, compiled as vLLM compiles it.

    Custom ops called inside an opaque op are compiled this way by vLLM; the eager
    form would be rounded to BF16 between kernels and differ from stock.
    """

    def __init__(self, activation: CustomOp) -> None:
        super().__init__()
        self.native = activation.maybe_compile(activation.forward_native)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the compiled native activation."""
        return self.native(x)


class DenseHead(nn.Module):
    """Gemma's dense MLP through its activation; the down projection follows."""

    def __init__(self, mlp: native.Gemma4MLP) -> None:
        super().__init__()
        self.gate_up_proj = mlp.gate_up_proj
        self.act_fn = mlp.act_fn

    def forward(self, x: Tensor) -> Tensor:
        """Return the activated gate-up product, the down projection's input."""
        gate_up, _ = self.gate_up_proj(x)
        return self.act_fn(gate_up)


class ProgressiveGemma4DecoderLayer(native.Gemma4DecoderLayer):
    """Native layer whose stock dense MLP runs beside the router and routed reader."""

    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config, cache_config=cache_config, quant_config=quant_config, prefix=prefix
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        # The op hides the dense MLP and router projection from the traced graph.
        # Stock's graph fuses the disabled GeluAndMul (one rounding), so compile
        # its native form here too.
        assert self.router is not None
        activation = self.mlp.act_fn
        if isinstance(activation, CustomOp) and not activation.enabled():
            self.mlp.act_fn = OpaqueActivation(activation)
        # Stock's graph also resolves GateLinear's Python token-count branch once,
        # at the profiling batch, so it never takes the cuteDSL M <= 16 tier
        # (SM90/SM100); this op projects per decode batch and must not either.
        mode = get_current_vllm_config().compilation_config.mode
        if mode in (CompilationMode.DYNAMO_TRACE_ONCE, CompilationMode.VLLM_COMPILE):
            self.router.proj.allow_ll_bf16_gemm = False
        # The dense and routed branches read only the post-attention residual
        # and are summed after their output norms. Reuse vLLM's shared-expert
        # aux stream, events, token threshold and per-ubatch output lifecycle.
        assert self.moe is not None
        self.dense_overlap: SharedExperts = SharedExperts(
            DenseHead(self.mlp),
            moe_config=self.moe.experts.moe_config,
            enable_dbo=get_current_vllm_config().parallel_config.enable_dbo,
            mk_can_overlap_shared_experts=lambda: False,
            is_multistream_safe=lambda: True,
        )
        # The fork is decided per call inside an opaque op, like vLLM's own
        # shared experts: in this traced forward a token-count decision would be
        # fixed at compile time. The op starts before the stock router
        # projection, so the dense branch also overlaps that GEMM.
        self.reader_stream = reader_stream()
        self.dense_done = torch.cuda.Event()
        cast(ProgressiveExperts, self.moe.experts.routed_experts).caller_prepares = True
        self.branches_name = f"{prefix}.feedforward_branches"
        context = get_current_vllm_config().compilation_config.static_forward_context
        if self.branches_name in context:
            raise ValueError(f"Duplicate layer name: {self.branches_name}")
        context[self.branches_name] = self

    def branches(
        self, dense_input: Tensor, routed_input: Tensor, router_input: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute the dense MLP and routed expert outputs from normalized inputs."""
        assert self.router is not None and self.moe is not None
        # Enqueue the router projection first, then fork the reader's
        # routing-independent input preparation, then the dense MLP, which
        # waits for neither: independent kernels dispatch in enqueue order.
        caller = torch.cuda.current_stream()
        self.reader_stream.wait_stream(caller)
        with torch.cuda.stream(self.reader_stream):
            router_logits, _ = self.router.proj(router_input)
        experts = cast(ProgressiveExperts, self.moe.experts.routed_experts)
        experts.prepare_reader_input(self.moe.experts, (routed_input,))
        # The dense all-reduce would share the custom all-reduce workspace with
        # the routed one before the join, so overlap only without TP.
        overlap = self.tp_size == 1 and self.dense_overlap.maybe_forward_async(
            dense_input
        )
        if not overlap:
            caller.wait_stream(self.reader_stream)
            return self.mlp(dense_input), self.moe(routed_input, router_logits)
        with torch.cuda.stream(self.reader_stream):
            routed = self.moe(routed_input, router_logits)
        # A down GEMM dispatched before the screen would take its SMs; gated on
        # routing, it is enqueued behind the screen and fills in around it.
        dense_stream = cast(torch.cuda.Stream, aux_stream())
        head = self.dense_overlap.output
        with torch.cuda.stream(dense_stream):
            dense_stream.wait_event(cast(ProgressiveGemmaMoE, self.moe).routed)
            dense, _ = self.mlp.down_proj(head)
            self.dense_done.record(dense_stream)
        caller.wait_stream(self.reader_stream)
        self.dense_done.wait(caller)
        return dense, routed

    def feedforward(self, residual: Tensor) -> Tensor:
        """Compute the native FFN output from the post-attention residual.

        The output is taken after ``post_feedforward_layernorm`` and before the
        residual add.
        """
        assert self.router is not None
        assert self.post_feedforward_layernorm_1 is not None
        assert self.pre_feedforward_layernorm_2 is not None
        assert self.post_feedforward_layernorm_2 is not None
        # Native Gemma4Router.forward up to its projection, which the op applies.
        router = self.router
        router_input = router.norm(residual)
        router_input = router_input * router.root_size.to(router_input.dtype)
        router_input = router_input * router.scale.to(router_input.dtype)
        dense, routed = torch.ops.vllm.tokenwise_gemma4_feedforward_branches(
            self.pre_feedforward_layernorm(residual),
            self.pre_feedforward_layernorm_2(residual),
            router_input,
            _encode_layer_name(self.branches_name),
        )
        dense = self.post_feedforward_layernorm_1(dense)
        routed = self.post_feedforward_layernorm_2(routed)
        return self.post_feedforward_layernorm(dense + routed)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        residual: Tensor | None,
        per_layer_input: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Any]:
        """Run the layer as native vLLM 0.29 does, with the FFN via ``feedforward``."""
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, **kwargs
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.feedforward(residual) + residual
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            assert self.per_layer_projection is not None
            assert self.post_per_layer_input_norm is not None
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            per_layer_contribution = self.per_layer_projection(gate * per_layer_input)
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            hidden_states = hidden_states + per_layer_contribution
        # Like native, return no residual: Gemma folds it into hidden_states.
        return hidden_states * cast(Tensor, self.layer_scalar), None


class ProgressiveGemma4ForCausalLM(native.Gemma4ForCausalLM):  # pyright: ignore[reportImplicitAbstractClass]
    """Gemma 4 text model with sparse decode operators on its routed experts."""

    options: dict[str, object]
    vllm_config: VllmConfig

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.options = validate_binding(vllm_config)
        self.vllm_config = vllm_config
        # Gemma4Model hardcodes Gemma4DecoderLayer, whose constructor hardcodes
        # Gemma4MoE. Substitute both only during one model build so native
        # attention/KV/MLP/PLE remain untouched and no duplicate expert weights
        # (not even an ephemeral first runner) are allocated.
        original = native.Gemma4DecoderLayer, native.Gemma4MoE
        native.Gemma4DecoderLayer = ProgressiveGemma4DecoderLayer
        native.Gemma4MoE = ProgressiveGemmaMoE
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            native.Gemma4DecoderLayer, native.Gemma4MoE = original

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load checkpoint weights with the native loader."""
        return native.Gemma4ForCausalLM.load_weights(self, weights)

    def process_weights_after_loading(self) -> None:
        """Bind the sparse decode operators once the native weights are final."""
        prepare_processed_weights(self)


class ProgressiveGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):  # pyright: ignore[reportImplicitAbstractClass]
    """Stock multimodal Gemma 4 wrapper around the sparse text model.

    Only text is loaded and evaluated; the inner causal model is replaced through
    vLLM's model override.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_binding(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not isinstance(self.language_model, ProgressiveGemma4ForCausalLM):
            raise TypeError("Gemma4 inner causal override was not applied")

    def process_weights_after_loading(self) -> None:
        """Bind the inner text model's operators once the native weights are final.

        vLLM calls only the outer model's hook, after native weight loading.
        """
        prepare_processed_weights(
            cast(PreparedModel, cast(object, self.language_model))
        )
