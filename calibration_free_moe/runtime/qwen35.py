"""Bind sparse routed-expert operators into Qwen3.5-MoE served by vLLM (text only).

Gated DeltaNet and full attention, the vision tower and the sigmoid-gated shared
expert stay native. On a single rank the router, shared expert and routed experts
run inside one opaque custom op so the shared expert overlaps the router and the
routed reader on separate CUDA streams.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor, nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from calibration_free_moe.runtime.experts import (
    ProgressiveExperts,
    prepare_processed_weights,
    reader_stream,
    validate_binding,
    validate_sparse_backend,
)


class ReducedSharedExpert(Qwen2MoeMLP):
    """Reduce the gated TP shards separately from the routed EP output."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply the shared expert and all-reduce its tensor-parallel shards."""
        return tensor_model_parallel_all_reduce(super().forward(x))


def _moe_branches(hidden_states: Tensor, layer_name: LayerNameType) -> Tensor:
    """Run a single-rank MoE block's gate, shared and routed experts in one op."""
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    return cast(ProgressiveQwen35MoE, layer).branches(hidden_states)


def _moe_branches_fake(hidden_states: Tensor, layer_name: LayerNameType) -> Tensor:
    """Return an output of the right shape for tracing (fake implementation)."""
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="tokenwise_qwen35_moe_branches",
    op_func=_moe_branches,
    fake_impl=_moe_branches_fake,
)


class ProgressiveQwen35MoE(Qwen3NextSparseMoeBlock):
    """Qwen3.5 MoE block with a sparse routed runner and a native shared expert.

    The sigmoid-gated shared MLP stays dense and outside the routed read statistics.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.is_sequence_parallel = False
        self.enable_eplb = False
        # This selects upstream's external-add branch; shared weights stay TP-sharded.
        self.replicate_shared_expert = True
        self.is_fused_shared_expert_enabled = False
        self.n_routed_experts = config.num_experts
        self.n_logical_experts = config.num_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = config.num_experts
        self.n_local_physical_experts = config.num_experts // self.ep_size
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = ReducedSharedExpert(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config.hidden_act,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            # Upstream accepts ReplicatedLinear here but annotates nn.Linear.
            expert_gate=self.shared_expert_gate,  # pyright: ignore[reportArgumentType]
            prefix=f"{prefix}.shared_expert",
        )
        # On one rank an opaque branch op runs the gate, the shared expert and
        # the routed runner, so the aux-stream overlap is decided on every call
        # (in this traced forward it would be fixed at compile time). Under TP
        # or EP the runner owns the gate and the shared all-reduce stays outside.
        self.single_rank = self.tp_size == 1 and self.ep_size == 1
        self.experts = FusedMoEFactory(
            gate=None if self.single_rank else self.gate,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.experts",
            routed_experts_cls=ProgressiveExperts,
        )
        validate_sparse_backend(self.experts)
        cast(ProgressiveExperts, self.experts.routed_experts).calibration_dense = (
            cast(dict[str, dict[str, object]], vllm_config.additional_config)[
                "tokenwise"
            ].get("policy")
            == "dense-calibration"
        )
        self.n_local_physical_experts = self.experts.routed_experts.local_num_experts
        if self.single_rank:
            # vLLM's shared-expert aux stream, events, token threshold and output
            # lifecycle, as the stock runner uses them.
            self.shared_overlap: SharedExperts = SharedExperts(
                self.shared_expert,
                moe_config=self.experts.moe_config,
                enable_dbo=vllm_config.parallel_config.enable_dbo,
                mk_can_overlap_shared_experts=lambda: False,
                is_multistream_safe=lambda: True,
            )
            self.reader_stream = reader_stream()
            cast(ProgressiveExperts, self.experts.routed_experts).caller_prepares = True
            self.branches_name = f"{prefix}.moe_branches"
            context = get_current_vllm_config().compilation_config
            if self.branches_name in context.static_forward_context:
                raise ValueError(f"Duplicate layer name: {self.branches_name}")
            context.static_forward_context[self.branches_name] = self

    def branches(self, hidden_states: Tensor) -> Tensor:
        """Compute the stock gate, shared expert and routed experts; sum as stock."""
        # Enqueue the gate first, then fork the reader's routing-independent
        # input preparation, then the shared expert, which waits for neither:
        # independent kernels dispatch in enqueue order, before the shared
        # GEMVs fill the SMs.
        caller = torch.cuda.current_stream()
        self.reader_stream.wait_stream(caller)
        with torch.cuda.stream(self.reader_stream):
            router_logits, _ = self.gate(hidden_states)
        experts = cast(ProgressiveExperts, self.experts.routed_experts)
        experts.prepare_reader_input(self.gate, (hidden_states,))
        if not self.shared_overlap.maybe_forward_async(hidden_states):
            caller.wait_stream(self.reader_stream)
            shared = self.shared_expert(hidden_states)
            return shared + self.experts(hidden_states, router_logits)
        with torch.cuda.stream(self.reader_stream):
            routed = self.experts(hidden_states, router_logits)
        caller.wait_stream(self.reader_stream)
        self.shared_overlap.wait()
        return self.shared_overlap.output + routed

    def forward(
        self, hidden_states: Tensor, already_sequence_parallel: bool = False
    ) -> Tensor:
        """Return the shared plus routed expert output, as the native block does."""
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.single_rank:
            final_hidden_states = torch.ops.vllm.tokenwise_qwen35_moe_branches(
                hidden_states, _encode_layer_name(self.branches_name)
            )
        else:
            shared_output = self.shared_expert(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
            final_hidden_states += shared_output

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        return final_hidden_states.view(orig_shape)


class ProgressiveQwen35DecoderLayer(Qwen3_5DecoderLayer):
    """Native Qwen3.5 decoder layer built with ``ProgressiveQwen35MoE``."""

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.use_attn_reduce_scatter_for_moe = False
        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=cast(Qwen3NextConfig, config),
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=True,
            )
        elif layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                cast(Qwen3NextConfig, config),
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=True,
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")
        self.mlp = ProgressiveQwen35MoE(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = nn.Parameter(torch.zeros(1, 1, config.hidden_size))


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class ProgressiveQwen35Model(Qwen3_5Model):
    """Native Qwen3.5 text model built from ``ProgressiveQwen35DecoderLayer``."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.num_redundant_experts = 0
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        def get_layer(prefix: str) -> ProgressiveQwen35DecoderLayer:
            return ProgressiveQwen35DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        # No fused-shared-expert weight remapping: the dense shared MLP has its
        # own native tensors, while the sparse runner only owns routed weights.
        self.is_fused_shared_expert_enabled = False
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = (
            GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.aux_hidden_state_layers: tuple[int, ...] = ()


class ProgressiveQwen35CausalLM(Qwen3_5MoeForCausalLM):
    """Native Qwen3.5-MoE language-model wrapper around ``ProgressiveQwen35Model``."""

    lora_manager: Any = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Match the native LM wrapper while constructing the inner model only
        # once, so there is never a second routed weight allocation.
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError("Qwen3.5 requires mamba_cache_mode=align")
        self.model = ProgressiveQwen35Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()


class ProgressiveQwen35ForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """Qwen3.5-MoE with sparse decode operators on its routed experts.

    The native multimodal wrapper and weight mapper stay; inputs are text only.
    """

    _processor_factory = Qwen3_5MoeForConditionalGeneration._processor_factory
    options: dict[str, object]
    vllm_config: VllmConfig
    lora_manager: Any = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        self.options = validate_binding(vllm_config)
        self.vllm_config = vllm_config
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        mm = vllm_config.model_config.multimodal_config
        if mm is None:
            raise ValueError("Qwen3.5 conditional generation needs multimodal config")
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = mm
        self.use_data_parallel = mm.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = mm.is_multimodal_pruning_enabled()
        self.video_pruning_rate = mm.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = ProgressiveQwen35CausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters()

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load checkpoint weights with the native loader."""
        return Qwen3_5MoeForConditionalGeneration.load_weights(self, weights)

    def process_weights_after_loading(self) -> None:
        """Bind the sparse decode operators once the native weights are final."""
        prepare_processed_weights(self)
