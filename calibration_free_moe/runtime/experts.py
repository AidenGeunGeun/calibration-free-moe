"""Bind sparse routed-expert operators into frozen MoE models served by vLLM.

The model classes here replace OLMoE and Qwen3-MoE through vLLM's
``model_class_overrides`` in method configurations only; stock workers import
neither this module nor the phase-metadata patch. Attention, routing, weight loading
and CUDA graphs stay native. After loading, every routed layer gets one decode
operator: the reader (``ProgressiveReader``) or a baseline (Prox, TEAL, WINA, CATS,
HiRE). Decode rows, and generated tokens recomputed in a prefill, take the
operator; prompt rows stay stock unless evaluation enables ``sparse_prefill``.

The forward context's ``num_decode_tokens`` and ``generated_prefill_split`` come
from ``patches/vllm-0.29.0-phase-metadata.patch``; the installed vLLM stubs do not
declare them, hence the scoped pyright ignores where they are read.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, get_args

import torch
from torch import Tensor, nn
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.olmoe import (
    OlmoeDecoderLayer,
    OlmoeForCausalLM,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    maybe_prefix,
)

from calibration_free_moe.activation import model_activation
from calibration_free_moe.analysis.allocation import (
    allocation_collect,
    allocation_finish,
    allocation_start,
    prepare_allocations,
    validate_allocation_options,
)
from calibration_free_moe.baselines.activation_sparsity import ActivationBaseline
from calibration_free_moe.baselines.hire import HireBaseline
from calibration_free_moe.baselines.prox import (
    ProxBaseline,
    ProxRecipe,
    ProxyScore,
    quantize_proxy,
)
from calibration_free_moe.model_config import (
    CALIBRATION_FREE_POLICIES,
    checkpoint_identity,
    validate_prox_keep_options,
)
from calibration_free_moe.reader.budget import (
    BudgetPlan,
    compile_budget,
    load_layer_budgets,
)
from calibration_free_moe.reader.progressive import (
    ProgressiveReader,
    ReaderWorkspace,
    SelectionScore,
    channel_major_down,
)
from calibration_free_moe.reader.sketch import require_sketch_acquisition

logger = init_logger(f"vllm.{__name__}")

# Recovery-map (``bank``) formats the loader accepts; each stores an encoder and a
# decoder.
RECOVERY_BANK_FORMATS = (
    "pca-joint-recovery-kl-v1",
    "native-plan-recovery-v1",
    "weight-subspace-recovery-v1",
)
# This worker's prepared operators, one per routed layer, for the work-report RPCs.
BASELINE_OPERATORS: list[ActivationBaseline | HireBaseline | ProxBaseline] = []
PREPARED_READERS: list[ProgressiveReader] = []


def reader_work_report(_worker: object) -> dict[str, object]:
    """Sum the reader's read cost and reference bytes over this worker's layers.

    Worker RPC, called once after evaluation. Read cost (``charged_bytes``) is the
    bytes the method must read, priced in 32-byte sectors of the accessed layouts;
    it is an analytical count, not DRAM traffic.
    """
    parts = [reader.report() for reader in PREPARED_READERS]
    charged = sum(float(part["charged_bytes"] or 0) for part in parts)
    logical = sum(float(part["logical_charged_bytes"] or 0) for part in parts)
    reference = sum(float(part["reference_bytes"] or 0) for part in parts)
    result: dict[str, object] = {
        "charged_bytes": charged,
        "logical_charged_bytes": logical,
        "reference_bytes": reference,
        "read_fraction": charged / reference if reference else None,
        "logical_read_fraction": logical / reference if reference else None,
        "accounting_basis": "analytical_sector_estimate",
        "actual_traffic_measured": False,
    }
    for key in (
        "native_expert_coefficient_bytes",
        "physical_native_expert_coefficient_bytes",
        "recovery_bytes",
        "physical_recovery_bytes",
        "selection_parameter_bytes",
        "physical_selection_parameter_bytes",
        "sketch_bytes",
        "physical_sketch_bytes",
        "sketch_resident_bytes",
        "selection_threshold_resident_bytes",
        "rotation_sign_bytes",
        "rotation_sign_resident_bytes",
        "rotation_input_io_bytes_per_token",
        "rotation_arithmetic_per_token",
        "grouped_rotation_input_io_bytes_per_token",
        "grouped_rotation_arithmetic_per_token",
        "screen_ranking_compare_exchanges_per_token",
        "screen_mask_arithmetic_per_token",
        "issued_sketch_bytes",
        "issued_gate_up_bytes",
        "issued_down_bytes",
        "issued_recovery_encoder_bytes",
        "grouped_rows",
        "grouped_issued_matrix_macs",
        "workspace_bytes",
        "batch_one_screen_split_io_bytes",
        "batched_screen_split_io_bytes_per_token",
        "batched_screen_split_adds_per_token",
        "grouped_screen_split_io_bytes_per_token",
        "grouped_screen_split_adds_per_token",
        "grouped_screen_union_io_bytes_per_token_upper_bound",
        # Small-batch top-k scratch counters (upper bounds where named).
        "small_batch_radix_scratch_io_bytes_per_token_upper_bound",
        "small_batch_radix_atomic_adds_per_token",
        "small_batch_radix_scratch_sector_bytes_per_token_upper_bound",
        "small_batch_radix_scan_adds_per_token_upper_bound",
        "small_batch_final_selection_score_io_bytes_per_token",
        "matrix_macs_per_token",
    ):
        result[key] = sum(float(part[key] or 0) for part in parts)
    if PREPARED_READERS and PREPARED_READERS[0].plan.screen_coordinates < 128:
        reader = PREPARED_READERS[0]
        result["screen_coordinates_per_block"] = reader.plan.screen_coordinates
        # Both screen paths read the same packed coordinate-major sketch.
        result["priced_sketch_layout"] = "coordinate_major"
        result["executed_sketch_layout"] = "coordinate_major"
        result["sketch_read_accounting"] = (
            "analytical batch-one sector count summed over tokens; cache reuse is not "
            "assumed (issued_sketch_bytes: executed coordinate-major requests, "
            "including grouped unions)"
        )
        # The recovery-aware score reads the residual variances.
        if reader.selection_score != "point":
            result["screen_residual_scope"] = (
                "kept coordinates only; dropped coordinates' contribution is "
                "not modelled"
            )
    return result


def reset_reader_work(_worker: object) -> None:
    """Zero the readers' work counters, excluding vLLM profiling and warmup."""
    for reader in PREPARED_READERS:
        reader.read_work.zero_()


def baseline_work_report(_worker: object) -> dict[str, object]:
    """Sum the baselines' coefficient and byte counters over this worker's layers.

    Worker RPC that synchronizes with the GPU once, after evaluation. Operators
    that do not price sectors count two bytes per BF16 coefficient read.
    """
    from calibration_free_moe.analysis.allocation import STATE

    if STATE.context is not None and not STATE.context.started:
        raise RuntimeError("The allocation was never started (no depth or requests)")
    parts = [operator.report() for operator in BASELINE_OPERATORS]
    gate = sum(float(cast(float, part["gate_coefficients"])) for part in parts)
    up = sum(float(cast(float, part["up_coefficients"])) for part in parts)
    down = sum(float(cast(float, part["down_coefficients"])) for part in parts)
    auxiliary = sum(
        float(cast(float, part.get("auxiliary_bf16_equivalent_reads", 0)))
        for part in parts
    )
    reference = sum(
        float(cast(float, part["routed_reference_coefficients"])) for part in parts
    )
    charged_bytes = sum(
        float(
            cast(
                float,
                part.get(
                    "charged_bytes",
                    2
                    * (
                        float(cast(float, part["gate_coefficients"]))
                        + float(cast(float, part["up_coefficients"]))
                        + float(cast(float, part["down_coefficients"]))
                        + float(
                            cast(float, part.get("auxiliary_bf16_equivalent_reads", 0))
                        )
                    ),
                ),
            )
        )
        for part in parts
    )
    reference_bytes = sum(
        float(
            cast(
                float,
                part.get(
                    "reference_bytes",
                    2 * float(cast(float, part["routed_reference_coefficients"])),
                ),
            )
        )
        for part in parts
    )
    basis = str(parts[0].get("accounting_basis", "logical"))
    result: dict[str, object] = {
        "gate_coefficients": gate,
        "up_coefficients": up,
        "down_coefficients": down,
        "auxiliary_bf16_equivalent_reads": auxiliary,
        "routed_reference_coefficients": reference,
        "read_fraction": charged_bytes / reference_bytes if reference_bytes else None,
        "charged_bytes": charged_bytes,
        # Sector-priced operators (Prox, HiRE) report their logical total too.
        "logical_charged_bytes": (
            sum(float(cast(float, part["logical_charged_bytes"])) for part in parts)
            if "logical_charged_bytes" in parts[0]
            else charged_bytes
        ),
        "accounting_basis": basis,
        "actual_traffic_measured": False,
        "reference_bytes": reference_bytes,
        "proxy_bytes": sum(
            float(cast(float, part.get("proxy_bytes", 0))) for part in parts
        ),
        "scale_bytes": sum(
            float(cast(float, part.get("scale_bytes", 0))) for part in parts
        ),
        "resident_calibration_bytes": sum(
            int(cast(int, part["resident_calibration_bytes_per_layer"]))
            for part in parts
        ),
        "layers": parts,
    }
    # Prox's layout fields; other operators need not report them.
    for key in (
        "row_major_sector_bytes",
        "coordinate_major_sector_bytes",
        "selection_parameter_bytes",
        # Allocation study: parameter bytes and read cost per token, summed over
        # layers and EP ranks. Per-layer count residuals stay in each layer's
        # "allocation" entry.
        "allocation_parameter_bytes",
        "allocation_charged_bytes_per_token",
        "allocation_reference_bytes_per_token",
    ):
        if key in parts[0]:
            result[key] = sum(float(cast(float, part[key])) for part in parts)
    for key in ("resident_norm_bytes",):
        if f"{key}_per_layer" in parts[0]:
            result[key] = sum(
                int(cast(int, part[f"{key}_per_layer"])) for part in parts
            )
    for key in (
        "priced_proxy_layout",
        "executed_proxy_layout",
        "native_execution",
        "proxy_score",
        "allocation_source",
    ):
        if key in parts[0]:
            result[key] = parts[0][key]
    return result


def reset_baseline_work(_worker: object) -> None:
    """Zero the baselines' work counters, excluding vLLM profiling and warmup."""
    for operator in BASELINE_OPERATORS:
        operator.work.zero_()
        if isinstance(operator, ProxBaseline):
            operator.allocation_work.zero_()


if TYPE_CHECKING:
    from vllm.lora.model_manager import LoRAModelManager


class PreparedModel(Protocol):
    """Fields shared by the routed model overrides at the post-load hook."""

    vllm_config: VllmConfig
    options: dict[str, object]

    def modules(self) -> Iterator[nn.Module]:
        """Iterate over every submodule, as ``nn.Module.modules`` does."""
        ...


def reader_rows(
    reader: ProgressiveReader | ActivationBaseline | HireBaseline | ProxBaseline,
    x: Tensor,
    ids: Tensor,
    weights: Tensor,
    padding: Tensor | None,
    rows: int,
    tokens: slice | Tensor | None = None,
) -> Tensor:
    """Run a per-token policy on any token count, in calls of at most ``rows``.

    Each token's output depends only on its own state and routing, so chunking
    changes results only through FP32 reduction order (a one-row tail takes the
    per-token path). The reader is not reentrant; it writes each chunk into its
    rows of one output, in order on the caller's stream. ``tokens`` (the rows'
    forward indices) are chunked alike for a Prox operator with an allocation.
    """

    def forward_rows(start: int, end: int) -> slice | Tensor | None:
        if isinstance(tokens, slice):
            first = tokens.start or 0
            return slice(first + start, first + min(end, x.shape[0]))
        return None if tokens is None else tokens[start:end]

    def call(*chunk: Any, start: int, end: int) -> Tensor:
        if isinstance(reader, ProxBaseline) and tokens is not None:
            return reader(*chunk, tokens=forward_rows(start, end))
        return reader(*chunk)

    if x.shape[0] <= rows:
        return call(x, ids, weights, padding, start=0, end=x.shape[0])
    output = torch.empty_like(x)
    for start in range(0, x.shape[0], rows):
        end = start + rows
        chunk = (
            x[start:end],
            ids[start:end],
            weights[start:end],
            padding[start:end] if padding is not None else None,
        )
        if isinstance(reader, ProgressiveReader):
            reader(*chunk, out=output[start:end])
        else:
            output[start:end] = call(*chunk, start=start, end=end)
    return output


class ProgressiveExperts(RoutedExperts):
    """Native routed experts that hand policy rows to a prepared sparse operator.

    The operator is bound after weight loading (``prepare_processed_weights``).
    Stock rows, and every row of a dense calibration worker, run vLLM's own
    modular fused-MoE path on the same weights.
    """

    calibration_dense: bool = False
    # Evaluation only: apply the policy to prefill rows too (every token sparse).
    sparse_prefill: bool = False
    reader: (
        ProgressiveReader | ActivationBaseline | HireBaseline | ProxBaseline | None
    ) = None
    # Rows per policy call: the prepared decode (or sparse-prefill) capacity.
    max_policy_rows: int = 0
    reader_input_prepared: bool = False
    # Branch ops (Gemma, single-rank Qwen3.5) prepare right after enqueuing
    # their router projection, not at routing.
    caller_prepares: bool = False

    def prepare_reader_input(self, _gate: nn.Module, args: tuple[Any, ...]) -> None:
        """Start the reader's routing-independent input work before routing.

        Called by the gate pre-hook, the routing wrapper or a branch op, so the
        work overlaps the router. Only the reader without expert parallelism, on a
        batch whose rows all take the policy in one call, prepares early; other
        batches prepare inside the reader call.
        """
        self.reader_input_prepared = False
        reader = self.reader
        if not isinstance(reader, ProgressiveReader) or reader.expert_map is not None:
            return
        x = cast(Tensor, args[0])
        context = get_forward_context()
        # Phase-metadata patch fields (see the module docstring).
        count = x.shape[0] if self.sparse_prefill else context.num_decode_tokens  # pyright: ignore[reportAttributeAccessIssue]
        # Mixed batches, generated-prefill gathers and chunked calls prepare
        # inside the reader call instead.
        if (
            context.generated_prefill_split is not None  # pyright: ignore[reportAttributeAccessIssue]
            or count != x.shape[0]
            or count == 0
            or count > self.max_policy_rows
        ):
            return
        self.reader_input_prepared = reader.prepare_input(x, context.is_padding)

    def forward_modular(
        self,
        x: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: Tensor | None = None,
    ) -> Tensor:
        """Run policy rows through the sparse operator and the rest through stock.

        The phase-metadata patch orders decode rows first, so policy rows are a
        prefix ``[0, count)`` unless a prefill recomputes generated tokens, in
        which case ``generated_prefill_split`` lists both row sets.
        """
        context = get_forward_context()
        # Phase-metadata patch fields (see the module docstring).
        count = context.num_decode_tokens  # pyright: ignore[reportAttributeAccessIssue]
        if count is None:
            raise RuntimeError("The vLLM phase-metadata patch must be enabled")
        # A prefill that recomputes generated tokens (after preemption) gives
        # them the policy, as in decode; its prompt rows stay stock.
        split = context.generated_prefill_split  # pyright: ignore[reportAttributeAccessIssue]
        if self.sparse_prefill:
            # Likelihood evaluation: every row takes the policy, including the
            # memory-profiling batch, so vLLM sizes the KV cache after the
            # baselines' prefill-sized transients. Padding is masked as in decode.
            count, split = x.shape[0], None
        if split is not None:
            count = split[0].numel()
        # Full-decode graphs specialize on their padded capture size; eager
        # steps use the real policy-row count, including generated prefill rows.
        if count == 0 and split is None:
            return super().forward_modular(
                x, topk_weights, topk_ids, shared_experts, shared_experts_input
            )
        if self.reader is None:
            if self.calibration_dense:
                return super().forward_modular(
                    x, topk_weights, topk_ids, shared_experts, shared_experts_input
                )
            raise RuntimeError("Sparse experts were not prepared by the weight loader")
        if split is None:
            sparse = self.policy(slice(0, count), x, topk_weights, topk_ids)
            if count == x.shape[0]:
                return sparse
            prefill = super().forward_modular(
                x[count:], topk_weights[count:], topk_ids[count:]
            )
            return torch.cat((sparse, prefill), dim=0)
        policy_rows, stock_rows = split
        output = torch.empty_like(x)
        output[policy_rows] = self.policy(policy_rows, x, topk_weights, topk_ids)
        if stock_rows.numel():
            output[stock_rows] = super().forward_modular(
                x[stock_rows], topk_weights[stock_rows], topk_ids[stock_rows]
            )
        return output

    def policy(
        self, rows: slice | Tensor, x: Tensor, topk_weights: Tensor, topk_ids: Tensor
    ) -> Tensor:
        """Run the sparse operator on ``rows``; its read statistics include them."""
        reader = cast(
            ProgressiveReader | ActivationBaseline | HireBaseline | ProxBaseline,
            self.reader,
        )
        padding = get_forward_context().is_padding
        row_padding = padding[rows] if padding is not None else None
        if isinstance(reader, ProgressiveReader) and self.reader_input_prepared:
            self.reader_input_prepared = False
            return reader(
                x[rows],
                topk_ids[rows],
                topk_weights[rows],
                row_padding,
                input_prepared=True,
            )
        return reader_rows(
            reader,
            x[rows],
            topk_ids[rows],
            topk_weights[rows],
            row_padding,
            self.max_policy_rows,
            rows
            if isinstance(reader, ProxBaseline) and reader.allocation is not None
            else None,
        )


_reader_stream: torch.cuda.Stream | None = None


def reader_stream() -> torch.cuda.Stream:
    """Return the one stream for routers and readers beside dense work.

    Callers enqueue the latency-bound router and reader kernels on it before the
    concurrent dense GEMVs, which would otherwise fill every SM's registers first.
    """
    global _reader_stream
    if _reader_stream is None:
        _reader_stream = torch.cuda.Stream()
    return _reader_stream


def prepare_before_routing(runner: MoERunner) -> None:
    """Fork reader input preparation beside routing when the gate is external.

    OLMoE computes router logits before the runner, so the earliest point
    inside the runner is its expert selection on the same hidden rows. Branch ops
    (Gemma, single-rank Qwen3.5) prepare themselves and never reach this hook.
    """
    experts = cast(ProgressiveExperts, runner.routed_experts)
    select = runner.router.select_experts

    def prepare(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        experts.prepare_reader_input(runner, (kwargs["hidden_states"],))
        return select(*args, **kwargs)

    runner.router.select_experts = prepare


def validate_sparse_backend(experts: MoERunner) -> None:
    """Reject any native backend that repacks aliased expert weights on load."""
    if experts.routed_experts.quant_method.is_monolithic:
        raise ValueError("Sparse decode requires a modular native MoE backend")
    method = experts.routed_experts.quant_method
    if (
        not isinstance(method, UnquantizedFusedMoEMethod)
        or method.unquantized_backend != UnquantizedMoeBackend.TRITON
    ):
        raise ValueError(
            "Sparse decode requires the unquantized Triton fused-MoE backend"
        )


def bind_experts(
    mlp: nn.Module, vllm_config: VllmConfig, prefix: str, *, gemma: bool = False
) -> None:
    """Rebuild the routed runner on ``ProgressiveExperts``, sharing native weights."""
    config = vllm_config.model_config.hf_text_config
    qwen = isinstance(mlp, Qwen3MoeSparseMoeBlock)
    width = getattr(config, "moe_intermediate_size", None) or config.intermediate_size
    name = f"{prefix}.experts"
    compilation = vllm_config.compilation_config
    previous = getattr(mlp, "experts", None)
    if previous is not None:
        # OLMoE's native decoder constructor registers a runner first. Qwen3
        # uses a thin decoder adapter and never constructs the native runner.
        if compilation.static_forward_context.get(name) is not previous:
            raise RuntimeError(f"Native routed expert registration is missing: {name}")
        del compilation.static_forward_context[name]
        compilation.static_all_moe_layers.remove(name)
    mlp.experts = FusedMoEFactory(
        num_experts=config.num_experts,
        top_k=config.top_k_experts if gemma else config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=width,
        renormalize=gemma or bool(getattr(config, "norm_topk_prob", False)),
        custom_routing_function=cast(Any, mlp.routing_function) if gemma else None,
        activation="gelu_tanh" if gemma else "silu",
        gate=mlp.gate if qwen else None,
        quant_config=vllm_config.quant_config,
        prefix=name,
        routed_experts_cls=ProgressiveExperts,
    )
    validate_sparse_backend(mlp.experts)
    cast(ProgressiveExperts, mlp.experts.routed_experts).calibration_dense = (
        cast(dict[str, dict[str, object]], vllm_config.additional_config)[
            "tokenwise"
        ].get("policy")
        == "dense-calibration"
    )


class ProgressiveDecoderLayer(OlmoeDecoderLayer):
    """Native OLMoE decoder layer with its routed runner on ``ProgressiveExperts``."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        bind_experts(self.mlp, vllm_config, f"{prefix}.mlp")


class ProgressiveQwenMoE(Qwen3MoeSparseMoeBlock):
    """Qwen3-MoE block that builds its routed runner once, on ProgressiveExperts."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.is_sequence_parallel = False
        self.enable_eplb = False
        self.n_routed_experts = config.num_experts
        self.n_logical_experts = config.num_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = config.num_experts
        self.n_local_physical_experts = config.num_experts
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert_gate = None
        self.shared_expert = None
        bind_experts(self, vllm_config, prefix)
        self.n_local_physical_experts = self.experts.routed_experts.local_num_experts


class ProgressiveQwenDecoderLayer(Qwen3MoeDecoderLayer):
    """Qwen's attention and norms with directly constructed routed experts."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=getattr(
                config, "dual_chunk_attention_config", None
            ),
        )
        self.mlp = ProgressiveQwenMoE(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


def validate_binding(vllm_config: VllmConfig) -> dict[str, object]:
    """Check the engine configuration and policy options; return the options.

    The options are ``additional_config["tokenwise"]``. Unsupported parallelism,
    dtypes, quantization or option combinations fail before any weight loads.
    """
    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_eplb
        or (parallel.tensor_parallel_size > 1 and not parallel.enable_expert_parallel)
        or parallel.use_sequence_parallel_moe
        or vllm_config.lora_config is not None
        or vllm_config.speculative_config is not None
        or vllm_config.model_config.dtype != torch.bfloat16
        or parallel.eplb_config.num_redundant_experts != 0
    ):
        raise ValueError(
            "Sparse binding requires BF16, TP with EP (or TP1), PP/DP1, "
            "no EPLB/sequence parallelism/LoRA/spec"
        )
    if vllm_config.quant_config is not None:
        raise ValueError("Sparse binding requires an unquantized BF16 checkpoint")
    additional = vllm_config.additional_config
    if (
        not isinstance(additional, dict)
        or additional.get("tokenwise_phase_metadata") is not True
    ):
        raise ValueError("Enable tokenwise_phase_metadata in additional_config")
    options = cast(dict[str, object], additional["tokenwise"])
    # Evaluation-only likelihood mode; deployment keeps stock prefill.
    sparse_prefill = options.get("sparse_prefill", False)
    if type(sparse_prefill) is not bool or (
        sparse_prefill and options.get("policy") == "dense-calibration"
    ):
        raise ValueError("sparse_prefill must be a boolean and needs a sparse policy")
    score = options.get("selection_score", "recovery-aware")
    reader_only = options.get("policy", "progressive") == "progressive"
    if options.get("layer_retention") is not None and not reader_only:
        raise ValueError("layer_retention requires the reader")
    thresholds = options.get("selection_thresholds")
    if thresholds is not None and (
        type(thresholds) is not str
        or not reader_only
        or options.get("retention") is None
        or options.get("layer_retention") is not None
    ):
        raise ValueError(
            "selection_thresholds is a calibration file replacing the reader's global "
            "top-F; the uniform retention is each token's cap"
        )
    if score not in get_args(SelectionScore) or (
        score != "recovery-aware" and not reader_only
    ):
        raise ValueError(
            f"selection_score is a reader option: {get_args(SelectionScore)}"
        )
    # A partial-sketch screen may use a recovery map; it reads the masked features.
    keep = options.get("screen_keep", 1.0)
    if not isinstance(keep, (int, float)) or (keep != 1 and not reader_only):
        raise ValueError("screen_keep is a reader fraction")
    validate_prox_keep_options(
        options.get("policy"),
        options.get("input_keep"),
        options.get("channel_keep"),
        options.get("channel_floor"),
        options.get("prox_recipe", "legacy"),
        options.get("proxy_score", "prox"),
    )
    validate_allocation_options(options)
    if (
        options.get("policy") == "prox"
        and options.get("allocation") is None
        and (options.get("retention") is not None or options.get("baseline") is None)
    ):
        raise ValueError("prox requires baseline thresholds, not retention")
    if (
        options.get("bank") is not None
        and options.get("policy", "progressive") != "progressive"
    ):
        raise ValueError("Only the reader takes a recovery bank")
    hf = vllm_config.model_config.hf_text_config
    if (
        getattr(hf, "shared_expert_intermediate_size", 0)
        and hf.model_type != "qwen3_5_moe_text"
    ) or getattr(hf, "mlp_only_layers", []):
        raise ValueError("Shared/dense-only routed expert layers are not supported")
    return options


def prepare_weights(
    model: ProgressiveOlmoeForCausalLM | ProgressiveQwen3MoeForCausalLM,
    weights: Iterable[tuple[str, Tensor]],
) -> set[str]:
    """Load checkpoint weights with the native model's loader."""
    if isinstance(model, ProgressiveOlmoeForCausalLM):
        loaded = OlmoeForCausalLM.load_weights(model, weights)
    else:
        loaded = Qwen3MoeForCausalLM.load_weights(model, weights)
    return loaded


def load_recovery_bank(
    model: PreparedModel, top_k: int, formats: tuple[str, ...] = RECOVERY_BANK_FORMATS
) -> tuple[Tensor, Tensor] | None:
    """Load and validate the configured recovery map as (encoder, decoder).

    Returns None when no map is configured (recovery disabled). The encoder is
    ``[layers, experts, 32, I]`` and the decoder ``[layers, H, 32]``; the map must
    name this checkpoint and revision. ``formats`` restricts the accepted formats.
    """
    if model.options.get("bank") is None:
        return None
    config = model.vllm_config.model_config
    hf = config.hf_text_config
    bank = cast(
        dict[str, object],
        torch.load(str(model.options["bank"]), map_location="cpu", weights_only=True),
    )
    identity = cast(dict[str, str], bank["model"])
    expected_identity = checkpoint_identity(Path(config.model))
    revision = (
        config.revision or getattr(hf, "_commit_hash", None) or Path(config.model).name
    )
    decoder, encoder = cast(Tensor, bank["decoder"]), cast(Tensor, bank["encoder"])
    if (
        bank["format"] not in formats
        or identity["revision"] != revision
        or identity.get("id", "allenai/OLMoE-1B-7B-0924") != expected_identity["id"]
        or bank.get(
            "dimensions",
            {
                "layers": decoder.shape[0],
                "experts": encoder.shape[1],
                "top_k": top_k,
                "hidden": decoder.shape[1],
                "expert_width": encoder.shape[-1],
            },
        )
        != {
            "layers": hf.num_hidden_layers,
            "experts": hf.num_experts,
            "top_k": top_k,
            "hidden": hf.hidden_size,
            "expert_width": getattr(hf, "moe_intermediate_size", None)
            or hf.intermediate_size,
        }
        # Maps fitted on expert-pruned checkpoints do not fit an unpruned model.
        or "pruned_from" in bank
        or "kept_experts" in bank
        or bank["layers"] != list(range(hf.num_hidden_layers))
        or decoder.shape != (hf.num_hidden_layers, hf.hidden_size, 32)
        or encoder.shape
        != (
            hf.num_hidden_layers,
            hf.num_experts,
            32,
            getattr(hf, "moe_intermediate_size", None) or hf.intermediate_size,
        )
    ):
        raise ValueError("Recovery bank does not match the loaded model/revision")
    require_sketch_acquisition(bank)
    # A map fitted for one budget needs the same selected-channel count, screen
    # width and score. The reader computes its selected channels exactly on every
    # layer, so a fit with a separate exact-channel count (key ``completion``) or
    # dense layers does not match.
    fitted = cast(dict[str, object], bank.get("plan", {}))
    if bank["format"] == "native-plan-recovery-v1" and (
        fitted.get("retention") != model.options.get("retention")
        or fitted.get("completion") is not None
        or bank.get("dense_layers", 0) != 0
        or ceil(128 * cast(float, fitted.get("screen_keep", 1.0)))
        != ceil(128 * cast(float, model.options.get("screen_keep", 1.0)))
        or fitted.get("selection_score", "recovery-aware")
        != model.options.get("selection_score", "recovery-aware")
    ):
        raise ValueError(f"Recovery bank was fitted for plan {fitted}")
    return encoder, decoder


def load_selection_thresholds(model: PreparedModel, plan: BudgetPlan) -> list[float]:
    """Load per-layer screen-score thresholds fitted for this reader configuration.

    The file (scripts/calibrate_selection_thresholds.py) must name the model
    revision, score, screen, bank and per-token cap F it was fitted under.
    """
    config = model.vllm_config.model_config
    hf = config.hf_text_config
    fitted = cast(
        dict[str, object],
        json.loads(Path(str(model.options["selection_thresholds"])).read_text()),
    )
    thresholds = cast(list[float], fitted["thresholds"])
    revision = (
        config.revision or getattr(hf, "_commit_hash", None) or Path(config.model).name
    )
    if (
        fitted["format"] != "selection-thresholds-v1"
        or fitted["model_revision"] != revision
        or fitted["selection_score"]
        != model.options.get("selection_score", "recovery-aware")
        or fitted["screen_coordinates"] != plan.screen_coordinates
        or fitted["cap"] != plan.final_count
        or fitted["bank"] != model.options.get("bank")
        or len(thresholds) != hf.num_hidden_layers
        or any(type(t) is not float or not 0 <= t < float("inf") for t in thresholds)
    ):
        raise ValueError("Selection thresholds do not match the reader configuration")
    return thresholds


@torch.no_grad()
def prepare_processed_weights(model: PreparedModel) -> None:
    """Re-lay the down projections and bind each routed layer's decode operator.

    vLLM calls this model-level hook after every routed quant method has finalized
    its weights and scales, and before warmup and CUDA graph capture. Only the
    routed layers change: W2 is re-laid channel-major in place, then each layer
    gets the configured reader or baseline. Stock and dense-only layers are
    untouched.
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    allocated_before = torch.cuda.memory_allocated()
    scheduler = model.vllm_config.scheduler_config
    capacity = max(
        scheduler.max_num_seqs,
        model.vllm_config.compilation_config.max_cudagraph_capture_size or 0,
    )
    if model.options.get("sparse_prefill") is True:
        # Evaluation scores whole prefills: one scheduler step is one reader call.
        capacity = max(capacity, scheduler.max_num_batched_tokens)
    for module in model.modules():
        if isinstance(module, ProgressiveExperts):
            module.sparse_prefill = model.options.get("sparse_prefill") is True
            # Recomputed generated rows run in chunks of at most one decode step.
            module.max_policy_rows = capacity
            original = cast(Tensor, module.w2_weight)
            physical = channel_major_down(original)
            # Keep vLLM's parameter and its EP/weight-loader attributes; only
            # replace its storage, before any CUDA graph captures its address.
            module.w2_weight.data = physical
            del original, physical
    config = model.vllm_config.model_config
    hf = config.hf_text_config
    activation = model_activation(hf)
    top_k = cast(int, getattr(hf, "num_experts_per_tok", None) or hf.top_k_experts)
    policy = str(model.options.get("policy", "progressive"))
    if policy == "dense-calibration":
        # Calibration workers bind their own operators after loading.
        return
    if policy != "progressive":
        if policy not in {
            "cats",
            "teal",
            "teal-greedy",
            "wina",
            "wina-greedy-topk",
            "prox",
            *CALIBRATION_FREE_POLICIES,
        }:
            raise ValueError(f"Unknown decode policy: {policy}")
        fraction = model.options.get("retention")
        # Top-K configurations reuse their thresholded policy's scores.
        base = policy if policy == "wina-greedy-topk" else policy.removesuffix("-topk")
        if policy not in CALIBRATION_FREE_POLICIES:
            fraction = None
        elif not isinstance(fraction, float) or not 0 < fraction <= 1:
            raise ValueError("Top-K baselines need retention in (0,1]")
        # The exact allocation score has no proxy; its thresholds are its own.
        exact = model.options.get("proxy_score") == "exact"
        if policy in CALIBRATION_FREE_POLICIES or exact:
            # No thresholds or data: every layer keeps the same fixed fraction.
            calibration: dict[str, object] = {"sparsity": 0.0}
            layers: list[dict[str, object]] = [{}] * hf.num_hidden_layers
        else:
            calibration = cast(
                dict[str, object],
                torch.load(
                    str(model.options["baseline"]),
                    map_location="cpu",
                    weights_only=True,
                ),
            )
            layers = cast(list[dict[str, object]], calibration["thresholds"])
            revision = (
                config.revision
                or getattr(hf, "_commit_hash", None)
                or Path(config.model).name
            )
            if (
                calibration["format"]
                != {"prox": "prox-v1"}.get(base, "activation-baselines-v1")
                or calibration["policy"] != base
                or calibration["model_revision"] != revision
                or len(layers) != hf.num_hidden_layers
            ):
                raise ValueError("Baseline calibration does not match policy/model")
            if base == "prox":
                if calibration.get("proxy_score", "prox") != model.options.get(
                    "proxy_score", "prox"
                ):
                    raise ValueError(
                        "Prox threshold score does not match the evaluation score"
                    )
                if calibration.get("recipe", "legacy") != model.options.get(
                    "prox_recipe", "legacy"
                ):
                    raise ValueError(
                        "Prox threshold recipe does not match the evaluation recipe"
                    )
                rules = {
                    key: model.options.get(key)
                    for key in ("input_keep", "channel_keep", "channel_floor")
                }
                fitted_rules = calibration.get("keep_rules", dict.fromkeys(rules))
                if fitted_rules != rules:
                    raise ValueError(
                        "Prox thresholds were fitted with different keep rules"
                    )
        BASELINE_OPERATORS.clear()
        # Only method workers import this binding. A string RPC avoids the
        # insecure callable-serialization fallback disabled by vLLM.
        from vllm.v1.worker.gpu_worker import Worker

        Worker.tokenwise_baseline_work_report = baseline_work_report  # pyright: ignore[reportAttributeAccessIssue]
        Worker.tokenwise_baseline_reset_work = reset_baseline_work  # pyright: ignore[reportAttributeAccessIssue]
        allocations = None
        if model.options.get("allocation") is not None:
            allocations = prepare_allocations(
                model.options,
                layers=hf.num_hidden_layers,
                candidates=top_k
                * (getattr(hf, "moe_intermediate_size", None) or hf.intermediate_size),
                max_tokens=max(capacity, scheduler.max_num_batched_tokens),
                max_requests=scheduler.max_num_seqs,
                device=torch.device("cuda", torch.cuda.current_device()),
                thresholds=None
                if exact
                else [float(cast(float, f["channel_threshold"])) for f in layers],
            )
            Worker.tokenwise_allocation_start = allocation_start  # pyright: ignore[reportAttributeAccessIssue]
            Worker.tokenwise_allocation_collect = allocation_collect  # pyright: ignore[reportAttributeAccessIssue]
            Worker.tokenwise_allocation_finish = allocation_finish  # pyright: ignore[reportAttributeAccessIssue]
        for module in model.modules():
            if isinstance(module, ProgressiveExperts):
                layer = extract_layer_index(module.layer_name)
                fitted = layers[layer]
                expert_map = module.expert_map
                if (
                    expert_map is not None
                    and module.expert_placement_strategy != "linear"
                ):
                    raise ValueError("Sparse decode requires linear expert placement")
                if policy == "hire-gate":
                    if expert_map is not None:
                        raise ValueError("The global top-F screen needs TP1 experts")
                    assert fraction is not None
                    module.reader = HireBaseline(
                        cast(Tensor, module.w13_weight),
                        cast(Tensor, module.w2_weight),
                        fraction,
                        activation=activation,
                    )
                    BASELINE_OPERATORS.append(module.reader)
                    continue
                if base in {"prox", "prox-free"}:
                    free = policy == "prox-free"
                    channel_fraction = cast(
                        float | None, model.options.get("channel_keep")
                    )
                    floor_fraction = cast(
                        float | None, model.options.get("channel_floor")
                    )
                    allocation = allocations[layer] if allocations is not None else None
                    joint = (
                        free
                        or channel_fraction is not None
                        or floor_fraction is not None
                        or allocation is not None
                    )
                    # Deterministic row-wise INT4 of this rank's native experts;
                    # identical to the calibration worker's proxy.
                    packed, scales = (
                        (torch.empty(0), torch.empty(0))
                        if exact
                        else quantize_proxy(cast(Tensor, module.w13_weight))
                    )
                    width = cast(Tensor, module.w2_weight).shape[2]
                    # Keep ceil(f n) of n candidates; the 1e-9 absorbs float error
                    # in f n. prox-free ranks each token's H inputs and K*I channels.
                    input_keep = cast(float | None, model.options.get("input_keep"))
                    keep_fraction = (
                        channel_fraction if channel_fraction is not None else fraction
                    )
                    module.reader = ProxBaseline(
                        cast(Tensor, module.w13_weight),
                        cast(Tensor, module.w2_weight),
                        packed,
                        scales,
                        None
                        if input_keep is not None or exact
                        else float(cast(float, fitted["input_threshold"])),
                        allocation.threshold
                        if allocation is not None
                        else None
                        if keep_fraction is not None
                        else float(cast(float, fitted["channel_threshold"])),
                        activation,
                        expert_map,
                        channel_keep=None
                        if keep_fraction is None
                        else max(
                            1,
                            ceil(
                                keep_fraction * (top_k if joint else 1) * width - 1e-9
                            ),
                        ),
                        input_keep=None
                        if input_keep is None
                        else max(1, ceil(input_keep * hf.hidden_size - 1e-9)),
                        channel_scope="token"
                        if joint or input_keep is not None
                        else "slot",
                        channel_floor=0
                        if floor_fraction is None
                        else max(1, ceil(floor_fraction * top_k * width - 1e-9)),
                        recipe=cast(
                            ProxRecipe, model.options.get("prox_recipe", "legacy")
                        ),
                        proxy_score=cast(
                            ProxyScore, model.options.get("proxy_score", "prox")
                        ),
                        allocation=allocation,
                    )
                    BASELINE_OPERATORS.append(module.reader)
                    continue
                gu = cast(Tensor, module.w13_weight)
                down = cast(Tensor, module.w2_weight)
                norms = cast(
                    tuple[Tensor | None, Tensor | None, Tensor | None],
                    (
                        (
                            gu[:, : down.shape[2]]
                            .float()
                            .norm(dim=1)
                            .to(torch.bfloat16),
                            gu[:, down.shape[2] :]
                            .float()
                            .norm(dim=1)
                            .to(torch.bfloat16)
                            if base == "wina"
                            else None,
                            down.float().norm(dim=1).to(torch.bfloat16)
                            if base == "wina"
                            else None,
                        )
                        if base in {"wina", "wina-greedy-topk"}
                        else (None, None, None)
                    ),
                )
                module.reader = ActivationBaseline(
                    gu,
                    down,
                    base,
                    cast(float, fitted.get("threshold", -1.0)),
                    input_threshold=float(
                        cast(float, fitted.get("input_threshold", -1.0))
                    ),
                    down_threshold=float(
                        cast(float, fitted.get("down_threshold", -1.0))
                    ),
                    gate_threshold=float(
                        cast(
                            float,
                            fitted.get(
                                "gate_threshold",
                                fitted.get("input_threshold", -1.0),
                            ),
                        )
                    ),
                    sparsity=float(cast(float, calibration["sparsity"])),
                    gate_norm=norms[0],
                    up_norm=norms[1],
                    down_norm=norms[2],
                    expert_map=expert_map,
                    activation=activation,
                    top_fraction=fraction,
                    projection_sparsities=(
                        cast(float, fitted["gate_sparsity"]),
                        cast(float, fitted["up_sparsity"]),
                        cast(float, fitted["down_sparsity"]),
                    )
                    if policy == "wina-greedy-topk"
                    else None,
                )
                BASELINE_OPERATORS.append(module.reader)
        logger.info(
            "Prepared %s baseline decode operators, calibration=%s",
            policy,
            model.options["baseline"],
        )
        return
    bank = load_recovery_bank(
        model,
        top_k,
        formats=("weight-subspace-recovery-v1",)
        if model.options.get("layer_retention") is not None
        else RECOVERY_BANK_FORMATS,
    )
    plans = (
        load_layer_budgets(
            str(model.options["layer_retention"]),
            layers=hf.num_hidden_layers,
            hidden_size=hf.hidden_size,
            expert_width=getattr(hf, "moe_intermediate_size", None)
            or hf.intermediate_size,
            top_k=top_k,
            screen_keep=float(cast(float, model.options.get("screen_keep", 1.0))),
        )
        if model.options.get("layer_retention") is not None
        else [
            compile_budget(
                hidden_size=hf.hidden_size,
                expert_width=getattr(hf, "moe_intermediate_size", None)
                or hf.intermediate_size,
                top_k=top_k,
                retention=float(cast(float, model.options["retention"])),
                screen_keep=float(cast(float, model.options.get("screen_keep", 1.0))),
            )
        ]
        * hf.num_hidden_layers
    )
    thresholds = (
        load_selection_thresholds(model, plans[0])
        if model.options.get("selection_thresholds") is not None
        else None
    )
    workspace_plan = max(plans, key=lambda p: p.final_count)
    # Layers run sequentially on one stream, so they share one transient scratch.
    workspace: ReaderWorkspace | None = None
    PREPARED_READERS.clear()
    from vllm.v1.worker.gpu_worker import Worker

    Worker.tokenwise_reader_work_report = reader_work_report  # pyright: ignore[reportAttributeAccessIssue]
    Worker.tokenwise_reader_reset_work = reset_reader_work  # pyright: ignore[reportAttributeAccessIssue]
    # Preparation time and allocation include the W2 re-layout above.
    preparation = {
        "sketch_resident_bytes": 0,
        "recovery_geometry_bytes": 0,
        "recovery_geometry_preparation_products": 0,
        "recovery_geometry_preparation_reduction_adds": 0,
        "workspace_bytes": 0,
        "recovery_response_preparation_macs": 0,
    }
    for module in model.modules():
        if isinstance(module, ProgressiveExperts):
            layer = extract_layer_index(module.layer_name)
            plan = plans[layer]
            expert_map = module.expert_map
            if expert_map is not None and module.expert_placement_strategy != "linear":
                raise ValueError("Sparse decode requires linear expert placement")
            gate_up = cast(Tensor, module.w13_weight)
            if workspace is None:
                workspace = ReaderWorkspace(
                    workspace_plan,
                    gate_up.shape[0],
                    bank[1].shape[-1] if bank is not None else 0,
                    capacity,
                    gate_up.device,
                    radix_select=expert_map is None and thresholds is None,
                )
            encoder = decoder = None
            if bank is not None:
                encoder = (
                    bank[0][layer, torch.where(expert_map.cpu() >= 0)[0]]
                    if expert_map is not None
                    else bank[0][layer]
                )
                decoder = bank[1][layer]
            module.reader = ProgressiveReader(
                gate_up,
                cast(Tensor, module.w2_weight),
                encoder,
                decoder,
                plan,
                capacity,
                expert_map,
                activation=activation,
                workspace=workspace,
                selection_score=cast(
                    SelectionScore,
                    model.options.get("selection_score", "recovery-aware"),
                ),
                selection_threshold=thresholds[layer] if thresholds else None,
            )
            PREPARED_READERS.append(module.reader)
            for name in preparation:
                preparation[name] += getattr(module.reader, name)
    # Qwen3 and multi-rank Qwen3.5 pass the hidden state to the runner, which
    # computes gate/top-k there; Gemma, OLMoE and single-rank Qwen3.5 route
    # precomputed logits inside it.
    # Register after preparation and before compilation/capture.
    for module in model.modules():
        if not isinstance(module, MoERunner):
            continue
        experts = module.routed_experts
        if (
            isinstance(experts, ProgressiveExperts)
            and isinstance(experts.reader, ProgressiveReader)
            and experts.reader.expert_map is None
        ):
            gate = module.gate
            if isinstance(gate, nn.Module):
                gate.register_forward_pre_hook(experts.prepare_reader_input)
            elif not experts.caller_prepares:
                prepare_before_routing(module)
    torch.cuda.synchronize()
    logger.info(
        "Reader preparation including the down-weight re-layout: seconds=%.6f "
        "allocated_delta_bytes=%d "
        "worker_peak_allocated_bytes=%d resources=%s recovery_bank=%s",
        time.perf_counter() - start,
        torch.cuda.memory_allocated() - allocated_before,
        torch.cuda.max_memory_allocated(),
        preparation,
        model.options.get("bank"),
    )


class ProgressiveOlmoeForCausalLM(OlmoeForCausalLM):
    """OLMoE with sparse decode operators on its routed experts."""

    lora_manager: LoRAModelManager | None = None
    options: dict[str, object]
    vllm_config: VllmConfig

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.options = validate_binding(vllm_config)
        self.vllm_config = vllm_config
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, layer_type=ProgressiveDecoderLayer
        )

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load checkpoint weights with the native loader."""
        return prepare_weights(self, weights)

    def process_weights_after_loading(self) -> None:
        """Bind the sparse decode operators once the native weights are final."""
        prepare_processed_weights(self)


class ProgressiveQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """Qwen3-MoE with sparse decode operators on its routed experts.

    The constructor mirrors the native one but builds the decoder layers with
    ``ProgressiveQwenDecoderLayer``, so the routed weights are allocated once.
    """

    lora_manager: LoRAModelManager | None = None
    options: dict[str, object]
    vllm_config: VllmConfig

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.options = validate_binding(vllm_config)
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = Qwen3MoeModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            decoder_layer_type=ProgressiveQwenDecoderLayer,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        moe_blocks = [
            cast(Qwen3MoeSparseMoeBlock, layer.mlp)
            for layer in self.model.layers
            if not isinstance(layer, PPMissingLayer)
        ]
        self.moe_layers = [block.experts for block in moe_blocks]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        example = moe_blocks[0]
        self.num_logical_experts = example.n_logical_experts
        self.num_physical_experts = example.n_physical_experts
        self.num_local_physical_experts = example.n_local_physical_experts
        self.num_routed_experts = example.n_routed_experts
        self.num_redundant_experts = example.n_redundant_experts

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load checkpoint weights with the native loader."""
        return prepare_weights(self, weights)

    def process_weights_after_loading(self) -> None:
        """Bind the sparse decode operators once the native weights are final."""
        prepare_processed_weights(self)
