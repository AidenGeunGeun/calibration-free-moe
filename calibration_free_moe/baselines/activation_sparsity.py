"""Activation-sparsity baselines for routed experts: CATS, TEAL and WINA.

Each policy follows its paper, restricted to the routed expert MLPs:

- CATS: Lee et al. (2024, arXiv:2404.08763), Sec. 4.1-4.3. A calibrated threshold
  on the gate activation magnitude masks the up and down projections.
- TEAL: Liu et al. (2024, arXiv:2408.14690), Sec. 4.2, with uniform MLP-only
  sparsity (no block-wise greedy allocation and no attention sparsity). Calibrated
  thresholds on |x| mask the gate/up inputs, and one on the down input's magnitude
  masks the down input.
- TEAL-greedy: Liu et al., Sec. 4.3 (Algorithm 1) and the official
  ``teal/greedyopt.py``: per-projection greedy output-error allocation.
- WINA: Chen et al. (2025, arXiv:2505.19427), Eq. (4): |input| times the
  weight-column L2 norm, with the uniform MLP allocation that Sec. 3.2 permits (no
  attention or weight rotation).
- WINA-greedy-topk: the official gate-only norm scoring, threshold-based greedy
  allocation and per-token top-K with truncated counts.

Calibration-free top-K variants (``top_fraction`` f) replace the thresholds by
fixed counts, uniform across layers and rounded to nearest-even. CATS keeps the fI
largest gate activation magnitudes per routed slot, with the same dense gate read.
TEAL and WINA keep their scores but take the fH largest inputs per token (TEAL) or
per routed slot (WINA) for gate and up, and the fI largest down inputs per slot
(WINA's own Sec. 3.2 top-K form; TEAL's per-token magnitude order).

Grouped expert GEMMs implement the operators; this is not a speed comparison with
the published kernels. Counters record the native coefficients each policy
selects, at two bytes per BF16 coefficient; the dense GEMM tiles read more.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import triton.language as tl
from torch import Tensor
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    try_get_optimal_moe_config,
)

SMALL_TILE_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
}


def local_ids(ids: Tensor, expert_map: Tensor | None) -> Tensor:
    """Map routed expert ids to rank-local indices; -1 for none or another EP rank.

    Unrouted slots (-1) stay -1 before any ``expert_map`` lookup.
    """
    if expert_map is None:
        return ids
    return torch.where(ids >= 0, expert_map[ids.long().clamp_min(0)], -1)


def top_mask(score: Tensor, keep: int) -> Tensor:
    """Mask the ``keep`` largest entries of each row (graph-safe)."""
    index = score.topk(keep, dim=-1, sorted=False).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, index, True)


def grouped_config(
    gate_up: Tensor, down: Tensor, top_k: int, tokens: int
) -> dict[str, int]:
    """Return vLLM's own fused-MoE tile config for stock's expert shapes and batch.

    Tiles only reassociate each output's FP32 accumulation; the BF16 output
    rounding of every projection is unchanged.
    """
    return try_get_optimal_moe_config(
        tuple(gate_up.shape),
        tuple(down.shape),
        top_k,
        None,
        tokens,
        None,
    )


def grouped_assignment(
    ids: Tensor,
    experts: int,
    expert_map: Tensor | None = None,
    config: dict[str, int] | None = None,
) -> tuple[Tensor | None, Tensor, Tensor, dict[str, int]]:
    """Align routed slots by expert once for every projection in a layer.

    Rows are routed slots (token-major, ``ids`` flattened); every projection of
    the layer shares the config's BLOCK_SIZE_M alignment. Graph-safe. Returns
    vLLM's sorted slot ids, block expert ids and padded slot count, plus the
    config.
    """
    slots = ids.numel()
    config = SMALL_TILE_CONFIG if config is None else config
    sorted_ids, expert_ids, count = _prepare_expert_assignment(
        ids.reshape(slots, 1), config, slots, 1, experts, expert_map
    )
    return sorted_ids, expert_ids, count, config


def grouped_projection(
    x: Tensor,
    weight: Tensor,
    assignment: tuple[Tensor | None, Tensor, Tensor, dict[str, int]],
    *,
    top_k: int = 1,
) -> Tensor:
    """Multiply routed slot inputs by a native strided expert weight view.

    ``x`` holds one row per slot (``top_k=1``) or one row per token shared by
    its ``top_k`` consecutive slots, which the kernel loads by token index.
    Rows of slots without a local expert (-1) are not written; callers must
    drop their contributions.
    """
    sorted_ids, expert_ids, count, config = assignment
    output = torch.empty(
        (x.shape[0] * top_k, 1, weight.shape[1]), device=x.device, dtype=torch.bfloat16
    )
    dispatch_fused_moe_kernel(
        x,
        weight,
        output,
        None,
        None,
        None,
        None,
        sorted_ids,
        expert_ids,
        count,
        False,
        top_k,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )
    return output[:, 0, :]


class ActivationBaseline:
    """CATS, TEAL or WINA decode operator on one routed layer's native weights.

    ``policy`` selects the method; calibrated thresholds or ``top_fraction`` set the
    masks. Work counters stay on the GPU so they accumulate across CUDA graph
    replays.
    """

    def __init__(
        self,
        gate_up: Tensor,
        down: Tensor,
        policy: str,
        threshold: Tensor | float,
        *,
        input_threshold: float = 0.0,
        down_threshold: float = 0.0,
        gate_threshold: float | None = None,
        gate_norm: Tensor | None = None,
        up_norm: Tensor | None = None,
        down_norm: Tensor | None = None,
        sparsity: float = 0.0,
        expert_map: Tensor | None = None,
        activation: Literal["silu", "gelu_tanh"] = "silu",
        top_fraction: float | None = None,
        projection_sparsities: tuple[float, float, float] | None = None,
    ) -> None:
        if top_fraction is not None and (
            policy not in {"cats", "teal", "wina"} or not 0 < top_fraction <= 1
        ):
            raise ValueError(
                "Top-K masks need CATS, TEAL or WINA and a fraction in (0,1]"
            )
        if policy == "wina-greedy-topk":
            if (
                projection_sparsities is None
                or len(projection_sparsities) != 3
                or any(not 0 <= level <= 1 for level in projection_sparsities)
            ):
                raise ValueError(
                    "WINA greedy top-K needs gate/up/down sparsities in [0,1]"
                )
        elif projection_sparsities is not None:
            raise ValueError("Per-projection sparsities require WINA greedy top-K")
        if policy not in {
            "cats",
            "teal",
            "teal-greedy",
            "wina",
            "wina-greedy-topk",
        }:
            raise ValueError(f"Unknown baseline policy: {policy}")
        self.gate_up = gate_up
        self.down = down
        self.expert_map = (
            expert_map.to(gate_up.device) if expert_map is not None else None
        )
        self.policy = policy
        if activation not in ("silu", "gelu_tanh"):
            raise ValueError(f"Unsupported routed expert activation: {activation}")
        self.activation = activation
        self.width = down.shape[2]
        self.hidden = down.shape[1]
        # Official WINA truncates; uniform top-K counts are rounded to nearest-even.
        self.top_counts: tuple[int, int, int] | None = None
        if projection_sparsities is not None:
            gate_level, up_level, down_level = projection_sparsities
            self.top_counts = (
                int(self.hidden * (1 - gate_level)),
                int(self.hidden * (1 - up_level)),
                int(self.width * (1 - down_level)),
            )
        elif top_fraction is not None:
            self.top_counts = (
                round(top_fraction * self.hidden),
                round(top_fraction * self.hidden),
                round(top_fraction * self.width),
            )
        self.threshold = torch.as_tensor(
            [] if policy == "wina-greedy-topk" else threshold,
            device=down.device,
            dtype=torch.float32,
        )
        self.input_threshold = input_threshold
        self.gate_threshold = (
            input_threshold if gate_threshold is None else gate_threshold
        )
        self.down_threshold = down_threshold
        if policy == "wina-greedy-topk":
            up_norm = down_norm = None
        self.gate_norm = gate_norm
        self.up_norm = up_norm
        self.down_norm = down_norm
        self.sparsity = sparsity
        wina = policy in {"wina", "wina-greedy-topk"}
        if policy == "wina" and any(n is None for n in (gate_norm, up_norm, down_norm)):
            raise ValueError("WINA requires expert column norms")
        if policy == "wina-greedy-topk" and gate_norm is None:
            raise ValueError("WINA greedy top-K requires expert gate column norms")
        if not (0 <= sparsity < 1):
            raise ValueError("Sparsity must lie in [0,1)")
        # [gate, up, down, routed-reference, auxiliary norm reads] on GPU.
        self.work = torch.zeros(5, device=down.device, dtype=torch.float64)
        # Calibration metadata bytes gathered per routed slot, at their stored dtypes.
        self.slot_metadata_bytes = sum(
            n.shape[-1] * n.element_size()
            for n in ((gate_norm, up_norm, down_norm) if wina else ())
            if n is not None
        )
        # Top-K configurations store no thresholds.
        calibrated = self.top_counts is None
        self.resident_bytes = (
            calibrated * self.threshold.numel() * self.threshold.element_size()
            + sum(
                n.numel() * n.element_size()
                for n in (gate_norm, up_norm, down_norm)
                if n is not None
            )
            + (12 if calibrated and policy in {"teal", "teal-greedy", "wina"} else 0)
            + (12 if projection_sparsities is not None else 0)
        )

    def __call__(
        self,
        hidden: Tensor,
        ids: Tensor,
        weights: Tensor,
        padding: Tensor | None = None,
    ) -> Tensor:
        """Compute the routed MoE output ``[tokens, H]`` under the policy's masks.

        ``hidden`` is ``[tokens, H]``; ``ids`` and ``weights`` are the router's
        ``[tokens, K]`` experts and weights; ``padding`` marks graph padding rows.
        """
        count, k = ids.shape
        h, i = self.hidden, self.width
        valid = (
            torch.ones(count, device=hidden.device, dtype=torch.bool)
            if padding is None
            else ~padding
        )
        active = valid[:, None].expand(-1, k)
        slots = count * k
        flat_ids = local_ids(ids, self.expert_map).contiguous().reshape(-1)
        active = active & (flat_ids.view(count, k) >= 0)
        safe_ids = flat_ids.clamp_min(0)
        assignment = grouped_assignment(
            ids.contiguous().reshape(-1),
            self.expert_map.numel()
            if self.expert_map is not None
            else self.gate_up.shape[0],
            self.expert_map,
            grouped_config(self.gate_up, self.down, k, count),
        )
        x = hidden.contiguous()
        wina = self.policy in {"wina", "wina-greedy-topk"}
        input_policy = self.policy in {"teal", "teal-greedy"} or wina
        shared_input = self.policy in {"teal", "teal-greedy"} and (
            self.top_counts is not None or self.gate_threshold == self.input_threshold
        )

        def keep(score: Tensor, threshold: float, projection: int) -> Tensor:
            """Keep scores above the calibrated threshold, or the top-K count."""
            if self.top_counts is None:
                return score > threshold
            return top_mask(score, self.top_counts[projection])

        if wina:
            assert self.gate_norm is not None
            gate_input = (
                keep(
                    x.float()[:, None, :].abs()
                    * self.gate_norm[safe_ids.view(count, k)].float(),
                    self.gate_threshold,
                    0,
                )
                & active[:, :, None]
            )
            up_score = x.float()[:, None, :].abs()
            if self.policy == "wina":
                assert self.up_norm is not None
                up_score = up_score * self.up_norm[safe_ids.view(count, k)].float()
            up_input = (
                keep(
                    up_score,
                    self.input_threshold,
                    1,
                )
                & active[:, :, None]
            )
            gate_input = gate_input.reshape(slots, h).contiguous()
            up_input = up_input.reshape(slots, h).contiguous()
        elif input_policy:
            gate_input = keep(x.abs(), self.gate_threshold, 0) & valid[:, None]
            up_input = (
                gate_input
                if shared_input
                else (x.abs() > self.input_threshold) & valid[:, None]
            )
        else:
            gate_input = up_input = None

        def operand(mask: Tensor | None) -> tuple[Tensor, int]:
            """Return the masked input rows and the number of slots sharing a row.

            TEAL's mask is per token and CATS reads the full input, so the kernel
            loads each slot's row by token index; WINA's expert-specific masks
            need one row per slot.
            """
            if mask is None:
                rows, top_k = x, k
            elif wina:
                rows = x[:, None, :].expand(count, k, h).reshape(slots, h) * mask
                top_k = 1
            else:
                rows, top_k = x * mask, k
            return rows.contiguous(), top_k

        def projection(source: tuple[Tensor, int], branch: int | None) -> Tensor:
            """Project one operand onto the gate (0), up (1) or both (None)."""
            columns = (
                slice(None) if branch is None else slice(branch * i, (branch + 1) * i)
            )
            values, top_k = source
            return grouped_projection(
                values,
                self.gate_up[:, columns],
                assignment,
                top_k=top_k,
            )

        gate_source = operand(gate_input)
        if gate_input is None or shared_input:
            # One operand for both branches: a single gate/up GEMM. The channel
            # masks below zero the unselected rows.
            gate, up = projection(gate_source, None).split(i, dim=-1)
        else:
            gate = projection(gate_source, 0)
            up = projection(operand(up_input), 1)
        channel_mask: Tensor | None = None
        activated = (
            torch.nn.functional.silu(gate)
            if self.activation == "silu"
            else torch.nn.functional.gelu(gate, approximate="tanh")
        ).to(hidden.dtype)
        if not input_policy:
            score = activated.abs()
            # CATS Eq. (5) rejects magnitudes strictly below the threshold.
            channel_mask = (
                top_mask(score, self.top_counts[2])
                if self.top_counts is not None
                else score >= self.threshold
            )
            up = up * channel_mask
        feature = (activated * up).to(hidden.dtype)
        if self.policy in {"teal", "teal-greedy"}:
            channel_mask = keep(feature.abs(), self.down_threshold, 2)
        elif wina:
            down_score = cast(Tensor, feature).float().abs()
            if self.policy == "wina":
                assert self.down_norm is not None
                down_score = down_score * self.down_norm[safe_ids].float()
            channel_mask = keep(
                down_score,
                self.down_threshold,
                2,
            )
        assert channel_mask is not None
        channel_mask = channel_mask & active.reshape(-1, 1)
        expert_output = grouped_projection(
            (feature * channel_mask).contiguous(),
            self.down,
            assignment,
        )
        gate_reads = (
            active.sum() * i * h
            if gate_input is None
            else (
                gate_input.sum() * i
                if wina
                else (gate_input[:, None, :] & active[:, :, None]).sum() * i
            )
        )
        up_reads = (
            channel_mask.sum() * h
            if up_input is None
            else (
                up_input.sum() * i
                if wina
                else (up_input[:, None, :] & active[:, :, None]).sum() * i
            )
        )
        # Norm metadata: one read per rank-local routed slot of a valid token.
        # Remote slots gather expert 0's norms through safe_ids only as padding and
        # are not counted, or EP ranks would count them more than once.
        norm_reads = active.sum() * self.slot_metadata_bytes // 2
        self.work += torch.stack(
            (
                gate_reads,
                up_reads,
                channel_mask.sum() * h,
                active.sum() * 3 * i * h,
                norm_reads,
            )
        ).to(self.work.dtype)
        # Slots without a local expert are never written by the grouped kernel.
        contribution = (
            expert_output.view(count, k, h).float() * weights.float()[..., None]
        )
        # A host-side finite check here would synchronize every decode layer.
        # vLLM likelihood validation checks non-finite logits at the caller.
        return torch.where(active[..., None], contribution, 0).sum(1).to(hidden.dtype)

    def report(self) -> dict[str, object]:
        """Return the accumulated coefficient reads and BF16 byte prices."""
        gate, up, down, reference, auxiliary = self.work.cpu().tolist()
        reference_bytes = 2 * reference
        charged_bytes = 2 * (gate + up + down) + 2 * auxiliary
        return {
            "gate_coefficients": gate,
            "up_coefficients": up,
            "down_coefficients": down,
            "auxiliary_bf16_equivalent_reads": auxiliary,
            "routed_reference_coefficients": reference,
            "read_fraction": charged_bytes / reference_bytes
            if reference_bytes
            else None,
            "charged_bytes": charged_bytes,
            "reference_bytes": reference_bytes,
            "resident_calibration_bytes_per_layer": self.resident_bytes,
        }
