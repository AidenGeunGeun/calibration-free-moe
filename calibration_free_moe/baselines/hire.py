"""HiRE baseline: a calibration-free INT4 channel screen for routed experts.

HiRE-Q (Samaga et al., arXiv:2402.09360, Sec. 3.2.1) shortlists FFN units with an
INT4 copy of the weights, with no training or calibration data, then computes the
shortlist exactly. Per token, the copy scores every channel of the router's top-K
experts on the full input, and one global top-F over the N = K*I candidates keeps
the largest w^2 ||d||^2 phi(g)^2 (g the INT4 gate estimate, w the router weight, d
the native down column). A routed expert may receive no channels. Skipped channels
are dropped (no recovery).

Differences from the paper:
- HiRE does not fix the quantizer; we use symmetric INT4 (scale max/7, nearest-even,
  [-7, 7], as in the Prox proxy) over groups of 128 inputs with FP16 scales, the
  common GPTQ/AWQ group layout, stored row-major.
- ``hire-gate`` quantizes only the gate rows; the up projection is not screened.
- F = fN rounded to nearest-even (possibly 0); the shortlist is not oversampled.
- HiRE's group-sparse transfer, common dense path and distributed top-k are TPU
  system components and are not modeled. TP1 only: the global top-F spans experts
  that expert parallelism would split.

Read cost per routed slot, in 32-byte sectors of contiguous per-expert blocks: the
INT4 gate rows, their group scales and the FP32 down-column norms; each selected
channel then reads its native gate and up rows and its channel-major down row
(3H BF16 values, 6H bytes). For H divisible by 128 these blocks are whole sectors,
so sector and logical bytes agree. Execution runs dense grouped native projections
followed by the channel mask; read cost follows the masks, not these kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from torch import Tensor

from calibration_free_moe.activation import GatedActivation, activate
from calibration_free_moe.baselines.activation_sparsity import (
    grouped_assignment,
    grouped_config,
    grouped_projection,
    top_mask,
)

if TYPE_CHECKING:
    constexpr = int
else:
    constexpr = tl.constexpr

GROUP = 128
SECTOR = 32


def sectors(size: int) -> int:
    """Return a contiguous block's size in bytes, padded to whole 32-byte sectors."""
    return -(-size // SECTOR) * SECTOR


def screen_bytes(hidden: int, width: int) -> int:
    """Return the sector-priced screen bytes per routed slot.

    The screen reads one expert's INT4 gate rows, their scales and its FP32
    down-column norms.
    """
    rows = width
    return (
        sectors(rows * hidden // 2)
        + sectors(rows * hidden // GROUP * 2)
        + sectors(4 * width)
    )


def matched_retention(read_fraction: float, hidden: int, width: int) -> float:
    """Return the retention whose total read cost equals ``read_fraction``.

    Relative to stock BF16 expert reads, the read cost is screen_bytes/(6IH) + F/N.
    F is fN rounded, so the realized fraction differs from the target by at most
    0.5/N.
    """
    return read_fraction - screen_bytes(hidden, width) / (6 * width * hidden)


def quantize_int4_groups(weight: Tensor, chunk: int = 8) -> tuple[Tensor, Tensor]:
    """Quantize ``[experts, rows, H]`` weights to signed INT4 in groups of 128.

    Returns packed codes ``[experts, rows, H/2]`` (two per byte) and FP16 scales
    ``[experts, rows, H/128]``. Deterministic per group, so chunking experts only
    bounds FP32 scratch.
    """
    experts, rows, hidden = weight.shape
    if hidden % GROUP:
        raise ValueError("Group-wise INT4 needs a hidden size divisible by 128")
    packed = torch.empty(
        (experts, rows, hidden // 2), dtype=torch.uint8, device=weight.device
    )
    scales = torch.empty(
        (experts, rows, hidden // GROUP), dtype=torch.float16, device=weight.device
    )
    for start in range(0, experts, chunk):
        groups = weight[start : start + chunk].float().unflatten(-1, (-1, GROUP))
        scale = (
            (groups.abs().amax(-1) / 7).half().clamp_min(torch.finfo(torch.half).tiny)
        )
        q = (groups / scale.float()[..., None]).round().clamp(-7, 7).flatten(-2)
        q = q.to(torch.int8).view(torch.uint8)
        packed[start : start + chunk] = (q[..., 0::2] & 15) | (q[..., 1::2] << 4)
        scales[start : start + chunk] = scale
    return packed, scales


def dequantize_int4_groups(packed: Tensor, scales: Tensor) -> Tensor:
    """Dequantize a packed INT4 copy to FP32 (analysis and tests; never resident)."""
    low = (packed & 15).to(torch.int8)
    high = (packed >> 4).to(torch.int8)
    q = torch.stack(
        (torch.where(low >= 8, low - 16, low), torch.where(high >= 8, high - 16, high)),
        dim=-1,
    ).flatten(-2)
    return (q.float().unflatten(-1, (-1, GROUP)) * scales.float()[..., None]).flatten(
        -2
    )


@triton.jit
def _int4_estimate(
    X,
    Packed,
    Scales,
    SortedIds,
    ExpertIds,
    Count,
    Out,
    H: constexpr,
    I: constexpr,
    BRANCHES: constexpr,
    SLOTS,  # Runtime: a constexpr slot count would recompile for every batch size.
    K: constexpr,
    NAIVE: constexpr,
    BLOCK: constexpr,
):
    block = tl.program_id(0)
    if block * BLOCK < tl.load(Count):
        expert = tl.load(ExpertIds + block).to(tl.int32)
        row = tl.arange(0, BLOCK)
        if NAIVE:
            # vLLM's low-latency assignment: one slot per block, in slot order.
            slot = tl.where(row == 0, block, SLOTS)
        else:
            slot = tl.load(SortedIds + block * BLOCK + row)
        channel = tl.program_id(1) * 32 + tl.arange(0, 32)
        written = (slot < SLOTS)[:, None] & (channel < I)[None, :]
        token = slot // K
        pair = tl.arange(0, 32)
        # Triton's stubs do not declare static_range iterable.
        for branch in tl.static_range(BRANCHES):  # pyright: ignore[reportGeneralTypeIssues]
            line = expert * BRANCHES * I + branch * I + channel
            acc = tl.zeros((BLOCK, 32), tl.float32)
            if expert >= 0:
                for group in range(H // 128):
                    # Integers and BF16 inputs are exact in BF16; FP32 dot sums.
                    part = tl.zeros((BLOCK, 32), tl.float32)
                    for half in tl.static_range(2):
                        coord = group * 128 + half * 64 + tl.arange(0, 64)
                        x = tl.load(
                            X + token[:, None] * H + coord[None, :],
                            (slot < SLOTS)[:, None],
                            other=0,
                        )
                        byte = tl.load(
                            Packed
                            + line[:, None] * (H // 2)
                            + group * 64
                            + half * 32
                            + pair[None, :],
                            (channel < I)[:, None],
                            other=0,
                        ).to(tl.int32)
                        low = (byte & 15) - ((byte & 8) << 1)
                        high = (byte >> 4) - ((byte & 128) >> 3)
                        weight = tl.interleave(low, high).to(tl.bfloat16)
                        part = tl.dot(x.to(tl.bfloat16), tl.trans(weight), part)
                    scale = tl.load(
                        Scales + line * (H // 128) + group, channel < I, other=0
                    ).to(tl.float32)
                    acc += part * scale[None, :]
            tl.store(
                Out + (branch * SLOTS + slot[:, None]) * I + channel[None, :],
                acc,
                written,
            )


def int4_estimates(
    x: Tensor,
    packed: Tensor,
    scales: Tensor,
    assignment: tuple[Tensor | None, Tensor, Tensor, dict[str, int]],
    top_k: int,
    width: int,
) -> Tensor:
    """Evaluate the packed INT4 rows on the full input as FP32 ``[branches, slots, I]``.

    ``branches`` is 1 for a gate-only copy and 2 for gate and up.
    """
    sorted_ids, expert_ids, total, config = assignment
    branches = packed.shape[1] // width
    slots = x.shape[0] * top_k
    block = config["BLOCK_SIZE_M"]
    out = torch.empty((branches, slots, width), device=x.device, dtype=torch.float32)
    blocks = (
        expert_ids.numel()
        if sorted_ids is None
        else triton.cdiv(sorted_ids.numel(), block)
    )
    _int4_estimate[(blocks, triton.cdiv(width, 32))](
        x.contiguous(),
        packed,
        scales,
        expert_ids if sorted_ids is None else sorted_ids,
        expert_ids,
        total,
        out,
        x.shape[1],
        width,
        branches,
        slots,
        top_k,
        sorted_ids is None,
        block,
    )
    return out


class HireBaseline:
    """Calibration-free INT4 screen, global top-F and exact selected channels."""

    def __init__(
        self,
        gate_up: Tensor,
        down: Tensor,
        retention: float,
        *,
        activation: GatedActivation = "silu",
    ) -> None:
        if gate_up.dtype != torch.bfloat16 or not 0 < retention <= 1:
            raise ValueError("HiRE screen needs BF16 experts and retention in (0,1]")
        self.gate_up, self.down = gate_up, down
        self.hidden, self.width = down.shape[1], down.shape[2]
        self.retention = retention
        self.activation: GatedActivation = activation
        self.packed, self.scales = quantize_int4_groups(gate_up[:, : self.width])
        # ||d_j|| of the channel-major down rows, FP32 (4 bytes per candidate).
        self.down_norm = torch.linalg.vector_norm(down, dim=1, dtype=torch.float32)
        self.screen_bytes = screen_bytes(self.hidden, self.width)
        self.resident_int4_bytes = (
            self.packed.numel() + self.scales.numel() * self.scales.element_size()
        )
        self.resident_bytes = self.resident_int4_bytes + self.down_norm.numel() * 4
        # [routed slots, selected channels, tokens] on the GPU.
        self.work = torch.zeros(3, device=down.device, dtype=torch.float64)

    def select(
        self,
        hidden: Tensor,
        ids: Tensor,
        weights: Tensor,
        active: Tensor,
        assignment: tuple[Tensor | None, Tensor, Tensor, dict[str, int]],
    ) -> Tensor:
        """Select each token's global top-F channels as a ``[slots, I]`` mask.

        Candidates are the channels of the token's active routed slots.
        """
        count, k = ids.shape
        i = self.width
        estimate = int4_estimates(hidden, self.packed, self.scales, assignment, k, i)
        proxy = activate(estimate[0], self.activation)
        # |w ||d|| phi(g)| orders channels as its square does.
        score = (
            weights.float()[..., None]
            * self.down_norm[ids.clamp_min(0)]
            * proxy.view(count, k, i)
        ).abs()
        # Inactive candidates rank last and are removed after selection.
        score = torch.where(active[..., None], score, -1.0).view(count, k * i)
        keep = round(self.retention * k * i)
        return top_mask(score, keep).view(count * k, i) & active.view(-1, 1)

    def __call__(
        self,
        hidden: Tensor,
        ids: Tensor,
        weights: Tensor,
        padding: Tensor | None = None,
    ) -> Tensor:
        """Compute the routed MoE output ``[tokens, H]`` from the screened channels.

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
        active = valid[:, None] & (ids >= 0)
        assignment = grouped_assignment(
            ids.contiguous().reshape(-1),
            self.gate_up.shape[0],
            None,
            grouped_config(self.gate_up, self.down, k, count),
        )
        mask = self.select(hidden, ids, weights, active, assignment)
        gate, up = grouped_projection(
            hidden.contiguous(), self.gate_up, assignment, top_k=k
        ).split(i, dim=-1)
        feature = (
            activate(gate * mask, self.activation).to(hidden.dtype) * (up * mask)
        ).to(hidden.dtype)
        output = grouped_projection(
            (feature * mask).contiguous(), self.down, assignment
        )
        self.work += torch.stack((active.sum(), mask.sum(), valid.sum())).to(
            self.work.dtype
        )
        contribution = output.view(count, k, h).float() * weights.float()[..., None]
        return torch.where(active[..., None], contribution, 0).sum(1).to(hidden.dtype)

    def report(self) -> dict[str, object]:
        """Return the accumulated reads, priced in 32-byte sectors."""
        slots, selected, tokens = self.work.cpu().tolist()
        h, i, unit = self.hidden, self.width, self.gate_up.element_size()
        rows = i
        proxy = slots * sectors(rows * h // 2)
        scale = slots * sectors(rows * h // GROUP * 2)
        norm = slots * sectors(4 * i)
        exact = selected * 3 * h * unit
        reference = slots * 3 * i * h * unit
        charged = slots * self.screen_bytes + exact
        return {
            "gate_coefficients": selected * h,
            "up_coefficients": selected * h,
            "down_coefficients": selected * h,
            "routed_reference_coefficients": slots * 3 * i * h,
            "proxy_bytes": proxy,
            "scale_bytes": scale,
            "norm_bytes": norm,
            "charged_bytes": charged,
            "logical_charged_bytes": slots * (rows * (h // 2 + h // GROUP * 2) + 4 * i)
            + exact,
            "reference_bytes": reference,
            "accounting_basis": "analytical_sector_estimate",
            "actual_traffic_measured": False,
            "read_fraction": charged / reference if reference else None,
            "retention": self.retention,
            "selected_per_token": selected / tokens if tokens else None,
            "proxy_matrix_macs": slots * rows * h,
            "matrix_macs": slots * rows * h + 3 * selected * h,
            "resident_int4_bytes_per_layer": self.resident_int4_bytes,
            "resident_calibration_bytes_per_layer": self.resident_bytes,
        }
