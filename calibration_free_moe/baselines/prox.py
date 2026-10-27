"""Prox two-stage gated-FFN baseline, adapted to independently routed experts.

Prox (Liu et al., arXiv:2607.27591v1, Sec. 4.1) thresholds |x| to sparsify the
inputs of INT4 gate/up proxies (Stage 1), thresholds |phi(proxy_gate) * proxy_up|
for one channel mask shared by gate, up and down, and computes the kept channels
from native weights (Stage 2). Thresholds are fixed per layer (App. A.6-A.7), with
stage sparsities around s2 = 0.7 and s1 in [0, 0.7] (Sec. 4.2, App. A.4).

Differences from the paper:
- Sec. 4.3 leaves the INT4 format open; we use symmetric per-row INT4 with scale
  max/7 in FP32, nearest-even rounding and two signed nibbles per byte.
- GeGLU experts (Gemma 4) use GELU(tanh) in place of SiLU in both stages, as Prox
  does for Gemma 3 (Sec. 5).
- Under expert parallelism each rank quantizes and scores only its resident experts
  and counts only its own routed slots; thresholds are shared per layer.
- Read cost assumes Prox's best layout: a coordinate-major proxy with two output
  channels per byte, so each kept input of a routed slot reads one sector-aligned
  run of I bytes (gate and up), plus one FP32 scale per row, and channel-major
  native down rows. Execution uses a row-major masked proxy and dense native
  grouped projections instead; read cost is analytical, not measured traffic.

Variants beyond the paper:
- ``prox-free`` replaces both thresholds by per-token top-k, with no calibration
  data: Stage 1 keeps the ``input_keep`` largest |x|, and Stage 2
  (``channel_scope="token"``) the ``channel_keep`` largest proxy scores jointly over
  the token's K routed slots.
- Mixed controls combine one top-k rule with the other threshold. ``channel_floor``
  adds the token's top-m channels to a threshold mask. Under expert parallelism the
  joint rules all-reduce count*K*I FP32 scores before ranking; every rank ranks all
  of them but counts only its local candidates.
- The ``output-unit`` score ranks w^2 n^2 (phi(g~) u~)^2 (w the router weight, n the
  native down-column L2 norm) and reads one FP32 norm per candidate.
- An ``allocation`` (``analysis/allocation.py``) keeps each token's top n_{t,l}
  channels jointly over its K slots. Its ``exact`` score uses each candidate's exact
  activation from the full native gate/up rows; it is an analysis-only score,
  priced as full gate/up reads of every candidate.

Recipes: ``legacy`` is the published Prox recipe (top-k ties in ``torch.topk``
order); ``study`` breaks top-k ties by the lowest input or flattened (slot, channel)
index.
"""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor

from calibration_free_moe.activation import GatedActivation, activate
from calibration_free_moe.analysis.allocation import LayerAllocation
from calibration_free_moe.baselines.activation_sparsity import (
    grouped_assignment,
    grouped_config,
    grouped_projection,
    local_ids,
    top_mask,
)

if TYPE_CHECKING:
    constexpr = int
else:
    constexpr = tl.constexpr

next_power_of_2 = cast(Callable[[int], int], triton.next_power_of_2)

# "prox": |phi(g~) u~|; "output-unit": w^2 n^2 (phi(g~) u~)^2 (see module docstring);
# "exact": w^2 n^2 a^2 with the exact activation (allocation analysis only).
ProxyScore = Literal["prox", "output-unit", "exact"]
ProxRecipe = Literal["legacy", "study"]


def prox_top_mask(score: Tensor, keep: int, stable_ties: bool = False) -> Tensor:
    """Mask the ``keep`` largest entries of each row.

    Ties follow ``torch.topk`` unless ``stable_ties`` (the study recipe) keeps the
    lowest index: top-k finds only the boundary value and a cumulative count over
    the tied entries resolves them, without a full sort or host sync.
    """
    if not stable_ties:
        return top_mask(score, keep)
    boundary = score.topk(keep, dim=-1, sorted=False).values.amin(-1, keepdim=True)
    above = score > boundary
    tied = score == boundary
    remaining = keep - above.sum(-1, keepdim=True)
    return above | (tied & (tied.cumsum(-1) <= remaining))


def prox_input_mask(
    x: Tensor,
    valid: Tensor,
    threshold: float | None,
    keep: int | None,
    stable_ties: bool,
) -> Tensor:
    """Mask Prox's Stage-1 inputs ``[tokens, H]`` by threshold or per-token top-k."""
    return (
        prox_top_mask(x.abs(), keep, stable_ties)
        if keep is not None
        else x.abs() >= cast(float, threshold)
    ) & valid[:, None]


def quantize_proxy(gate_up: Tensor, chunk: int = 8) -> tuple[Tensor, Tensor]:
    """Quantize native ``[E, 2I, H]`` BF16 weights to packed signed INT4.

    Returns codes ``[E, 2I, H/2]`` (adjacent inputs share a byte) and FP32 row
    scales ``[E, 2I]``. Row-wise, so chunking experts only bounds FP32 scratch.
    """
    assert gate_up.shape[-1] % 2 == 0
    experts, rows, hidden = gate_up.shape
    packed = torch.empty(
        (experts, rows, hidden // 2), dtype=torch.uint8, device=gate_up.device
    )
    scales = torch.empty((experts, rows), dtype=torch.float32, device=gate_up.device)
    for start in range(0, experts, chunk):
        weights = gate_up[start : start + chunk].float()
        scale = weights.abs().amax(-1).clamp_min(1e-30) / 7
        q = (weights / scale[..., None]).round().clamp(-7, 7).to(torch.uint8)
        packed[start : start + chunk] = (q[..., 0::2] & 15) | (q[..., 1::2] << 4)
        scales[start : start + chunk] = scale
    return packed, scales


@triton.jit
def _gated(gate, up, GELU: constexpr):
    if GELU:
        # tanh(z) = 2 sigmoid(2z) - 1, so GELU_tanh(g) = g sigmoid(2z).
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        return gate * tl.sigmoid(2 * inner) * up
    return gate * tl.sigmoid(gate) * up


@triton.jit
def _proxy(
    X,  # [tokens, H]
    Ids,  # [tokens * K] rank-local expert ids, -1 if not on this rank
    Packed,  # [E, 2I, H/2] uint8 INT4 codes
    Scales,  # [E, 2I] fp32
    InputMask,  # [tokens, H] bool, Stage-1 mask
    Valid,  # [tokens] bool, False on padding rows
    Out,  # [tokens * K, I] fp32 proxy scores
    H: constexpr,
    I: constexpr,
    K: constexpr,
    B: constexpr,
    GELU: constexpr,
):
    slot = tl.program_id(0)
    channel = tl.program_id(1) * 16 + tl.arange(0, 16)
    pair = tl.arange(0, B)
    token = slot // K
    # Rank-local expert index; -1 marks an expert resident on another EP rank.
    expert = tl.load(Ids + slot).to(tl.int32)
    active = (tl.load(Valid + token) != 0) & (expert >= 0)
    even = 2 * pair
    a = (tl.load(InputMask + token * H + even, even < H, other=0) != 0) & active
    b = (tl.load(InputMask + token * H + even + 1, even + 1 < H, other=0) != 0) & active
    xa = tl.load(X + token * H + even, even < H, other=0).to(tl.float32)
    xb = tl.load(X + token * H + even + 1, even + 1 < H, other=0).to(tl.float32)
    both = a | b
    first = tl.load(
        Packed + (expert * 2 * I + channel[:, None]) * (H // 2) + pair[None, :],
        (channel < I)[:, None] & both[None, :] & (pair < H // 2)[None, :],
        other=0,
    ).to(tl.int32)
    second = tl.load(
        Packed + (expert * 2 * I + I + channel[:, None]) * (H // 2) + pair[None, :],
        (channel < I)[:, None] & both[None, :] & (pair < H // 2)[None, :],
        other=0,
    ).to(tl.int32)
    gate_scale = tl.load(
        Scales + expert * 2 * I + channel, (channel < I) & active, other=0
    )
    up_scale = tl.load(
        Scales + expert * 2 * I + I + channel, (channel < I) & active, other=0
    )
    ga = (first & 15) - ((first & 8) << 1)
    gb = (first >> 4) - ((first & 128) >> 3)
    ua = (second & 15) - ((second & 8) << 1)
    ub = (second >> 4) - ((second & 128) >> 3)
    gate = (
        tl.sum(
            ga * tl.where(a[None, :], xa[None, :], 0)
            + gb * tl.where(b[None, :], xb[None, :], 0),
            1,
        )
        * gate_scale
    )
    up = (
        tl.sum(
            ua * tl.where(a[None, :], xa[None, :], 0)
            + ub * tl.where(b[None, :], xb[None, :], 0),
            1,
        )
        * up_scale
    )
    value = _gated(gate, up, GELU)
    tl.store(Out + slot * I + channel, value, channel < I)


@triton.jit
def _grouped_proxy(
    X,  # [tokens, H] bf16
    Packed,  # [E, 2I, H/2] uint8 INT4 codes
    Scales,  # [E, 2I] fp32
    InputMask,  # [tokens, H] bool, Stage-1 mask
    SortedIds,  # [padded slots] slot ids grouped by expert; padding >= tokens * K
    ExpertIds,  # [num_blocks] rank-local expert per block, -1 if remote
    Count,  # [1] number of entries in SortedIds that are used
    Out,  # [tokens * K, I] fp32 proxy scores
    H: constexpr,
    I: constexpr,
    SLOTS,  # Runtime: a constexpr slot count would recompile for every batch size.
    K: constexpr,
    GELU: constexpr,
    BLOCK: constexpr,
):
    block = tl.program_id(0)
    if block * BLOCK < tl.load(Count):
        # Rank-local expert index; -1 marks a block of remote EP experts.
        expert = tl.load(ExpertIds + block).to(tl.int32)
        slot = tl.load(SortedIds + block * BLOCK + tl.arange(0, BLOCK))
        channel = tl.program_id(1) * 32 + tl.arange(0, 32)
        output = Out + slot[:, None] * I + channel[None, :]
        written = (slot < SLOTS)[:, None] & (channel < I)[None, :]
        if expert >= 0:
            pair = tl.arange(0, 32)
            # Decode integers exactly to BF16. Masked native BF16 inputs and INT4
            # integers have exact BF16 representations; tl.dot accumulates in FP32.
            gate_acc = tl.full((BLOCK, 32), 0, tl.float32)
            up_acc = tl.full((BLOCK, 32), 0, tl.float32)
            for start in range(H // 64):
                coord = start * 64 + tl.arange(0, 64)
                token = slot // K
                x = tl.load(
                    X + token[:, None] * H + coord[None, :],
                    (slot < SLOTS)[:, None],
                    other=0,
                )
                mask = tl.load(
                    InputMask + token[:, None] * H + coord[None, :],
                    (slot < SLOTS)[:, None],
                    other=0,
                )
                x = tl.where(mask != 0, x, 0).to(tl.bfloat16)
                offset = (
                    (expert * 2 * I + channel[:, None]) * (H // 2)
                    + start * 32
                    + pair[None, :]
                )
                gate = tl.load(Packed + offset, (channel < I)[:, None], other=0).to(
                    tl.int32
                )
                up = tl.load(
                    Packed + offset + I * (H // 2), (channel < I)[:, None], other=0
                ).to(tl.int32)
                ga = (gate & 15) - ((gate & 8) << 1)
                gb = (gate >> 4) - ((gate & 128) >> 3)
                ua = (up & 15) - ((up & 8) << 1)
                ub = (up >> 4) - ((up & 128) >> 3)
                gate_acc = tl.dot(
                    x, tl.trans(tl.interleave(ga, gb).to(tl.bfloat16)), gate_acc
                )
                up_acc = tl.dot(
                    x, tl.trans(tl.interleave(ua, ub).to(tl.bfloat16)), up_acc
                )
            gs = tl.load(Scales + expert * 2 * I + channel, channel < I, other=0)
            us = tl.load(Scales + expert * 2 * I + I + channel, channel < I, other=0)
            g = gate_acc * gs[None, :]
            u = up_acc * us[None, :]
            tl.store(output, _gated(g, u, GELU), written)
        else:
            tl.store(output, tl.full((BLOCK, 32), 0, tl.float32), written)


def proxy_scores(
    x: Tensor,
    ids: Tensor,
    packed: Tensor,
    scales: Tensor,
    input_threshold: float | None,
    valid: Tensor | None = None,
    assignment: tuple[Tensor | None, Tensor, Tensor, dict[str, int]] | None = None,
    activation: GatedActivation = "silu",
    expert_map: Tensor | None = None,
    input_keep: int | None = None,
    stable_ties: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return FP32 proxy scores ``[tokens*K, I]`` and the Stage-1 mask ``[tokens, H]``.

    Stage 1 keeps |x| >= ``input_threshold`` or, if given, each token's
    ``input_keep`` largest |x|. An ``assignment`` must use the same ``expert_map``.
    Only valid tokens' rank-local slots get meaningful scores; the grouped path
    leaves rows of unrouted (-1) slots unwritten, so callers mask inactive slots.
    """
    count, k = ids.shape
    h = x.shape[-1]
    i = scales.shape[-1] // 2
    valid = (
        torch.ones(count, dtype=torch.bool, device=x.device) if valid is None else valid
    )
    mask = prox_input_mask(x, valid, input_threshold, input_keep, stable_ties)
    score = torch.empty((count * k, i), device=x.device, dtype=torch.float32)
    if (
        assignment is not None
        and assignment[0] is not None
        and x.dtype == torch.bfloat16
        and h % 64 == 0
    ):
        sorted_ids, expert_ids, total, config = assignment
        assert sorted_ids is not None
        block = config["BLOCK_SIZE_M"]
        _grouped_proxy[(triton.cdiv(sorted_ids.numel(), block), triton.cdiv(i, 32))](
            x.contiguous(),
            packed,
            scales,
            mask,
            sorted_ids,
            expert_ids,
            total,
            score,
            h,
            i,
            count * k,
            k,
            activation == "gelu_tanh",
            block,
        )
    else:
        _proxy[(count * k, triton.cdiv(i, 16))](
            x.contiguous(),
            local_ids(ids, expert_map).contiguous().reshape(-1),
            packed,
            scales,
            mask,
            valid,
            score,
            h,
            i,
            k,
            next_power_of_2(h // 2),
            activation == "gelu_tanh",
        )
    return score, mask


def output_unit_scores(
    proxy: Tensor, weights: Tensor, norms: Tensor, slot_experts: Tensor
) -> Tensor:
    """Score candidates by |w| n |p|, the square root of w^2 n^2 p^2.

    The square root ranks identically and avoids FP32 underflow. Calibration,
    thresholds and top-k all use this value; callers exclude inactive slots.
    """
    return (
        weights.float().abs().reshape(-1, 1)
        * norms[slot_experts.reshape(-1).clamp_min(0).long()]
        * proxy.abs()
    )


class ProxBaseline:
    """Prox decode operator on one routed layer's native weights.

    Stage 1 masks inputs, an INT4 proxy scores channels, Stage 2 masks channels and
    the kept channels are computed from native weights. Work counters stay on the
    GPU so they accumulate across CUDA graph replays.
    """

    def __init__(
        self,
        gate_up: Tensor,
        down: Tensor,
        packed: Tensor,
        scales: Tensor,
        input_threshold: float | None,
        channel_threshold: float | None,
        activation: GatedActivation = "silu",
        expert_map: Tensor | None = None,
        channel_keep: int | None = None,
        input_keep: int | None = None,
        channel_scope: Literal["slot", "token"] = "slot",
        proxy_score: ProxyScore = "prox",
        channel_floor: int = 0,
        recipe: ProxRecipe = "legacy",
        *,
        allocation: LayerAllocation | None = None,
    ) -> None:
        """Configure the Stage-1 input rule and the Stage-2 channel rule.

        Inputs are kept by ``input_threshold`` or each token's ``input_keep``
        largest. Channels are kept by ``channel_threshold`` of each slot or the
        ``channel_keep`` largest of each token's K slots jointly
        (``channel_scope="token"``).

        ``proxy_score="output-unit"`` takes a keep count, or a threshold fitted
        under the study recipe with input top-k. ``allocation`` sets each token's
        count jointly (no channel keep or floor); ``channel_threshold`` then only
        feeds its threshold source and features.
        """
        self.width, self.hidden = down.shape[2], down.shape[1]
        exact = proxy_score == "exact"
        if (allocation is not None and (channel_keep is not None or channel_floor)) or (
            exact and (allocation is None or input_keep)
        ):
            raise ValueError(
                "An allocation replaces channel keep/floor; the exact score needs an "
                "allocation without input top-k"
            )
        self.allocation: LayerAllocation | None = allocation
        self.exact: bool = exact
        if recipe not in {"legacy", "study"}:
            raise ValueError("Prox recipe must be legacy or study")
        self.recipe: ProxRecipe = recipe
        if (
            proxy_score == "output-unit"
            and channel_keep is None
            and (recipe != "study" or input_keep is None)
        ):
            raise ValueError(
                "Output-unit thresholds need the study recipe and input top-k"
            )
        if (not exact and (input_threshold is None) == (input_keep is None)) or (
            input_keep is not None and not 0 < input_keep <= self.hidden
        ):
            raise ValueError("Give an input threshold or a keep count in [1, H]")
        # A per-token keep count ranks K*I scores; topk rejects one above that.
        if (
            (
                allocation is None
                and (channel_threshold is None) == (channel_keep is None)
            )
            or (channel_keep is not None and channel_keep < 1)
            or (channel_scope == "slot" and channel_keep is not None)
        ):
            raise ValueError("Give a channel threshold or a joint per-token keep count")
        if channel_scope not in {"slot", "token"}:
            raise ValueError("Channel scope must be slot or token")
        if channel_floor < 0 or (
            channel_floor and (channel_threshold is None or channel_scope != "token")
        ):
            raise ValueError("A channel floor requires a joint channel threshold")
        self.gate_up, self.down = gate_up, down
        self.activation: GatedActivation = activation
        self.expert_map = expert_map.to(down.device) if expert_map is not None else None
        self.packed, self.scales = packed, scales
        self.input_threshold, self.channel_threshold = (
            input_threshold,
            channel_threshold,
        )
        self.channel_keep: int | None = channel_keep
        self.input_keep: int | None = input_keep
        self.channel_scope: Literal["slot", "token"] = channel_scope
        self.channel_floor = channel_floor
        if self.expert_map is not None and (
            (channel_scope == "token" and channel_keep is not None)
            or channel_floor
            or allocation is not None
        ):
            from vllm.distributed import get_ep_group

            self.ep_group = get_ep_group().device_group
        else:
            self.ep_group = None
        self.candidate_channels = 0
        self.proxy_score: ProxyScore = proxy_score
        # FP32 native down-column norms [E, I], one expert at a time (transient).
        self.norms: Tensor | None = (
            torch.stack([expert.float().norm(dim=0) for expert in down])
            if proxy_score in {"output-unit", "exact"}
            else None
        )
        if allocation is not None:
            # Native bytes per kept channel: BF16 gate, up and down rows (down only
            # under the exact score).
            allocation.context.channel_bytes[allocation.layer] = float(
                2 * self.hidden * (1 if exact else 3)
            )
        # work[:11]: proxy bytes, scale bytes, gate/up/down coefficients, dense
        # BF16 bytes, kept inputs, routed slots, tokens, row-major proxy sector
        # bytes, FP32 norm bytes. work[11:112]: histogram of selected channels per
        # token in 1% bins of K*I. work[112:114]: min and max selected count.
        # One tensor, so the warmup reset clears every counter.
        self.work = torch.zeros(114, device=down.device, dtype=torch.float64)
        # Allocation: parameter bytes and the sum of joint counts.
        self.allocation_work = torch.zeros(2, device=down.device, dtype=torch.float64)
        self.resident_bytes = (
            packed.numel() * packed.element_size()
            + scales.numel() * scales.element_size()
            + 8
        )
        self.resident_norm_bytes: int = (
            4 * self.norms.numel() if self.norms is not None else 0
        )

    def __call__(
        self,
        hidden: Tensor,
        ids: Tensor,
        weights: Tensor,
        padding: Tensor | None = None,
        *,
        diagnostics: dict[str, Tensor] | None = None,
        tokens: slice | Tensor | None = None,
    ) -> Tensor:
        """Compute the routed MoE output ``[tokens, H]`` under Prox's two stages.

        ``hidden`` is ``[tokens, H]``; ``ids`` and ``weights`` are the router's
        ``[tokens, K]`` experts and weights; ``padding`` marks graph padding rows.
        ``diagnostics``, if given, receives this call's masks and proxy scores
        without copies (for untimed analysis callers). ``tokens`` are these rows'
        forward indices, required with an allocation.
        """
        count, k = ids.shape
        h, i = self.hidden, self.width
        self.candidate_channels = k * i
        if max(self.channel_floor, self.channel_keep or 0) > (
            k * i if self.channel_scope == "token" else i
        ):
            raise ValueError("Channel keep/floor exceeds the candidate count")
        valid = (
            torch.ones(count, device=hidden.device, dtype=torch.bool)
            if padding is None
            else ~padding
        )
        # Valid tokens' routed slots whose expert is resident on this rank.
        slot_experts = local_ids(ids, self.expert_map)
        local = valid[:, None] & (slot_experts >= 0)
        assignment = None
        exact_feature: Tensor | None = None
        allocated: Tensor | None = None
        if self.exact:
            assignment = grouped_assignment(
                ids.contiguous().reshape(-1),
                self.expert_map.numel()
                if self.expert_map is not None
                else self.gate_up.shape[0],
                self.expert_map,
                grouped_config(self.gate_up, self.down, k, count),
            )
            # The exact activation of every candidate from its full native rows.
            gate, up = grouped_projection(
                hidden.contiguous(), self.gate_up, assignment, top_k=k
            ).split(i, dim=-1)
            exact_feature = (activate(gate, self.activation).to(hidden.dtype) * up).to(
                hidden.dtype
            )
            score = exact_feature.float()
            input_mask = torch.zeros((count, h), dtype=torch.bool, device=hidden.device)
        else:
            assignment = grouped_assignment(
                ids.contiguous().reshape(-1),
                self.expert_map.numel()
                if self.expert_map is not None
                else self.gate_up.shape[0],
                self.expert_map,
                grouped_config(self.gate_up, self.down, k, count),
            )
            score, input_mask = proxy_scores(
                hidden,
                ids,
                self.packed,
                self.scales,
                self.input_threshold,
                valid,
                assignment,
                self.activation,
                self.expert_map,
                self.input_keep,
                self.recipe == "study",
            )
        salience = score.abs()
        if self.norms is not None:
            salience = output_unit_scores(score, weights, self.norms, slot_experts)
        joint = self.channel_scope == "token" and self.channel_keep is not None
        if self.allocation is not None:
            # One owner per routed slot; the joint vector is identical on EP ranks.
            ranked = torch.where(local.reshape(-1, 1), salience, -1.0).view(
                count, k * i
            )
            if self.expert_map is not None:
                dist.all_reduce(ranked, op=dist.ReduceOp.MAX, group=self.ep_group)
            if tokens is None:
                raise ValueError("An allocation needs the call's forward rows")
            joint_mask = self.allocation(ranked, valid, tokens)
            allocated = joint_mask.sum(-1)  # the joint count, identical on EP ranks
            channel_mask = joint_mask.view(count * k, i)
        elif joint or self.channel_floor:
            # One owner per routed slot; inactive slots stay -1 on every rank.
            # Torch allocates ranked in the graph pool during capture, so the
            # in-place collective reuses its captured storage on every replay.
            ranked = torch.where(local.reshape(-1, 1), salience, -1.0).view(
                count, k * i
            )
            if self.expert_map is not None:
                dist.all_reduce(ranked, op=dist.ReduceOp.MAX, group=self.ep_group)
            joint_mask = prox_top_mask(
                ranked,
                cast(int, self.channel_keep) if joint else self.channel_floor,
                self.recipe == "study",
            ).view(count * k, i)
            channel_mask = (
                joint_mask
                if joint
                else (salience >= cast(float, self.channel_threshold)) | joint_mask
            )
        else:
            channel_mask = salience >= cast(float, self.channel_threshold)
        channel_mask = channel_mask & local.reshape(-1, 1)
        if self.allocation is not None and self.allocation.context.recording:
            assert tokens is not None
            charged = self.allocation.token_bytes(
                self._token_bytes(input_mask, channel_mask, local, valid), valid
            )
            if self.expert_map is not None:
                dist.all_reduce(charged, group=self.ep_group)
            self.allocation.record_bytes(tokens, charged)
        if diagnostics is not None:
            diagnostics.update(
                input_mask=input_mask,
                channel_mask=channel_mask,
                proxy=score,
                local=local,
            )
        if exact_feature is not None:
            assert assignment is not None
            output = grouped_projection(
                (exact_feature * channel_mask).contiguous(), self.down, assignment
            )
        else:
            assert assignment is not None
            # Dense native grouped GEMMs with the shared channel mask; the read
            # cost is computed from the masks, not from these kernels.
            gate, up = grouped_projection(
                hidden.contiguous(), self.gate_up, assignment, top_k=k
            ).split(i, dim=-1)
            gate, up = gate * channel_mask, up * channel_mask
            feature = (activate(gate, self.activation).to(hidden.dtype) * up).to(
                hidden.dtype
            )
            output = grouped_projection(
                (feature * channel_mask).contiguous(), self.down, assignment
            )
        self._record_work(input_mask, channel_mask, local, valid, allocated)
        # Slots without a local expert are never written by the grouped kernels.
        mixed = torch.where(
            local[..., None],
            output.view(count, k, h).float() * weights.float()[..., None],
            0,
        ).sum(1)
        return mixed.to(hidden.dtype)

    def _token_bytes(
        self, input_mask: Tensor, channel_mask: Tensor, local: Tensor, valid: Tensor
    ) -> Tensor:
        """Return each token's read cost on this rank.

        Matches ``report``'s ``charged_bytes`` before ``LayerAllocation.token_bytes``
        adds the token's share of per-call parameter reads.
        """
        count, k = local.shape
        h, i = self.hidden, self.width
        slots = local.sum(-1)
        selected = channel_mask.view(count, k * i).sum(-1)
        native = slots * i if self.exact else selected
        proxy = input_mask.sum(-1) * slots * (-(-i // 32) * 32)
        scale = slots * 2 * i * self.scales.element_size() * (not self.exact)
        norms = slots * 4 * i * (self.norms is not None)
        return proxy + scale + 2 * h * (2 * native + selected) + norms

    def _record_work(
        self,
        input_mask: Tensor,
        channel_mask: Tensor,
        local: Tensor,
        valid: Tensor,
        allocated: Tensor | None = None,
    ) -> None:
        count, k = local.shape
        h, i = self.hidden, self.width
        slots_per_token = local.sum(-1)
        active = slots_per_token.sum()
        selected = channel_mask.sum()
        # The exact score reads every candidate's full gate/up rows and no proxy:
        # its input mask is all false and its scale term is zero.
        retained_inputs = (input_mask.sum(-1) * slots_per_token).sum()
        native_rows = active * i if self.exact else selected
        pair_mask = input_mask[:, 0::2] | input_mask[:, 1::2]
        # Row-major proxy sector estimate, reported for comparison.
        pair_sectors = pair_mask.reshape(count, -1, 32).any(-1).sum(-1)
        tokens = valid.sum()
        counts = channel_mask.view(count, k * i).sum(-1)
        bins = counts * 100 // (k * i)
        self.work[11:112].scatter_add_(0, bins, valid.to(self.work.dtype))
        minimum = counts.masked_fill(~valid, k * i).amin().to(self.work.dtype)
        self.work[112] = torch.where(
            tokens > 0,
            torch.where(self.work[8] > 0, self.work[112].minimum(minimum), minimum),
            self.work[112],
        )
        self.work[113] = self.work[113].maximum(counts.amax())
        self.work[:11] += torch.stack(
            (
                retained_inputs * (-(-i // 32) * 32),
                active * 2 * i * self.scales.element_size() * (not self.exact),
                native_rows * h,
                native_rows * h,
                selected * h,
                active * 3 * i * h * 2,
                retained_inputs,
                active,
                tokens,
                (pair_sectors * slots_per_token).sum() * 2 * i * 32,
                active * 4 * i * (self.norms is not None),
            )
        ).to(self.work.dtype)
        if self.allocation is not None and allocated is not None:
            self.allocation_work += torch.stack(
                (
                    self.allocation.call_parameter_bytes(valid),
                    torch.where(valid, allocated, 0).sum(),
                )
            ).to(self.allocation_work.dtype)

    def _allocation_report(
        self,
        charged: float,
        reference: float,
        tokens: float,
        parameter_bytes: float,
        allocated: float,
    ) -> dict[str, object]:
        """Report an allocation's parameter bytes, per-token reads and count residual.

        Per-token bytes are this rank's share (they sum over EP ranks); the joint
        counts and their target are global, identical on every rank.
        """
        allocation = self.allocation
        if allocation is None:
            return {}
        target = allocation.target_count
        realized = allocated / tokens if tokens else None
        return {
            "allocation_parameter_bytes": parameter_bytes,
            "allocation_charged_bytes_per_token": charged / tokens if tokens else 0.0,
            "allocation_reference_bytes_per_token": reference / tokens
            if tokens
            else 0.0,
            "allocation": {
                "layer": allocation.layer,
                "top_count": allocation.top_count,
                "target_count": target,
                "realized_count": realized,
                "count_residual": realized - target
                if realized is not None and target is not None
                else None,
                "overhead": allocation.overhead,
                "instrumentation": "uncharged: recorded features and charged bytes"
                if allocation.context.recording
                else None,
            },
        }

    def report(self) -> dict[str, object]:
        """Return the accumulated reads, sector prices and selection statistics."""
        (
            proxy,
            scale,
            gate,
            up,
            down,
            reference,
            inputs,
            experts,
            tokens,
            row_sectors,
            norms,
        ) = self.work[:11].cpu().tolist()
        histogram = [int(n) for n in self.work[11:112].cpu().tolist()]
        minimum, maximum = self.work[112:].cpu().tolist()
        n = self.candidate_channels

        def interval(bin_index: int) -> list[int]:
            return [
                ceil(bin_index * n / 100),
                min(n, ceil((bin_index + 1) * n / 100) - 1),
            ]

        def percentile(q: float) -> list[int] | None:
            cumulative = 0
            for bin_index, frequency in enumerate(histogram):
                cumulative += frequency
                if tokens and cumulative >= ceil(q * tokens):
                    return interval(bin_index)
            return None

        # FP32 norm runs (I divisible by 8 in every supported model) fill whole
        # 32-byte sectors, so logical and sector-priced bytes agree.
        auxiliary = norms
        allocation_bytes, allocated = self.allocation_work.cpu().tolist()
        charged = proxy + scale + 2 * (gate + up + down) + auxiliary + allocation_bytes
        ranked_inputs = tokens * self.hidden if self.input_keep is not None else 0
        ranked_channels = (
            experts * self.width
            if self.channel_keep is not None
            or self.channel_floor
            or (
                self.allocation is not None
                and self.allocation.context.source != "threshold"
            )
            else 0
        )
        return {
            "proxy_bytes": proxy,
            "scale_bytes": scale,
            "selection_parameter_bytes": norms,
            "gate_coefficients": gate,
            "up_coefficients": up,
            "down_coefficients": down,
            "routed_reference_coefficients": reference / 2,
            "charged_bytes": charged,
            "logical_charged_bytes": charged,
            "row_major_sector_bytes": row_sectors
            + scale
            + 2 * (gate + up + down)
            + auxiliary
            + allocation_bytes,
            "coordinate_major_sector_bytes": charged,
            "accounting_basis": "analytical_sector_estimate",
            "priced_proxy_layout": "coordinate_major",
            "executed_proxy_layout": "row_major",
            "native_execution": "dense_grouped_masked",
            "actual_traffic_measured": False,
            "reference_bytes": reference,
            "matrix_macs": inputs * 2 * self.width + gate + up + down,
            "proxy_matrix_macs": inputs * 2 * self.width,
            # |w| n |p|: two multiplies per ranked candidate.
            "output_unit_score_multiplies": 2 * experts * self.width
            if self.norms is not None
            else 0,
            "activation_evaluations": experts * self.width + gate / self.hidden,
            # Each stage thresholds or ranks its scores: H per token in Stage 1,
            # I per slot in Stage 2 (ranked per slot or jointly per token).
            "threshold_comparisons": (
                tokens * self.hidden
                if self.input_keep is None and not self.exact
                else 0
            )
            + (experts * self.width if self.channel_keep is None else 0),
            "ranked_input_scores": ranked_inputs,
            "ranked_channel_scores": ranked_channels,
            "tie_break_scan_elements": ranked_inputs + ranked_channels
            if self.recipe == "study"
            else 0,
            "input_keep": self.input_keep,
            "channel_keep": self.channel_keep,
            "channel_scope": self.channel_scope,
            "channel_floor": self.channel_floor,
            "allocation_source": self.allocation.context.source
            if self.allocation is not None
            else None,
            **self._allocation_report(
                charged, reference, tokens, allocation_bytes, allocated
            ),
            "selected_channels_per_token": {
                "scope": "rank_local" if self.expert_map is not None else "token",
                "tokens": int(tokens),
                "candidates": n,
                "min": int(minimum) if tokens else None,
                "max": int(maximum) if tokens else None,
                "mean": down / self.hidden / tokens if tokens else None,
                "p1_interval": percentile(0.01),
                "p5_interval": percentile(0.05),
                "p50_interval": percentile(0.5),
                "histogram": [
                    {"count_interval": interval(b), "tokens": frequency}
                    for b, frequency in enumerate(histogram)
                    if frequency
                ],
                "histogram_bin_width": "1% of K*I (integer count intervals)",
            },
            "proxy_score": self.proxy_score,
            "recipe": self.recipe,
            "topk_tie_break": "lowest_index"
            if self.recipe == "study"
            else "torch_topk",
            "read_fraction": charged / reference if reference else None,
            "resident_calibration_bytes_per_layer": self.resident_bytes,
            "resident_norm_bytes_per_layer": self.resident_norm_bytes,
            "measurement_counter_bytes_per_layer": self.work.numel()
            * self.work.element_size(),
        }
