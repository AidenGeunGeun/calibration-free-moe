"""Sparse decode operator for the routed experts of one MoE layer.

A token's candidate channels are the intermediate channels of its top-k experts. A
resident ternary sketch scores every candidate (the screen), the top-k channels per
token and layer are computed exactly from the native BF16 weights, and an optional
low-rank recovery map (`bank`) estimates the output of the skipped channels. The
vLLM integration calls the operator once per MoE layer and decode step.

Kernel arguments use these symbols: H hidden size, I intermediate channels per
expert, E local experts, K top-k experts per token, N = K * I candidate channels,
F selected channels per token, and R rank of the recovery map.

Stages of `ProgressiveReader.__call__`:

1. Rotate each token's input by a block-Hadamard transform (`_prepare_input`).
2. Sort each token's top-k experts by id, so score ties break by expert (`_prepare`).
3. Screen: score every candidate channel from the sketch (`_sketch_screen`).
4. Keep the top-F channels per token, or those above a threshold (`_select`).
5. Exact computation: native gate and up projections of selected channels (`_acquire`).
6. Down projection of the selected channels, per expert (`_down`).
7. Recovery, on a side stream from stage 4 on: encode the skipped channels
   (`_encode`, `_latent`).
8. Mix experts by router weight and add the recovery decoder term (`_output`).

From `GROUPED_TOKENS` tokens per call, stages 5-7 run expert-major (`_group_*`
kernels), and from `GROUPED_SCREEN_TOKENS` the screen does too: a block of
`GROUP_ROWS` (token, expert) pairs reads its expert's rows once: exact computation and
the down projection read the union of channels its pairs selected, recovery the union
they skipped, and the screen the union of their input coordinates.

Read cost is the bytes the method must read, priced in 32-byte sectors (report key
`charged_bytes`); `_count_reads` adds it to `read_work`. Requested bytes are the loads
as the kernels issue them, before cache reuse (report keys `issued_*`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor

from calibration_free_moe.activation import GatedActivation
from calibration_free_moe.reader.budget import (
    BudgetPlan,
    sketch_code_stride,
    sketch_storage_bytes,
)
from calibration_free_moe.reader.sketch import block_hadamard_signs, build_sketch


def channel_major_down(weight: Tensor) -> Tensor:
    """Re-lay a down weight so each channel's H outputs are contiguous.

    Returns one physical [E, I, H] allocation viewed as the logical [E, H, I]
    weight, so stock kernels and the reader share a single layout.
    """
    return weight.transpose(1, 2).contiguous().transpose(1, 2)


# Triton reads the `constexpr` annotations; the type checker sees plain ints. Lines
# where Triton's stubs fall short carry narrow `pyright: ignore` comments.
if TYPE_CHECKING:
    constexpr = int
    constexpr_tuple = tuple[int, ...]
    static_range = range
else:
    constexpr = tl.constexpr
    constexpr_tuple = tl.constexpr
    static_range = tl.static_range

# Host dimensions use the integer overloads, not Triton's constexpr overloads.
next_power_of_2 = cast(Callable[[int], int], triton.next_power_of_2)
cdiv = cast(Callable[[int, int], int], triton.cdiv)

# (token, expert) pairs per expert-major block.
GROUP_ROWS = 16
# Batch sizes (tokens per call) from which exact computation, down projection and
# encoding, and then also the screen, run expert-major. Tuned on an RTX 4090 for
# the four evaluated model shapes.
GROUPED_TOKENS = 32
GROUPED_SCREEN_TOKENS = 64
# Coordinates per screen chunk. Each lane loads one four-byte word of a packed
# sector, and one chunk covers a whole rotation block.
SCREEN_COORDINATES = 128
# Launch shapes tuned on an RTX PRO 6000 for Qwen3-Thinking at the 18% read budget:
# narrow screen CTAs, two rotation blocks prefetched per screen CTA for one
# token and one for larger batches, and 64-coordinate chunks for batches. The
# same tuning chose the output tile widths and small-batch row counts in
# `ProgressiveReader.__call__`.
SCREEN_WARPS = 2
SCREEN_PREFETCH_BLOCKS = 2
SCREEN_BATCH_PREFETCH_BLOCKS = 1
SCREEN_BATCH_COORDINATES = 64
# Multi-token screens cap registers so that 8 two-warp CTAs fit on an SM.
# Uncapped, ptxas picks between 128 and 168 registers depending on unrelated
# code; 168 leaves 6 CTAs per SM, and a two-token Gemma screen (1152 CTAs) then
# no longer fits one wave on 188 SMs.
SCREEN_BATCH_REGISTERS = 128
# Channels per program of the split-screen reduction, which also scores. Small
# tiles expose more parallel special-function work in the score.
SCREEN_REDUCTION_CHANNELS = 32
# Radix selection for small batches. Nonnegative FP32 score keys use 31 bits
# (adding 0.0 first maps -0.0 to +0.0). A histogram over the top RADIX_BITS bits
# finds the bucket holding the cutoff, and a hashed cache of up to RADIX_CAPACITY
# keys per bucket narrows the search inside it. The last bucket saturates, which
# keeps any high bit pattern in bounds without changing the unsigned order.
RADIX_BITS = 12
RADIX_HASH_BITS = 8
RADIX_CAPACITY = 128
RADIX_TOKENS = 4
# Small batches stream weights evict-first and keep reused metadata evict-last;
# larger batches keep the default policies so weights can be reused across the
# batch.
CACHE_PRIORITY_TOKENS = 4
# Channel score of the screen. "recovery-aware" is the five-point Gauss-Hermite
# estimate of the output error left by skipping a channel, given the recovery
# map; "point" ignores recovery and the sketch residual variances and ranks
# w^2 n^2 (phi(g) u)^2 (router weight w, down-column norm n).
SelectionScore = Literal["recovery-aware", "point"]


def aligned_pairs(pairs: int, experts: int) -> int:
    """Return the pair capacity of the expert sort after padding to blocks.

    Each expert's pairs are padded to a multiple of GROUP_ROWS. Padding adds at
    most GROUP_ROWS - 1 rows per expert that holds a pair, and no more experts
    than pairs can hold one.
    """
    padded = pairs + experts * (GROUP_ROWS - 1)
    return min(pairs * GROUP_ROWS, padded) if pairs < experts else padded


def aligned_blocks(pairs: int, experts: int) -> int:
    """Return the number of GROUP_ROWS blocks the expert sort may fill."""
    return (aligned_pairs(pairs, experts) + GROUP_ROWS - 1) // GROUP_ROWS


@triton.jit
def _prepare(
    RouterIds,
    RouterWeights,
    ExpertMap,
    Ids,  # [tokens, K] int32, sorted local expert ids
    Weights,  # [tokens, K] fp32 router weights in sorted order
    Keys,  # [tokens, K] int32 expert-major grouping keys
    Padding,  # [tokens] bool
    H: constexpr,
    K: constexpr,
    BK: constexpr,
    HAS_PADDING: constexpr,
    EP: constexpr,
    GROUPED: constexpr = False,
    E: constexpr = 0,
    X=None,
    Signs=None,
    Planes=None,
    Split=None,
    Norms=None,
    SCREEN: constexpr = False,
    GROUPED_SCREEN: constexpr = False,
    KEEP: constexpr = 128,
    Coordinates=None,
    ScreenScores=None,
    Completed=None,
    Selected=None,
    N: constexpr = 0,
    BN: constexpr = 1,
    Work=None,
    ACCOUNT: constexpr = False,
    CHARGES: constexpr_tuple = (),
    R: constexpr = 0,
    F: constexpr = 0,
    Histogram=None,
    BucketCounts=None,
    RADIX: constexpr = False,
    HB: constexpr = 512,
    BINS: constexpr = 1 << RADIX_BITS,
    HASH: constexpr = 1 << RADIX_HASH_BITS,
    INPUT_PREPARED: constexpr = False,
):
    """Sort each token's experts and, when the screen runs, rotate its input.

    Grid (tokens, H // 128 if the input is rotated here, else 1). Program (t, 0)
    also zeroes the token's rows for the expert-major screen and, with ACCOUNT,
    adds its fixed read statistics. INPUT_PREPARED skips the rotation and the
    bucket reset, which `_prepare_input` already did.
    """
    token = tl.program_id(0)
    active = not tl.load(Padding + token) if HAS_PADDING else True
    if RADIX and not INPUT_PREPARED:
        _reset_score_buckets(Histogram, BucketCounts, token, active, N, HB, BINS, HASH)
    if SCREEN and not INPUT_PREPARED:
        _rotate_sketch_block(
            X,
            Signs,
            Planes,
            Split,
            Norms,
            token,
            tl.program_id(1),
            active,
            H,
            GROUPED_SCREEN,
            KEEP,
            Coordinates,
        )
    if tl.program_id(1) == 0:
        if SCREEN and GROUPED_SCREEN and ScreenScores is not None:
            # Remote and padded pairs have no expert block: their scores must be
            # zero before the EP all-reduce, and no flag of the previous call may
            # survive. Split screens instead reset every row when they score.
            channel = tl.arange(0, BN)
            tl.store(ScreenScores + token * N + channel, 0, channel < N)
            tl.store(Completed + token * N + channel, 0, channel < N)
            tl.store(Selected + token * N + channel, 0, channel < N)
        _route(
            RouterIds,
            RouterWeights,
            ExpertMap,
            Ids,
            Weights,
            Keys,
            token,
            active,
            K,
            BK,
            EP,
            GROUPED,
            E,
        )
        if ACCOUNT:
            _account_fixed(
                RouterIds, Work, token, active, K, BK, N, R, F, GROUPED, CHARGES
            )


@triton.jit
def _account_fixed(
    RouterIds,
    Work,
    token,
    active,
    K: constexpr,
    BK: constexpr,
    N: constexpr,
    R: constexpr,
    F: constexpr,
    GROUPED: constexpr,
    CHARGES: constexpr_tuple,
):
    """Add one token's selection-independent read statistics to Work[0:18].

    CHARGES holds them (see `ProgressiveReader._fixed_charges`). Per-token
    execution with a recovery map replaces the requested encoder bytes, Work[14],
    by 2 R bytes per skipped local candidate.
    """
    counter = tl.arange(0, 32)
    value = tl.zeros((32,), tl.int64)
    for column in static_range(18):
        value = tl.where(counter == column, CHARGES[column], value)
    if not GROUPED and R > 0:
        slots = tl.arange(0, BK)
        routed = tl.load(RouterIds + token * K + slots, slots < K, -1)
        candidates = tl.sum(((slots < K) & (routed >= 0)).to(tl.int64), 0) * (N // K)
        value = tl.where(counter == 14, 2 * R * (candidates - F), value)
    tl.atomic_add(
        Work + counter,
        value.to(tl.float64),
        (counter < 18) & active,
        sem="relaxed",
    )


@triton.jit
def _reset_score_buckets(
    Histogram,
    BucketCounts,
    token,
    active,
    N: constexpr,
    HB: constexpr,
    BINS: constexpr,
    HASH: constexpr,
):
    """Clear HB radix histogram bins of a token; program (t, 0) also clears buckets."""
    bucket = tl.program_id(1) * HB + tl.arange(0, HB)
    # Padding rows have all-zero scores; keep their tie rule without atomics.
    zeros = tl.where(active, 0, N)
    tl.store(
        Histogram + token * BINS + bucket,
        tl.where(bucket == 0, zeros, 0),
        bucket < BINS,
    )
    if tl.program_id(1) == 0:
        hashed = tl.arange(0, HASH)
        tl.store(BucketCounts + token * HASH + hashed, tl.where(hashed == 0, zeros, 0))


@triton.jit
def _prepare_input(
    X,
    Signs,
    Planes,
    Split,
    Norms,
    Coordinates,
    Padding,
    Histogram,
    BucketCounts,
    H: constexpr,
    N: constexpr,
    HAS_PADDING: constexpr,
    GROUPED_SCREEN: constexpr,
    KEEP: constexpr,
    RADIX: constexpr,
    HB: constexpr,
    BINS: constexpr = 1 << RADIX_BITS,
    HASH: constexpr = 1 << RADIX_HASH_BITS,
):
    """Rotate the input and reset score buckets before the router finishes.

    Grid (tokens, H // 128), on the side stream; `_prepare` then skips both.
    """
    token = tl.program_id(0)
    active = not tl.load(Padding + token) if HAS_PADDING else True
    if RADIX:
        _reset_score_buckets(Histogram, BucketCounts, token, active, N, HB, BINS, HASH)
    _rotate_sketch_block(
        X,
        Signs,
        Planes,
        Split,
        Norms,
        token,
        tl.program_id(1),
        active,
        H,
        GROUPED_SCREEN,
        KEEP,
        Coordinates,
    )


@triton.jit
def _route(
    RouterIds,
    RouterWeights,
    ExpertMap,
    Ids,
    Weights,
    Keys,
    token,
    active,
    K: constexpr,
    BK: constexpr,
    EP: constexpr,
    GROUPED: constexpr,
    E: constexpr,
):
    """Sort one token's top-k experts by id; store ids, weights and grouping keys.

    Under expert parallelism the ids are local, -1 for remote experts.
    """
    # Sorting by expert id breaks channel-score ties by expert, not router slot.
    slot = tl.arange(0, BK)
    original = tl.load(RouterIds + token * K + slot, slot < K, 0).to(tl.int32)
    routed = tl.load(RouterWeights + token * K + slot, (slot < K) & active, 0)
    ordered = tl.sort(tl.where(slot < K, original, 0x7FFFFFFF), descending=False)
    match = (ordered[:, None] == original[None, :]) & (slot[None, :] < K)
    weights = tl.sum(tl.where(match, routed[None, :], 0), axis=1)
    if EP:
        # A negative router id (e.g. in a profiling batch) stays -1 instead of
        # indexing the map.
        ordered = tl.load(ExpertMap + ordered, (slot < K) & (ordered >= 0), -1)
    tl.store(Ids + token * K + slot, ordered, slot < K)
    tl.store(Weights + token * K + slot, weights, slot < K)
    if GROUPED:
        # Expert-major grouping keys; padding and remote experts get the
        # out-of-range key E, which the alignment leaves out of every block.
        key = tl.where((ordered >= 0) & active, ordered, E)
        tl.store(Keys + token * K + slot, key, slot < K)


@triton.jit
def _rotate_sketch_block(
    X,
    Signs,
    Planes,
    Split,
    Norms,
    token,
    block,
    active,
    H: constexpr,
    GROUPED: constexpr,
    KEEP: constexpr,
    Coordinates,
):
    """Rotate one 128-coordinate block of a token's input for the screen.

    Computes H_128 diag(signs) x / sqrt(128) in FP32 and, when KEEP < 128, zeroes
    all but the KEEP largest magnitudes. Writes the squared norm of the kept values
    and either the three-term BF16 split (expert-major screen, [tokens, H // 128,
    3, 128]) or the compact kept values and their coordinates ([tokens, H // 128,
    KEEP]; coordinates only when KEEP < 128).
    """
    coordinate = tl.arange(0, 128)
    offset = block * 128 + coordinate
    value = tl.load(X + token * H + offset, active, 0).to(tl.float32)
    value *= tl.load(Signs + offset, active, 0).to(tl.float32)
    for stage in static_range(7):
        partner = tl.gather(value, coordinate ^ (1 << stage), axis=0)
        value = tl.where(
            coordinate & (1 << stage) == 0, value + partner, partner - value
        )
    value *= 0.08838834764831845
    if KEEP < 128:
        # Keep the KEEP largest magnitudes, breaking ties in coordinate order.
        magnitude = tl.abs(value)
        ordered = tl.sort(magnitude, descending=True)
        threshold = tl.sum(tl.where(coordinate == KEEP - 1, ordered, 0.0), axis=0)
        above = magnitude > threshold
        tied = magnitude == threshold
        ties = KEEP - tl.sum(above.to(tl.int32), axis=0)
        kept = above | (tied & (tl.cumsum(tied.to(tl.int32), axis=0) <= ties))
        value = tl.where(kept, value, 0.0)
    else:
        kept = tl.full((128,), True, tl.int1)
    cell = token * (H // 128) + block
    tl.store(Norms + cell, tl.sum(value * value, axis=0))
    if GROUPED:
        # Three BF16 terms represent each FP32 input exactly, so FP32-accumulated
        # MMAs keep FP32 projections.
        high = value.to(tl.bfloat16)
        rest = value - high.to(tl.float32)
        middle = rest.to(tl.bfloat16)
        low = (rest - middle.to(tl.float32)).to(tl.bfloat16)
        tl.store(Split + cell * 384 + coordinate, high)
        tl.store(Split + cell * 384 + 128 + coordinate, middle)
        tl.store(Split + cell * 384 + 256 + coordinate, low)
    else:
        # Compact in coordinate order, not magnitude order, for reproducible sums.
        if KEEP < 128:
            position = tl.cumsum(kept.to(tl.int32), axis=0) - 1
            tl.store(Coordinates + cell * KEEP + position, coordinate, kept)
        else:
            position = coordinate
        tl.store(Planes + cell * KEEP + position, value, kept)


@triton.jit
def _nearest(value):
    """Return the nearest integer as a float; exact for |value| < 2**22."""
    return (value + 12582912.0) - 12582912.0


@triton.jit
def _screen_route(RouterIds, RouterWeights, token, slot, active, K: constexpr):
    """Return the expert id and router weight of one slot in id order.

    Every screen CTA repeats the small sort of the token's K router ids, so no CTA
    reads another's stores.
    """
    slots = tl.arange(0, triton.next_power_of_2(K))
    original = tl.load(RouterIds + token * K + slots, slots < K, 0).to(tl.int32)
    ordered = tl.sort(tl.where(slots < K, original, 0x7FFFFFFF), descending=False)
    expert = tl.sum(tl.where(slots == slot, ordered, 0), 0)
    routed = tl.load(RouterWeights + token * K + slots, (slots < K) & active, 0)
    weight = tl.sum(tl.where((original == expert) & (slots < K), routed, 0), 0)
    return expert.to(tl.int64), weight


@triton.jit
def _sketch_screen(
    Codes,  # [E, sketch_storage_bytes(2 I, H)] uint8, five trits per byte
    Statistics,  # [E, 2 I, H // 128, 2] fp16 scale and residual variance
    Planes,  # [tokens, H // 128, KEEP] fp32 kept rotated input values
    Coordinates,  # [tokens, H // 128, KEEP] int32 their coordinates (KEEP < 128)
    Norms,  # [tokens, H // 128] fp32 squared norm of the kept values
    Ids,  # [tokens, K] int32 local expert ids in id order
    Partial,  # [tokens, 2, N] fp32 gate and up projections
    Variances,  # [tokens, 2, N] fp32 their variances
    Parts,  # [tokens, SPLITS, 2, N] fp32 per-split projections
    VarianceParts,  # [tokens, SPLITS, 2, N] fp32 per-split variances
    Padding,  # [tokens] bool
    Weights,  # [tokens, K] fp32 router weights
    DownNorms,  # [E, I] fp32 down-column norms
    Geometry,  # [E, I, 2] fp32 recovery geometry (alpha, beta)
    Scores,  # [tokens, N] fp32
    Completed,  # [tokens, N] bool, channel computed exactly
    Selected,  # [tokens, N] bool, channel selected
    H: constexpr,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    HAS_PADDING: constexpr,
    KEEP: constexpr,
    STRIDE: constexpr,
    STORAGE: constexpr,
    SPLITS: constexpr,
    GELU: constexpr,
    GEOMETRY: constexpr,
    POINT: constexpr = False,
    BX: constexpr = SCREEN_COORDINATES,
    Histogram=None,
    BucketCounts=None,
    BucketKeys=None,
    RADIX: constexpr = False,
    Counters=None,
    LAST_BLOCK: constexpr = False,
    BS: constexpr = 1,
    PREFETCH: constexpr = SCREEN_PREFETCH_BLOCKS,
    PRIORITIZE: constexpr = False,
    BATCH: constexpr = False,
    RouterIds=None,
    RouterWeights=None,
    ROUTE: constexpr = False,
    Work=None,
    ACCOUNT: constexpr = False,
    CHARGES: constexpr_tuple = (),
    R: constexpr = 0,
    F: constexpr = 0,
):
    """Screen 80 candidate channels of one top-k expert of one token.

    Grid (cdiv(I, 80), K, tokens * SPLITS); program (c, k, t * SPLITS + s) covers
    channels [80 c, 80 c + 80) of the token's k-th expert in id order, over split
    s of its 128-coordinate rotation blocks. Per kept coordinate it reads one
    32-byte code sector, five trits per byte for 80 gate and 80 up rows. Per
    rotation block it reads the rows' FP16 scales, five sectors per 80 rows, and
    for the recovery-aware score their FP16 residual variances. The trits are
    decoded in registers and projected onto the rotated input.

    Without splits the program scores directly (`_store_screen`). With splits it
    writes Parts and VarianceParts; with LAST_BLOCK the last split to arrive sums
    them in split order and scores, otherwise `_acquire` does. With ROUTE it also
    sorts the token's experts (`_screen_route`) and, with ACCOUNT, adds the
    token's fixed read statistics.
    """
    token = tl.program_id(2) // SPLITS
    split = tl.program_id(2) % SPLITS
    slot = tl.program_id(1)
    reused: tl.constexpr = "evict_last" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
    active = not tl.load(Padding + token) if HAS_PADDING else True
    blocks: constexpr = H // 128
    byte = tl.program_id(0) * 32 + tl.arange(0, 32)
    route_weight = 0.0
    if ROUTE:
        expert, route_weight = _screen_route(  # pyright: ignore[reportGeneralTypeIssues]
            RouterIds, RouterWeights, token, slot, active, K
        )
        # One writer per slot; downstream kernels consume these only after the screen.
        if tl.program_id(0) == 0 and split == 0:
            tl.store(Ids + token * K + slot, expert.to(tl.int32))
            tl.store(Weights + token * K + slot, route_weight)
            if ACCOUNT and slot == 0:
                _account_fixed(
                    RouterIds,
                    Work,
                    token,
                    active,
                    K,
                    triton.next_power_of_2(K),
                    N,
                    R,
                    F,
                    False,
                    CHARGES,
                )
    else:
        expert = tl.load(Ids + token * K + slot).to(tl.int64)
    # Per-digit (32,) byte-lane vectors; digit d of byte b is channel b // 2 * 5 + d.
    zero = tl.zeros((32,), tl.float32)
    estimate = (zero, zero, zero, zero, zero)
    variance = estimate
    span: constexpr = tl.cdiv(blocks, SPLITS)
    prefetch: constexpr = min(PREFETCH, span)
    end_block = tl.minimum((split + 1) * span, blocks)
    # Code addresses depend on the kept coordinates: load each group's one group
    # ahead, so no code load in the loop waits on a coordinate load.
    staged = ()
    for offset in static_range(prefetch):
        block = split * span + offset
        staged += (
            _screen_block_coordinates(
                Coordinates,
                token,
                block,
                active & (expert >= 0) & (block < end_block),
                blocks,
                KEEP,
                BX,
            ),
        )
    for begin_block in range(split * span, end_block, prefetch):
        coordinates = staged
        loaded = ()
        # Start this group's code and metadata loads before decoding its trits.
        # Keeping codes packed bounds the registers held by the prefetch.
        for offset in static_range(prefetch):
            block = begin_block + offset
            used = (
                True
                if blocks % SPLITS == 0 and span % prefetch == 0
                else block < end_block
            )
            loaded += (
                _screen_block_inputs(
                    Codes,
                    Statistics,
                    Planes,
                    Norms,
                    expert,
                    token,
                    block,
                    tl.program_id(0),
                    active & (expert >= 0) & used,
                    coordinates[offset],
                    I,
                    blocks,
                    KEEP,
                    STRIDE,
                    STORAGE,
                    POINT,
                    BX,
                    PRIORITIZE,
                    BATCH,
                ),
            )
        staged = ()
        for offset in static_range(prefetch):
            block = begin_block + prefetch + offset
            staged += (
                _screen_block_coordinates(
                    Coordinates,
                    token,
                    block,
                    active & (expert >= 0) & (block < end_block),
                    blocks,
                    KEEP,
                    BX,
                ),
            )
        for offset in static_range(prefetch):
            used = (
                True
                if blocks % SPLITS == 0 and span % prefetch == 0
                else begin_block + offset < end_block
            )
            if used:
                words, inputs, scale, residual, norm2 = loaded[offset]
                estimate, variance = _screen_block_projection(
                    words,
                    inputs,
                    scale,
                    residual,
                    norm2,
                    estimate,
                    variance,
                    KEEP,
                    BX,
                    POINT,
                    BATCH,
                )
    if SPLITS > 1:
        for digit in static_range(5):
            channel = byte // 2 * 5 + digit
            address = ((token * SPLITS + split) * 2 + byte % 2) * N + slot * I + channel
            tl.store(Parts + address, estimate[digit], channel < I)
            if not POINT:
                tl.store(VarianceParts + address, variance[digit], channel < I)
        if LAST_BLOCK:
            counter = Counters + (token * K + slot) * tl.cdiv(I, 80) + tl.program_id(0)
            # All lanes publish their partials before the single counter RMW.
            # GPU-scope acq_rel chains the arrivals (release is cumulative over the
            # CTA's stores after the barrier); the barrier after the RMW orders its
            # acquire before every lane's reads. No spinning or grid barrier; the
            # last CTA reads by split index, never arrival order.
            tl.debug_barrier()
            arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
            tl.debug_barrier()
            if arrived == SPLITS - 1:
                local = tl.arange(0, 128)
                channels = tl.program_id(0) * 80 + local
                valid_channels = (local < 80) & (channels < I)
                weight = (
                    route_weight
                    if ROUTE
                    else tl.load(Weights + token * K + slot, active, 0)
                )
                norm = tl.load(
                    DownNorms + expert * I + channels,
                    valid_channels & active & (expert >= 0),
                    0,
                    eviction_policy=reused,
                )
                gate, up, vg, vu = _screen_sums(
                    Parts,
                    VarianceParts,
                    token,
                    tl.where(valid_channels, slot * I + channels, N),
                    N,
                    SPLITS,
                    BS,
                    POINT,
                    LAST_BLOCK=True,
                )
                _store_screen(
                    gate,
                    up,
                    vg,
                    vu,
                    token,
                    slot,
                    channels,
                    expert,
                    valid_channels,
                    active,
                    Partial,
                    Variances,
                    weight,
                    DownNorms,
                    Geometry,
                    Scores,
                    Completed,
                    Selected,
                    I,
                    N,
                    GELU,
                    GEOMETRY,
                    POINT,
                    Histogram,
                    BucketCounts,
                    BucketKeys,
                    RADIX,
                    NormValue=norm,
                    PRIORITIZE=PRIORITIZE,
                )
                # Every split has arrived, and calls and graph replays share one
                # ordered caller stream, so no CTA of this launch increments again.
                tl.store(counter, 0)
    else:
        gate, up = tl.split(_digit_tile(estimate).reshape(8, 16, 2))
        vg, vu = tl.split(_digit_tile(variance).reshape(8, 16, 2))
        local = tl.arange(0, 128)
        order = local % 5 * 16 + local // 5
        gate = tl.gather(gate.reshape(128), order, axis=0)
        up = tl.gather(up.reshape(128), order, axis=0)
        vg = tl.gather(vg.reshape(128), order, axis=0)
        vu = tl.gather(vu.reshape(128), order, axis=0)
        channels = tl.program_id(0) * 80 + local
        _store_screen(
            gate,
            up,
            vg,
            vu,
            token,
            slot,
            channels,
            expert,
            (local < 80) & (channels < I),
            active,
            Partial,
            Variances,
            route_weight if ROUTE else tl.load(Weights + token * K + slot, active, 0),
            DownNorms,
            Geometry,
            Scores,
            Completed,
            Selected,
            I,
            N,
            GELU,
            GEOMETRY,
            POINT,
            Histogram,
            BucketCounts,
            BucketKeys,
            RADIX,
            PRIORITIZE=PRIORITIZE,
        )


@triton.jit
def _load_screen_words(
    Codes,
    expert,
    block,
    coordinate,
    tile,
    used,
    I: constexpr,
    BLOCKS: constexpr,
    STRIDE: constexpr,
    STORAGE: constexpr,
    DENSE: constexpr = False,
    PACKED: constexpr = False,
    PRIORITIZE: constexpr = False,
    BATCH: constexpr = False,
):
    """Load the packed codes of one 80-channel tile for a chunk of coordinates.

    Each coordinate's tile is one 32-byte sector, loaded as eight aligned uint32
    words and returned as [coordinates, 8] words when PACKED, else [coordinates,
    32] bytes. A final word may include two padding bytes, never another sector;
    short tails of adjacent rotation blocks share a sector (see `build_sketch`).
    BATCH indexes in words up to the 16-byte padded stride, allowing 16-byte
    vector loads. Padding holds the zero code 121, also the masked-load value, so
    a padding word is 0x79797979 either way.
    """
    if BATCH:
        word = tile * 8 + tl.arange(0, 8)
        # Index in words: every byte offset is a multiple of four, so the INT32
        # in-expert index (below STORAGE // 4) is the byte offset over four and
        # the INT64 expert base is added once per load.
        body: constexpr = STRIDE // 32 * 8  # pyright: ignore[reportRedeclaration]
        offset = coordinate[:, None] * body + (block * 128 * body + word)[None, :]
        if STRIDE % 32:
            tail = BLOCKS * 128 * body + block // 2 * 1024 + block % 2 * 4 - body
            tail_offset = coordinate[:, None] * 8 + (tail + word)[None, :]
            offset = tl.where(word[None, :] < body, offset, tail_offset)
        address = (
            Codes.to(tl.pointer_type(tl.uint32)) + expert * (STORAGE // 4) + offset
        )
        bound = word < STRIDE // 4
    else:
        byte = tile * 32 + tl.arange(0, 8) * 4
        body: constexpr = STRIDE // 32 * 32
        offset = (block * 128 + coordinate[:, None]) * body + byte[None, :]
        if STRIDE % 32:
            tail_offset = (
                BLOCKS * 128 * body
                + (block // 2 * 128 + coordinate[:, None]) * 32
                + block % 2 * 16
                + byte[None, :]
                - body
            )
            offset = tl.where(byte[None, :] < body, offset, tail_offset)
        address = (
            Codes.to(tl.pointer_type(tl.uint32)) + (expert * STORAGE + offset) // 4
        )
        bound = byte < 2 * tl.cdiv(I, 5)
    priority: tl.constexpr = "evict_first" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
    words = tl.load(
        address,
        bound[None, :] if DENSE else used[:, None] & bound[None, :],
        0x79797979,
        eviction_policy=priority,  # pyright: ignore[reportArgumentType]
    )
    return words if PACKED else _screen_bytes(words)


@triton.jit
def _screen_bytes(words):
    """Split [rows, 8] uint32 words into [rows, 32] bytes, low byte first."""
    shifts = tl.arange(0, 4) * 8
    return ((words[:, :, None] >> shifts[None, None, :]) & 255).reshape(
        words.shape[0], 32
    )


@triton.jit
def _screen_biased(words):
    """Place each byte under the 2**23 exponent (0x4B000000 | byte) by one PRMT.

    Read as FP32, each lane equals 2**23 + byte: the byte-to-float trick of integer
    dequantizers, with the lanes of `_screen_bytes` but no shift, mask or convert.
    """
    lanes = tl.arange(0, 4)[None, None, :]
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, $1, 0x4B000000, $2;",
        "=r,r,r",
        [words[:, :, None], lanes | 0x7440],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).reshape(words.shape[0], 32)


@triton.jit
def _screen_block_coordinates(
    Coordinates,
    token,
    block,
    active,
    BLOCKS: constexpr,
    KEEP: constexpr,
    BX: constexpr,
):
    """Return the kept coordinates of one rotation block in BX-coordinate chunks.

    The full screen keeps every coordinate and reads nothing; the partial screen
    reads its compact INT32 indices.
    """
    coordinates = ()
    for begin in static_range(0, KEEP, BX):
        position = begin + tl.arange(
            0,
            BX if KEEP - begin >= BX else triton.next_power_of_2(KEEP - begin),
        )
        cell = (token * BLOCKS + block) * KEEP + position
        used = active & (position < KEEP)
        coordinates += (
            position if KEEP == 128 else tl.load(Coordinates + cell, used, 0),
        )
    return coordinates


@triton.jit
def _screen_block_inputs(
    Codes,
    Statistics,
    Planes,
    Norms,
    expert,
    token,
    block,
    tile,
    active,
    coordinates,
    I: constexpr,
    BLOCKS: constexpr,
    KEEP: constexpr,
    STRIDE: constexpr,
    STORAGE: constexpr,
    POINT: constexpr,
    BX: constexpr,
    PRIORITIZE: constexpr,
    BATCH: constexpr,
):
    """Start one rotation block's screen loads before any trit arithmetic.

    Returns the packed code words and FP32 rotated inputs per BX-coordinate chunk,
    the FP16 scales (and residual variances) of the tile's 80 gate and 80 up rows
    as five per-digit byte-lane vectors, and the block's squared input norm.
    """
    words = ()
    inputs = ()
    for begin in static_range(0, KEEP, BX):
        position = begin + tl.arange(
            0,
            BX if KEEP - begin >= BX else triton.next_power_of_2(KEEP - begin),
        )
        cell = (token * BLOCKS + block) * KEEP + position
        used = active & (position < KEEP)
        inputs += (tl.load(Planes + cell, used, 0),)
        words += (
            _load_screen_words(
                Codes,
                expert,
                block,
                coordinates[begin // BX],
                tile,
                used,
                I,
                BLOCKS,
                STRIDE,
                STORAGE,
                PACKED=True,
                PRIORITIZE=PRIORITIZE,
                BATCH=BATCH,
            ),
        )
    # Load each digit's gate/up metadata in its byte-lane order: the same
    # sectors as one contiguous load, without an on-chip permutation.
    byte = tl.arange(0, 32)
    reused: tl.constexpr = "evict_last" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
    scale = ()
    residual = ()
    for digit in static_range(5):
        channel = tile * 80 + byte // 2 * 5 + digit
        address = (expert * BLOCKS + block) * 4 * I + byte % 2 * I + channel
        used = active & (channel < I)
        scale += (tl.load(Statistics + address, used, 0, eviction_policy=reused),)  # pyright: ignore[reportArgumentType]
        if not POINT:
            residual += (
                tl.load(Statistics + address + 2 * I, used, 0, eviction_policy=reused),  # pyright: ignore[reportArgumentType]
            )
    if POINT:
        norm2 = 0.0
    else:
        norm2 = tl.load(
            Norms + token * BLOCKS + block,
            active,
            0,
            eviction_policy=reused,  # pyright: ignore[reportArgumentType]
        )
    return words, inputs, scale, residual, norm2


@triton.jit
def _screen_block_projection(
    words,
    inputs,
    scale,
    residual,
    norm2,
    estimate,
    variance,
    KEEP: constexpr,
    BX: constexpr,
    POINT: constexpr,
    BATCH: constexpr,
):
    """Add one rotation block's projections and variances to the running sums.

    A row's projection gains scale * <trits, rotated x>; for the recovery-aware
    score its variance gains the residual variance times the squared input norm.
    """
    zero = tl.zeros((32,), tl.float32)
    dots = (zero, zero, zero, zero, zero)
    for chunk in static_range(tl.cdiv(KEEP, BX)):
        if BATCH:  # noqa: SIM108
            code = _screen_biased(words[chunk])
        else:
            code = _screen_bytes(words[chunk])
        dots = _screen_dots(code, inputs[chunk], dots, BATCH)
    estimated = ()
    varied = ()
    for digit in static_range(5):
        estimated += (estimate[digit] + dots[digit] * scale[digit].to(tl.float32),)
        if POINT:
            varied += (variance[digit],)
        else:
            varied += (variance[digit] + residual[digit].to(tl.float32) * norm2,)
    return estimated, varied


@triton.jit
def _screen_dots(code, x, dots, BIASED: constexpr):
    """Add one chunk's five per-digit dot products with x to the running dots.

    A byte holds five balanced trits t_d in {-1, 0, 1} as 121 + sum_d t_d 3**d,
    so digit d of the centered code v is q_d - 3 q_{d+1} with q_d the nearest
    integer to v / 3**d. Each q_d is computed directly rather than by peeling
    digits, so the five quotients are independent and can overlap. A biased byte
    (`_screen_biased`) gives v by one exact subtraction of 2**23 + 121.
    """
    if BIASED:
        value = code.to(tl.float32, bitcast=True) - 8388729.0
    else:
        value = code.to(tl.float32) - 121.0
    q1 = _nearest(value * (1 / 3))
    dot0 = dots[0] + tl.sum((value - 3.0 * q1) * x[:, None], axis=0)
    q2 = _nearest(value * (1 / 9))
    dot1 = dots[1] + tl.sum((q1 - 3.0 * q2) * x[:, None], axis=0)
    q3 = _nearest(value * (1 / 27))
    dot2 = dots[2] + tl.sum((q2 - 3.0 * q3) * x[:, None], axis=0)
    q4 = _nearest(value * (1 / 81))
    dot3 = dots[3] + tl.sum((q3 - 3.0 * q4) * x[:, None], axis=0)
    dot4 = dots[4] + tl.sum(q4 * x[:, None], axis=0)
    return dot0, dot1, dot2, dot3, dot4


@triton.jit
def _digit_tile(digits):
    """Stack five per-digit byte-lane vectors as an (8, 32) digit-major tile."""
    zero = tl.zeros((32,), tl.float32)
    low = tl.cat(tl.cat(digits[0], digits[1]), tl.cat(digits[2], digits[3]))
    high = tl.cat(tl.cat(digits[4], zero), tl.cat(zero, zero))
    return tl.cat(low, high).reshape(8, 32)


@triton.jit
def _screen_statistics(
    Statistics,
    expert,
    block,
    tile,
    active,
    I: constexpr,
    BLOCKS: constexpr,
    POINT: constexpr,
    PRIORITIZE: constexpr = False,
):
    """Read one tile's FP16 scales (and variances) once as (8, 32) digit-major tiles.

    Used by the expert-major screen; the layout matches `_digit_tile`.
    """
    local = tl.arange(0, 256)
    channel = tile * 80 + local % 128
    valid = (local % 128 < 80) & (channel < I) & active
    address = (expert * BLOCKS + block) * 4 * I + local // 128 * I + channel
    priority: tl.constexpr = "evict_last" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
    scale = tl.load(Statistics + address, valid, 0, eviction_policy=priority).to(  # pyright: ignore[reportArgumentType]
        tl.float32
    )
    if POINT:
        residual = tl.zeros((256,), tl.float32)
    else:
        residual = tl.load(
            Statistics + address + 2 * I,
            valid,
            0,
            eviction_policy=priority,  # pyright: ignore[reportArgumentType]
        ).to(tl.float32)
    byte = tl.arange(0, 32)
    digit = tl.arange(0, 8)
    order = byte[None, :] % 2 * 128 + byte[None, :] // 2 * 5 + digit[:, None]
    return (
        tl.gather(scale, order.reshape(256), axis=0).reshape(8, 32),
        tl.gather(residual, order.reshape(256), axis=0).reshape(8, 32),
    )


@triton.jit
def _store_screen(
    gate,
    up,
    vg,
    vu,
    token,
    slot,
    channel,
    expert,
    valid,
    active,
    Partial,
    Variances,
    weight,
    DownNorms,
    Geometry,
    Scores,
    Completed,
    Selected,
    I: constexpr,
    N: constexpr,
    GELU: constexpr,
    GEOMETRY: constexpr,
    POINT: constexpr,
    Histogram=None,
    BucketCounts=None,
    BucketKeys=None,
    RADIX: constexpr = False,
    NormValue=None,
    PRIORITIZE: constexpr = False,
):
    """Store screened projections and their scores.

    Writes the FP32 gate and up projections and, unless POINT, their variances;
    reads the down-column norm (unless the caller passes NormValue) and, with
    GEOMETRY, the channel's recovery geometry; writes the score, adds it to the
    radix buckets and clears the channel's exact-computation and selection flags.
    """
    # The caller loads router weights with a scalar or per-token mask before
    # channel broadcasting: a scalar pointer cannot take a channel-block mask.
    logical = slot * I + channel
    address = token * 2 * N + logical
    tl.store(Partial + address, gate, valid)
    tl.store(Partial + address + N, up, valid)
    # POINT scoring and recovery consume projections, never residual variances.
    if not POINT:
        tl.store(Variances + address, vg, valid)
        tl.store(Variances + address + N, vu, valid)
    local = valid & active & (expert >= 0)
    column = expert * I + channel
    priority: tl.constexpr = "evict_last" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
    if GEOMETRY:
        alpha = tl.load(Geometry + 2 * column, local, 0, eviction_policy=priority)  # pyright: ignore[reportArgumentType]
        beta = tl.load(Geometry + 2 * column + 1, local, 0, eviction_policy=priority)  # pyright: ignore[reportArgumentType]
    else:
        alpha = tl.full(gate.shape, 0, tl.float32)
        beta = tl.full(gate.shape, 0, tl.float32)
    score = _screen_score(
        gate,
        up,
        vg,
        vu,
        tl.load(DownNorms + column, local, 0, eviction_policy=priority)  # pyright: ignore[reportArgumentType]
        if NormValue is None
        else NormValue,
        weight,
        alpha,
        beta,
        GELU,
        POINT,
    )
    tl.store(Scores + token * N + logical, score, valid)
    if RADIX:
        _bucket_scores(
            score, valid & active, token, Histogram, BucketCounts, BucketKeys
        )
    tl.store(Completed + token * N + logical, 0, valid)
    tl.store(Selected + token * N + logical, 0, valid)


@triton.jit
def _bucket_scores(
    score,
    valid,
    token,
    Histogram,
    BucketCounts,
    BucketKeys,
    SHIFT: constexpr = 31 - RADIX_BITS,
    BINS: constexpr = 1 << RADIX_BITS,
    HASH: constexpr = 1 << RADIX_HASH_BITS,
    CAP: constexpr = RADIX_CAPACITY,
):
    """Count scores in the token's radix histogram and cache their keys.

    Counts are exact per bucket ([tokens, BINS]); keys go to a bounded hashed
    cache ([tokens, HASH, CAP]). Overflow never drops a candidate: `_select`
    falls back to the full scores when the chosen bucket's cache exceeds CAP.
    Hash collisions only affect speed.
    """
    key = (score + 0.0).to(tl.uint32, bitcast=True)
    bucket = tl.minimum(key >> SHIFT, BINS - 1)
    tl.atomic_add(Histogram + token * BINS + bucket, 1, valid, sem="relaxed")
    hashed = bucket & (HASH - 1)
    offset = tl.atomic_add(
        BucketCounts + token * HASH + hashed, 1, valid, sem="relaxed"
    )
    tl.store(
        BucketKeys + (token * HASH + hashed) * CAP + offset, key, valid & (offset < CAP)
    )


@triton.jit
def _screen_sums(
    Parts,
    VarianceParts,
    token,
    logical,
    N: constexpr,
    SPLITS: constexpr,
    BS: constexpr,
    POINT: constexpr,
    LAST_BLOCK: constexpr = False,
):
    """Sum a split screen's projections and variances in split order.

    Used by the last screen CTA to arrive and by `_acquire`. The fixed split order
    makes the sum independent of arrival order and CTA layout.
    """
    valid = logical < N
    address = token * SPLITS * 2 * N + logical
    # L1-bypass policy for partials written by other CTAs (a performance hint;
    # ordering comes from the counter's acquire/release and CTA barriers).
    cache: constexpr = ".cg" if LAST_BLOCK else ""  # pyright: ignore[reportAssignmentType]
    gate = tl.load(Parts + address, valid, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
    up = tl.load(Parts + address + N, valid, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
    if POINT:
        vg = tl.zeros_like(gate)
        vu = tl.zeros_like(up)
    else:
        vg = tl.load(VarianceParts + address, valid, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
        vu = tl.load(VarianceParts + address + N, valid, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
    for split in static_range(1, BS):
        address = (token * SPLITS + split) * 2 * N + logical
        mask = valid & (split < SPLITS)
        gate += tl.load(Parts + address, mask, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
        up += tl.load(Parts + address + N, mask, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
        if not POINT:
            vg += tl.load(VarianceParts + address, mask, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
            vu += tl.load(VarianceParts + address + N, mask, 0, cache_modifier=cache)  # pyright: ignore[reportArgumentType]
    return gate, up, vg, vu


@triton.jit
def _activate(g, GELU: constexpr):
    """Apply the gate activation phi: SiLU, or the tanh approximation of GELU."""
    if GELU:
        # (1 + tanh(u)) / 2 = sigmoid(2u); retain the tanh GELU approximation.
        return g * tl.sigmoid(1.5957691216057308 * (g + 0.044715 * g * g * g))
    return g * tl.sigmoid(g)


@triton.jit
def _screen_score(
    gate,
    up,
    gate_variance,
    up_variance,
    norm,
    weight,
    alpha,
    beta,
    GELU: constexpr,
    POINT: constexpr = False,
):
    """Score channels from their FP32 sketch projections.

    The recovery-aware score is w^2 times the expected squared output error of
    skipping the channel (`_gh5_risk`), with router weight w. POINT ranks
    w^2 n^2 (phi(g) u)^2 instead. Projections are first rounded to BF16, as the
    native activations are.
    """
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    if POINT:
        point = norm * (_activate(g, GELU) * u)
        risk = point * point
    else:
        risk = _gh5_risk(g, u, gate_variance, up_variance, norm, alpha, beta, GELU)
    return risk * weight * weight


@triton.jit
def _gh5_risk(g, u, gate_variance, up_variance, norm, alpha, beta, GELU: constexpr):
    """Return the expected squared output error of skipping one channel.

    The native gate and up projections are modeled as Gaussian around the sketch
    estimates g and u, with the sketch residual variances. With activation
    a = phi(gate) * up, down-column norm n, and the recovery response split into
    alpha along the down column and beta across it, skipping the channel and
    recovering it from the sketch value p = phi(g) u leaves

        n^2 Var[a] + (n E[a] - alpha p)^2 + (beta p)^2.

    Without a recovery map alpha = beta = 0, leaving n^2 E[a^2]. Expectations over
    the gate use five Gauss-Hermite nodes.
    """
    sg = tl.sqrt(gate_variance)
    su = tl.sqrt(up_variance)
    variance = su * su
    nodes = (
        -2.8569700138728056,
        -1.355626179974266,
        0.0,
        1.355626179974266,
        2.8569700138728056,
    )
    quadrature = (
        0.011257411327720693,
        0.2220759220056126,
        0.5333333333333333,
        0.2220759220056126,
        0.011257411327720693,
    )
    energy = tl.zeros_like(g)
    first_moment = tl.zeros_like(g)
    paid_partial = tl.zeros_like(g)
    for j in static_range(5):
        shifted = g + nodes[j] * sg
        activation = _activate(shifted, GELU)
        mean = u
        energy += quadrature[j] * activation * activation * (mean * mean + variance)
        first_moment += quadrature[j] * activation * mean
        if j == 2:
            paid_partial = (activation * u).to(tl.bfloat16).to(tl.float32)
    parallel = norm * first_moment - alpha * paid_partial
    perpendicular = beta * paid_partial
    return (
        norm * norm * tl.maximum(energy - first_moment * first_moment, 0.0)
        + parallel * parallel
        + perpendicular * perpendicular
    )


@triton.jit
def _pdl_signal():
    """Release this grid's writes, then let the dependent grid launch (PDL, SM90+)."""
    # PTX griddepcontrol (ISA Section 9.7.15.14): release before signaling
    # same-domain grids.
    tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; // dummy $0",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.extra.cuda.gdc_launch_dependents()  # pyright: ignore[reportAttributeAccessIssue]


@triton.jit
def _acquire(
    NativeGU,  # [E, 2 I, H] bf16 native gate and up rows
    X,  # [tokens, H] bf16 input
    Ids,  # [tokens, K] int32
    Weights,  # [tokens, K] fp32
    Live,  # [tokens, COUNT] int32 candidate ids kept by `_select`
    Partial,  # [tokens, 2, N] fp32 gate and up projections
    Variances,  # [tokens, 2, N] fp32
    Activations,  # [tokens, N] bf16 activations of exactly computed channels
    Completed,  # [tokens, N] bool
    Selected,  # [tokens, N] bool
    Norms,  # [E, I] fp32 down-column norms
    RecoveryGeometry,  # [E, I, 2] fp32
    Scores,  # [tokens, COUNT] fp32
    Padding,  # [tokens] bool
    H: constexpr,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    COUNT: constexpr,
    BEGIN: constexpr,
    END: constexpr,
    HAS_PADDING: constexpr,
    SCORE: constexpr,
    ALL_SELECTED: constexpr,
    GELU: constexpr,
    GROUPED: constexpr = False,
    BR: constexpr = 16,
    BH: constexpr = 128,
    GEOMETRY: constexpr = True,
    POINT: constexpr = False,
    ScreenParts=None,
    ScreenVarianceParts=None,
    SCREEN_SPLITS: constexpr = 1,
    BS: constexpr = 1,
    Histogram=None,
    BucketCounts=None,
    BucketKeys=None,
    RADIX: constexpr = False,
    PDL: constexpr = False,
    PREFETCH: constexpr = False,
    BK: constexpr = 1,
    X_WIDTH: constexpr = 1,
    PRIORITIZE: constexpr = False,
):
    """Finish a split screen, or compute the selected channels exactly.

    Grid (cdiv(COUNT, BR), tokens); program (r, t) handles rows [BR r, BR r + BR)
    of token t's COUNT candidates: all N, or the ids `_select` kept, where -1 ends
    a thresholded token's list. The stage's input range [BEGIN, END) selects the
    work:

    - END == 0 (split screen without last-block reduction): sum the per-split
      projections (`_screen_sums`), store them and, with SCORE, write the scores
      and fill the radix buckets.
    - END == H (exact computation): per-token execution reads each channel's
      native BF16 gate and up rows (4 H bytes) and accumulates FP32 dots with the
      input; expert-major execution loads what `_group_complete` stored. Either
      way it stores the projections and the BF16 activation and sets the
      channel's exact-computation flag.

    When all N candidates enter, it first clears their exact-computation and
    (unless every channel is selected) selection flags. With PDL the program
    waits for `_select`; with PREFETCH it first loads the token's input and
    expert ids, which do not depend on the selection.
    """
    token = tl.program_id(1)
    streaming: tl.constexpr = "evict_first" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
    reused: tl.constexpr = "evict_last" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
    # Padding/X: caller-owned inputs, unchanged throughout this call.
    active = True
    native_input = tl.zeros((X_WIDTH,), tl.float32)
    native_ids = tl.full((BK,), -1, tl.int32)
    if PREFETCH:
        active = not tl.load(Padding + token) if HAS_PADDING else True
        input_coordinate = tl.arange(0, X_WIDTH)
        native_input = tl.load(
            X + token * H + input_coordinate, (input_coordinate < H) & active, 0
        ).to(tl.float32)
        # Ids: _prepare or the routing screen, both before _select.
        slots = tl.arange(0, BK)
        native_ids = tl.load(Ids + token * K + slots, slots < K, -1)
    if PDL:
        # The candidate list comes from the immediately preceding `_select`.
        tl.extra.cuda.gdc_wait()
        _pdl_signal()
    if not PREFETCH:
        active = not tl.load(Padding + token) if HAS_PADDING else True
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    valid = row < COUNT
    if COUNT == N:
        logical = row
    else:
        logical = tl.load(Live + token * COUNT + row, valid, 0)
        # A selection threshold marks the rows past a token's own count with -1.
        valid = valid & (logical >= 0)
    slot, channel = logical // I, logical % I
    expert = (
        tl.gather(native_ids, slot, 0)
        if PREFETCH
        else tl.load(Ids + token * K + slot, valid, 0)
    ).to(tl.int64)
    address = token * 2 * N + logical
    # Grouped kernels write only local active rows, which already hold the exact
    # projections; the others read as zero, as their masked reads do.
    state = valid & (expert >= 0) & active if GROUPED else valid
    vg = tl.zeros((BR,), tl.float32)
    vu = tl.zeros((BR,), tl.float32)
    if END == 0 and SCREEN_SPLITS > 1:
        gate, up, vg, vu = _screen_sums(
            ScreenParts,
            ScreenVarianceParts,
            token,
            tl.where(state, logical, N) if GROUPED else logical,
            N,
            SCREEN_SPLITS,
            BS,
            POINT,
        )
        tl.store(Partial + address, gate, valid)
        tl.store(Partial + address + N, up, valid)
        tl.store(Variances + address, vg, valid)
        tl.store(Variances + address + N, vu, valid)
    elif END == H and not GROUPED:
        gate = tl.full((BR,), 0.0, tl.float32)
        up = tl.full((BR,), 0.0, tl.float32)
    else:
        gate = tl.load(Partial + address, state, 0)
        up = tl.load(Partial + address + N, state, 0)
    if END == 0 or COUNT == N:
        # Every candidate enters here, so clear the flags of the previous call.
        tl.store(Completed + token * N + logical, 0, valid)
        if not ALL_SELECTED:
            tl.store(Selected + token * N + logical, 0, valid)
    if END == H and not GROUPED:
        # Serial native full-row arithmetic, BR rows per program. Rows past a
        # thresholded token's own count are masked and load nothing.
        for begin in range(0, H, BH):
            coordinate = begin + tl.arange(0, BH)
            if PREFETCH:
                x_full = tl.gather(native_input, coordinate, 0)
            else:
                x_full = tl.load(
                    X + token * H + coordinate, (coordinate < H) & active, 0
                ).to(tl.float32)
            offset_full = (
                expert[:, None] * (2 * I) + channel[:, None]
            ) * H + coordinate[None, :]
            mask_full = (
                valid[:, None]
                & (expert[:, None] >= 0)
                & (coordinate[None, :] < H)
                & active
            )
            g_full = tl.load(
                NativeGU + offset_full, mask_full, 0.0, eviction_policy=streaming
            ).to(tl.float32)
            u_full = tl.load(
                NativeGU + offset_full + I * H,
                mask_full,
                0.0,
                eviction_policy=streaming,
            ).to(tl.float32)
            gate += tl.sum(g_full * x_full[None, :], axis=1)
            up += tl.sum(u_full * x_full[None, :], axis=1)
    if END == H:
        tl.store(Partial + address, gate, valid)
        tl.store(Partial + address + N, up, valid)
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    if END == H:
        activation = _activate(g, GELU).to(tl.bfloat16).to(tl.float32)
        value = (activation * u).to(tl.bfloat16).to(tl.float32)
        tl.store(Activations + token * N + logical, value, valid)
        tl.store(Completed + token * N + logical, 1, valid)
    elif SCORE:
        local = valid & (expert >= 0) & active
        if GEOMETRY:
            geometry = (expert * I + channel) * 2
            alpha = tl.load(
                RecoveryGeometry + geometry, local, 0, eviction_policy=reused
            )
            beta = tl.load(
                RecoveryGeometry + geometry + 1, local, 0, eviction_policy=reused
            )
        else:
            # Without score geometry (no recovery map, or point) alpha = beta = 0.
            alpha = tl.zeros((BR,), tl.float32)
            beta = tl.zeros((BR,), tl.float32)
        if SCREEN_SPLITS == 1:
            vg = tl.load(Variances + address, state, 0)
            vu = tl.load(Variances + address + N, state, 0)
        score = _screen_score(
            gate,
            up,
            vg,
            vu,
            tl.load(Norms + expert * I + channel, local, 0, eviction_policy=reused),
            tl.load(Weights + token * K + slot, valid, 0),
            alpha,
            beta,
            GELU,
            POINT,
        )
        tl.store(Scores + token * COUNT + row, score, valid)
        if RADIX:
            _bucket_scores(
                score, valid & active, token, Histogram, BucketCounts, BucketKeys
            )


@triton.jit
def _select(
    Scores,  # [tokens, COUNT] fp32, nonnegative
    Live,  # [tokens, COUNT] int32 candidate ids (implicit when COUNT == N)
    Next,  # [tokens, KEEP] int32 kept candidate ids
    Selected,  # [tokens, N] bool
    Starts,  # [tokens, K] int32 each expert's first entry in Next
    Counts,  # [tokens, K] int32 each expert's selected channels
    TileStarts,  # [tokens, K] int32 each expert's first DC-channel tile
    COUNT: constexpr,
    KEEP: constexpr,
    N: constexpr,
    I: constexpr,
    K: constexpr,
    FINAL: constexpr,
    BLOCK: constexpr,
    DC: constexpr,
    BK: constexpr = 1,
    Histogram=None,
    BucketCounts=None,
    BucketKeys=None,
    RADIX: constexpr = False,
    SHIFT: constexpr = 31 - RADIX_BITS,
    BINS: constexpr = 1 << RADIX_BITS,
    HASH: constexpr = 1 << RADIX_HASH_BITS,
    CAP: constexpr = RADIX_CAPACITY,
    Threshold=None,
    THRESHOLD: constexpr = False,
    BI: constexpr = 1,
    PARTITION: constexpr = False,
):
    """Keep the KEEP best-scoring candidates of each token (top-k channels).

    Grid (tokens, K with PARTITION, else 1). Nonnegative FP32 scores order as
    their bit patterns, so a binary search over the bits finds the KEEP-th
    largest score without a full sort; ties at the cutoff are kept in candidate
    order. RADIX first narrows the search to one histogram bucket. With
    THRESHOLD the program instead keeps every score at or above the layer
    threshold, at most KEEP, and pads the token's row of Next with -1.

    Writes the kept candidate ids to Next in candidate order. With FINAL it also
    sets the selection flags and writes each expert's start, count and first
    tile for `_down`. With PARTITION each program compacts one expert.
    """
    token = tl.program_id(0)
    if COUNT == N:
        # Dense candidates use independent per-expert channel scans.
        slot = tl.arange(0, BK)
        channel = tl.arange(0, BI)
        row = slot[:, None] * I + channel[None, :]
        valid = (slot < K)[:, None] & (channel < I)[None, :]
    else:
        row = tl.arange(0, BLOCK)
        valid = row < COUNT
    scores = tl.load(Scores + token * COUNT + row, valid, 0)
    keys = (scores + 0.0).to(tl.uint32, bitcast=True)
    if THRESHOLD:
        # Nonnegative FP32 bits order as the values, so the key compare is score >= tau.
        bound = (tl.load(Threshold) + 0.0).to(tl.uint32, bitcast=True)
        passing = tl.sum((valid & (keys >= bound)).to(tl.int32))
        # Within capacity the threshold is the boundary: every tie at it is kept.
        within = passing <= KEEP
        low = tl.where(within, bound, tl.min(tl.where(valid, keys, 0xFFFFFFFF)))
        high = tl.where(within, bound, tl.max(tl.where(valid, keys, 0)))
    elif RADIX:
        bins = tl.arange(0, BINS)
        prefix = tl.cumsum(tl.load(Histogram + token * BINS + bins), 0)
        bucket = tl.min(tl.where(prefix > COUNT - KEEP, bins, BINS), 0)
        above = COUNT - tl.sum(tl.where(bins == bucket, prefix, 0), 0)
        hashed = bucket & (HASH - 1)
        cached = tl.load(BucketCounts + token * HASH + hashed)
        if cached <= CAP:
            index = tl.arange(0, CAP)
            candidates = tl.load(
                BucketKeys + (token * HASH + hashed) * CAP + index, index < cached, 0
            ).to(tl.uint32)
            inside = (index < cached) & (
                tl.minimum(candidates >> SHIFT, BINS - 1) == bucket
            )
            low = tl.min(tl.where(inside, candidates, 0xFFFFFFFF), 0)
            high = tl.max(tl.where(inside, candidates, 0), 0)
            while low < high:
                middle = low + (high - low + 1) // 2
                retained = tl.sum((inside & (candidates >= middle)).to(tl.int32), 0)
                enough = retained >= KEEP - above
                low = tl.where(enough, middle, low)
                high = tl.where(
                    retained == KEEP - above, middle, tl.where(enough, high, middle - 1)
                )
        else:
            low = tl.maximum(
                tl.min(tl.where(valid, keys, 0xFFFFFFFF)),
                bucket.to(tl.uint32) << SHIFT,
            )
            high = tl.minimum(
                tl.max(tl.where(valid, keys, 0)),
                tl.where(
                    bucket == BINS - 1,
                    0xFFFFFFFF,
                    ((bucket.to(tl.uint32) + 1) << SHIFT) - 1,
                ),
            )
    else:
        low = tl.min(tl.where(valid, keys, 0xFFFFFFFF))
        high = tl.max(tl.where(valid, keys, 0))
    # Nonnegative FP32 score bits preserve order; find membership, not a full sort.
    while low < high:
        middle = low + (high - low + 1) // 2
        retained = tl.sum((valid & (keys >= middle)).to(tl.int32))
        enough = retained >= KEEP
        low = tl.where(enough, middle, low)
        high = tl.where(retained == KEEP, middle, tl.where(enough, high, middle - 1))
    above = valid & (keys > low)
    equal = valid & (keys == low)
    # Disjoint fields share one scan without carrying between the two counts.
    if COUNT < 65536:
        flag_shift: constexpr = 16
        flag_mask: constexpr = 65535
        packed = (above.to(tl.uint32) << flag_shift) | equal.to(tl.uint32)
    else:
        flag_shift: constexpr = 32
        flag_mask: constexpr = 0xFFFFFFFF
        packed = (above.to(tl.uint64) << flag_shift) | equal.to(tl.uint64)
    if COUNT == N:
        row_totals = tl.sum(packed, axis=1)
        before = tl.cumsum(row_totals, axis=0) - row_totals
        above_count = (row_totals >> flag_shift).to(tl.int32)
        equal_count = (row_totals & flag_mask).to(tl.int32)
        above_start = (before >> flag_shift).to(tl.int32)
        equal_start = (before & flag_mask).to(tl.int32)
        ties_needed = KEEP - tl.sum(above_count, axis=0)
        count = above_count + tl.minimum(
            equal_count, tl.maximum(ties_needed - equal_start, 0)
        )
        start = above_start + tl.minimum(equal_start, ties_needed)
        if FINAL:
            slot = tl.arange(0, BK)
            tiles = tl.cdiv(count, DC)
            tile_start = tl.cumsum(tiles, axis=0) - tiles
            owned = (slot < K) & ((slot == tl.program_id(1)) if PARTITION else True)
            tl.store(Starts + token * K + slot, start, owned)
            tl.store(Counts + token * K + slot, count, owned)
            tl.store(TileStarts + token * K + slot, tile_start, owned)
        if PARTITION:
            # Each CTA repeats the small cutoff search but compacts only one
            # expert. No grid barrier or additional launch is needed.
            partition = tl.program_id(1)
            slot = tl.arange(0, BK)
            offset_before = tl.sum(tl.where(slot == partition, before, 0), 0)
            row = partition * I + tl.arange(0, BI)
            valid = tl.arange(0, BI) < I
            keys = (tl.load(Scores + token * COUNT + row, valid, 0) + 0.0).to(
                tl.uint32, bitcast=True
            )
            above = valid & (keys > low)
            equal = valid & (keys == low)
            packed = (above.to(packed.dtype) << flag_shift) | equal.to(packed.dtype)
            scanned = tl.cumsum(packed, axis=0) + offset_before
        else:
            scanned = tl.cumsum(packed, axis=1) + before[:, None]
        above_prefix = (scanned >> flag_shift).to(tl.int32)
        equal_prefix = (scanned & flag_mask).to(tl.int32)
    else:
        ties_needed = KEEP - tl.sum(above.to(tl.int32), axis=0)
        scanned = tl.cumsum(packed, axis=0)
        above_prefix = (scanned >> flag_shift).to(tl.int32)
        equal_prefix = (scanned & flag_mask).to(tl.int32)
    keep = above | (equal & (equal_prefix <= ties_needed))
    prefix = above_prefix + tl.minimum(equal_prefix, ties_needed)
    offset = prefix - 1
    # The initial candidate list is dense and is never overwritten.
    if COUNT == N:  # noqa: SIM108
        logical = row
    else:
        logical = tl.load(Live + token * COUNT + row, valid, 0)
    tl.store(Next + token * KEEP + offset, logical, keep)
    if THRESHOLD:
        # Exact computation and the down projection stop at the row's first -1.
        total = tl.sum(keep.to(tl.int32))
        tl.store(Next + token * KEEP + row, -1, valid & (row >= total) & (row < KEEP))
    if FINAL:
        tl.store(Selected + token * N + logical, 1, keep)
    if FINAL and COUNT != N:
        start = tl.full((), 0, tl.int32)
        tile_start = tl.full((), 0, tl.int32)
        for slot in static_range(K):
            count = tl.sum((keep & (logical // I == slot)).to(tl.int32), axis=0)
            tl.store(Starts + token * K + slot, start)
            tl.store(Counts + token * K + slot, count)
            tl.store(TileStarts + token * K + slot, tile_start)
            start += count
            tile_start += tl.cdiv(count, DC)


@triton.jit
def _down(
    D,  # [E, H, I] bf16 down weight, physical [E, I, H]
    Activations,  # [tokens, N] bf16 activations
    Ids,  # [tokens, K] int32
    Final,  # [tokens, F] int32 selected candidate ids
    Starts,  # [tokens, K] int32, from `_select`
    Counts,  # [tokens, K] int32
    TileStarts,  # [tokens, K] int32
    Parts,  # [tokens, K, SPLITS, H] fp32 partial sums
    Padding,  # [tokens] bool
    H: constexpr,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    F: constexpr,
    SPLITS: constexpr,
    BK: constexpr,
    HAS_PADDING: constexpr,
    BC: constexpr = 128,
    BO: constexpr = 64,
    PDL: constexpr = False,
    PRIORITIZE: constexpr = False,
):
    """Multiply one tile of a token's selected activations by the down weight.

    Grid (cdiv(H, BO), cdiv(F, BC) + K - 1, tokens). The second axis walks the
    token's selected channels expert by expert in BC-channel tiles; programs past
    the token's last tile do nothing. Program (o, c, t) writes the FP32 partial
    sum of outputs [BO o, BO o + BO).
    """
    output = tl.program_id(0) * BO + tl.arange(0, BO)
    tile = tl.program_id(1)
    token = tl.program_id(2)
    # Padding: caller-owned input, unchanged throughout the reader call.
    active = not tl.load(Padding + token) if HAS_PADDING else True
    slots = tl.arange(0, BK)
    # _acquire waits for _select, then cumulatively publishes its writes with
    # its GPU release/signal (PTX ISA Section 8.9.5); this same-domain grid may
    # preload them.
    starts = tl.load(TileStarts + token * K + slots, slots < K, 0)
    counts = tl.load(Counts + token * K + slots, slots < K, 0)
    match = (slots < K) & (tile >= starts) & (tile < starts + tl.cdiv(counts, BC))
    if tl.sum(match.to(tl.int32), axis=0) > 0:
        slot = tl.sum(tl.where(match, slots, 0), axis=0)
        local_tile = tile - tl.sum(tl.where(match, starts, 0), axis=0)
        count = tl.sum(tl.where(match, counts, 0), axis=0)
        # Starts: final selection; Ids: _prepare or the routing screen.
        start = tl.load(Starts + token * K + slot)
        expert = tl.load(Ids + token * K + slot).to(tl.int64)
        row = local_tile * BC + tl.arange(0, BC)
        if F == N:
            logical = start + row
        else:
            # Final: written by `_select`, two launches earlier on the main stream.
            logical = tl.load(Final + token * F + start + row, row < count, 0)
        channel = logical % I
        # D: immutable native weights, not _acquire's outputs. Physical [E,I,H].
        streaming: tl.constexpr = "evict_first" if PRIORITIZE else ""
        coefficient = tl.load(
            D + (expert * I + channel[:, None]) * H + output[None, :],
            (output[None, :] < H) & (row[:, None] < count) & (expert >= 0) & active,
            0.0,
            eviction_policy=streaming,
        ).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()
            _pdl_signal()
        # Activations come from `_acquire`; no read of them may precede the GDC wait.
        value = tl.load(Activations + token * N + logical, row < count, 0).to(
            tl.float32
        )
        partial = tl.sum(coefficient * value[:, None], axis=0)
        tl.store(
            Parts + ((token * K + slot) * SPLITS + local_tile) * H + output,
            partial,
            output < H,
        )
    # Empty CTAs read no predecessor outputs and implicitly signal at exit.


@triton.jit
def _encode(
    B,  # [E, I, R] bf16 recovery encoder
    Partial,  # [tokens, 2, N] fp32 gate and up projections
    Activations,  # [tokens, N] bf16 exact activations
    Completed,  # [tokens, N] bool
    Selected,  # [tokens, N] bool
    Ids,  # [tokens, K] int32
    Parts,  # [tokens, K, SPLITS, R] fp32 partial latents
    Padding,  # [tokens] bool
    I: constexpr,
    K: constexpr,
    N: constexpr,
    R: constexpr,
    SPLITS: constexpr,
    BR: constexpr,
    HAS_PADDING: constexpr,
    GELU: constexpr,
    BC: constexpr = 64,
    Weights=None,
    Latent=None,
    Counters=None,
    LAST_BLOCK: constexpr = False,
    BS: constexpr = 1,
    BK: constexpr = 1,
):
    """Encode one tile of a top-k expert's skipped channels into the latent.

    Grid (cdiv(I, BC), tokens * K); program (c, t * K + k) covers channels
    [BC c, BC c + BC) of token t's k-th expert. Each skipped channel contributes
    its BF16 activation: the exact value if it was computed exactly, else
    phi(g) u of its sketch projections. With LAST_BLOCK the token's last program
    to arrive also runs `_latent`.
    """
    tile = tl.program_id(0)
    slot = tl.program_id(1)
    token, candidate = slot // K, slot % K
    active = not tl.load(Padding + token) if HAS_PADDING else True
    channel = tile * BC + tl.arange(0, BC)
    rank = tl.arange(0, BR)
    # Recovery encodes only the skipped channels.
    valid = (channel < I) & ~tl.load(
        Selected + token * N + candidate * I + channel, channel < I, 0
    )
    logical = candidate * I + channel
    offset = token * N + logical
    completed = tl.load(Completed + offset, valid, 0)
    outside = valid & ~completed
    g = (
        tl.load(Partial + token * 2 * N + logical, outside, 0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    u = (
        tl.load(Partial + token * 2 * N + N + logical, outside, 0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    partial_value = (_activate(g, GELU) * u).to(tl.bfloat16).to(tl.float32)
    exact = tl.load(Activations + offset, valid & completed, 0).to(tl.float32)
    value = tl.where(completed, exact, partial_value)
    expert = tl.load(Ids + slot).to(tl.int64)
    coefficient = tl.load(
        B + (expert * I + channel[:, None]) * R + rank[None, :],
        valid[:, None] & (rank[None, :] < R) & (expert >= 0) & active,
        0,
    ).to(tl.float32)
    partial = tl.sum(value[:, None] * coefficient, axis=0)
    tl.store(Parts + (slot * SPLITS + tile) * R + rank, partial, rank < R)
    if LAST_BLOCK:
        # Publish every lane's partial before the token's single arrival RMW.
        # The last CTA acquires all arrivals and reduces by split index, not
        # arrival order; no CTA waits or spins.
        counter = Counters + token
        tl.debug_barrier()
        arrived = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if arrived == K * SPLITS - 1:
            _latent(
                Parts,
                Weights,
                Latent,
                K,
                R,
                SPLITS,
                BS,
                BR,
                Token=token,
                LAST_BLOCK=True,
                BK=BK,
            )
            # All CTAs have arrived; the caller-stream join orders workspace reuse.
            tl.store(counter, 0)


@triton.jit
def _latent(
    Parts,  # [tokens, K, S, R] fp32 partial latents
    Weights,  # [tokens, K] fp32
    Latent,  # [tokens, R] fp32
    K: constexpr,
    R: constexpr,
    S: constexpr,
    BS: constexpr,
    BR: constexpr,
    Ids=None,
    Padding=None,
    HAS_PADDING: constexpr = False,
    GROUPED: constexpr = False,
    Token=0,
    LAST_BLOCK: constexpr = False,
    BK: constexpr = 1,
):
    """Mix a token's per-expert latents by router weight.

    Grid (tokens,), or called for Token by the last `_encode` program. Each
    expert's partials are summed in split order and rounded to BF16 first.
    """
    rank = tl.arange(0, BR)
    split = tl.arange(0, BS)
    if LAST_BLOCK:
        slot = tl.arange(0, BK)
        # One vectorized load replaces the serial expert load/reduce chain.
        # Bypass L1 for other CTAs' partials; ordering comes from the counter.
        partial = tl.load(
            Parts
            + ((Token * K + slot[:, None, None]) * S + split[None, :, None]) * R
            + rank[None, None, :],
            (slot[:, None, None] < K)
            & (split[None, :, None] < S)
            & (rank[None, None, :] < R),
            0,
            cache_modifier=".cg",
        )
        # The fixed split-order reduction rounds each whole expert sum before
        # router mixing, as the separate launch does.
        value = tl.sum(partial, axis=1).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(Weights + Token * K + slot, slot < K, 0)
        result = tl.sum(value * weight[:, None], axis=0)
        tl.store(Latent + Token * R + rank, result, rank < R)
    else:
        token = tl.program_id(0)
        result = tl.full((BR,), 0.0, tl.float32)
        for slot in static_range(K):
            used = split[:, None] < S
            if GROUPED:
                # Grouped encoding writes only local active pairs; others are zero.
                active = not tl.load(Padding + token) if HAS_PADDING else True
                used = used & (tl.load(Ids + token * K + slot) >= 0) & active
            partial = tl.load(
                Parts + ((token * K + slot) * S + split[:, None]) * R + rank[None, :],
                used & (rank[None, :] < R),
                0,
            )
            value = tl.sum(partial, axis=0).to(tl.bfloat16).to(tl.float32)
            weight = tl.load(Weights + token * K + slot)
            result += value * weight
        tl.store(Latent + token * R + rank, result, rank < R)


@triton.jit
def _output(
    DParts,  # [tokens, K, S, H] fp32 down partial sums
    Counts,  # [tokens, K] int32 selected channels per expert
    Weights,  # [tokens, K] fp32
    Latent,  # [tokens, R] fp32
    U,  # [H, R] bf16 recovery decoder
    Output,  # [tokens, H]
    Padding,  # [tokens] bool
    H: constexpr,
    K: constexpr,
    R: constexpr,
    S: constexpr,
    BS: constexpr,
    BR: constexpr,
    HAS_PADDING: constexpr,
    RECOVERY: constexpr,
    Ids=None,
    GROUPED: constexpr = False,
    DC: constexpr = 128,
    BO: constexpr = 128,
    PDL: constexpr = False,
    PRIORITIZE: constexpr = False,
):
    """Write one tile of each token's routed-expert output.

    Grid (cdiv(H, BO), tokens). Sums each expert's down partials in split order,
    rounds to BF16, mixes the experts by router weight and, with RECOVERY, adds
    the BF16-rounded decoder term U latent.
    """
    token = tl.program_id(1)
    # Padding: caller-owned input, unchanged throughout the reader call.
    active = not tl.load(Padding + token) if HAS_PADDING else True
    output = tl.program_id(0) * BO + tl.arange(0, BO)
    split = tl.arange(0, BS)
    counts = ()
    routes = ()
    if PDL:
        # _down's wait/release/signal cumulatively republishes selection/routing.
        for slot in static_range(K):
            # Counts: `_select`, before `_acquire` and `_down`.
            counts += (tl.load(Counts + token * K + slot),)
            # Weights: _prepare or the routing screen, before selection.
            routes += (tl.load(Weights + token * K + slot),)
    correction = tl.full((BO,), 0.0, tl.float32)
    if RECOVERY:
        rank = tl.arange(0, BR)
        # U: immutable resident decoder rows; no dependency on _down.
        priority: tl.constexpr = "evict_last" if PRIORITIZE else ""  # pyright: ignore[reportAssignmentType]
        coefficient = tl.load(
            U + output[:, None] * R + rank[None, :],
            (output[:, None] < H) & (rank[None, :] < R) & active,
            0,
            eviction_policy=priority,  # pyright: ignore[reportArgumentType]
        ).to(tl.float32)
        if PDL:
            tl.extra.cuda.gdc_wait()  # pyright: ignore[reportAttributeAccessIssue]
            _pdl_signal()
        # Latent: _encode/_latent on the recovery stream. Keep this after the
        # event join and GDC wait; captured PDL may include that producer edge.
        # EP joins its all-reduce before _output, so the global mixture is
        # rounded once.
        latent = (
            tl.load(Latent + token * R + rank, rank < R, 0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        correction = (
            tl.sum(coefficient * latent[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
        )
    elif PDL:
        tl.extra.cuda.gdc_wait()  # pyright: ignore[reportAttributeAccessIssue]
        _pdl_signal()
    routed = tl.full((BO,), 0.0, tl.float32)
    for slot in static_range(K):
        if GROUPED:
            # One full FP32 channel sum per local active pair, in split 0.
            local = (tl.load(Ids + token * K + slot) >= 0) & active
            used = (split[:, None] == 0) & local
        else:
            count = counts[slot] if PDL else tl.load(Counts + token * K + slot)
            used = split[:, None] < tl.cdiv(count, DC)
        part = (token * K + slot) * S + split[:, None]
        # DParts: immediately preceding _down; read only after the GDC wait.
        partial = tl.load(
            DParts + part * H + output[None, :],
            used & (output[None, :] < H),
            0,
        )
        value = tl.sum(partial, axis=0).to(tl.bfloat16).to(tl.float32)
        weight = routes[slot] if PDL else tl.load(Weights + token * K + slot)
        routed += value * weight
    tl.store(Output + token * H + output, routed + correction, output < H)


@triton.jit
def _count_reads(
    Ids,
    Padding,
    Completed,
    Selected,
    BlockExperts,
    PostPad,
    UnionCounts,
    Work,
    TOKENS,
    BLOCKS,
    UNION_STRIDE,
    H: constexpr,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    Q: constexpr,  # plan.completion_count, channels computed exactly; equals F
    F: constexpr,
    R: constexpr,
    E: constexpr,
    NATIVE: constexpr,
    REFERENCE: constexpr,
    RECOVERY: constexpr,
    SELECTION: constexpr,
    SKETCH: constexpr,
    UNIT: constexpr,
    EP: constexpr,
    OWNER: constexpr,
    HAS_PADDING: constexpr,
    GROUPED: constexpr,
    GROUPED_SCREEN: constexpr,
    SHARED_B: constexpr,
    ENCODER_TILES: constexpr,
    BT: constexpr,
    BK: constexpr,
    BC: constexpr,
    BB: constexpr = 256,
    SCREEN_BYTES: constexpr = 12,
    ISSUED_SKETCH: constexpr = 0,
    PHYSICAL_SKETCH: constexpr = 0,
    THRESHOLD: constexpr = False,
):
    """Add one call's read statistics to Work (slots listed at `read_work`).

    Grid (cdiv(TOKENS, BT) + cdiv(BLOCKS, BB) if GROUPED, else cdiv(TOKENS, BT)).
    Each token program adds the read cost of BT tokens, from fixed per-token terms
    or, when the count varies per token (EP, a selection threshold, or a narrow
    shared B row), from the exact-computation and selection flags. With GROUPED,
    each block program adds the requested bytes and tile MACs of BB expert-major
    blocks; `_group_screen` adds its own sketch sectors.

    SCREEN_BYTES is the per-candidate screen metadata (FP32 norm and, with score
    geometry, two FP32 values). SKETCH is a token's batch-one sketch bytes priced
    exactly; PHYSICAL_SKETCH and ISSUED_SKETCH price them in sectors for the read
    cost and the requested bytes. With THRESHOLD each token also reads the layer's
    FP32 threshold (one sector) and Work[19] counts active tokens. Every counter is
    an integer sum, so FP64 atomic partial sums equal the call totals exactly.
    """
    token_programs = tl.cdiv(TOKENS, BT)
    program = tl.program_id(0)
    if program < token_programs:
        slot = tl.arange(0, BK)
        channel = tl.arange(0, BC)
        active_count = tl.full((), 0, tl.int64)
        local_slots = tl.full((), 0, tl.int64)
        completed_rows = tl.full((), 0, tl.int64)
        selected_rows = tl.full((), 0, tl.int64)
        b_sectors = tl.full((), 0, tl.int64)
        for j in range(BT):
            token = program * BT + j
            if token < TOKENS:
                active = not tl.load(Padding + token) if HAS_PADDING else True
                ids = tl.load(Ids + token * K + slot, slot < K, -1)
                local = (slot < K) & (ids >= 0) & active
                active_count += active.to(tl.int64)
                local_slots += tl.sum(local.to(tl.int64), axis=0)
                if EP or SHARED_B or THRESHOLD:
                    for s in static_range(K):
                        held = tl.sum(tl.where(slot == s, local, 0), axis=0) > 0
                        for begin in static_range(0, I, BC):
                            rows = held & (begin + channel < I)
                            offset = token * N + s * I + begin + channel
                            done = rows & (tl.load(Completed + offset, rows, 0) != 0)
                            chosen = tl.load(Selected + offset, rows, 0) != 0
                            completed_rows += tl.sum(done.to(tl.int64), axis=0)
                            selected_rows += tl.sum((rows & chosen).to(tl.int64), 0)
                            if SHARED_B:
                                # Narrow B rows: 16 // R skipped rows per sector.
                                omitted = (rows & ~chosen).to(tl.int32)
                                sector = tl.max(
                                    omitted.reshape(BC * R // 16, 16 // R), 1
                                )
                                b_sectors += tl.sum(sector.to(tl.int64), axis=0)
        candidates = local_slots * I
        if EP or THRESHOLD:
            completed = completed_rows
            selected = selected_rows
            native = UNIT * (2 * completed * H + selected * H)
            # REFERENCE is per local slot under EP and per token otherwise.
            reference = local_slots * REFERENCE if EP else active_count * REFERENCE
            recovery = (
                2 * R * (candidates - selected + (active_count * H if OWNER else 0))
            )
            if Q < N:  # noqa: SIM108
                selection = SCREEN_BYTES * candidates
            else:
                selection = candidates * 0
            sketch = local_slots * (SKETCH // K)
            physical_sketch = local_slots * (PHYSICAL_SKETCH // K)
            issued_sketch = local_slots * (ISSUED_SKETCH // K)
        else:
            completed = active_count * Q
            selected = active_count * F
            native = active_count * NATIVE
            reference = active_count * REFERENCE
            recovery = active_count * RECOVERY
            selection = active_count * SELECTION
            sketch = active_count * SKETCH
            physical_sketch = active_count * PHYSICAL_SKETCH
            issued_sketch = active_count * ISSUED_SKETCH
        physical_selection = selection
        if THRESHOLD:
            # Every rank selects, so each reads the FP32 layer threshold per token.
            selection += 4 * active_count
            physical_selection += 32 * active_count
            tl.atomic_add(Work + 19, active_count.to(tl.float64), sem="relaxed")
        physical_recovery = recovery
        if SHARED_B:
            physical_recovery = (
                recovery - 2 * R * (candidates - selected) + 32 * b_sectors
            )
        # Logical, then sector-priced, bytes (see `read_work` for the slots).
        tl.atomic_add(Work + 0, native.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 1, reference.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 3, recovery.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 4, selection.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 5, sketch.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 6, native.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 7, physical_recovery.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 9, physical_selection.to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 17, physical_sketch.to(tl.float64), sem="relaxed")
        if Q < N:
            tl.atomic_add(Work + 10, (active_count * H).to(tl.float64), sem="relaxed")
        if GROUPED:
            tl.atomic_add(Work + 16, active_count.to(tl.float64), sem="relaxed")
            if not GROUPED_SCREEN:
                # Grouped batches below the screen crossover screen per token.
                tl.atomic_add(Work + 11, issued_sketch.to(tl.float64), sem="relaxed")
        else:
            tl.atomic_add(Work + 11, issued_sketch.to(tl.float64), sem="relaxed")
            tl.atomic_add(
                Work + 12, (UNIT * 2 * H * completed).to(tl.float64), sem="relaxed"
            )
            tl.atomic_add(
                Work + 13, (UNIT * H * selected).to(tl.float64), sem="relaxed"
            )
            tl.atomic_add(
                Work + 14,
                (2 * R * (candidates - selected)).to(tl.float64),
                sem="relaxed",
            )
    elif GROUPED:
        # Requested gate/up, down and encoder bytes of the grouped kernels, then
        # their tile MACs (16-row blocks, 64-channel union tiles and the grouped
        # screen's three BF16 terms over 128 coordinates).
        block = (program - token_programs) * BB + tl.arange(0, BB)
        inside = block < BLOCKS
        expert = tl.load(BlockExperts + block, inside, -1)
        in_use = inside & (block * 16 < tl.load(PostPad)) & (expert >= 0) & (expert < E)
        valid = tl.sum(in_use.to(tl.int64), axis=0)
        full_width = valid * ((I + 63) // 64 * 64)
        if Q < N:
            first = tl.load(UnionCounts + block * 2, inside, 0).to(tl.int64)
            completion = tl.sum(first, axis=0)
            completion_tiles = tl.sum((first + 63) // 64 * 64, axis=0)
        else:
            completion = valid * I
            completion_tiles = full_width
        if Q < N and GROUPED_SCREEN:
            # Union-masked loads still run dense padded MMA tiles.
            screen_macs = valid * tl.cdiv(I, 80) * 256 * H * 3
        else:
            screen_macs = valid * 0
        if F < N:
            last = UnionCounts + UNION_STRIDE + block * 2
            held = tl.load(last, inside, 0).to(tl.int64)
            omitted = tl.sum(tl.load(last + 1, inside, 0).to(tl.int64), axis=0)
            readout = tl.sum(held, axis=0)
            readout_tiles = tl.sum((held + 63) // 64 * 64, axis=0)
        else:
            omitted = valid * 0
            readout = valid * I
            readout_tiles = full_width
        macs = 16 * (
            screen_macs
            + 2 * H * completion_tiles
            + H * readout_tiles
            + valid * 64 * ENCODER_TILES
        )
        tl.atomic_add(Work + 12, (4 * H * completion).to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 13, (2 * H * readout).to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 14, (2 * R * omitted).to(tl.float64), sem="relaxed")
        tl.atomic_add(Work + 15, macs.to(tl.float64), sem="relaxed")


# Expert-major (grouped) execution for batches. _group_sort sorts (token, slot)
# pairs, indexed token * K + slot, by local expert into BM-row blocks, padded
# with the pair count PAIRS; each block reads its expert's weights once and, per
# stage, only the union of channels or coordinates its pairs need. Per-pair masks
# keep each token's contractions and FP32/BF16 boundaries those of the per-token
# path. Only the FP32 summation order differs, which can flip a selection whose
# score lies within that reassociation error of the cutoff.


@triton.jit
def _group_sort(
    Keys,
    Pairs,
    BlockExperts,
    PostPad,
    PAIRS,
    E: constexpr,
    BM: constexpr,
    BE: constexpr,
    BP: constexpr = 1024,
):
    """Sort (token, slot) pairs by expert key into BM-row blocks, stably.

    Grid (E,); program e places expert e's pairs. Experts appear in ascending
    order, each padded with PAIRS to a multiple of BM, and keep ascending pair
    order, so block membership (hence union order) is a function of the keys.
    Out-of-range keys (E) join no block. Writes the sorted pair list Pairs,
    each block's expert BlockExperts, and the padded pair count PostPad. This
    replaces vLLM's moe_align_block_size (same layout), whose atomic placement
    orders pairs nondeterministically from 1024 pairs.
    """
    expert = tl.program_id(0)
    bins = tl.arange(0, BE)
    counts = tl.zeros((BE,), tl.int32)
    for begin in range(0, PAIRS, BP):
        pair = begin + tl.arange(0, BP)
        counts += tl.histogram(tl.load(Keys + pair, pair < PAIRS, E), BE)
    padded = tl.where(bins < E, (counts + BM - 1) // BM * BM, 0)
    start = tl.sum(tl.where(bins < expert, padded, 0), axis=0)
    count = tl.sum(tl.where(bins == expert, counts, 0), axis=0)
    rows = tl.sum(tl.where(bins == expert, padded, 0), axis=0)
    held = tl.full((), 0, tl.int32)
    for begin in range(0, PAIRS, BP):
        pair = begin + tl.arange(0, BP)
        mine = tl.load(Keys + pair, pair < PAIRS, E) == expert
        rank = held + tl.cumsum(mine.to(tl.int32), axis=0) - 1
        tl.store(Pairs + start + rank, pair, mine)
        held += tl.sum(mine.to(tl.int32), axis=0)
    tail = count + tl.arange(0, BM)
    tl.store(Pairs + start + tail, PAIRS, tail < rows)
    for block in range(rows // BM):
        tl.store(BlockExperts + start // BM + block, expert)
    if expert == 0:
        tl.store(PostPad, tl.sum(padded, axis=0))


@triton.jit
def _block_expert(BlockExperts, PostPad, E: constexpr, BM: constexpr):
    """Return the program's block expert, or E for blocks past the padded pairs."""
    block = tl.program_id(0)
    expert = tl.load(BlockExperts + block)
    expert = tl.where((block * BM < tl.load(PostPad)) & (expert >= 0), expert, E)
    return expert.to(tl.int64)


@triton.jit
def _group_screen_union(
    Split,  # [tokens, H // 128, 3, 128] bf16 three-term split of the rotated input
    Pairs,  # [padded pairs] int32 token * K + slot; padding entries equal PAIRS
    BlockExperts,  # [blocks] int32
    PostPad,  # [1] int32 padded pair count
    Union,  # [blocks, H] bool coordinate union
    Dense,  # [blocks, H // 128] bool, union holds all 128 coordinates
    PAIRS,
    H: constexpr,
    K: constexpr,
    E: constexpr,
    BM: constexpr,
):
    """Mark the rotated coordinates that are nonzero for any pair of an expert block.

    Grid (blocks, H // 128). A coordinate that is zero for every pair reads no
    sketch code in `_group_screen`. Deciding this once per block, rather than in
    every channel tile of the screen, avoids repeating it.
    """
    expert = _block_expert(BlockExperts, PostPad, E, BM)
    if expert < E:
        group, block = tl.program_id(0), tl.program_id(1)
        pair = tl.load(Pairs + group * BM + tl.arange(0, BM))
        coordinate = tl.arange(0, 128)
        address = (
            Split
            + ((pair // K)[:, None] * (H // 128) + block) * 384
            + coordinate[None, :]
        )
        high = tl.load(address, (pair < PAIRS)[:, None], 0)
        middle = tl.load(address + 128, (pair < PAIRS)[:, None], 0)
        low = tl.load(address + 256, (pair < PAIRS)[:, None], 0)
        union = (
            tl.max(((high != 0) | (middle != 0) | (low != 0)).to(tl.int32), axis=0) != 0
        )
        tl.store(Union + group * H + block * 128 + coordinate, union)
        tl.store(
            Dense + group * (H // 128) + block,
            tl.sum(union.to(tl.int32), axis=0) == 128,
        )


@triton.jit
def _group_screen(
    Codes,
    Statistics,
    Split,
    Norms,
    Pairs,
    BlockExperts,
    PostPad,
    Partial,
    Variances,
    Work,
    Weights,
    DownNorms,
    Geometry,
    Scores,
    Completed,
    Selected,
    Parts,
    VarianceParts,
    CoordinateUnion,
    Dense,
    PAIRS,
    H: constexpr,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    E: constexpr,
    BM: constexpr,
    STRIDE: constexpr,
    STORAGE: constexpr,
    GELU: constexpr,
    GEOMETRY: constexpr,
    POINT: constexpr,
    SPLITS: constexpr = 1,
):
    """Expert-major variant of `_sketch_screen` for the BM pairs of one block.

    Grid (blocks, cdiv(I, 80), SPLITS). Reads code sectors only for the block's
    coordinate union and contracts the decoded trits with the pairs' three-term
    BF16 inputs on tensor cores, which keeps FP32 projections. With splits,
    `_acquire` reduces and scores. Program (g, 0, s) adds the requested sketch
    sectors to Work[11].
    """
    expert = _block_expert(BlockExperts, PostPad, E, BM)
    # Padding entries of the aligned pair list equal the pair count.
    pair = tl.load(Pairs + tl.program_id(0) * BM + tl.arange(0, BM))
    member, token, slot = pair < PAIRS, pair // K, pair % K
    if expert < E:
        blocks: constexpr = H // 128
        split = tl.program_id(2)
        span: constexpr = tl.cdiv(blocks, SPLITS)
        begin_block = tl.minimum(split * span, blocks)
        end_block = tl.minimum((split + 1) * span, blocks)
        estimate = tl.zeros((BM, 256), tl.float32)
        variance = tl.zeros((BM, 256), tl.float32)
        coordinates_read = tl.full((), 0, tl.int32)
        for block in tl.range(begin_block, end_block, num_stages=2):
            coordinate = tl.arange(0, 128)
            digits = (
                Split + (token[:, None] * blocks + block) * 384 + coordinate[None, :]
            )
            high = tl.load(digits, member[:, None], 0)
            middle = tl.load(digits + 128, member[:, None], 0)
            low = tl.load(digits + 256, member[:, None], 0)
            dense = tl.load(Dense + tl.program_id(0) * blocks + block)
            if dense:
                # The cached flag is an exact all-members union, not a density
                # heuristic. Byte/tail bounds still apply to every vector load.
                code = _load_screen_words(
                    Codes,
                    expert,
                    block,
                    coordinate,
                    tl.program_id(1),
                    None,
                    I,
                    blocks,
                    STRIDE,
                    STORAGE,
                    DENSE=True,
                )
                used_coordinates = 128
            else:
                union = tl.load(
                    CoordinateUnion + tl.program_id(0) * H + block * 128 + coordinate
                )
                code = _load_screen_words(
                    Codes,
                    expert,
                    block,
                    coordinate,
                    tl.program_id(1),
                    union,
                    I,
                    blocks,
                    STRIDE,
                    STORAGE,
                )
                used_coordinates = tl.sum(union.to(tl.int32), axis=0)
            value = code.to(tl.float32) - 121.0
            coordinates_read += used_coordinates
            # One padded N=256 tile trades extra zero-column MACs for three dots
            # instead of nine, avoiding repeated digit-tile layout conversions.
            # The padding is on-chip only: sketch and metadata reads are unchanged.
            q1 = _nearest(value * (1 / 3))
            q2 = _nearest(value * (1 / 9))
            q3 = _nearest(value * (1 / 27))
            q4 = _nearest(value * (1 / 81))
            first = tl.cat(value - 3.0 * q1, q1 - 3.0 * q2, dim=1)
            second = tl.cat(q2 - 3.0 * q3, q3 - 3.0 * q4, dim=1)
            tail = tl.cat(
                tl.cat(q4, tl.zeros((128, 32), tl.float32), dim=1),
                tl.zeros((128, 64), tl.float32),
                dim=1,
            )
            trit = tl.cat(tl.cat(first, second, dim=1), tail, dim=1).to(tl.bfloat16)
            dot = tl.dot(low, trit)
            dot = tl.dot(middle, trit, dot)
            dot = tl.dot(high, trit, dot)
            # Load metadata after the decode and MMA: earlier loads spill the
            # padded tile even though the same contraction fits in registers.
            scale, residual = _screen_statistics(
                Statistics, expert, block, tl.program_id(1), True, I, blocks, POINT
            )
            estimate += dot * scale.reshape(256)[None, :]
            if not POINT:
                norm2 = tl.load(Norms + token * blocks + block, member, 0)
                variance += residual.reshape(256)[None, :] * norm2[:, None]
        gate, up = tl.split(estimate.reshape(BM, 8, 16, 2))
        vg, vu = tl.split(variance.reshape(BM, 8, 16, 2))
        local = tl.arange(0, 128)
        order = tl.broadcast_to((local % 5 * 16 + local // 5)[None, :], (BM, 128))
        gate = tl.gather(gate.reshape(BM, 128), order, axis=1)
        up = tl.gather(up.reshape(BM, 128), order, axis=1)
        vg = tl.gather(vg.reshape(BM, 128), order, axis=1)
        vu = tl.gather(vu.reshape(BM, 128), order, axis=1)
        channels = tl.program_id(1) * 80 + local[None, :]
        written = member[:, None] & (local < 80)[None, :] & (channels < I)
        if SPLITS > 1:
            address = ((token[:, None] * SPLITS + split) * 2) * N
            address += slot[:, None] * I + channels
            tl.store(Parts + address, gate, written)
            tl.store(Parts + address + N, up, written)
            if not POINT:
                tl.store(VarianceParts + address, vg, written)
                tl.store(VarianceParts + address + N, vu, written)
        else:
            _store_screen(
                gate,
                up,
                vg,
                vu,
                token[:, None],
                slot[:, None],
                channels,
                expert,
                written,
                True,
                Partial,
                Variances,
                tl.load(Weights + token * K + slot, member, 0)[:, None],
                DownNorms,
                Geometry,
                Scores,
                Completed,
                Selected,
                I,
                N,
                GELU,
                GEOMETRY,
                POINT,
            )
        if tl.program_id(1) == 0:
            # Sector reads of this 16-pair expert block, counted once across tiles.
            requested = coordinates_read.to(tl.int64) * tl.cdiv(STRIDE, 32) * 32 + (
                end_block - begin_block
            ) * tl.cdiv(4 * I, 32) * 32 * (2 - POINT)
            tl.atomic_add(Work + 11, requested.to(tl.float64), sem="relaxed")


@triton.jit
def _group_union(
    Member,  # [tokens, N] bool per-pair channel flags
    Pairs,
    BlockExperts,
    PostPad,
    Union,  # [blocks, I] int32 channel list
    Counts,  # [blocks, 2] int32
    PAIRS,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    E: constexpr,
    BM: constexpr,
    BI: constexpr = 256,
):
    """List, in ascending order, the channels any pair of an expert block holds.

    Grid (blocks,). Counts receives the list's length and the number of channels
    some pair skips, which bounds the encoder rows the block reads.
    """
    expert = _block_expert(BlockExperts, PostPad, E, BM)
    block = tl.program_id(0)
    held = tl.full((), 0, tl.int32)
    missing = tl.full((), 0, tl.int32)
    if expert < E:
        for begin in range(0, I, BI):
            channel = begin + tl.arange(0, BI)
            union = tl.zeros((BI,), tl.int32)
            outside = tl.zeros((BI,), tl.int32)
            for row in static_range(BM):
                pair = tl.load(Pairs + block * BM + row)
                used = (pair < PAIRS) & (channel < I)
                inside = tl.load(
                    Member + (pair // K) * N + (pair % K) * I + channel, used, 0
                ).to(tl.int32)
                union = tl.maximum(union, inside)
                outside = tl.maximum(outside, (used & (inside == 0)).to(tl.int32))
            position = held + tl.cumsum(union, axis=0) - 1
            tl.store(Union + block * I + position, channel, union > 0)
            held += tl.sum(union, axis=0)
            missing += tl.sum(outside, axis=0)
    tl.store(Counts + block * 2, held)
    tl.store(Counts + block * 2 + 1, missing)


@triton.jit
def _group_complete(
    NativeGU,
    X,
    Pairs,
    BlockExperts,
    PostPad,
    Union,
    Counts,
    Member,
    Partial,
    PAIRS,
    H: constexpr,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    E: constexpr,
    BM: constexpr,
    ALL: constexpr,
    BC: constexpr = 64,
    BH: constexpr = 128,
):
    """Compute native gate and up projections for one expert block on tensor cores.

    Grid (blocks, cdiv(I, BC)); program (g, c) covers union channels
    [BC c, BC c + BC) (all channels with ALL), reading their BF16 rows once for
    the block. Stores FP32 projections only for pairs that hold the channel;
    `_acquire` then forms the activations.
    """
    expert = _block_expert(BlockExperts, PostPad, E, BM)
    # Padding entries of the aligned pair list equal the pair count.
    pair = tl.load(Pairs + tl.program_id(0) * BM + tl.arange(0, BM))
    member, token, slot = pair < PAIRS, pair // K, pair % K
    block = tl.program_id(0)
    column = tl.program_id(1) * BC + tl.arange(0, BC)
    count = I if ALL else tl.load(Counts + block * 2)
    if (expert < E) & (tl.program_id(1) * BC < count):
        in_union = column < count
        if ALL:  # noqa: SIM108
            channel = column
        else:
            channel = tl.load(Union + block * I + column, in_union, 0)
        row = expert * 2 * I + channel
        gate = tl.zeros((BM, BC), tl.float32)
        up = tl.zeros((BM, BC), tl.float32)
        for begin in range(0, H, BH):
            coordinate = begin + tl.arange(0, BH)
            x = tl.load(
                X + token[:, None] * H + coordinate[None, :], member[:, None], 0
            )
            offset = row[None, :] * H + coordinate[:, None]
            g = tl.load(NativeGU + offset, in_union[None, :], 0)
            gate = tl.dot(x, g, gate)
            u = tl.load(NativeGU + offset + I * H, in_union[None, :], 0)
            up = tl.dot(x, u, up)
        logical = slot[:, None] * I + channel[None, :]
        written = member[:, None] & in_union[None, :]
        if not ALL:
            written = written & (
                tl.load(Member + token[:, None] * N + logical, written, 0) != 0
            )
        address = token[:, None] * 2 * N + logical
        tl.store(Partial + address, gate, written)
        tl.store(Partial + address + N, up, written)


@triton.jit
def _group_down(
    D,
    Activations,
    Selected,
    Pairs,
    BlockExperts,
    PostPad,
    Union,
    Counts,
    Parts,
    Work,
    PAIRS,
    H: constexpr,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    E: constexpr,
    SPLITS: constexpr,
    BM: constexpr,
    ALL: constexpr,
    BC: constexpr = 64,
    BO: constexpr = 128,
    ACCOUNT: constexpr = False,
    R: constexpr = 0,
    ENCODER_TILES: constexpr = 0,
    SCREEN_MACS: constexpr = 0,
):
    """Expert-major variant of `_down` for one expert block, on tensor cores.

    Grid (blocks, cdiv(H, BO)). Reads the down rows of the union channels once,
    masks the activations to each pair's own selected channels, and writes each
    pair's whole FP32 channel sum to split 0. With ACCOUNT, program (g, 0) adds
    the block's requested bytes and tile MACs to Work[12:16].
    """
    expert = _block_expert(BlockExperts, PostPad, E, BM)
    # Padding entries of the aligned pair list equal the pair count.
    pair = tl.load(Pairs + tl.program_id(0) * BM + tl.arange(0, BM))
    member, token, slot = pair < PAIRS, pair // K, pair % K
    if expert < E:
        block = tl.program_id(0)
        output = tl.program_id(1) * BO + tl.arange(0, BO)
        count = I if ALL else tl.load(Counts + block * 2)
        if ACCOUNT and tl.program_id(1) == 0:
            omitted = 0 if ALL else tl.load(Counts + block * 2 + 1)
            counter = tl.arange(0, 4)
            wide_count = tl.full((), 0, tl.int64) + count
            macs = 16 * (
                SCREEN_MACS + 3 * H * tl.cdiv(wide_count, 64) * 64 + 64 * ENCODER_TILES
            )
            value = tl.where(counter == 0, 4 * H * wide_count, 0)
            value += tl.where(counter == 1, 2 * H * wide_count, 0)
            value += tl.where(counter == 2, 2 * R * omitted, 0)
            value += tl.where(counter == 3, macs, 0)
            tl.atomic_add(Work + 12 + counter, value.to(tl.float64), sem="relaxed")
        result = tl.zeros((BM, BO), tl.float32)
        for begin in range(0, count, BC):
            column = begin + tl.arange(0, BC)
            in_union = column < count
            if ALL:
                channel = column
            else:
                channel = tl.load(Union + block * I + column, in_union, 0)
            logical = token[:, None] * N + slot[:, None] * I + channel[None, :]
            held = member[:, None] & in_union[None, :]
            if not ALL:
                held = held & (tl.load(Selected + logical, held, 0) != 0)
            value = tl.load(Activations + logical, held, 0)
            # Physical D is [expert, I, H]; each union channel owns an H row.
            coefficient = tl.load(
                D + (expert * I + channel[:, None]) * H + output[None, :],
                in_union[:, None],
                0,
            )
            result = tl.dot(value, coefficient, result)
        # The whole FP32 channel sum precedes the per-expert BF16 rounding.
        tl.store(
            Parts + (token[:, None] * K + slot[:, None]) * SPLITS * H + output[None, :],
            result,
            member[:, None],
        )


@triton.jit
def _group_encode(
    B,
    Partial,
    Activations,
    Completed,
    Selected,
    Pairs,
    BlockExperts,
    PostPad,
    Parts,
    PAIRS,
    I: constexpr,
    K: constexpr,
    N: constexpr,
    R: constexpr,
    E: constexpr,
    SPLITS: constexpr,
    BM: constexpr,
    BR: constexpr,
    GELU: constexpr,
    BC: constexpr = 64,
):
    """Expert-major variant of `_encode` for one tile of an expert block.

    Reads the encoder rows of channels that some pair skips once for the block.
    """
    expert = _block_expert(BlockExperts, PostPad, E, BM)
    # Padding entries of the aligned pair list equal the pair count.
    pair = tl.load(Pairs + tl.program_id(0) * BM + tl.arange(0, BM))
    member, token, slot = pair < PAIRS, pair // K, pair % K
    if expert < E:
        tile = tl.program_id(1)
        channel = tile * BC + tl.arange(0, BC)
        rank = tl.arange(0, BR)
        logical = slot[:, None] * I + channel[None, :]
        offset = token[:, None] * N + logical
        # Recovery encodes only each token's skipped channels.
        rows = member[:, None] & (channel < I)[None, :]
        omitted = rows & (tl.load(Selected + offset, rows, 1) == 0)
        completed = tl.load(Completed + offset, omitted, 0) != 0
        outside = omitted & ~completed
        state = token[:, None] * 2 * N + logical
        g = tl.load(Partial + state, outside, 0).to(tl.bfloat16).to(tl.float32)
        u = tl.load(Partial + state + N, outside, 0).to(tl.bfloat16).to(tl.float32)
        partial_value = (_activate(g, GELU) * u).to(tl.bfloat16)
        exact = tl.load(Activations + offset, omitted & completed, 0)
        value = tl.where(completed, exact, partial_value)
        read = tl.max(omitted.to(tl.int32), axis=0) > 0
        coefficient = tl.load(
            B + (expert * I + channel[:, None]) * R + rank[None, :],
            read[:, None] & (rank[None, :] < R),
            0,
        )
        result = tl.dot(value, coefficient)
        tl.store(
            Parts
            + ((token[:, None] * K + slot[:, None]) * SPLITS + tile) * R
            + rank[None, :],
            result,
            member[:, None] & (rank[None, :] < R),
        )


class ReaderWorkspace:
    """Hold the device scratch, streams and events of one reader call.

    Readers of sequential layers may share one workspace. Its `BudgetPlan` fixes
    the largest shapes; a screened layer with a smaller final_count may share it.
    Buffers are sized for `capacity` tokens; per-token, expert-major and
    split-screen scratch is allocated only for the batch sizes that use it.
    `bytes` is the total allocation, reported once by the first reader bound to
    the workspace.

    Every buffer a call reads is rewritten (or constant) within that call, and a
    call joins its recovery side stream before its output kernel, so the next
    layer on the caller stream is ordered after this layer's uses. Layers must
    run sequentially on one stream.
    """

    @torch.no_grad()
    def __init__(
        self,
        plan: BudgetPlan,
        experts: int,
        rank: int,
        capacity: int,
        device: torch.device,
        grouped_tokens: int = GROUPED_TOKENS,
        grouped_screen_tokens: int = GROUPED_SCREEN_TOKENS,
        radix_select: bool = True,
        screen_block_splits: int = 8,
        grouped_screen_splits: int = 2,
    ) -> None:
        if capacity < 1 or grouped_tokens < 2:
            raise ValueError("Workspace needs a positive capacity and grouped size")
        if screen_block_splits not in (1, 2, 4, 8) or grouped_screen_splits not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("Screen block splits must be 1, 2, 4 or 8")
        self.key = (
            plan,
            experts,
            rank,
            capacity,
            device,
            grouped_tokens,
            grouped_screen_tokens,
        )
        h, i, k, n = (
            plan.hidden_size,
            plan.expert_width,
            plan.top_k,
            plan.candidate_count,
        )
        # Batches from grouped_tokens run expert-major.
        grouped = 0 if capacity < grouped_tokens else capacity
        per_token = min(capacity, grouped_tokens - 1)
        # Batches from max(grouped_tokens, grouped_screen_tokens) also screen
        # expert-major; smaller ones screen per token.
        screen_from = max(grouped_tokens, grouped_screen_tokens)
        grouped_screen = 0 if capacity < screen_from else capacity
        per_token_screen = min(capacity, screen_from - 1)
        self.ids = torch.empty((capacity, k), dtype=torch.int32, device=device)
        self.weights = torch.empty((capacity, k), dtype=torch.float32, device=device)
        self.partial = torch.empty((capacity, 2, n), dtype=torch.float32, device=device)
        self.variances = torch.empty_like(self.partial)
        # Compact rotated values and their coordinate indices for sparse gathers.
        self.sketch_planes = torch.empty(
            (per_token_screen, h // 128, plan.screen_coordinates),
            dtype=torch.float32,
            device=device,
        )
        self.sketch_coordinates = (
            torch.empty_like(self.sketch_planes, dtype=torch.int32)
            if plan.screen_coordinates < 128
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        # Rotation-block splits partition the original reads, not the sketch
        # layout. Scratch is reused by per-token and expert-major screens.
        self.screen_splits = min(8, h // 128) if plan.completion_count < n else 1
        batch_splits = min(screen_block_splits, h // 128)
        group_splits = min(grouped_screen_splits, h // 128)
        split_rows = (
            max(
                self.screen_splits if self.screen_splits > 1 else 0,
                per_token_screen * batch_splits
                if per_token_screen >= 2 and batch_splits > 1
                else 0,
                grouped_screen * group_splits if group_splits > 1 else 0,
            )
            if plan.completion_count < n
            else 0
        )
        self.screen_parts = torch.empty(
            (split_rows, 2, n),
            dtype=torch.float32,
            device=device,
        )
        self.screen_variance_parts = torch.empty_like(self.screen_parts)
        # One arrival counter per per-token (slot, 80-channel tile), not per
        # split. Initialized only at allocation; the last CTA resets each call.
        self.screen_counters = torch.zeros(
            (per_token if split_rows else 0, k, cdiv(i, 80)),
            dtype=torch.int32,
            device=device,
        )
        self.activations = torch.empty(
            (capacity, n), dtype=torch.bfloat16, device=device
        )
        self.completed = torch.empty((capacity, n), dtype=torch.bool, device=device)
        # Full retention selects every channel up front; otherwise the first
        # stage clears the flags and `_select` sets them.
        self.selected = torch.full(
            (capacity, n), plan.final_count == n, dtype=torch.bool, device=device
        )
        # The dense initial candidate list is implicit in the kernels.
        self.candidate_ids = [torch.empty((0,), dtype=torch.int32, device=device)]
        self.scores: list[Tensor] = []
        for step in plan.rounds:
            if step.retained_count == step.entering_count:
                continue
            self.candidate_ids.append(
                torch.empty(
                    (capacity, step.retained_count), dtype=torch.int32, device=device
                )
            )
            self.scores.append(
                torch.empty(
                    (capacity, step.entering_count), dtype=torch.float32, device=device
                )
            )
        self.starts = torch.empty((capacity, k), dtype=torch.int32, device=device)
        self.counts = torch.empty_like(self.starts)
        self.tile_starts = torch.empty_like(self.starts)
        self.down_splits, self.encoder_splits = (i + 127) // 128, (i + 63) // 64
        screened_final = plan.final_count == plan.completion_count < n
        radix_tokens = (
            min(per_token_screen, RADIX_TOKENS)
            if radix_select and screened_final and n >= 1024
            else 0
        )
        self.histogram = torch.empty(
            (radix_tokens, 1 << RADIX_BITS), dtype=torch.int32, device=device
        )
        self.bucket_counts = torch.empty(
            (radix_tokens, 1 << RADIX_HASH_BITS), dtype=torch.int32, device=device
        )
        self.bucket_keys = torch.empty(
            (radix_tokens, 1 << RADIX_HASH_BITS, RADIX_CAPACITY),
            dtype=torch.int32,
            device=device,
        )
        if plan.final_count == n:
            # All-channel membership is known before execution; no decision is needed.
            slots = torch.arange(k, device=device, dtype=torch.int32)
            self.starts[:] = slots * i
            self.counts.fill_(i)
            self.tile_starts[:] = slots * self.down_splits
        # The per-token down projection writes 128-channel parts; the grouped one
        # writes the whole pair sum.
        self.down_parts = torch.empty(
            (max(per_token * self.down_splits, grouped) * k * h,),
            dtype=torch.float32,
            device=device,
        )
        self.encoder_parts = torch.empty(
            (capacity, k, self.encoder_splits, rank), dtype=torch.float32, device=device
        )
        # Small BF16 recovery uses one last-arrival counter per token. The last
        # encoder CTA resets it, including padding and captured graph replays.
        self.recovery_counters = torch.zeros(
            (min(per_token, 8) if rank else 0,),
            dtype=torch.int32,
            device=device,
        )
        self.latent = torch.empty((capacity, rank), dtype=torch.float32, device=device)
        blocks = aligned_blocks(grouped * k, experts)
        self.keys = torch.empty((grouped, k), dtype=torch.int32, device=device)
        self.pairs = torch.empty(
            (blocks * GROUP_ROWS,), dtype=torch.int32, device=device
        )
        self.block_experts = torch.empty((blocks,), dtype=torch.int32, device=device)
        self.padded_pairs = torch.empty((1,), dtype=torch.int32, device=device)
        # One compact channel union per block, shared by exact computation and
        # the down projection, which use the same selected channels. Counts hold
        # [union length, channels some pair skips].
        unions = 0 if plan.final_count == n else 1
        self.unions = torch.empty((unions, blocks, i), dtype=torch.int32, device=device)
        self.union_counts = torch.empty(
            (unions, blocks, 2), dtype=torch.int32, device=device
        )
        # The expert-major screen's exact three-term BF16 split of the coordinates,
        # and every screened row's per-block squared norms.
        self.sketch_split = torch.empty(
            (grouped_screen, h // 128, 3, 128), dtype=torch.bfloat16, device=device
        )
        self.sketch_norms = torch.empty(
            (capacity, h // 128), dtype=torch.float32, device=device
        )
        mask_groups = blocks if grouped_screen and plan.completion_count < n else 0
        self.screen_coordinate_union = torch.empty(
            (mask_groups, h), dtype=torch.bool, device=device
        )
        self.screen_dense = torch.empty(
            (mask_groups, h // 128), dtype=torch.bool, device=device
        )
        buffers = (
            self.histogram,
            self.bucket_counts,
            self.bucket_keys,
            self.ids,
            self.weights,
            self.partial,
            self.variances,
            self.sketch_planes,
            self.sketch_coordinates,
            self.screen_parts,
            self.screen_variance_parts,
            self.screen_counters,
            self.activations,
            self.completed,
            self.selected,
            *self.candidate_ids,
            *self.scores,
            self.starts,
            self.counts,
            self.tile_starts,
            self.down_parts,
            self.encoder_parts,
            self.recovery_counters,
            self.latent,
            self.keys,
            self.pairs,
            self.block_experts,
            self.padded_pairs,
            self.unions,
            self.union_counts,
            self.sketch_split,
            self.sketch_norms,
            self.screen_coordinate_union,
            self.screen_dense,
        )
        self.bytes = sum(t.numel() * t.element_size() for t in buffers)
        # The first reader bound to this workspace reports its bytes once.
        self.charged = False
        # The highest priority: encode is enqueued just before the multi-wave
        # exact computation and must not queue behind it or concurrent dense work.
        highest = torch.cuda.current_stream(device).priority_range()[1]
        self.recovery_stream = torch.cuda.Stream(device=device, priority=highest)
        self.recovery_ready = torch.cuda.Event()
        self.routed, self.aligned = torch.cuda.Event(), torch.cuda.Event()
        self.statistics_ready = torch.cuda.Event()
        self.recovery_done = torch.cuda.Event()
        self.input_ready, self.input_done = torch.cuda.Event(), torch.cuda.Event()


class ProgressiveReader:
    """Sparse decode operator of one MoE layer's routed experts.

    Construct one per layer, then call it once per decode step with the layer's
    hidden rows and router outputs (see `__call__`). The operator reads the
    native weights in place, in the storage stock prefill uses, and owns no KV or
    serving state. Its workspace is reused across calls and is not reentrant.
    """

    @torch.no_grad()
    def __init__(
        self,
        gate_up: Tensor,
        down: Tensor,
        encoder: Tensor | None,
        decoder: Tensor | None,
        plan: BudgetPlan,
        max_tokens: int,
        expert_map: Tensor | None = None,
        sketch: tuple[Tensor, Tensor] | None = None,
        activation: GatedActivation = "silu",
        grouped_tokens: int = GROUPED_TOKENS,
        workspace: ReaderWorkspace | None = None,
        grouped_screen_tokens: int = GROUPED_SCREEN_TOKENS,
        selection_score: SelectionScore = "recovery-aware",
        fused_accounting: bool = True,
        radix_select: bool = True,
        screen_block_splits: int = 8,
        grouped_screen_splits: int = 2,
        selection_threshold: float | None = None,
        screen_last_block_reduction: bool = True,
        recovery_last_block_reduction: bool = True,
        pdl: bool = True,
    ) -> None:
        """Prepare the sketch, recovery geometry and workspace of one layer.

        Args:
            gate_up: Native BF16 gate and up rows [E, 2 I, H], contiguous.
            down: Channel-major down weight, logical [E, H, I] over physical
                [E, I, H] (see `channel_major_down`).
            encoder: Recovery-map (`bank`) encoder B [E, R, I], or None.
            decoder: Recovery-map decoder U [H, R], or None. With both None,
                recovery is disabled: nothing is recovered or read for it, and
                the screen ranks the skipped energy alone (alpha = beta = 0),
                exactly as an all-zero map would.
            plan: The layer's `BudgetPlan`. Fewer than 128 screen coordinates
                (the partial screen) screen only each block's largest rotated
                coordinates, and recovery then encodes those partial estimates.
            max_tokens: Largest batch per call.
            expert_map: For expert parallelism, global to local expert ids (-1
                for remote experts); ranks then all-reduce scores and latents.
            sketch: Prebuilt (codes, statistics) from `build_sketch` with the
                seed-0 rotation; built here if None.
            activation: Gate activation, "silu" or "gelu_tanh".
            grouped_tokens: Batch size from which stages run expert-major.
            workspace: A `ReaderWorkspace` shared with layers run in sequence.
            grouped_screen_tokens: Batch size from which the screen runs
                expert-major too.
            selection_score: The `SelectionScore` the screen ranks by.
            fused_accounting: Add the selection-independent read statistics
                inside existing launches.
            radix_select: Narrow the top-k search with a radix histogram.
            screen_block_splits: Splits of the rotation blocks for batched
                per-token screens.
            grouped_screen_splits: Splits for the expert-major screen.
            selection_threshold: For calibrated-threshold baselines, keep every
                candidate scoring at least this layer threshold, at most F per
                token, instead of each token's top F.
            screen_last_block_reduction: Let the last split CTA to arrive score,
                instead of a separate launch.
            recovery_last_block_reduction: Let the last encoder CTA mix small
                BF16 latents, instead of a separate `_latent` launch.
            pdl: Overlap kernel prologues by programmatic dependent launch on
                SM90 and newer (per-token screened layers with fused accounting).

        The scheduling options (grouped_tokens, grouped_screen_tokens,
        fused_accounting, radix_select, the split counts, the last-block
        reductions and pdl) leave the arithmetic unchanged, except that
        expert-major execution and screen splits change the FP32 summation order,
        which can flip a selection whose score lies within rounding error of the
        top-k cutoff.
        """
        h, i, k = plan.hidden_size, plan.expert_width, plan.top_k
        e = gate_up.shape[0]
        rank = decoder.shape[1] if decoder is not None else 0
        if selection_threshold is not None and (
            not selection_threshold >= 0
            or selection_threshold == float("inf")
            or not plan.final_count == plan.completion_count < plan.candidate_count
        ):
            raise ValueError(
                "A selection threshold is finite, nonnegative and needs a screened plan"
            )
        # The threshold's kept count varies per token: the fixed-count radix search
        # and the fused, selection-independent read statistics do not apply.
        thresholded = selection_threshold is not None
        # FP32 explicitly: vLLM prepares readers under its BF16 default dtype.
        self.selection_threshold = (
            torch.tensor(
                [selection_threshold], device=gate_up.device, dtype=torch.float32
            )
            if thresholded
            else None
        )
        if screen_block_splits not in (1, 2, 4, 8) or grouped_screen_splits not in (
            1,
            2,
            4,
            8,
        ):
            raise ValueError("Screen block splits must be 1, 2, 4 or 8")
        self.batch_screen_splits = min(screen_block_splits, h // 128)
        self.grouped_screen_splits = min(grouped_screen_splits, h // 128)
        self.radix_select = (
            radix_select
            and not thresholded
            and expert_map is None
            and plan.candidate_count >= 1024
            and plan.final_count == plan.completion_count < plan.candidate_count
        )
        shared_b = 0 < rank < 16 and 16 % rank == 0 and i % (16 // rank) == 0
        self.fused_accounting = (
            fused_accounting
            and not thresholded
            and expert_map is None
            and plan.completion_count == plan.final_count
            and not shared_b
        )
        if (
            gate_up.shape != (e, 2 * i, h)
            or down.shape != (e, h, i)
            or (encoder is None) != (decoder is None)
            or (encoder is not None and encoder.shape != (e, rank, i))
            or (decoder is not None and (decoder.shape != (h, rank) or rank < 1))
            or not gate_up.is_cuda
            or down.device != gate_up.device
            or gate_up.dtype != torch.bfloat16
            or down.dtype != gate_up.dtype
            or not gate_up.is_contiguous()
            or down.stride() != (h * i, 1, h)
            or (expert_map is None and k > e)
            or max_tokens < 1
            or grouped_tokens < 2
        ):
            raise ValueError(
                "Expected native GU and channel-major D CUDA experts, "
                "matching recovery and capacity"
            )
        self.pdl = (
            pdl
            and expert_map is None
            and self.fused_accounting
            and plan.final_count == plan.completion_count < plan.candidate_count
            and torch.cuda.get_device_capability(gate_up.device)[0] >= 9
        )
        self.screen_last_block_reduction = (
            screen_last_block_reduction and expert_map is None
        )
        # Bound the last CTA's vectorized table to 4096 FP32 entries. Wider
        # banks, EP and grouped encoding use the separate `_latent` launch.
        self.recovery_last_block_reduction = (
            recovery_last_block_reduction
            and expert_map is None
            and rank > 0
            and next_power_of_2(k)
            * next_power_of_2(cdiv(i, 64))
            * next_power_of_2(rank)
            <= 4096
        )
        self.selection_score: SelectionScore = selection_score
        self.expert_map = (
            expert_map.to(device=gate_up.device, dtype=torch.int32)
            if expert_map is not None
            else None
        )
        if self.expert_map is not None:
            from vllm.distributed import get_ep_group

            self.ep_group = get_ep_group().device_group
            self.recovery_owner = get_ep_group().rank_in_group == 0
        else:
            self.ep_group = None
            self.recovery_owner = True
        if self.expert_map is not None and (
            self.expert_map.ndim != 1 or int((self.expert_map >= 0).sum()) != e
        ):
            raise ValueError("Expert map must cover exactly the native local experts")
        self.plan, self.max_tokens, self.rank = plan, max_tokens, rank
        self.activation: GatedActivation = activation
        device = gate_up.device
        self.sketch_signs = block_hadamard_signs(h, device)
        if sketch is None:
            sketch = build_sketch(gate_up, self.sketch_signs)
        self.sketch_codes, self.sketch_statistics = sketch
        if (
            self.sketch_codes.shape != (e, sketch_storage_bytes(2 * i, h))
            or self.sketch_codes.dtype != torch.uint8
            or self.sketch_codes.device != device
            or not self.sketch_codes.is_contiguous()
            or self.sketch_statistics.shape != (e, 2 * i, h // 128, 2)
            or self.sketch_statistics.dtype != torch.float16
            or self.sketch_statistics.device != device
            or self.sketch_statistics.stride() != (4 * i * (h // 128), 1, 4 * i, 2 * i)
        ):
            raise ValueError("Sketch must match the local native routed experts")
        self.sketch_resident_bytes = (
            self.sketch_codes.numel() + self.sketch_statistics.numel() * 2
        )
        self.norms = down.float().square().sum(1).sqrt()
        self.native_gu = gate_up.detach()
        self.down = down.detach()
        # Without a recovery map, no map or score geometry is resident or read.
        self.encoder: Tensor | None = None
        self.decoder: Tensor | None = None
        self.recovery_geometry: Tensor | None = None
        prepared = False
        if encoder is not None and decoder is not None:
            self.encoder = (
                encoder.to(device=device, dtype=torch.bfloat16)
                .transpose(1, 2)
                .contiguous()
            )
            self.decoder = decoder.to(device=device, dtype=torch.bfloat16).contiguous()
            # The point score reads no geometry, so none is resident or read.
            geometry = (
                torch.zeros((e, i, 2), device=device, dtype=torch.float32)
                if selection_score != "point"
                else None
            )
            self.recovery_geometry = geometry
            # Each channel's recovery response U B_i splits into alpha, its
            # component along the channel's down column, and beta, the norm of
            # the rest. Only this screening geometry needs the response; one
            # expert at a time bounds setup workspace, and no expanded response
            # is retained.
            if geometry is not None:
                prepared = True
                decoder_float = self.decoder.float().T
                for expert in range(e):
                    original = down[expert].float().T
                    response = self.encoder[expert].float() @ decoder_float
                    norm = self.norms[expert]
                    alpha = (original * response).sum(-1) / torch.where(
                        norm > 0, norm, torch.ones_like(norm)
                    )
                    beta = (
                        (response.square().sum(-1) - alpha.square()).clamp_min(0).sqrt()
                    )
                    geometry[expert, :, 0].copy_(alpha)
                    geometry[expert, :, 1].copy_(beta)
        n = plan.candidate_count
        if workspace is None:
            workspace = ReaderWorkspace(
                plan,
                e,
                rank,
                max_tokens,
                device,
                grouped_tokens,
                grouped_screen_tokens,
                radix_select=self.radix_select,
                screen_block_splits=screen_block_splits,
                grouped_screen_splits=grouped_screen_splits,
            )
        workspace_plan = workspace.key[0]
        compatible_plan = workspace_plan == plan or (
            workspace_plan.hidden_size == h
            and workspace_plan.expert_width == i
            and workspace_plan.top_k == k
            and workspace_plan.screen_coordinates == plan.screen_coordinates
            and plan.final_count == plan.completion_count
            and workspace_plan.final_count == workspace_plan.completion_count
            and plan.final_count <= workspace_plan.final_count < n
        )
        if not compatible_plan or workspace.key[1:] != (
            e,
            rank,
            max_tokens,
            device,
            grouped_tokens,
            grouped_screen_tokens,
        ):
            raise ValueError(
                "Shared workspace must match the plan, experts and capacity"
            )
        self.workspace = workspace
        self.grouped_tokens = grouped_tokens
        self.grouped_screen_tokens = grouped_screen_tokens
        w = workspace
        self.histogram, self.bucket_counts, self.bucket_keys = (
            w.histogram,
            w.bucket_counts,
            w.bucket_keys,
        )
        # Optional paths may share scratch with generic readers when it fits.
        if self.radix_select and self.histogram.shape[0] < min(
            w.sketch_planes.shape[0], RADIX_TOKENS
        ):
            raise ValueError(
                "Shared workspace lacks the selected decode tiling capacity"
            )
        self.ids, self.weights, self.partial, self.variances = (
            w.ids,
            w.weights,
            w.partial,
            w.variances,
        )
        self.sketch_planes = w.sketch_planes
        self.sketch_coordinates = w.sketch_coordinates
        self.screen_splits = w.screen_splits
        self.screen_parts = w.screen_parts
        self.screen_variance_parts = w.screen_variance_parts
        self.screen_counters = w.screen_counters
        split_rows = max(
            w.sketch_planes.shape[0] * self.batch_screen_splits
            if w.sketch_planes.shape[0] >= 2 and self.batch_screen_splits > 1
            else 0,
            w.sketch_split.shape[0] * self.grouped_screen_splits
            if self.grouped_screen_splits > 1
            else 0,
        )
        if plan.completion_count < n and self.screen_parts.numel() < split_rows * 2 * n:
            raise ValueError(
                "Shared workspace lacks the selected screen split capacity"
            )
        self.activations, self.completed, self.selected = (
            w.activations,
            w.completed,
            w.selected,
        )
        self.candidate_ids, self.scores = w.candidate_ids, w.scores
        if workspace_plan != plan:
            # Kernels index packed token*F lists, not tensor strides. A column
            # slice of the larger list would leave gaps between token rows.
            self.candidate_ids = [
                w.candidate_ids[0],
                w.candidate_ids[1]
                .view(-1)[: max_tokens * plan.final_count]
                .view(max_tokens, plan.final_count),
            ]
        self.starts, self.counts, self.tile_starts = w.starts, w.counts, w.tile_starts
        self.down_splits, self.encoder_splits = w.down_splits, w.encoder_splits
        self.down_parts, self.encoder_parts, self.latent = (
            w.down_parts,
            w.encoder_parts,
            w.latent,
        )
        self.recovery_counters = w.recovery_counters
        self.keys, self.pairs, self.block_experts, self.padded_pairs = (
            w.keys,
            w.pairs,
            w.block_experts,
            w.padded_pairs,
        )
        self.unions, self.union_counts = w.unions, w.union_counts
        self.sketch_split, self.sketch_norms = w.sketch_split, w.sketch_norms
        self.screen_coordinate_union = w.screen_coordinate_union
        self.screen_dense = w.screen_dense
        self.recovery_stream = w.recovery_stream
        self.recovery_ready, self.recovery_done = w.recovery_ready, w.recovery_done
        self.input_ready, self.input_done = w.input_ready, w.input_done
        self.statistics_ready = w.statistics_ready
        self.routed, self.aligned = w.routed, w.aligned
        # Shared scratch is reported once, by the first reader bound to it.
        self.workspace_bytes = 0 if w.charged else w.bytes
        w.charged = True
        self.recovery_geometry_bytes = (
            self.recovery_geometry.numel() * 4
            if self.recovery_geometry is not None
            else 0
        )
        # The point score holds no geometry.
        self.recovery_geometry_preparation_products = prepared * 2 * e * i * h
        self.recovery_geometry_preparation_reduction_adds = (
            prepared * 2 * e * i * (h - 1)
        )
        self.recovery_response_preparation_macs = prepared * e * i * h * rank
        self.matrix_macs = (
            2 * n * h * (plan.completion_count != n)
            + 2 * plan.completion_count * h
            + h * plan.final_count
            + rank * (n - plan.final_count + h)
        )
        # Read statistics, summed on device so they also accumulate under CUDA
        # graph replay. Logical (exact) bytes: 0 native weights, 1 dense
        # reference, 3 recovery B and U, 4 selection metadata, 5 sketch. Read
        # cost (sector-priced) bytes: 6 native, 7 recovery, 9 selection,
        # 17 sketch; slot 10, the rotation signs, counts toward both. Requested
        # bytes: 11 sketch, 12 gate/up, 13 down, 14 encoder. Then 15 grouped tile
        # MACs, 16 grouped rows, 19 active tokens of a thresholded reader. Slots
        # 2, 8 and 18 are unused.
        self.read_work = torch.zeros(20, device=device, dtype=torch.float64)

    def report(self) -> dict[str, float | None]:
        """Return the accumulated read statistics, work and workspace counters.

        Reads `read_work` back to the host (a device sync). Byte counts are
        totals since `read_work` was last zeroed; per-token entries are
        analytical prices or bounds for one token. "charged_bytes" is the read
        cost and "read_fraction" its ratio to the dense reference; "logical_*"
        entries price exact bytes instead of sectors, and "issued_*" entries are
        the requested bytes.
        """
        (
            native,
            reference,
            _,
            recovery,
            selection,
            sketch,
            physical_native,
            physical_recovery,
            _,
            physical_selection,
            rotation_signs,
            issued_sketch,
            issued_gate_up,
            issued_down,
            issued_encoder,
            grouped_macs,
            grouped_rows,
            physical_sketch,
            _,
            threshold_tokens,
        ) = self.read_work.cpu().tolist()
        h = self.plan.hidden_size
        logical = native + recovery + selection + sketch + rotation_signs
        charged = (
            physical_native
            + physical_recovery
            + physical_selection
            + physical_sketch
            + rotation_signs
        )
        screened = float(self.plan.completion_count != self.plan.candidate_count)
        input_sparse = screened * (self.plan.screen_coordinates < 128)
        counter_groups = (
            self.plan.top_k
            * cdiv(self.plan.expert_width, 80)
            * self.screen_last_block_reduction
            * screened
        )
        route_programs = (
            self.plan.top_k
            * cdiv(self.plan.expert_width, 80)
            * max(self.screen_splits, self.batch_screen_splits)
            * screened
        )
        route_slots = next_power_of_2(self.plan.top_k)
        route_stages = route_slots.bit_length() - 1
        n, k = self.plan.candidate_count, self.plan.top_k
        # Scores entering the selection that keeps the final count.
        final_entering = next(
            (
                step.entering_count
                for step in self.plan.rounds
                if step.retained_count == self.plan.final_count < step.entering_count
            ),
            0,
        )
        # Up to batch eight, the dense final selection runs one CTA per routed
        # expert; each repeats the cutoff search over all n scores.
        partitioned = float(
            self.selection_threshold is None and n >= 1024 and final_entering == n
        )
        repeats = 1 + (k - 1) * partitioned
        return {
            "charged_bytes": charged,
            "reference_bytes": reference,
            "native_expert_coefficient_bytes": native,
            "physical_native_expert_coefficient_bytes": physical_native,
            "recovery_bytes": recovery,
            "physical_recovery_bytes": physical_recovery,
            "selection_parameter_bytes": selection,
            "physical_selection_parameter_bytes": physical_selection,
            "sketch_bytes": sketch,
            "physical_sketch_bytes": physical_sketch,
            "sketch_resident_bytes": float(self.sketch_resident_bytes),
            "selection_threshold_resident_bytes": 4.0
            * (self.selection_threshold is not None),
            "rotation_sign_bytes": rotation_signs,
            "rotation_sign_resident_bytes": float(h),
            # Early, non-grouped screens repeat the small router sort per CTA.
            # Bound INT64 IDs plus FP32 weights, then one metadata write per slot.
            "early_screen_route_metadata_io_bytes_per_token_upper_bound": float(
                12 * self.plan.top_k * route_programs + 8 * self.plan.top_k * screened
            ),
            "early_screen_route_compare_exchanges_per_token_upper_bound": float(
                route_programs * route_slots * route_stages * (route_stages + 1) // 4
            ),
            "early_screen_route_reduction_adds_per_token_upper_bound": float(
                route_programs * 2 * (route_slots - 1)
            ),
            # Full retention (every channel selected) skips the screen, so no
            # input rotation runs. Per-token rows read BF16 input and write compact FP32
            # values, INT32 coordinates and per-block FP32 norms (signs are
            # priced above): sign, seven butterfly, scale and squared-norm
            # multiply-add operations per coordinate.
            "rotation_input_io_bytes_per_token": float(
                2 * h
                + ((8 if input_sparse else 4) * self.plan.screen_coordinates + 4)
                * (h // 128)
            )
            * screened,
            "rotation_arithmetic_per_token": float(11 * h) * screened,
            # Grouped-screen rows instead write three BF16 planes, adding two
            # residual subtractions per coordinate (plus three BF16 conversions).
            "grouped_rotation_input_io_bytes_per_token": float(8 * h + 4 * (h // 128))
            * screened,
            "grouped_rotation_arithmetic_per_token": float(13 * h) * screened,
            # Partial-sketch screens rank every 128-value block by a bitonic sort
            # (28 stages of 64 compare-exchanges). Per block the mask then takes
            # 9 x 128 elementwise operations (abs; index compare and select of the
            # threshold; above, tied and scan-budget compares; and, or; the final
            # select), 2 x 128 Boolean-to-integer casts, 3 x 127 adds (threshold
            # and above reductions, tie scan) and the tie-budget subtraction.
            "screen_ranking_compare_exchanges_per_token": float(h // 128 * 1792)
            * input_sparse,
            "screen_mask_arithmetic_per_token": float(
                h // 128 * (9 * 128 + 2 * 128 + 3 * 127 + 1)
            )
            * input_sparse,
            "screen_compaction_scan_adds_per_token": float(h // 128 * 127)
            * input_sparse,
            "batch_one_screen_split_io_bytes": float(
                16
                * self.screen_splits
                * self.plan.candidate_count
                * (1 + (self.selection_score != "point"))
            )
            * (self.screen_splits > 1)
            * screened,
            # Per-token B>=2 and grouped screens can use different split counts.
            # Both write/read each FP32 partial once, with no new parameter reads.
            "batched_screen_split_io_bytes_per_token": float(
                16
                * self.batch_screen_splits
                * self.plan.candidate_count
                * (1 + (self.selection_score != "point"))
            )
            * (self.batch_screen_splits > 1)
            * screened,
            "batched_screen_split_adds_per_token": float(
                2
                * (next_power_of_2(self.batch_screen_splits) - 1)
                * self.plan.candidate_count
                * (1 + (self.selection_score != "point"))
            )
            * screened,
            # Counter traffic is transient workspace, not a parameter read:
            # SPLITS 4-byte atomic RMWs and one 4-byte reset per channel tile.
            "batch_one_screen_counter_io_bytes": float(
                counter_groups * (8 * self.screen_splits + 4)
            )
            * (self.screen_splits > 1),
            "batched_screen_counter_io_bytes_per_token": float(
                counter_groups * (8 * self.batch_screen_splits + 4)
            )
            * (self.batch_screen_splits > 1),
            # Conservative no-cache bound: a sector read/write per atomic, plus a
            # sector store for the reset, reported beside the read cost.
            "batch_one_screen_counter_sector_bytes_upper_bound": float(
                counter_groups * (64 * self.screen_splits + 32)
            )
            * (self.screen_splits > 1),
            "batched_screen_counter_sector_bytes_per_token_upper_bound": float(
                counter_groups * (64 * self.batch_screen_splits + 32)
            )
            * (self.batch_screen_splits > 1),
            "grouped_screen_split_io_bytes_per_token": float(
                16
                * self.grouped_screen_splits
                * self.plan.candidate_count
                * (1 + (self.selection_score != "point"))
            )
            * (self.grouped_screen_splits > 1)
            * screened,
            "grouped_screen_split_adds_per_token": float(
                2
                * (next_power_of_2(self.grouped_screen_splits) - 1)
                * self.plan.candidate_count
                * (1 + (self.selection_score != "point"))
            )
            * screened,
            # Worst case: each routed pair occupies its own expert block. The
            # prepass reads three BF16 input planes and alignment metadata; it
            # writes the mask and dense flag. Each channel tile reads both.
            # Scalar metadata requests are priced as full 32-byte sectors.
            "grouped_screen_union_io_bytes_per_token_upper_bound": float(
                6 * h * self.plan.top_k
                + self.plan.top_k * (h + 160 * (h // 128))
                + self.plan.top_k
                * cdiv(self.plan.expert_width, 80)
                * (h + 32 * (h // 128))
            )
            * screened,
            # Batch-one lanes; batched launches' 64-wide chunks plus a tail
            # request no more, so this bounds them.
            "per_token_screen_dot_terms": float(
                self.plan.top_k
                * ((self.plan.expert_width + 79) // 80)
                * 160
                * (h // 128)
                * next_power_of_2(self.plan.screen_coordinates)
            )
            * screened,
            "read_fraction": charged / reference if reference else None,
            "logical_charged_bytes": logical,
            "logical_read_fraction": logical / reference if reference else None,
            # Rows requested by the executed kernels, before cache reuse; the
            # per-token read cost above is unaffected.
            "issued_sketch_bytes": issued_sketch,
            "issued_gate_up_bytes": issued_gate_up,
            "issued_down_bytes": issued_down,
            "issued_recovery_encoder_bytes": issued_encoder,
            # Rows executed expert-major and the MACs their tensor-core tiles
            # perform, including padded rows, union tiles and the three-term
            # screen; per-token rows follow matrix_macs_per_token.
            "grouped_rows": grouped_rows,
            "grouped_issued_matrix_macs": grouped_macs,
            "workspace_bytes": float(self.workspace_bytes),
            # On-chip input staging for one token, separate from the allocated
            # global workspace.
            "batch_one_completion_shared_bytes_per_cta_upper_bound": float(
                4 * next_power_of_2(max(h, 1024)) * self.pdl
            ),
            # Only small per-token BF16 recovery uses these transient counters:
            # one 4-byte atomic RMW per encoder CTA and one reset per token.
            "small_batch_recovery_counter_io_bytes_per_token": float(
                (8 * self.plan.top_k * self.encoder_splits + 4)
                * self.recovery_last_block_reduction
            ),
            "small_batch_recovery_counter_sector_bytes_per_token_upper_bound": float(
                (64 * self.plan.top_k * self.encoder_splits + 32)
                * self.recovery_last_block_reduction
            ),
            # Mode-specific request bounds, not DRAM traffic: the histogram is
            # zeroed once and scanned by every selection CTA; each score makes
            # two 4-byte atomic RMWs and at most one cache store. Overflow falls
            # back to the full Scores already in the workspace.
            "small_batch_radix_scratch_io_bytes_per_token_upper_bound": float(
                4 * (1 << RADIX_BITS)
                + 4 * (1 << RADIX_HASH_BITS)
                + 20 * n
                + repeats * (4 * (1 << RADIX_BITS) + 4 + 4 * RADIX_CAPACITY)
            )
            * self.radix_select,
            "small_batch_radix_atomic_adds_per_token": float(
                2 * self.plan.candidate_count
            )
            * self.radix_select,
            # Conservative requested-sector bound: each atomic is a sector read
            # plus write and each scattered key store a full sector.
            "small_batch_radix_scratch_sector_bytes_per_token_upper_bound": float(
                4 * (1 << RADIX_BITS)
                + 4 * (1 << RADIX_HASH_BITS)
                + 160 * n
                + repeats * (4 * (1 << RADIX_BITS) + 32 + 4 * RADIX_CAPACITY)
            )
            * self.radix_select,
            "small_batch_radix_scan_adds_per_token_upper_bound": float(
                (1 << RADIX_BITS) * RADIX_BITS
            )
            * repeats
            * self.radix_select,
            # Each final-selection CTA loads all n scores for the cutoff search;
            # partitioned CTAs reload their own expert's scores to compact them.
            "small_batch_final_selection_score_io_bytes_per_token": float(
                4 * final_entering * (repeats + partitioned)
            ),
            # Analytical per-token contraction MACs (the reported work axis),
            # not the tiles a batched execution runs.
            "matrix_macs_per_token": float(
                (self.matrix_macs - self.rank * h)
                / (self.ep_group.size() if self.ep_group is not None else 1)
                + (self.rank * h if self.recovery_owner else 0)
            )
            if self.selection_threshold is None
            else self._threshold_matrix_macs(native, reference, threshold_tokens),
        }

    def _validate_input(self, hidden: Tensor, padding: Tensor | None) -> int:
        """Check the hidden rows and padding mask; return the token count."""
        tokens = hidden.shape[0]
        if (
            not 1 <= tokens <= self.max_tokens
            or hidden.shape != (tokens, self.plan.hidden_size)
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.native_gu.device
            or not hidden.is_contiguous()
            or (
                padding is not None
                and (
                    padding.shape != (tokens,)
                    or padding.device != hidden.device
                    or padding.dtype != torch.bool
                    or not padding.is_contiguous()
                )
            )
        ):
            raise ValueError(
                "Sparse input exceeds the prepared token/candidate contract"
            )
        return tokens

    @torch.no_grad()
    def prepare_input(self, hidden: Tensor, padding: Tensor | None = None) -> bool:
        """Start the routing-independent input rotation on the side stream.

        Call this before the router finishes, then pass input_prepared=True to
        the next `__call__` with the same hidden rows [tokens, H] and padding
        [tokens]; that call joins the side stream before screening. Returns
        False, launching nothing, under EP or for unscreened plans, which keep
        the single-launch preparation. The workspace remains nonreentrant,
        including between preparation and its consuming call.
        """
        p = self.plan
        if self.expert_map is not None or p.completion_count == p.candidate_count:
            return False
        tokens = self._validate_input(hidden, padding)
        grouped_screen = (
            tokens >= self.grouped_tokens and tokens >= self.grouped_screen_tokens
        )
        main_stream = torch.cuda.current_stream(self.native_gu.device)
        self.input_ready.record(main_stream)
        with torch.cuda.stream(self.recovery_stream):
            self.input_ready.wait(self.recovery_stream)
            cast(triton.runtime.JITFunction, _prepare_input)[
                (tokens, p.hidden_size // 128)
            ](
                hidden,
                self.sketch_signs,
                self.sketch_planes,
                self.sketch_split,
                self.sketch_norms,
                self.sketch_coordinates,
                padding,
                self.histogram,
                self.bucket_counts,
                p.hidden_size,
                p.candidate_count,
                padding is not None,
                grouped_screen,
                p.screen_coordinates,
                self.radix_select and tokens <= RADIX_TOKENS and not grouped_screen,
                next_power_of_2(cdiv(1 << RADIX_BITS, p.hidden_size // 128)),
                num_warps=1,
            )
            self.input_done.record(self.recovery_stream)
        return True

    @torch.no_grad()
    def __call__(
        self,
        hidden: Tensor,
        ids: Tensor,
        weights: Tensor,
        padding: Tensor | None = None,
        out: Tensor | None = None,
        *,
        input_prepared: bool = False,
    ) -> Tensor:
        """Compute the layer's routed-expert output for a batch of tokens.

        hidden is the BF16 input [tokens, H]; ids and weights are the router's
        top-k expert ids and weights [tokens, K]; padding marks padding rows
        [tokens], which read no expert weights and add no read statistics.
        Writes the output [tokens, H] into out (allocated if absent), returns it,
        and adds the call's read statistics to `read_work`. Pass
        input_prepared=True only after `prepare_input` returned True for the
        same rows.
        """
        tokens = self._validate_input(hidden, padding)
        p = self.plan
        h, i, k, n, rank = (
            p.hidden_size,
            p.expert_width,
            p.top_k,
            p.candidate_count,
            self.rank,
        )
        if (
            ids.shape != (tokens, k)
            or weights.shape != ids.shape
            or ids.device != hidden.device
            or weights.device != hidden.device
            or ids.dtype not in (torch.int32, torch.int64)
            or weights.dtype not in (torch.float32, torch.bfloat16)
            or not ids.is_contiguous()
            or not weights.is_contiguous()
            or (
                out is not None
                and (
                    out.shape != hidden.shape
                    or out.dtype != hidden.dtype
                    or out.device != hidden.device
                    or not out.is_contiguous()
                )
            )
        ):
            raise ValueError(
                "Sparse input exceeds the prepared token/candidate contract"
            )
        if out is None:
            out = torch.empty_like(hidden)
        has_padding = padding is not None
        # Both paths compute each pair's contractions with the same rounding
        # boundaries, up to FP32 reassociation (rare near-threshold flips).
        grouped = tokens >= self.grouped_tokens
        pdl = self.pdl and not grouped
        grouped_screen = grouped and tokens >= self.grouped_screen_tokens
        radix_select = (
            self.radix_select and tokens <= RADIX_TOKENS and not grouped_screen
        )
        prioritize_cache = tokens <= CACHE_PRIORITY_TOKENS
        experts = self.native_gu.shape[0]
        # Without early preparation, rotation stays in the routing launch.
        screen = p.completion_count < n
        splits = (
            self.grouped_screen_splits
            if grouped_screen
            else self.screen_splits
            if tokens == 1
            else (self.batch_screen_splits if tokens >= 2 else 1)
        )
        last_block_reduction = (
            self.screen_last_block_reduction and screen and splits > 1 and not grouped
        )
        screen_route = (
            input_prepared and screen and not grouped and self.expert_map is None
        )
        if not screen_route:
            _prepare.run(
                ids,
                weights,
                self.expert_map,
                self.ids,
                self.weights,
                self.keys,
                padding,
                h,
                k,
                next_power_of_2(k),
                has_padding,
                self.expert_map is not None,
                grouped,
                experts,
                hidden,
                self.sketch_signs,
                self.sketch_planes,
                self.sketch_split,
                self.sketch_norms,
                screen,
                grouped_screen,
                p.screen_coordinates,
                self.sketch_coordinates,
                self.scores[0] if screen and grouped_screen and splits == 1 else None,
                self.completed,
                self.selected,
                n,
                next_power_of_2(n),
                Work=self.read_work,
                ACCOUNT=self.fused_accounting,
                CHARGES=self._fixed_charges(grouped, grouped_screen)
                if self.fused_accounting
                else (),
                R=rank,
                F=p.final_count,
                Histogram=self.histogram,
                BucketCounts=self.bucket_counts,
                RADIX=radix_select,
                HB=next_power_of_2(cdiv(1 << RADIX_BITS, h // 128)),
                INPUT_PREPARED=input_prepared,
                grid=(tokens, h // 128 if screen and not input_prepared else 1),
                warmup=False,
                num_warps=1,
            )
        group_blocks = aligned_blocks(tokens * k, experts)
        main_stream = torch.cuda.current_stream(self.native_gu.device)
        if grouped:
            # The per-token screen does not need the expert blocks, so they are
            # sorted beside it; the expert-major screen waits for them.
            self.routed.record(main_stream)
            with torch.cuda.stream(self.recovery_stream):
                self.routed.wait(self.recovery_stream)
                _group_sort[(experts,)](
                    self.keys,
                    self.pairs,
                    self.block_experts,
                    self.padded_pairs,
                    tokens * k,
                    experts,
                    GROUP_ROWS,
                    next_power_of_2(experts + 1),
                )
                self.aligned.record(self.recovery_stream)
            if grouped_screen:
                self.aligned.wait(main_stream)
        if input_prepared:
            self.input_done.wait(main_stream)
        group_rows = (self.pairs, self.block_experts, self.padded_pairs)
        selection = 0
        candidate_ids = self.candidate_ids[0]
        for step in p.rounds:
            count, keep = step.entering_count, step.retained_count
            prune = keep < count
            if step.input_end == 0:
                if grouped_screen:
                    cast(triton.runtime.JITFunction, _group_screen_union)[
                        (group_blocks, h // 128)
                    ](
                        self.sketch_split,
                        *group_rows,
                        self.screen_coordinate_union,
                        self.screen_dense,
                        tokens * k,
                        h,
                        k,
                        experts,
                        GROUP_ROWS,
                        num_warps=4,
                    )
                    _group_screen.run(
                        self.sketch_codes,
                        self.sketch_statistics,
                        self.sketch_split,
                        self.sketch_norms,
                        *group_rows,
                        self.partial,
                        self.variances,
                        self.read_work,
                        self.weights,
                        self.norms,
                        self.recovery_geometry,
                        self.scores[selection],
                        self.completed,
                        self.selected,
                        self.screen_parts,
                        self.screen_variance_parts,
                        self.screen_coordinate_union,
                        self.screen_dense,
                        tokens * k,
                        h,
                        i,
                        k,
                        n,
                        experts,
                        GROUP_ROWS,
                        sketch_code_stride(2 * i),
                        sketch_storage_bytes(2 * i, h),
                        self.activation == "gelu_tanh",
                        self.recovery_geometry is not None,
                        self.selection_score == "point",
                        SPLITS=splits,
                        grid=(group_blocks, cdiv(i, 80), splits),
                        warmup=False,
                        num_warps=4,
                        num_stages=2,
                    )
                else:
                    _sketch_screen[(cdiv(i, 80), k, tokens * splits)](
                        self.sketch_codes,
                        self.sketch_statistics,
                        self.sketch_planes,
                        self.sketch_coordinates,
                        self.sketch_norms,
                        self.ids,
                        self.partial,
                        self.variances,
                        self.screen_parts,
                        self.screen_variance_parts,
                        padding,
                        self.weights,
                        self.norms,
                        self.recovery_geometry,
                        self.scores[selection],
                        self.completed,
                        self.selected,
                        h,
                        i,
                        k,
                        n,
                        has_padding,
                        p.screen_coordinates,
                        sketch_code_stride(2 * i),
                        sketch_storage_bytes(2 * i, h),
                        splits,
                        self.activation == "gelu_tanh",
                        self.recovery_geometry is not None,
                        POINT=self.selection_score == "point",
                        BX=min(
                            SCREEN_BATCH_COORDINATES,
                            next_power_of_2(p.screen_coordinates),
                        )
                        if tokens >= 2
                        else next_power_of_2(p.screen_coordinates),
                        Histogram=self.histogram,
                        BucketCounts=self.bucket_counts,
                        BucketKeys=self.bucket_keys,
                        RADIX=radix_select,
                        Counters=self.screen_counters,
                        LAST_BLOCK=last_block_reduction,
                        BS=next_power_of_2(splits),
                        PREFETCH=SCREEN_PREFETCH_BLOCKS
                        if tokens == 1
                        else SCREEN_BATCH_PREFETCH_BLOCKS,
                        PRIORITIZE=prioritize_cache,
                        # One token keeps the byte-offset loads and integer
                        # decode: for one token the batch path measured slightly
                        # slower on Qwen3 (1.3%) and Gemma (0.3%).
                        BATCH=tokens >= 2,
                        RouterIds=ids,
                        RouterWeights=weights,
                        ROUTE=screen_route,
                        Work=self.read_work,
                        ACCOUNT=self.fused_accounting and screen_route,
                        CHARGES=self._fixed_charges(False, False)
                        if self.fused_accounting and screen_route
                        else (),
                        R=rank,
                        F=p.final_count,
                        num_warps=SCREEN_WARPS,  # pyright: ignore[reportCallIssue]
                        maxnreg=SCREEN_BATCH_REGISTERS if tokens >= 2 else None,  # pyright: ignore[reportCallIssue]
                    )
            if step.input_end == h and grouped:
                if not grouped_screen:
                    self.aligned.wait(main_stream)
                if count < n:
                    # The screen's selection is also the set computed exactly.
                    self._union(
                        self.selected,
                        0,
                        tokens,
                        group_blocks,
                    )
                _group_complete[(group_blocks, cdiv(i, 64))](
                    self.native_gu,
                    hidden,
                    *group_rows,
                    self.unions[0] if count < n else None,
                    self.union_counts[0] if count < n else None,
                    self.selected,
                    self.partial,
                    tokens * k,
                    h,
                    i,
                    k,
                    n,
                    experts,
                    GROUP_ROWS,
                    count == n,
                    BH=128,
                    num_stages=3,  # pyright: ignore[reportCallIssue]
                )
            # Only per-token exact computation (not grouped) is matrix-shaped;
            # every other stage is pointwise per channel. Small per-token batches
            # spread rows over more programs and use fewer input-loop iterations.
            completion = step.input_end == h and not grouped
            rows = (
                (1 if tokens == 1 else 2 if tokens <= 4 else 16) if completion else 256
            )
            if step.input_end == 0 and splits > 1:
                rows = (
                    SCREEN_REDUCTION_CHANNELS
                    if tokens == 1
                    else (256 if grouped_screen else 128)
                )
            # Unsplit and last-arrival screens score directly. Other split paths
            # use this parallel reduction and scoring launch.
            if step.input_end != 0 or (splits > 1 and not last_block_reduction):
                _acquire[(cdiv(count, rows), tokens)](
                    self.native_gu,
                    hidden,
                    self.ids,
                    self.weights,
                    candidate_ids,
                    self.partial,
                    self.variances,
                    self.activations,
                    self.completed,
                    self.selected,
                    self.norms,
                    self.recovery_geometry,
                    self.scores[selection] if prune else None,
                    padding,
                    h,
                    i,
                    k,
                    n,
                    count,
                    step.input_start,
                    step.input_end,
                    has_padding,
                    prune,
                    p.final_count == n,
                    self.activation == "gelu_tanh",
                    grouped,
                    BR=rows,
                    BH=(min(2048, next_power_of_2(h)) if tokens == 1 else 1024)
                    if completion and tokens <= 4
                    else 128,
                    GEOMETRY=self.recovery_geometry is not None,
                    POINT=self.selection_score == "point",
                    ScreenParts=self.screen_parts,
                    ScreenVarianceParts=self.screen_variance_parts,
                    SCREEN_SPLITS=splits if step.input_end == 0 else 1,
                    BS=next_power_of_2(splits),
                    Histogram=self.histogram,
                    BucketCounts=self.bucket_counts,
                    BucketKeys=self.bucket_keys,
                    RADIX=radix_select and step.input_end == 0,
                    PDL=pdl and completion,
                    PREFETCH=pdl and completion and tokens == 1,
                    BK=next_power_of_2(k),
                    X_WIDTH=next_power_of_2(max(h, 1024)),
                    launch_pdl=pdl and completion,  # pyright: ignore[reportCallIssue]
                    PRIORITIZE=prioritize_cache,
                )
            # Recovery encodes the skipped channels, so the selection precedes
            # the recovery fork.
            if prune:
                if self.expert_map is not None:
                    # Scores are in global expert/channel order on every rank.
                    # Exchange only the screen scores, never expert weights.
                    dist.all_reduce(
                        self.scores[selection][:tokens],
                        group=self.ep_group,
                    )
                partition_select = (
                    count == n
                    and keep == p.final_count
                    and n >= 1024
                    and tokens <= 8
                    and self.selection_threshold is None
                )
                _select.run(
                    self.scores[selection],
                    candidate_ids,
                    self.candidate_ids[selection + 1],
                    self.selected,
                    self.starts,
                    self.counts,
                    self.tile_starts,
                    count,
                    keep,
                    n,
                    i,
                    k,
                    keep == p.final_count,
                    next_power_of_2(count),
                    128,
                    BK=next_power_of_2(k),
                    BI=next_power_of_2(i),
                    Histogram=self.histogram,
                    BucketCounts=self.bucket_counts,
                    BucketKeys=self.bucket_keys,
                    RADIX=radix_select,
                    Threshold=self.selection_threshold,
                    THRESHOLD=self.selection_threshold is not None,
                    PARTITION=partition_select,
                    grid=(tokens, k if partition_select else 1),
                    warmup=False,
                    num_warps=4 if partition_select else 8,
                )
                selection += 1
                candidate_ids = self.candidate_ids[selection]
            # The screen's selection is final: skipped channels keep their
            # sketch projections and are never computed exactly, so recovery
            # forks before exact computation, which writes only selected rows.
            # Without a recovery map nothing forks.
            if self.encoder is not None and (
                (keep == p.final_count < count) or (p.final_count == n == count)
            ):
                recovery_last_block_reduction = (
                    self.recovery_last_block_reduction
                    and not grouped
                    and tokens <= self.recovery_counters.numel()
                )
                self.recovery_ready.record(main_stream)
                with torch.cuda.stream(self.recovery_stream):
                    self.recovery_ready.wait(self.recovery_stream)
                    if grouped:
                        _group_encode[(group_blocks, self.encoder_splits)](
                            self.encoder,
                            self.partial,
                            self.activations,
                            self.completed,
                            self.selected,
                            *group_rows,
                            self.encoder_parts,
                            tokens * k,
                            i,
                            k,
                            n,
                            rank,
                            experts,
                            self.encoder_splits,
                            GROUP_ROWS,
                            max(16, next_power_of_2(rank)),
                            self.activation == "gelu_tanh",
                        )
                    else:
                        _encode[(self.encoder_splits, tokens * k)](
                            self.encoder,
                            self.partial,
                            self.activations,
                            self.completed,
                            self.selected,
                            self.ids,
                            self.encoder_parts,
                            padding,
                            i,
                            k,
                            n,
                            rank,
                            self.encoder_splits,
                            next_power_of_2(rank),
                            has_padding,
                            self.activation == "gelu_tanh",
                            Weights=self.weights,
                            Latent=self.latent,
                            Counters=self.recovery_counters,
                            LAST_BLOCK=recovery_last_block_reduction,
                            BS=next_power_of_2(self.encoder_splits),
                            BK=next_power_of_2(k),
                            num_warps=4,  # pyright: ignore[reportCallIssue]
                        )
                    if not recovery_last_block_reduction:
                        _latent[(tokens,)](
                            self.encoder_parts,
                            self.weights,
                            self.latent,
                            k,
                            rank,
                            self.encoder_splits,
                            next_power_of_2(self.encoder_splits),
                            next_power_of_2(rank),
                            self.ids,
                            padding,
                            has_padding,
                            grouped,
                        )
                    if self.fused_accounting:
                        self.recovery_done.record(self.recovery_stream)
        # Selection-dependent read statistics are added beside the down
        # projection. With fused accounting, `_prepare` (or the routing screen)
        # and `_group_down` below add them instead.
        if not self.fused_accounting:
            self.statistics_ready.record(main_stream)
            with torch.cuda.stream(self.recovery_stream):
                self.statistics_ready.wait(self.recovery_stream)
                self._charge(tokens, padding, grouped, grouped_screen, group_blocks)
                self.recovery_done.record(self.recovery_stream)
        output_tile = 32 if tokens == 1 else 64
        # The grouped down projection writes each pair's whole channel sum in one part.
        down_splits = 1 if grouped else self.down_splits
        if grouped:
            _group_down[(group_blocks, cdiv(h, 128))](
                self.down,
                self.activations,
                self.selected,
                *group_rows,
                self.unions[-1] if p.final_count < n else None,
                self.union_counts[-1] if p.final_count < n else None,
                self.down_parts,
                self.read_work,
                tokens * k,
                h,
                i,
                k,
                n,
                experts,
                down_splits,
                GROUP_ROWS,
                p.final_count == n,
                ACCOUNT=self.fused_accounting,
                R=rank,
                ENCODER_TILES=self.encoder_splits * max(16, next_power_of_2(rank))
                if rank
                else 0,
                SCREEN_MACS=cdiv(i, 80) * 256 * h * 3
                if grouped_screen and screen
                else 0,
            )
        else:
            _down[
                (
                    cdiv(h, output_tile),
                    cdiv(p.final_count, 128) + k - 1,
                    tokens,
                )
            ](
                self.down,
                self.activations,
                self.ids,
                self.candidate_ids[-1],
                self.starts,
                self.counts,
                self.tile_starts,
                self.down_parts,
                padding,
                h,
                i,
                k,
                n,
                p.final_count,
                self.down_splits,
                next_power_of_2(k),
                has_padding,
                BO=output_tile,
                PDL=pdl,
                launch_pdl=pdl,  # pyright: ignore[reportCallIssue]
                PRIORITIZE=prioritize_cache,
                num_warps=1,  # pyright: ignore[reportCallIssue]
            )
        # Recovery reads only the skipped channels' screen state, so the down
        # projection runs alongside it.
        if self.encoder is not None or not self.fused_accounting:
            self.recovery_done.wait(main_stream)
        recovery = self.decoder is not None
        if self.ep_group is not None and recovery:
            dist.all_reduce(self.latent[:tokens], group=self.ep_group)
        # A single token's output spreads over more programs, which is faster.
        output_width = 32 if tokens == 1 else 128
        _output[(cdiv(h, output_width), tokens)](
            self.down_parts,
            self.counts,
            self.weights,
            self.latent,
            self.decoder,
            out,
            padding,
            h,
            k,
            rank,
            down_splits,
            next_power_of_2(down_splits),
            next_power_of_2(rank) if recovery else 1,
            has_padding,
            recovery and self.recovery_owner,
            self.ids,
            grouped,
            BO=output_width,
            PDL=pdl,
            launch_pdl=pdl,  # pyright: ignore[reportCallIssue]
            num_warps=1,  # pyright: ignore[reportCallIssue]
            PRIORITIZE=prioritize_cache,
        )
        return out

    def _threshold_matrix_macs(
        self, native: float, reference: float, tokens: float
    ) -> float:
        """Return this rank's realized MACs per active token when thresholded.

        Each local candidate slot screens 2 H I; each selected channel (native
        bytes / 6 H) costs 3 H; recovery encodes the skipped local channels and
        the owner decodes R x H. Rank values add across EP ranks.
        """
        if not tokens:
            return 0.0
        p = self.plan
        h, i = p.hidden_size, p.expert_width
        slots = reference / (p.native_read_bytes() // p.top_k)
        kept = native / (2 * 3 * h)
        local = 2 * h * i * slots + 3 * h * kept + self.rank * (i * slots - kept)
        return local / tokens + (self.rank * h if self.recovery_owner else 0)

    def _fixed_charges(self, grouped: bool, grouped_screen: bool) -> tuple[int, ...]:
        """Return the per-token read statistics that do not depend on the selection.

        Valid without EP or a threshold, when every token selects the same number
        of channels. The tuple follows `read_work` slots 0 to 17.
        """
        p = self.plan
        native = p.selected_weight_bytes()
        recovery = p.recovery_read_bytes(self.rank)
        selection = p.selection_read_bytes(self.recovery_geometry is not None)
        residual = self.selection_score != "point"
        sketch = p.sketch_read_bytes(residual)
        h, unit = p.hidden_size, 2
        return (
            native,
            p.native_read_bytes(),
            0,
            recovery,
            selection,
            p.sketch_read_bytes(residual, sectors=False),
            native,
            recovery,
            0,
            selection,
            h if p.completion_count < p.candidate_count else 0,
            0 if grouped_screen else sketch,
            0 if grouped else unit * 2 * h * p.completion_count,
            0 if grouped else unit * h * p.final_count,
            0,
            0,
            int(grouped),
            sketch,
        )

    def _union(self, member: Tensor, index: int, tokens: int, blocks: int) -> None:
        """Compact each expert block's union of member channels into unions[index]."""
        p = self.plan
        _group_union[(blocks,)](
            member,
            self.pairs,
            self.block_experts,
            self.padded_pairs,
            self.unions[index],
            self.union_counts[index],
            tokens * p.top_k,
            p.expert_width,
            p.top_k,
            p.candidate_count,
            self.native_gu.shape[0],
            GROUP_ROWS,
        )

    def _charge(
        self,
        tokens: int,
        padding: Tensor | None,
        grouped: bool,
        grouped_screen: bool,
        blocks: int,
    ) -> None:
        """Add this call's read statistics in one kernel, after selection."""
        p = self.plan
        h, i, k, n, rank = (
            p.hidden_size,
            p.expert_width,
            p.top_k,
            p.candidate_count,
            self.rank,
        )
        # Without a recovery map (rank 0) no B row is read, shared or not.
        shared_b = 0 < rank < 16 and 16 % rank == 0 and i % (16 // rank) == 0
        geometry = self.recovery_geometry is not None
        # Point scores do not load the separate residual plane.
        residual = self.selection_score != "point"
        ep = self.expert_map is not None
        token_rows = 1 if tokens <= 256 else 16
        token_programs = cdiv(tokens, token_rows)
        _count_reads[(token_programs + (cdiv(blocks, 256) if grouped else 0),)](
            self.ids,
            padding,
            self.completed,
            self.selected,
            self.block_experts,
            self.padded_pairs,
            self.union_counts,
            self.read_work,
            tokens,
            blocks,
            (self.unions.shape[0] - 1) * self.union_counts.shape[1] * 2,
            h,
            i,
            k,
            n,
            p.completion_count,
            p.final_count,
            rank,
            self.native_gu.shape[0],
            p.selected_weight_bytes(),
            p.native_read_bytes() // k if ep else p.native_read_bytes(),
            p.recovery_read_bytes(rank),
            p.selection_read_bytes(geometry),
            p.sketch_read_bytes(residual, sectors=False),
            2,
            ep,
            self.recovery_owner,
            padding is not None,
            grouped,
            grouped_screen,
            shared_b,
            self.encoder_splits * max(16, next_power_of_2(rank)) if rank else 0,
            token_rows,
            next_power_of_2(k),
            min(max(16, next_power_of_2(i)), 1024),
            SCREEN_BYTES=12 if geometry else 4,
            ISSUED_SKETCH=p.sketch_read_bytes(residual),
            PHYSICAL_SKETCH=p.sketch_read_bytes(residual),
            THRESHOLD=self.selection_threshold is not None,
        )
