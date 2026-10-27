"""Compile read budgets into `BudgetPlan`s and price their reads per token.

Symbols follow the reader: H hidden size, I intermediate channels per expert, K
top-k experts per token, N = K * I candidate channels, F selected channels.
"""

import json
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path


def sketch_code_stride(rows: int) -> int:
    """Return the code bytes per coordinate of one expert's sketch.

    Each byte pair holds five gate and five up channels (one trit each); the
    stride pads the pairs to a multiple of 16 bytes.
    """
    return (2 * ((rows // 2 + 4) // 5) + 15) // 16 * 16


def sketch_storage_bytes(rows: int, hidden: int) -> int:
    """Return the packed code bytes of one expert's sketch.

    Full 32-byte tiles of each coordinate are stored coordinate-major; a
    16-byte tail shares its sector with the same coordinate of the adjacent
    rotation block (see `build_sketch`).
    """
    stride = sketch_code_stride(rows)
    blocks = hidden // 128
    return (
        blocks * 128 * (stride // 32 * 32) + (blocks + 1) // 2 * 128 * (stride % 32) * 2
    )


@dataclass(frozen=True)
class PaidRound:
    """One stage of a plan: read inputs [input_start, input_end), then keep the best.

    entering_count channels enter and the best retained_count leave. A stage with
    input_end == 0 is the screen, which reads no native rows; a stage with
    input_end == hidden_size computes its channels exactly from their native gate
    and up rows. The last stage keeps final_count (F) channels.
    """

    input_start: int
    input_end: int
    entering_count: int
    retained_count: int


@dataclass(frozen=True)
class BudgetPlan:
    """Token-independent read plan of one MoE layer.

    Fixes how many channels each token computes exactly (completion_count) and
    selects (final_count, F), and the stages (`rounds`) that pick them; which
    channels are picked depends on the token.
    """

    hidden_size: int
    expert_width: int
    top_k: int
    completion_count: int
    final_count: int
    rounds: tuple[PaidRound, ...]
    # Rotated coordinates the screen keeps per 128-coordinate block: 128 for
    # the full screen, fewer for the partial screen.
    screen_coordinates: int = 128

    def __post_init__(self) -> None:
        # Each 80-channel CTA must begin and end its FP16 metadata reads on
        # sector boundaries, including the last CTA and the up-plane offset.
        if self.expert_width % 16:
            raise ValueError(
                "Reader expert width must be divisible by 16 "
                "for sector-aligned sketch metadata"
            )

    @property
    def candidate_count(self) -> int:
        """Return the candidate channels per token, N = K * I.

        These are all intermediate channels of the token's naturally routed
        experts.
        """
        return self.top_k * self.expert_width

    def native_read_bytes(self) -> int:
        """Return the dense routed-expert BF16 weight bytes per token."""
        return 2 * 3 * self.candidate_count * self.hidden_size

    def selected_weight_bytes(self) -> int:
        """Return the native weight bytes per token the plan reads.

        These are the BF16 gate and up rows of the completion_count channels
        computed exactly and the down columns of the F selected channels,
        excluding the sketch.
        """
        return 2 * self.hidden_size * (2 * self.completion_count + self.final_count)

    def sketch_read_bytes(self, residual: bool = True, sectors: bool = True) -> int:
        """Return the sketch bytes one token's screen reads.

        These are the packed trits, FP16 scales and, with residual (for scores
        that read them), FP16 residual variances of every candidate row, in the
        single resident layout. sectors=False prices exact bytes instead of
        32-byte sectors.

        Each kept coordinate has 2*ceil(I/5) useful code bytes in 32-byte tiles;
        vector-load padding stays inside those sectors. A short final tile shares
        its sector with the next rotation block, not another coordinate in the
        same load, so each tile reads one sector. Scale and residual planes are
        separate contiguous FP16 runs per block; the required 16-channel width
        alignment keeps CTAs from sharing a sector. This is the batch-one price;
        the expert-major screen's requested bytes use the coordinate union.
        """
        if self.completion_count == self.candidate_count:
            return 0
        blocks = self.hidden_size // 128
        rows = 2 * self.expert_width

        def run(size: int) -> int:
            return -(-size // 32) * 32 if sectors else size

        codes = self.screen_coordinates * run(2 * ((self.expert_width + 4) // 5))
        return self.top_k * blocks * (codes + run(2 * rows) * (1 + residual))

    def recovery_read_bytes(self, rank: int) -> int:
        """Return the recovery-map (`bank`) bytes per token.

        These are the BF16 encoder rows B of the N - F skipped channels and the
        BF16 decoder U read by the output kernel; 0 without a map (rank 0).
        """
        return 2 * rank * (self.candidate_count - self.final_count + self.hidden_size)

    def selection_read_bytes(self, geometry: bool = True) -> int:
        """Return the screening metadata bytes per token.

        Each candidate's FP32 down-column norm and, with score geometry, its two
        FP32 recovery geometry values. Without score geometry (no recovery map, or the
        point score) only the norm is read; full retention reads none.
        """
        if self.completion_count < self.candidate_count:
            # FP32 norm and, with score geometry, two FP32 values per channel.
            return (12 if geometry else 4) * self.candidate_count
        return 0


def compile_budget(
    *,
    hidden_size: int,
    expert_width: int,
    top_k: int,
    retention: float,
    screen_keep: float = 1.0,
) -> BudgetPlan:
    """Compile a retention fraction into a token-independent read plan.

    The plan keeps F of the N = top_k * expert_width candidate channels, F being
    retention * N rounded half to even. Fractions must be finite and in (0, 1],
    and the rounded count must satisfy 1 <= F <= N. The resident ternary sketch
    screens all candidates, then the F selected channels are computed exactly
    from their native rows (completion_count equals F, odd counts included).
    Full retention (F = N) skips the screen, which would prune nothing.
    Expert widths must be divisible by 16 for sector-aligned sketch metadata.
    Invalid dimensions or budgets raise ValueError before expert execution.
    screen_keep < 1 (the partial screen) reads only each token's top
    ceil(128 * screen_keep) |R x| coordinates of every rotation block.
    """
    for name, value in (
        ("hidden_size", hidden_size),
        ("expert_width", expert_width),
        ("top_k", top_k),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, fraction in (
        ("retention", retention),
        ("screen_keep", screen_keep),
    ):
        if (
            not isinstance(fraction, (int, float))
            or not isfinite(fraction)
            or not 0 < fraction <= 1
        ):
            raise ValueError(f"{name} must be finite and in (0, 1]")

    n = top_k * expert_width
    f = round(retention * n)
    q = f
    if not 1 <= f <= n:
        raise ValueError("Rounded budgets must satisfy 1 <= F <= N")
    if hidden_size % 128:
        raise ValueError("Sketch hidden width must be divisible by 128")
    kept = ceil(screen_keep * 128)
    if kept < 128 and q == n:
        raise ValueError(
            "A partial-sketch screen needs fewer selected than candidate channels"
        )
    rounds = (
        (PaidRound(0, hidden_size, n, f),)
        if q == n
        else (PaidRound(0, 0, n, q), PaidRound(0, hidden_size, q, f))
    )
    return BudgetPlan(hidden_size, expert_width, top_k, q, f, rounds, kept)


def load_layer_budgets(
    path: str | Path,
    *,
    layers: int,
    hidden_size: int,
    expert_width: int,
    top_k: int,
    screen_keep: float,
) -> list[BudgetPlan]:
    """Load per-layer final counts from an allocation JSON and compile their plans.

    The file lists one count per MoE layer in decoder-layer order under
    "final_counts"; optional "dimensions" and "screen_keep" entries must match
    the model and reader. Every layer is screened (F < N) and computes exactly the
    channels it selects.
    """
    allocation = json.loads(Path(path).read_text())
    counts = allocation["final_counts"]
    if not isinstance(counts, list) or len(counts) != layers:
        raise ValueError("Layer allocation must give one count per MoE layer")
    dimensions = allocation.get("dimensions")
    if dimensions is not None and any(
        dimensions.get(name) != value
        for name, value in (
            ("num_hidden_layers", layers),
            ("hidden_size", hidden_size),
            ("expert_width", expert_width),
            ("num_experts_per_tok", top_k),
        )
    ):
        raise ValueError("Layer allocation dimensions do not match the model")
    if "screen_keep" in allocation and ceil(128 * allocation["screen_keep"]) != ceil(
        128 * screen_keep
    ):
        raise ValueError("Layer allocation screen_keep does not match the reader")
    n = top_k * expert_width
    if any(type(f) is not int or not 1 <= f < n for f in counts):
        raise ValueError("Layer counts must be integers in [1, candidate_count)")
    return [
        compile_budget(
            hidden_size=hidden_size,
            expert_width=expert_width,
            top_k=top_k,
            retention=f / n,
            screen_keep=screen_keep,
        )
        for f in counts
    ]
