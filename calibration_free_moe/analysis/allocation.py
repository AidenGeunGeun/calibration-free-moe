"""Per-token, per-layer channel counts for Prox's joint channel selection.

Used by the Section 4 study. With ``allocation``, policy ``prox`` keeps each token's
top n_{t,l} channels jointly over its K routed slots, ranked by Prox's INT4 proxy
score (or, for analysis only, the exact output contribution,
``proxy_score="exact"``). The count n_{t,l} comes from one source:

- ``threshold``: what the calibrated layer threshold keeps.
- ``uniform``: a per-layer mean k_l (``scale`` times k, or ``allocation_counts`` from
  ``scripts/allocation_study.py layer-counts``).
- ``shuffled``: a recorded threshold run's per-token count vectors, permuted across the
  tokens of each panel.
- ``schedule``: k_l times a piecewise-constant function of the answer depth.

Real-valued targets are rounded stochastically, n = floor(target + u), with u hashed
from (request, depth, layer), so every arm of one request draws the same u. Counts
are fixed before the selected channels are read (the exact-score control reads every
candidate's native gate and up rows first). Ties keep the lowest (slot, channel)
index.

Answer depth is the position minus the request's answer start. The worker writes depth,
request slot and sampled step into fixed buffers before each forward, so CUDA graphs can
read them. With ``allocation_recording`` the first tensor-parallel rank stores, per
request and sampled step, the counts, score features and read cost of the row that
produced that step. ``uniform`` and ``schedule`` arms are matched to a threshold arm by
per-layer read cost, not by count, because Prox's read cost also depends on each
token's kept input coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample.sampler import Sampler

AllocationSource = Literal["threshold", "uniform", "shuffled", "schedule"]
SOURCES: tuple[str, ...] = get_args(AllocationSource)
# Per-token, per-layer statistics of the ranked scores s (all K*I candidates):
# log sum, log max, share of the sum in the top-k (the uniform count at scale 1),
# entropy of s / sum s over log(K*I), and the share at or above the layer threshold.
FEATURES = ("log_mass", "log_max", "top_mass", "entropy", "above_threshold")
KEY = "tokenwise_allocation_key"
OFFSET = "tokenwise_answer_offset"
RECORD_FORMAT = "allocation-record-v1"
# log2-score histogram for fitting exact-score thresholds (about 2.2% per bin).
HISTOGRAM_BINS = 2048
HISTOGRAM_LOG2 = (-48.0, 16.0)
MASK32 = 0xFFFFFFFF


def request_key(panel: Path, document_id: str) -> str:
    """Return a run-independent request key: the resolved panel path and document ID."""
    return json.dumps([str(panel.resolve()), document_id])


def mix32(x: Any) -> Any:
    """Hash ints or int64 tensors to 32 bits with two xor-shift-multiply rounds.

    Products stay below 2^59, so int64 tensors never overflow.
    """
    for _ in range(2):
        x = ((x >> 16) ^ x) * 0x45D9F3B & MASK32
    return (x >> 16) ^ x


def rounding_seed(key: object) -> int:
    """Return the request's 32-bit rounding seed; keyless one-token controls use 0."""
    if not isinstance(key, str):
        return 0
    digest = hashlib.blake2b(f"rounding/{key}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def count_mask(score: Tensor, counts: Tensor) -> Tensor:
    """Mask each row's ``counts`` largest entries; ties keep the lowest index.

    The boundary is the counts-th largest value (``topk(k).amin`` in
    ``prox_top_mask``) and the tie scan is that function's study rule, so a constant
    count gives its mask exactly. A zero count keeps nothing; entries below every
    kept score (inactive slots at -1) are never kept.
    """
    ordered = score.sort(dim=-1, descending=True).values
    index = (counts - 1).clamp(0, score.shape[-1] - 1)[:, None]
    boundary = ordered.gather(-1, index)
    above = score > boundary
    tied = score == boundary
    remaining = counts[:, None] - above.sum(-1, keepdim=True)
    keep = above | (tied & (tied.cumsum(-1) <= remaining))
    return keep & (counts[:, None] > 0)


def integer_counts(
    target: Tensor, seeds: Tensor, depth: Tensor, layer: int, limit: int
) -> Tensor:
    """Convert targets to integers floor(target + u), clamped to [0, limit].

    u is uniform in [0, 1), hashed from (seed, depth, layer).
    """
    salt = mix32((layer * 0x9E3779B9 + 1) & MASK32)
    hashed = mix32(seeds ^ mix32((depth.clamp_min(0) & MASK32) ^ salt))
    uniform = hashed.double() / 2.0**32
    return torch.floor(target.double() + uniform).clamp(0, limit).long()


def score_features(score: Tensor, top_count: int, threshold: float | None) -> Tensor:
    """Compute FEATURES of nonnegative joint scores [rows, N] (inactive at zero)."""
    mass = score.sum(-1)
    safe = mass.clamp_min(1e-30)
    share = score / safe[:, None]
    entropy = -(share * share.clamp_min(1e-30).log()).sum(-1) / math.log(
        score.shape[-1]
    )
    top = score.topk(min(top_count, score.shape[-1]), dim=-1, sorted=False).values
    above = (
        (score >= threshold).float().mean(-1)
        if threshold is not None
        else torch.full_like(mass, float("nan"))
    )
    return torch.stack(
        (
            safe.log(),
            score.amax(-1).clamp_min(1e-30).log(),
            top.sum(-1) / safe,
            entropy,
            above,
        ),
        dim=-1,
    )


@dataclass
class Schedule:
    """Depth-bin multipliers of k; ``edges`` are increasing lower edges from 0."""

    edges: Tensor
    multipliers: Tensor

    @property
    def parameter_bytes(self) -> int:
        """Return the bytes of the edges and multipliers, read once per call."""
        return sum(t.numel() * t.element_size() for t in (self.edges, self.multipliers))

    def __call__(self, depth: Tensor) -> Tensor:
        """Return each row's multiplier for its answer depth (negative depth as 0)."""
        bins = torch.searchsorted(self.edges, depth.clamp_min(0).double(), right=True)
        return self.multipliers[bins - 1]


def load_schedule(path: Path, device: torch.device) -> Schedule:
    """Load a depth schedule JSON and normalize it to mean multiplier one.

    The mean is taken under the schedule's own depth-bin ``weights``.
    """
    spec = cast(dict[str, Any], json.loads(path.read_text()))
    edges = torch.tensor(spec["edges"], dtype=torch.float64)
    multipliers = torch.tensor(spec["multipliers"], dtype=torch.float64)
    weights = torch.tensor(spec["weights"], dtype=torch.float64)
    if (
        spec.get("format") != "allocation-schedule-v1"
        or edges.ndim != 1
        or edges.shape != multipliers.shape
        or edges.shape != weights.shape
        or edges[0] != 0
        or not bool((edges[1:] > edges[:-1]).all())
        or not bool((multipliers > 0).all())
        or not bool((weights >= 0).all())
        or weights.sum() <= 0
    ):
        raise ValueError(f"{path} is not an allocation schedule")
    # Mean multiplier one under the schedule's own depth distribution.
    multipliers = multipliers / ((weights * multipliers).sum() / weights.sum())
    return Schedule(edges.to(device), multipliers.to(device))


def load_records(directory: Path) -> dict[str, dict[str, Tensor]]:
    """Load every recorded request of a record directory, keyed by request key.

    Each record holds ``steps`` and ``depth`` [S], ``counts`` and ``charged_bytes``
    [S, L] and ``features`` [S, L, F] for S sampled steps and L layers.
    """
    records: dict[str, dict[str, Tensor]] = {}
    for path in sorted(directory.glob("records-*.pt")):
        saved = cast(dict[str, Any], torch.load(path, weights_only=True))
        if saved["format"] != RECORD_FORMAT:
            raise ValueError(f"{path} is not an allocation record")
        for key, record in cast(dict[str, dict[str, Tensor]], saved["records"]).items():
            if key in records:
                raise ValueError(f"Request {key} is recorded twice in {directory}")
            records[key] = record
    if not records:
        raise ValueError(f"No allocation records in {directory}")
    return records


def load_channel_bytes(directory: Path) -> Tensor:
    """Load each layer's native bytes per kept channel, as recorded with the counts."""
    path = min(directory.glob("records-*.pt"))
    saved = cast(dict[str, Any], torch.load(path, weights_only=True))
    channel_bytes = cast(Tensor, saved["channel_bytes"]).double()
    if not bool((channel_bytes > 0).all()):
        raise ValueError(f"{directory} records no native bytes per channel")
    return channel_bytes


def shuffled_counts(
    records: dict[str, dict[str, Tensor]], seed: int
) -> dict[str, tuple[Tensor, Tensor]]:
    """Permute whole per-token count vectors across the tokens of each panel."""
    panels: dict[str, list[str]] = defaultdict(list)
    for key in sorted(records):
        panels[cast(str, json.loads(key)[0])].append(key)
    out: dict[str, tuple[Tensor, Tensor]] = {}
    for panel, keys in sorted(panels.items()):
        counts = torch.cat([records[key]["counts"] for key in keys])
        generator = torch.Generator().manual_seed(seed)
        permuted = counts[torch.randperm(len(counts), generator=generator)]
        start = 0
        for key in keys:
            steps = records[key]["steps"]
            out[key] = (steps, permuted[start : start + len(steps)])
            start += len(steps)
        del panel
    return out


class TokenContext:
    """Fixed per-row and per-request buffers shared by every layer's allocation."""

    def __init__(
        self,
        source: AllocationSource,
        layers: int,
        max_tokens: int,
        max_requests: int,
        device: torch.device,
        *,
        records: Path | None = None,
        seed: int = 0,
        histogram: bool = False,
        recording: bool = False,
    ) -> None:
        self.source: AllocationSource = source
        self.layers, self.seed, self.device = layers, seed, device
        self.recording = recording
        self.started = False

        def rows(*shape: int, dtype: torch.dtype = torch.int64) -> Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        self.depth, self.step, self.slot = (
            rows(max_tokens),
            rows(max_tokens),
            rows(max_tokens),
        )
        self.answer_start = rows(max_requests)
        self.seeds = rows(max_requests)
        self.counts = rows(max_tokens, layers, dtype=torch.int32)
        # Recording only: instrumentation rows of features and charged bytes.
        recorded_rows = max_tokens if recording else 1
        self.features = rows(recorded_rows, layers, len(FEATURES), dtype=torch.float32)
        self.charged = rows(recorded_rows, layers, dtype=torch.float64)
        # Native bytes per kept channel of each layer (set by its reader).
        self.channel_bytes: list[float] = [0.0] * layers
        self.histograms = (
            rows(layers, HISTOGRAM_BINS, dtype=torch.float64) if histogram else None
        )
        self.histogram_tokens = rows(layers, dtype=torch.float64) if histogram else None
        self.shuffled: dict[str, tuple[Tensor, Tensor]] = {}
        self.table = rows(1, 1, layers, dtype=torch.int32)
        if source == "shuffled":
            if records is None:
                raise ValueError(f"The {source} allocation needs a record directory")
            recorded = load_records(records)
            if any(r["counts"].shape[-1] != layers for r in recorded.values()):
                raise ValueError("Recorded counts have another layer count")
            self.shuffled = shuffled_counts(recorded, seed)
            steps = max(int(r["steps"].max()) for r in recorded.values()) + 1
            self.table = rows(max_requests, steps, layers, dtype=torch.int32)

    def start_request(
        self, slot: int, answer_start: int, key: object, max_tokens: int | None
    ) -> None:
        """Bind a new request's answer start, rounding stream and recorded counts.

        A one-token request (an unforced control) never reaches sparse decode.
        """
        self.answer_start[slot] = answer_start
        self.seeds[slot] = rounding_seed(key)
        if max_tokens == 1:
            return
        if not isinstance(key, str):
            raise TypeError("Allocated requests need request keys")
        if self.source != "shuffled":
            return
        if key not in self.shuffled:
            raise ValueError(f"No recorded counts for request {key}")
        steps, counts = self.shuffled[key]
        self.table[slot].zero_()
        self.table[slot, steps.to(self.device)] = counts.to(self.device, torch.int32)

    def observe_batch(self, batch: InputBatch, prompt_len: Tensor) -> None:
        """Store the answer depth, request slot and sampled step of every real row."""
        tokens = batch.num_tokens
        requests = np.repeat(
            np.arange(batch.num_reqs), batch.num_scheduled_tokens[: batch.num_reqs]
        )
        if requests.size != tokens:
            raise RuntimeError("Allocation needs one scheduled row per forward token")
        slots = torch.from_numpy(batch.idx_mapping_np[requests].astype(np.int64)).to(
            self.device, non_blocking=True
        )
        positions = batch.positions[:tokens]
        self.slot[:tokens].copy_(slots)
        self.depth[:tokens].copy_(positions - self.answer_start[slots])
        self.step[:tokens].copy_(positions + 1 - prompt_len[slots])


class LayerAllocation:
    """One layer's count rule over the joint [rows, K*I] score of a Prox call."""

    def __init__(
        self,
        context: TokenContext,
        layer: int,
        candidates: int,
        top_count: int,
        target: float,
        threshold: float | None,
        schedule: Schedule | None = None,
    ) -> None:
        source = context.source
        if (
            (source == "threshold" and threshold is None)
            or (source == "schedule") != (schedule is not None)
            or not 0 <= target <= candidates
            or not 1 <= top_count <= candidates
        ):
            raise ValueError(f"Invalid {source} allocation for layer {layer}")
        self.context, self.layer, self.candidates = context, layer, candidates
        self.top_count, self.target, self.threshold = top_count, target, threshold
        self.schedule = schedule
        # Parameter bytes read per call, added to the read cost.
        self.parameter_bytes: int = (
            schedule.parameter_bytes if schedule is not None else 0
        )

    @property
    def target_count(self) -> float | None:
        """Return the configured mean count k_l (None for threshold and shuffled)."""
        return self.target if self.context.source in {"uniform", "schedule"} else None

    @property
    def overhead(self) -> str:
        """Describe the source's read cost beyond the ranked scores."""
        return {
            "threshold": "none beyond the threshold comparisons",
            "uniform": "none beyond the ranked scores",
            "schedule": "charged: schedule parameter reads per call",
            "shuffled": "offline analysis control: recorded-count table, not charged",
        }[self.context.source]

    def call_parameter_bytes(self, valid: Tensor) -> Tensor:
        """Return this call's parameter bytes, charged once if it has tokens."""
        return (valid.sum() > 0) * self.parameter_bytes

    def token_bytes(self, charged: Tensor, valid: Tensor) -> Tensor:
        """Add each valid token's share of the call's parameter bytes.

        The tokens of a call then sum to the rank's read cost for the call.
        """
        share = self.call_parameter_bytes(valid).double() / valid.sum().clamp_min(1)
        return charged.double() + torch.where(valid, share, 0.0)

    def record_bytes(self, tokens: slice | Tensor, charged: Tensor) -> None:
        """Store each token's charged bytes in this layer (recording only)."""
        if self.context.recording:
            self.context.charged[tokens, self.layer] = charged

    def __call__(self, ranked: Tensor, valid: Tensor, tokens: slice | Tensor) -> Tensor:
        """Return the joint keep mask over ``ranked`` ``[tokens, K*I]``.

        Inactive entries of ``ranked`` are -1 and never kept.
        """
        context = self.context
        score = ranked.clamp_min(0)
        features = (
            score_features(score, self.top_count, self.threshold)
            if context.recording
            else None
        )
        depth = context.depth[tokens]
        if context.source == "threshold":
            mask = ranked >= cast(float, self.threshold)
            counts = mask.sum(-1)
        else:
            if context.source == "shuffled":
                table = context.table
                counts = table[
                    context.slot[tokens].clamp(0, table.shape[0] - 1),
                    context.step[tokens].clamp(0, table.shape[1] - 1),
                    self.layer,
                ].long()
            else:
                target = torch.full_like(depth, self.target, dtype=torch.float64)
                if self.schedule is not None:
                    target = target * self.schedule(depth)
                seeds = context.seeds[context.slot[tokens]]
                counts = integer_counts(
                    target, seeds, depth, self.layer, self.candidates
                )
            counts = torch.where(valid, counts.clamp(0, self.candidates), 0)
            mask = count_mask(ranked, counts)
        context.counts[tokens, self.layer] = counts.int()
        if features is not None and context.recording:
            context.features[tokens, self.layer] = features
        if context.histograms is not None and context.histogram_tokens is not None:
            # Static shapes for graph capture: inactive entries add zero weight.
            low, high = HISTOGRAM_LOG2
            bins = (score.clamp_min(2.0**low).log2() - low) / (high - low)
            bins = (bins * HISTOGRAM_BINS).long().clamp(0, HISTOGRAM_BINS - 1)
            counted = ((ranked >= 0) & valid[:, None]).double()
            context.histograms[self.layer].scatter_add_(
                0, bins.flatten(), counted.flatten()
            )
            context.histogram_tokens[self.layer] += valid.sum()
        return mask


class WorkerState:
    """This worker's prepared context and installed sampler wrapper."""

    context: TokenContext | None = None
    recording: AllocationSampler | None = None


STATE = WorkerState()


def validate_allocation_options(options: dict[str, object]) -> None:
    """Validate the binding's allocation options; allocations need policy prox."""
    source = options.get("allocation")
    extra = (
        "allocation_input",
        "allocation_scale",
        "allocation_seed",
        "allocation_counts",
    )
    if options.get("allocation_recording", False) not in {True, False}:
        raise ValueError("allocation_recording is a boolean")
    if source is None:
        if (
            any(options.get(key) is not None for key in extra)
            or options.get("allocation_recording")
            or options.get("proxy_score") == "exact"
        ):
            raise ValueError("Allocation options and the exact score need allocation")
        return
    retention = options.get("retention")
    scale = options.get("allocation_scale")
    path = options.get("allocation_input")
    seed = options.get("allocation_seed")
    score = options.get("proxy_score", "prox")
    if (
        source not in SOURCES
        or options.get("policy") != "prox"
        or any(
            options.get(key) is not None
            for key in ("channel_keep", "channel_floor", "bank")
        )
        or type(retention) is not float
        or not 0 < retention <= 1
        or score not in {"prox", "exact"}
        or (score == "prox" and options.get("baseline") is None)
        or (score == "exact" and source == "threshold" and not options.get("baseline"))
    ):
        raise ValueError(
            "An allocation needs policy prox, retention (the top-k count) and the "
            "prox score with its calibration or the exact score (thresholds for its "
            "threshold source), without channel keep rules or a bank"
        )
    counts = options.get("allocation_counts")
    if (path is None and source in {"shuffled", "schedule"}) or (
        path is not None
        and (source in {"threshold", "uniform"} or type(path) is not str)
    ):
        raise ValueError(
            "shuffled takes a record directory, schedule a JSON, "
            "uniform and threshold nothing"
        )
    if counts is not None and (
        type(counts) is not str
        or source not in {"uniform", "schedule"}
        or scale is not None
    ):
        raise ValueError(
            "allocation_counts (per-layer base counts) is a JSON path of "
            "uniform/schedule, without allocation_scale"
        )
    if scale is not None and (
        type(scale) is not float or scale <= 0 or source not in {"uniform", "schedule"}
    ):
        raise ValueError("allocation_scale is a positive multiplier of the top-k count")
    if seed is not None and (type(seed) is not int or source != "shuffled"):
        raise ValueError("allocation_seed is an integer of shuffled")


def load_exact_thresholds(path: Path, layers: int, top_count: int) -> list[float]:
    """Load per-layer exact-score thresholds fitted for this top-k count."""
    spec = cast(dict[str, Any], json.loads(path.read_text()))
    thresholds = cast(list[float], spec["thresholds"])
    if (
        spec.get("format") != "allocation-thresholds-v1"
        or spec.get("score") != "exact"
        or spec.get("target_count") != top_count
        or len(thresholds) != layers
        or any(type(t) is not float or not 0 < t < math.inf for t in thresholds)
    ):
        raise ValueError(f"{path} holds no exact thresholds for count {top_count}")
    return thresholds


def prepare_allocations(
    options: dict[str, object],
    *,
    layers: int,
    candidates: int,
    max_tokens: int,
    max_requests: int,
    device: torch.device,
    thresholds: list[float] | None,
) -> list[LayerAllocation]:
    """Build the shared context and every layer's allocation (``thresholds``: Prox's).

    The top-k count k = ceil(retention K I) is ``prox --channel-keep``'s joint count.
    """
    source = cast(AllocationSource, options["allocation"])
    top_count = max(1, math.ceil(cast(float, options["retention"]) * candidates - 1e-9))
    path = (
        Path(str(options["allocation_input"]))
        if options.get("allocation_input")
        else None
    )
    if thresholds is None and options.get("baseline"):
        thresholds = load_exact_thresholds(
            Path(str(options["baseline"])), layers, top_count
        )
    targets = [
        top_count * float(cast(float, options.get("allocation_scale") or 1.0))
    ] * layers
    schedule = load_schedule(path, device) if path and source == "schedule" else None
    if options.get("allocation_counts"):
        counts = Path(str(options["allocation_counts"]))
        spec = cast(dict[str, Any], json.loads(counts.read_text()))
        targets = [float(value) for value in spec["layer_counts"]]
        if spec.get("format") != "allocation-layer-counts-v1" or len(targets) != layers:
            raise ValueError(f"{counts} holds no per-layer counts for {layers} layers")
    context = STATE.context = TokenContext(
        source,
        layers,
        max_tokens,
        max_requests,
        device,
        records=path if source == "shuffled" else None,
        seed=cast(int, options.get("allocation_seed") or 0),
        histogram=options.get("proxy_score") == "exact",
        recording=options.get("allocation_recording") is True,
    )
    return [
        LayerAllocation(
            context,
            layer,
            candidates,
            top_count,
            targets[layer],
            thresholds[layer] if thresholds is not None else None,
            schedule,
        )
        for layer in range(layers)
    ]


class AllocationSampler:
    """Delegate to the native sampler; bind requests and record allocations.

    Every rank binds answer starts and recorded counts at ``add_request``. The
    recording rank also stores, per tracked request and sampled step s >= 1, the
    counts, features, charged bytes and depth of the forward row whose logits
    produced step s.
    """

    def __init__(
        self, sampler: Sampler, context: TokenContext, record: Path | None, steps: int
    ) -> None:
        self._sampler: Sampler = sampler
        self.context: TokenContext = context
        self.record: Path | None = record
        self.batches = 0
        slots = sampler.req_states.max_num_reqs
        device = context.device
        self.keys: dict[int, str] = {}
        self.lengths = torch.zeros(slots, dtype=torch.int64, device=device)
        size = (slots, steps if record is not None else 1)
        self.seen = torch.zeros(size, dtype=torch.bool, device=device)
        self.depth = torch.zeros(size, dtype=torch.int64, device=device)
        self.counts = torch.zeros(
            (*size, context.layers), dtype=torch.int32, device=device
        )
        self.features = torch.zeros(
            (*size, context.layers, len(FEATURES)), dtype=torch.float32, device=device
        )
        self.charged = torch.zeros(
            (*size, context.layers), dtype=torch.float64, device=device
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sampler, name)

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        """Register a request natively, bind its allocation state and track it."""
        self._sampler.add_request(req_idx, prompt_len, sampling_params)
        extra = sampling_params.extra_args or {}
        key = extra.get(KEY)
        self.context.start_request(
            req_idx,
            prompt_len - int(extra.get(OFFSET, 0)),
            key,
            sampling_params.max_tokens,
        )
        if self.record is None or not isinstance(key, str):
            return
        if req_idx in self.keys:
            raise RuntimeError("A recorded request's slot was reused before collection")
        steps = sampling_params.max_tokens
        if steps is None or steps > self.seen.shape[1]:
            raise ValueError("Recorded request exceeds the configured step count")
        self.keys[req_idx] = key
        self.lengths[req_idx] = steps
        self.seen[req_idx] = False

    def __call__(self, logits: Tensor, input_batch: InputBatch) -> SamplerOutput:
        """Copy tracked rows' recorded allocation state, then sample natively."""
        rows = [
            row
            for row, slot in enumerate(input_batch.idx_mapping_np.tolist())
            if slot in self.keys
        ]
        if rows:
            index = torch.tensor(rows, device=self.context.device)
            slots = input_batch.idx_mapping[index].long()
            token_rows = input_batch.logits_indices[index].long()
            prompt = self._sampler.req_states.prompt_len.gpu[slots]
            steps = (input_batch.positions[token_rows] + 1 - prompt).long()
            valid = (steps >= 1) & (steps < self.lengths[slots])
            steps = steps.clamp(0, self.seen.shape[1] - 1)
            for buffer, value in (
                (self.counts, self.context.counts[token_rows]),
                (self.features, self.context.features[token_rows]),
                (self.charged, self.context.charged[token_rows]),
                (self.depth, self.context.depth[token_rows]),
            ):
                keep = valid.view(-1, *([1] * (value.ndim - 1)))
                buffer[slots, steps] = torch.where(keep, value, buffer[slots, steps])
            self.seen[slots, steps] |= valid
        return self._sampler(logits, input_batch)

    def collect(self) -> dict[str, int]:
        """Save every tracked request's recorded steps; return their step counts."""
        if self.record is None:
            return {}
        records: dict[str, dict[str, Tensor]] = {}
        for slot, key in sorted(self.keys.items()):
            steps = self.seen[slot].nonzero().squeeze(-1)
            records[key] = {
                "steps": steps.cpu(),
                "depth": self.depth[slot, steps].cpu(),
                "counts": self.counts[slot, steps].cpu(),
                "features": self.features[slot, steps].cpu(),
                "charged_bytes": self.charged[slot, steps].cpu(),
            }
        self.keys.clear()
        if records:
            torch.save(
                {
                    "format": RECORD_FORMAT,
                    "features": FEATURES,
                    "channel_bytes": torch.tensor(self.context.channel_bytes),
                    "records": records,
                },
                self.record / f"records-{self.batches:05d}.pt",
            )
            self.batches += 1
        return {key: len(record["steps"]) for key, record in records.items()}


def allocation_start(worker: Any, record: str | None, steps: int) -> bool:
    """Install the allocation's batch observer and sampler on this worker.

    Worker RPC, called once before the first request and before any fidelity
    sampler. The first TP rank records when ``record`` is given. Returns whether
    this worker records.
    """
    from vllm.distributed import get_tensor_model_parallel_rank

    from calibration_free_moe.eval.fidelity import FidelitySampler

    context = STATE.context
    runner = worker.model_runner
    if context is None or context.started:
        raise RuntimeError("No prepared allocation, or it was already started")
    if isinstance(runner.sampler, FidelitySampler) or runner.sampler is None:
        raise RuntimeError("Start the allocation before the fidelity sampler")
    if runner.batch_sharder is not None or runner.rejection_sampler is not None:
        raise RuntimeError("Allocation needs unsharded, non-speculative sampling")
    if (record is not None) != context.recording:
        raise RuntimeError("A record directory needs allocation_recording, and back")
    prepare = runner.prepare_inputs

    def prepare_inputs(*args: Any, **kwargs: Any) -> Any:
        batch = prepare(*args, **kwargs)
        context.observe_batch(batch, runner.req_states.prompt_len.gpu)
        return batch

    runner.prepare_inputs = prepare_inputs
    recording = record is not None and get_tensor_model_parallel_rank() == 0
    directory = Path(record) if recording and record is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    STATE.recording = AllocationSampler(runner.sampler, context, directory, steps)
    runner.sampler = STATE.recording
    if context.histograms is not None and context.histogram_tokens is not None:
        context.histograms.zero_()
        context.histogram_tokens.zero_()
    context.started = True
    return recording


def allocation_collect(_worker: Any) -> dict[str, int] | None:
    """Save and report the recording rank's steps (worker RPC after each batch)."""
    if STATE.recording is None or STATE.recording.record is None:
        return None
    return STATE.recording.collect()


def allocation_finish(_worker: Any) -> bool:
    """Save the recording rank's exact-score histograms, if any (worker RPC)."""
    context, recording = STATE.context, STATE.recording
    if recording is None or recording.record is None or context is None:
        return False
    if context.histograms is not None and context.histogram_tokens is not None:
        torch.save(
            {
                "format": "allocation-histogram-v1",
                "log2_range": HISTOGRAM_LOG2,
                "histograms": context.histograms.cpu(),
                "tokens": context.histogram_tokens.cpu(),
            },
            recording.record / "histogram.pt",
        )
    return True


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the allocation options to an evaluation CLI (policy prox, --retention)."""
    parser.add_argument(
        "--allocation",
        choices=SOURCES,
        help="prox with --retention: keep each token's top n channels jointly, n from "
        "this source (threshold: the calibrated threshold's own count)",
    )
    parser.add_argument(
        "--allocation-scale",
        type=float,
        help="uniform/schedule: multiplier of the top-k count (uniform sweep)",
    )
    parser.add_argument(
        "--allocation-input",
        help="schedule: depth schedule JSON; shuffled: record directory",
    )
    parser.add_argument(
        "--allocation-counts",
        help="uniform/schedule: per-layer base counts JSON (allocation_study.py "
        "layer-counts), matching a threshold arm's charged bytes in two passes",
    )
    parser.add_argument("--allocation-seed", type=int)
    parser.add_argument(
        "--allocation-record",
        type=Path,
        help="New directory for per-token counts, score features and charged bytes "
        "(first TP rank)",
    )


def cli_options(args: argparse.Namespace) -> dict[str, object]:
    """Convert the CLI arguments to binding options.

    Recording becomes a flag, fixed before graph capture; the record path stays
    with the evaluator.
    """
    return {
        "allocation": args.allocation,
        "allocation_scale": args.allocation_scale,
        "allocation_input": args.allocation_input,
        "allocation_seed": args.allocation_seed,
        "allocation_counts": args.allocation_counts,
        "allocation_recording": args.allocation_record is not None,
    }
