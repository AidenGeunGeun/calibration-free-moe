"""Write which MoE-side quantity carries the depth drift of calibrated kept counts.

A threshold fitted on chat text keeps fewer channels deeper into a reasoning answer.
This script splits that drift into score scale, score shape, layer type and
candidate mediators on saved chat and trace MoE states, and writes one JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from calibration_free_moe.activation import (
    GatedActivation,
    activate,
    model_activation,
)
from calibration_free_moe.model_config import model_dimensions

VARIANTS = {
    "exact": (True, True),
    "without_router": (False, True),
    "without_down": (True, False),
    "activation": (False, False),
}
SCALE_STATISTICS = ("rms", "median")
TAIL_MULTIPLES = (0.5, 2**-0.5, 1.0, 2**0.5, 2.0)
NORM_KEYS = (
    "residual_in_norm",
    "attention_out_norm",
    "residual_mid_norm",
    "moe_out_norm",
    "ffn_out_norm",
    "shared_out_norm",
    "dense_out_norm",
    "dense_branch_norm",
    "moe_branch_norm",
)
WINDOW = 16
BOOTSTRAP = 1000


@dataclass
class Experts:
    """One layer's routed-expert weights needed for the channel score."""

    gate: Tensor  # [E, I, H]
    up: Tensor  # [E, I, H]
    down: Tensor  # [E, I] down-column norms
    router: Tensor | None  # [E, H]; Gemma 4 routes from the residual, not x


@dataclass
class Rows:
    """Saved MoE input rows of one layer with their routing and pre-activations."""

    states: Tensor  # [T, H]
    norm: Tensor  # [T]
    weight: Tensor  # [T, K] applied router weights, descending
    ids: Tensor  # [T, K]
    logits: Tensor | None  # [T, E]
    gate: Tensor  # [T, K, I] gate pre-activations
    up: Tensor  # [T, K, I]
    down: Tensor  # [T, K, I]


class Checkpoint:
    """Lazy reader of a safetensors checkpoint's routed-expert weights."""

    def __init__(self, model: Path, device: str) -> None:
        self.model = model
        self.device = device
        self.config = model_dimensions(model)
        self.architecture = str(self.config["architecture"])
        if self.architecture not in {
            "Qwen3MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Gemma4ForConditionalGeneration",
        }:
            raise ValueError(f"Unsupported checkpoint layout: {self.architecture}")
        index = json.loads((model / "model.safetensors.index.json").read_text())
        self.index: dict[str, str] = index["weight_map"]
        self.files: dict[str, Any] = {}
        self.activation: GatedActivation = model_activation(
            SimpleNamespace(**self.config)
        )

    def tensor(self, name: str) -> Tensor:
        """Return one checkpoint tensor, opening its shard once."""
        file = self.index[name]
        if file not in self.files:
            self.files[file] = safe_open(
                self.model / file, framework="pt", device=self.device
            )
        return self.files[file].get_tensor(name)

    def layer(self, layer: int) -> Experts:
        """Return one layer's gate and up rows, down-column norms and router."""
        if self.architecture == "Qwen3MoeForCausalLM":
            prefix = f"model.layers.{layer}.mlp"
            experts = range(int(self.config["num_experts"]))
            names = [f"{prefix}.experts.{e}" for e in experts]
            return Experts(
                torch.stack([self.tensor(f"{n}.gate_proj.weight") for n in names]),
                torch.stack([self.tensor(f"{n}.up_proj.weight") for n in names]),
                torch.stack(
                    [
                        self.tensor(f"{n}.down_proj.weight").float().norm(dim=0)
                        for n in names
                    ]
                ),
                self.tensor(f"{prefix}.gate.weight"),
            )
        prefix = f"model.language_model.layers.{layer}"
        gemma = self.architecture.startswith("Gemma4")
        experts = prefix + (".experts" if gemma else ".mlp.experts")
        gate, up = self.tensor(f"{experts}.gate_up_proj").chunk(2, dim=1)
        return Experts(
            gate,
            up,
            self.tensor(f"{experts}.down_proj").float().norm(dim=1),
            None if gemma else self.tensor(f"{prefix}.mlp.gate.weight"),
        )


def project(
    payload: dict[str, Tensor], experts: Experts, config: dict[str, Any]
) -> Rows:
    """Return exact pre-activations of every row's own top-k experts.

    Uses the recorded routing when the payload has it, else the checkpoint router.
    Slots are sorted by descending router weight.
    """
    x = payload["states"].float()
    logits = payload.get("router_logits")
    if "route_ids" in payload:
        weight, ids = payload["route_weights"].float(), payload["route_ids"].long()
    elif experts.router is not None:
        logits = x @ experts.router.float().T
        weight, ids = logits.softmax(-1).topk(int(config["num_experts_per_tok"]), -1)
        if config.get("norm_topk_prob", False):
            weight = weight / weight.sum(-1, keepdim=True)
    else:
        raise ValueError("Gemma 4 states need recorded routing")
    weight, order = weight.sort(-1, descending=True)
    ids = ids.gather(-1, order)
    width = int(config["expert_width"])
    gate = x.new_empty(*ids.shape, width)
    up = x.new_empty(*ids.shape, width)
    for expert in ids.unique().tolist():
        token, slot = (ids == expert).nonzero(as_tuple=True)
        gate[token, slot] = x[token] @ experts.gate[expert].float().T
        up[token, slot] = x[token] @ experts.up[expert].float().T
    return Rows(
        x,
        x.norm(dim=-1),
        weight,
        ids,
        None if logits is None else logits.float(),
        gate,
        up,
        experts.down[ids],
    )


def score(
    rows: Rows,
    activation: GatedActivation,
    router: bool = True,
    down: bool = True,
) -> Tensor:
    """Return [T, N] channel scores of every row's own routed candidates.

    Channel j of routed expert k scores w_k |act(g_j) u_j| ||d_j|| (applied router
    weight, exact gated activation, down-column norm); `router` and `down` switch
    the first and last factor.
    """
    gate, up = rows.gate, rows.up
    values = (activate(gate, activation) * up).abs()
    if router:
        values = values * rows.weight[..., None]
    if down:
        values = values * rows.down
    return values.flatten(1)


def threshold(scores: Tensor, kept: int) -> float:
    """Return the tau whose mean kept count (scores > tau) over these rows is `kept`."""
    pool = scores.flatten()
    return float(pool.kthvalue(pool.numel() - kept * scores.shape[0]).values.item())


def token_scale(scores: Tensor, statistic: str) -> Tensor:
    """Return each row's RMS or median candidate score."""
    if statistic == "rms":
        return scores.square().mean(-1).sqrt()
    return scores.median(-1).values


class Grid:
    """Trace rows as documents x columns (0: prompt, b + 1: answer bin b).

    The cohort is the traces whose answer reaches the last edge and that have rows in
    every answer bin. ``weight`` gives each cohort trace equal weight within a bin (each
    row 1 / (its trace's rows in that bin x cohort size)), so pooled estimates equal
    trace-balanced means however many rows each trace contributes to a bin.
    """

    def __init__(
        self,
        document: Tensor,
        position: Tensor,
        prompt: Tensor,
        used: Tensor,
        width: int,
        end: int,
    ) -> None:
        self.bins = end // width
        document = document.long()
        answer = position.long() - prompt[document]
        column = torch.where(answer < 0, 0, answer // width + 1)
        column[answer >= end] = -1
        inside = used[document] & (column > 0)
        present = torch.zeros(len(used), self.bins + 1, device=used.device)
        present[document[inside], column[inside]] = 1
        self.cohort = used & (present[:, 1:] > 0).all(1)
        self.documents = int(self.cohort.sum())
        slot = torch.full_like(used, -1, dtype=torch.long)
        slot[self.cohort] = torch.arange(self.documents, device=used.device)
        row_slot = slot[document]
        self.valid = (row_slot >= 0) & (column >= 0)
        self.column = torch.where(self.valid, column, -1)
        self.cell = row_slot[self.valid] * (self.bins + 1) + column[self.valid]
        self.position = position.long()
        count = torch.bincount(self.cell, minlength=self.documents * (self.bins + 1))
        weight = torch.zeros(len(document), dtype=torch.float64, device=used.device)
        weight[self.valid] = 1 / (count[self.cell].double() * self.documents)
        self.weight = torch.where(self.column > 0, weight, 0)

    def mean(self, values: Tensor) -> Tensor:
        """Return per-document, per-column means of finite values: [D, B + 1] on CPU."""
        values = values[self.valid].double()
        finite = torch.isfinite(values)
        cells = self.documents * (self.bins + 1)
        sums = values.new_zeros(cells).index_add_(0, self.cell[finite], values[finite])
        counts = values.new_zeros(cells).index_add_(
            0, self.cell[finite], torch.ones_like(values[finite])
        )
        return (sums / counts).view(self.documents, self.bins + 1).cpu()

    def pooled(self, values: Tensor) -> Tensor:
        """Return trace-balanced means of per-row values per answer bin: [B] on CPU."""
        answer = self.column > 0
        return (
            torch.zeros(self.bins + 1, dtype=torch.float64, device=values.device)
            .index_add_(
                0, self.column[answer], self.weight[answer] * values[answer].double()
            )[1:]
            .cpu()
        )

    def bin(self, index: int) -> Tensor:
        """Return the row mask of answer bin `index`."""
        return self.column == index + 1

    def windows(self) -> Tensor:
        """Return [G, WINDOW] row indices of windows within one document and column.

        Each window holds consecutive sampled rows; a trailing partial window drops.
        """
        rows = self.valid.nonzero().squeeze(1)
        span = int(self.position.max()) + 1
        order = (self.cell * span + self.position[rows]).argsort()
        rows, cells = rows[order], self.cell[order]
        rank = torch.arange(len(cells), device=cells.device) - torch.searchsorted(
            cells, cells
        )
        count = torch.bincount(cells, minlength=self.documents * (self.bins + 1))
        starts = ((rank % WINDOW == 0) & (rank + WINDOW <= count[cells])).nonzero()
        return rows[starts + torch.arange(WINDOW, device=rows.device)]


def prefix_cosine(states: Tensor, document: Tensor, position: Tensor) -> Tensor:
    """Return each row's cosine to the sum of its document's earlier sampled rows.

    The first sampled row of a document has no earlier rows and gets NaN.
    """
    span = int(position.max()) + 1
    order = (document.long() * span + position.long()).argsort()
    ordered = document[order]
    values = states[order].double()
    total = values.cumsum(0)
    start = torch.searchsorted(ordered, ordered)
    earlier = total - values - (total[start] - values[start])
    cosine = torch.nn.functional.cosine_similarity(values, earlier, dim=-1)
    cosine[torch.arange(len(order), device=order.device) == start] = math.nan
    result = torch.empty_like(cosine)
    result[order] = cosine
    return result


def participation(states: Tensor, windows: Tensor) -> Tensor:
    """Return each window's participation ratio of centered states, per row.

    Rows outside every window get NaN.
    """
    block = states[windows].double()
    block = block - block.mean(1, keepdim=True)
    gram = block @ block.transpose(1, 2)
    ratio = gram.diagonal(dim1=1, dim2=2).sum(-1).square() / gram.square().sum((1, 2))
    result = states.new_full((states.shape[0],), math.nan, dtype=torch.float64)
    result[windows.flatten()] = ratio.repeat_interleave(WINDOW)
    return result


def profile(matrix: Tensor) -> dict[str, Any]:
    """Return trace-balanced bin means with bootstrap-by-trace SEs.

    `matrix` is [traces, 1 + bins] with the prompt in column 0; traces missing a
    bin drop out. `delta` and `delta_se` are the change versus the first bin.
    """
    answer = matrix[:, 1:]
    complete = torch.isfinite(answer).all(1)
    answer, prompt = answer[complete], matrix[complete, 0]
    count = answer.shape[0]
    if count == 0:
        return {"documents": 0}
    draws = torch.randint(
        count, (BOOTSTRAP, count), generator=torch.Generator().manual_seed(0)
    )
    boot = answer[draws].mean(1)
    mean = answer.mean(0)
    result: dict[str, Any] = {
        "documents": count,
        "mean": mean,
        "se": boot.std(0),
        "delta": mean - mean[0],
        "delta_se": (boot - boot[:, :1]).std(0),
    }
    if bool(torch.isfinite(prompt).all()):
        result["prompt"] = prompt.mean()
        result["prompt_se"] = prompt[draws].mean(1).std()
    return result


def bootstrap_mean(values: Tensor) -> dict[str, Any]:
    """Return the mean of per-document values with its bootstrap SE."""
    draws = torch.randint(
        len(values),
        (BOOTSTRAP, len(values)),
        generator=torch.Generator().manual_seed(0),
    )
    return {
        "documents": len(values),
        "mean": values.mean(),
        "se": values[draws].mean(1).std(),
    }


def align(scores: Tensor, grid: Grid, tau: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-bin log scale versus bin 0, observed and reconstructed kept fraction.

    The log scale log c_b is the mean log ratio of bin-b to bin-0 quantiles at
    upper-tail probabilities f0 x TAIL_MULTIPLES (f0: bin-0 kept fraction), that
    is, near the cutoff. The reconstructed fraction P0(s > tau / c_b) on bin-0
    scores is the count if depth only rescaled them. Each bin's channel scores carry
    their row's trace-balanced weight, so quantiles and fractions above a cut are
    trace-balanced too. The quantile at level q is the smallest score whose
    cumulative weight reaches q (less a 1e-9 relative slack for rounding).
    """
    candidates = scores.shape[1]
    bins = []
    for b in range(grid.bins):
        rows = grid.bin(b)
        values, order = scores[rows].flatten().sort()
        weight = (grid.weight[rows] / candidates).repeat_interleave(candidates)
        bins.append((values, weight[order].cumsum(0)))

    def above(values: Tensor, mass: Tensor, cut: Tensor) -> Tensor:
        index = torch.searchsorted(values, cut.reshape(1).to(values), right=True)
        below = torch.where(index > 0, mass[(index - 1).clamp(min=0)], 0)
        return ((mass[-1] - below) / mass[-1]).squeeze()

    def quantile(values: Tensor, mass: Tensor, tail: Tensor) -> Tensor:
        level = ((1 - tail) * mass[-1] * (1 - 1e-9)).reshape(1)
        index = torch.searchsorted(mass, level).clamp(max=len(values) - 1)
        return values[index].double().squeeze()

    base, base_mass = bins[0]
    cutoff = torch.tensor(tau, dtype=torch.float64, device=scores.device)
    tail = above(base, base_mass, cutoff)
    log_scale, observed, reconstructed = [], [], []
    for values, mass in bins:
        shift = torch.stack(
            [
                quantile(values, mass, tail * m).log()
                - quantile(base, base_mass, tail * m).log()
                for m in TAIL_MULTIPLES
            ]
        ).mean()
        log_scale.append(shift)
        observed.append(above(values, mass, cutoff))
        reconstructed.append(above(base, base_mass, cutoff / shift.exp()))
    return (
        torch.stack(log_scale).cpu(),
        torch.stack(observed).cpu(),
        torch.stack(reconstructed).cpu(),
    )


def document_means(values: Tensor, document: Tensor) -> Tensor:
    """Return the mean of row values per document (ascending document IDs), on CPU."""
    _, inverse = document.unique(return_inverse=True)
    groups = int(inverse.max()) + 1
    sums = torch.zeros(groups, dtype=torch.float64, device=values.device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, inverse, values.double())
    counts.index_add_(0, inverse, torch.ones_like(values, dtype=torch.float64))
    return (sums / counts).cpu()


def plain(value: Any) -> Any:
    """Return a JSON-ready copy: tensors to lists, non-finite floats to null."""
    if isinstance(value, Tensor):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load(directory: Path, layer: int, device: str) -> dict[str, Tensor]:
    """Load one layer's saved MoE states (`layer-NN.pt`)."""
    return torch.load(
        directory / f"layer-{layer:02d}.pt", map_location=device, weights_only=True
    )


def main() -> None:
    """Score every layer's chat and trace rows, then write and print the analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chat-states", type=Path, required=True)
    parser.add_argument("--trace-states", type=Path, required=True)
    parser.add_argument(
        "--trace-requests",
        type=Path,
        required=True,
        help="Requests the trace panel was built from (prompt + target per ID)",
    )
    parser.add_argument("--retention", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width", type=int, default=500)
    parser.add_argument(
        "--answer-end",
        type=int,
        help="Last bin edge; default: the shortest answer, rounded down to a bin",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device
    checkpoint = Checkpoint(args.model, device)
    config = checkpoint.config
    layers, top_k = int(config["num_hidden_layers"]), int(config["num_experts_per_tok"])
    width = int(config["expert_width"])
    candidates = top_k * width
    kept_count = round(args.retention * candidates)

    requests = {
        r["id"]: r for r in json.loads(args.trace_requests.read_text())["requests"]
    }
    panel = json.loads((args.trace_states / "collection.json").read_text())["documents"]
    # Trace requests prefill the question plus the first ``depth`` answer tokens.
    prompt = torch.tensor(
        [
            len(requests[d["id"]]["prompt"]) - requests[d["id"]].get("depth", 0)
            for d in panel
        ]
    )
    if any(
        d["tokens"]
        != len(requests[d["id"]]["prompt"]) + len(requests[d["id"]]["target"])
        for d in panel
    ):
        raise ValueError("Trace requests do not match the trace panel")
    answer_length = torch.tensor([d["tokens"] for d in panel]) - prompt
    end = args.answer_end or int(answer_length.min()) // args.bin_width * args.bin_width
    if end < 2 * args.bin_width:
        parser.error("Fewer than two answer bins")
    if end % args.bin_width:
        parser.error("--answer-end must be a multiple of --bin-width")
    used = answer_length >= end

    chat0, trace0 = (
        load(args.chat_states, 0, device),
        load(args.trace_states, 0, device),
    )
    # Thresholds are fitted on even chat documents; odd ones are held out.
    fit = chat0["document"] % 2 == 0
    held_document = chat0["document"][~fit].long()
    grid = Grid(
        trace0["document"],
        trace0["position"],
        prompt.to(device),
        used.to(device),
        args.bin_width,
        end,
    )
    windows = grid.windows()
    cohort = grid.cohort.tolist()
    bins = grid.bins
    layer_types: list[str] = config.get("layer_types") or ["full_attention"] * layers

    thresholds: dict[str, list[float]] = {}
    tables: dict[str, list[Tensor]] = {}
    chat_kept = torch.zeros(len(held_document), device=device, dtype=torch.float64)
    chat_layer: list[float] = []
    alignment: dict[str, list[Tensor]] = {
        "log_scale": [],
        "observed": [],
        "reconstructed": [],
    }

    def table(name: str, values: Tensor) -> None:
        tables.setdefault(name, []).append(grid.mean(values))

    for layer in range(layers):
        experts = checkpoint.layer(layer)
        chat = load(args.chat_states, layer, device)
        trace = load(args.trace_states, layer, device)
        if not torch.equal(chat["document"], chat0["document"]) or not torch.equal(
            trace["position"], trace0["position"]
        ):
            raise ValueError(f"Layer {layer} rows differ from layer 0")
        chat_rows, trace_rows = (
            project(chat, experts, config),
            project(trace, experts, config),
        )
        activation = checkpoint.activation

        # 1. Kept fraction by score variant, then with each token's scores divided
        # by their own RMS or median (tau refitted on chat likewise).
        exact_chat = score(chat_rows, activation)
        exact_trace = score(trace_rows, activation)
        exact_tau = threshold(exact_chat[fit], kept_count)
        held = (exact_chat[~fit] > exact_tau).double().mean(-1)
        chat_kept += held
        chat_layer.append(float(document_means(held, held_document).mean()))
        for name, (router, down) in VARIANTS.items():
            if name == "exact":
                tau, trace_scores = exact_tau, exact_trace
            else:
                chat_scores = score(chat_rows, activation, router, down)
                tau = threshold(chat_scores[fit], kept_count)
                trace_scores = score(trace_rows, activation, router, down)
            thresholds.setdefault(name, []).append(tau)
            table(f"kept_{name}", (trace_scores > tau).float().mean(-1))
        for statistic in SCALE_STATISTICS:
            chat_scores = exact_chat / token_scale(exact_chat, statistic)[:, None]
            tau = threshold(chat_scores[fit], kept_count)
            thresholds.setdefault(f"normalized_{statistic}", []).append(tau)
            trace_scores = exact_trace / token_scale(exact_trace, statistic)[:, None]
            table(f"normalized_{statistic}", (trace_scores > tau).float().mean(-1))
        del exact_chat, chat_rows

        # 2. One-dimensional scale alignment to the first answer bin.
        for key, value in zip(
            alignment, align(exact_trace, grid, exact_tau), strict=True
        ):
            alignment[key].append(value)
        del exact_trace

        # 3. Mediators. Attention output is divided by residual_in, every other
        # output norm by residual_mid.
        table("input_norm", trace_rows.norm)
        table("top1_weight", trace_rows.weight[:, 0])
        if trace_rows.logits is not None:
            log_probability = trace_rows.logits.log_softmax(-1)
            table("router_entropy", -(log_probability.exp() * log_probability).sum(-1))
            table(
                "topk_probability_mass",
                log_probability.exp().topk(top_k, dim=-1).values.sum(-1),
            )
        for key in NORM_KEYS:
            if key in trace:
                table(key, trace[key].float())
                if key == "attention_out_norm":
                    table(
                        "attention_over_residual",
                        trace[key] / trace["residual_in_norm"],
                    )
                elif key not in ("residual_in_norm", "residual_mid_norm"):
                    name = key.removesuffix("_norm") + "_over_residual"
                    table(name, trace[key] / trace["residual_mid_norm"])
        table(
            "prefix_cosine",
            prefix_cosine(trace_rows.states, trace["document"], trace["position"]),
        )
        table("participation_ratio", participation(trace_rows.states, windows))
        del trace_rows, chat, trace
        print(f"layer {layer} tau {exact_tau:.3e}", flush=True)

    stacked = {name: torch.stack(values) for name, values in tables.items()}
    profiles = {name: profile(values.mean(0)) for name, values in stacked.items()}

    observed = torch.stack(alignment["observed"])
    reconstructed = torch.stack(alignment["reconstructed"])
    observed_mean, reconstructed_mean = observed.mean(0), reconstructed.mean(0)
    drop = observed_mean - observed_mean[0]
    gain = reconstructed_mean - observed_mean[0]
    # Share of the observed drop that pure rescaling reproduces, per bin and pooled.
    explained = torch.where(drop != 0, gain / drop, math.nan)
    explained[0] = math.nan
    scale_alignment = {
        "log_scale": torch.stack(alignment["log_scale"]),
        "observed": observed_mean,
        "reconstructed": reconstructed_mean,
        "explained_share": explained,
        "explained_share_lsq": float((gain * drop).sum() / drop.square().sum()),
        "per_layer_observed": observed,
        "per_layer_reconstructed": reconstructed,
    }

    # Offset of each layer from the last full-attention layer (the staircase).
    offsets: list[int | None] = []
    last_full: int | None = None
    for index, kind in enumerate(layer_types):
        last_full = index if kind == "full_attention" else last_full
        offsets.append(None if last_full is None else index - last_full)
    per_layer = [profile(values) for values in stacked["kept_exact"]]
    first_bin = torch.stack([p["mean"][0] for p in per_layer])
    last_bin = torch.stack([p["mean"][-1] for p in per_layer])
    relative = last_bin / first_bin - 1

    def group(keys: list[Any]) -> dict[str, dict[str, float]]:
        groups: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            if key is not None:
                groups.setdefault(str(key), []).append(index)
        return {
            key: {
                "layers": len(index),
                "relative_drift": float(relative[index].mean()),
                "drift": float((last_bin - first_bin)[index].mean()),
            }
            for key, index in groups.items()
        }

    by_type, by_offset = group(layer_types), group(offsets)
    layer_table = {
        "type": layer_types,
        "offset_from_full_attention": offsets,
        "first_bin": first_bin,
        "last_bin": last_bin,
        "drift": last_bin - first_bin,
        "drift_se": torch.stack([p["delta_se"][-1] for p in per_layer]),
        "relative_drift": relative,
        "by_type": by_type,
        "by_offset": by_offset,
    }

    chat_means = document_means(chat_kept / layers, held_document)
    exact = profiles["kept_exact"]
    level = {
        "chat_heldout": bootstrap_mean(chat_means),
        "trace_prompt": {"mean": exact.get("prompt"), "se": exact.get("prompt_se")},
        "trace_first_bin": {"mean": exact["mean"][0], "se": exact["se"][0]},
        "trace_last_bin": {"mean": exact["mean"][-1], "se": exact["se"][-1]},
        "per_layer": {
            "chat_heldout": chat_layer,
            "trace_prompt": stacked["kept_exact"][:, :, 0].nanmean(1),
            "trace_first_bin": first_bin,
            "trace_last_bin": last_bin,
        },
    }
    variants = [*VARIANTS, *(f"normalized_{s}" for s in SCALE_STATISTICS)]
    mediators = {
        name: profiles[name] | {"per_layer_mean": stacked[name][:, :, 1:].nanmean(1)}
        for name in profiles
        if name not in {f"kept_{v}" for v in VARIANTS} and name not in variants
    }
    result = {
        "model": str(args.model),
        "architecture": checkpoint.architecture,
        "retention": args.retention,
        "kept_count": kept_count,
        "candidates": candidates,
        "layers": layers,
        "bin_width": args.bin_width,
        "answer_end": end,
        "bin_edges": list(range(0, end + 1, args.bin_width)),
        "documents": {
            "used": [d["id"] for d, u in zip(panel, cohort, strict=True) if u],
            "dropped": [d["id"] for d, u in zip(panel, cohort, strict=True) if not u],
        },
        "rows": {
            "trace": int(grid.valid.sum()),
            "chat_fit": int(fit.sum()),
            "chat_heldout": int((~fit).sum()),
            "participation_windows": len(windows),
        },
        "thresholds": thresholds,
        "kept": {
            name: profiles[f"kept_{name}" if name in VARIANTS else name]
            for name in variants
        },
        "scale_alignment": scale_alignment,
        "layer_drift": layer_table,
        "level": level,
        "mediators": mediators,
    }
    args.output.write_text(json.dumps(plain(result)) + "\n")

    def change(name: str) -> str:
        p = profiles[name]
        if not p["documents"]:
            return "no complete trace"
        return (
            f"{float(p['mean'][0]):.4g} -> {float(p['mean'][-1]):.4g} "
            f"(delta {float(p['delta'][-1]):+.4g} +- {float(p['delta_se'][-1]):.2g})"
        )

    print(f"F={kept_count} of N={candidates}; {grid.documents} traces, {bins} bins")
    for name in variants:
        print(f"kept {name}: {change(f'kept_{name}' if name in VARIANTS else name)}")
    print(
        f"scale alignment explains {float(explained[-1]):.3f} of the last-bin drop "
        f"(pooled {scale_alignment['explained_share_lsq']:.3f})"
    )
    print(
        f"level: chat held-out {float(chat_means.mean()):.4g}; "
        f"trace {change('kept_exact')}"
    )
    for key, value in by_type.items():
        print(f"layer type {key}: {value}")
    for key, value in by_offset.items():
        print(f"offset {key}: {value}")
    for name in mediators:
        print(f"mediator {name}: {change(name)}")


if __name__ == "__main__":
    main()
