"""Measure the routed-output energy that top-F selection captures per screen (OLMoE).

A screen is a resident approximation of the routed experts' gate and up rows. For
each screen and score, every token keeps its top F of N = K*I candidate channels
with exact contributions and no recovery; the JSON report gives per-layer captured
energy, feature error and each screen's read and resident bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import Tensor

from calibration_free_moe.reader.sketch import (
    block_hadamard_signs,
    build_sketch,
    rotate_blocks,
    unpack_sketch,
)

FRACTIONS = (0.05, 0.10, 0.30)
BLOCK = 128
# Probabilists' Gauss-Hermite nodes and weights of the reader's five-point
# Gauss-Hermite score.
NODES = (
    -2.8569700138728056,
    -1.355626179974266,
    0.0,
    1.355626179974266,
    2.8569700138728056,
)
WEIGHTS = (
    0.011257411327720693,
    0.2220759220056126,
    0.5333333333333333,
    0.2220759220056126,
    0.011257411327720693,
)

# A quantizer maps one expert's rows [I, H] to (dequantized rows, per-block
# residual mean square or None).
Quantizer = Callable[[Tensor], tuple[Tensor, Tensor | None]]


@dataclass(frozen=True)
class Screen:
    """One resident screen: its quantizer, coordinates and per-coordinate bits."""

    name: str
    quantize: Quantizer
    rotated: bool
    gate_only: bool
    code_bits: float  # per coordinate of each screened row
    scale_bits: float
    residual_bits: float  # 0 when no residual is stored
    input_keep: float = 1.0  # fraction of input coordinates read (Prox stage 1)


def ternary(rows: Tensor) -> tuple[Tensor, Tensor]:
    """Return the reader's ternary sketch of rows already in screen coordinates.

    The sketch is `calibration_free_moe.reader.sketch.build_sketch`; the result is
    the dequantized rows [I, H] and the per-block residual mean square [I, H/128].
    """
    codes, moments = build_sketch(rows[None])
    trits = (
        unpack_sketch(codes[0], rows.shape[0], rows.shape[1])
        .float()
        .unflatten(-1, (-1, BLOCK))
    )
    scale, residual = moments[0].float().unbind(-1)
    return (trits * scale[..., None]).flatten(-2), residual


def int4_row(rows: Tensor) -> tuple[Tensor, None]:
    """Return per-row symmetric INT4 rows with max/7 FP32 scales.

    This matches `calibration_free_moe.baselines.prox.quantize_proxy`.
    """
    value = rows.float()
    scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 7
    return (value / scale).round().clamp(-7, 7) * scale, None


def screens(hidden: int) -> list[Screen]:
    """Return the screen catalog for hidden size `hidden`."""
    group = 16 / BLOCK  # one FP16 value per 128-coordinate block
    row = 32 / hidden  # one FP32 value per row
    out = [
        Screen("ternary-rot", ternary, True, False, 26 * 8 / BLOCK, group, group),
        Screen("int4-row-gate", int4_row, False, True, 4.0, row, 0.0),
    ]
    # Input-sparse screens read only each token's top-|x| coordinates (Prox stage 1).
    for keep in (0.5, 0.3):
        tag = f"-x{round(keep * 100)}"
        out.append(
            Screen(f"int4-row{tag}", int4_row, False, False, 4.0, row, 0.0, keep)
        )
    return out


def byte_shares(screen: Screen, score: str, hidden: int) -> dict[str, float]:
    """Return read bits per coordinate and read and resident shares of BF16 bytes.

    These are analytical bytes: codes, stored scales and, for the `plain` score,
    residuals. Per channel, BF16 routed bytes are 3 rows x H x 2. Norm-weighted
    scores also read and keep one FP32 down-column norm per channel; Prox's
    unweighted score does not. Input-sparse and partial-sketch screens read a
    subset of coordinates but keep the whole proxy.
    """
    rows = 1 if screen.gate_only else 2
    stored = screen.scale_bits + (screen.residual_bits if score == "plain" else 0.0)
    read = screen.code_bits * screen.input_keep + stored
    norm = 0.0 if score == "prox" else 32.0
    routed = 3 * hidden * 16
    return {
        "bits": read,
        "read_share": (rows * hidden * read + norm) / routed,
        "resident_share": (rows * hidden * (screen.code_bits + stored) + norm) / routed,
    }


def silu_moments(g: Tensor, variance: Tensor) -> Tensor:
    """Return E[silu(g)^2] for g ~ N(g_hat, variance) by five-point Gauss-Hermite."""
    sd = variance.sqrt()
    out = torch.zeros_like(g)
    for z, q in zip(NODES, WEIGHTS, strict=True):
        out += q * F.silu(g + z * sd).square()
    return out


class Layer:
    """One OLMoE MoE layer's native weights and the TUNE token sample's routing."""

    def __init__(
        self,
        model: Path,
        layer: int,
        x: Tensor,
        top_k: int,
        device: str,
    ):
        index = json.loads((model / "model.safetensors.index.json").read_text())
        prefix = f"model.layers.{layer}.mlp"

        def read(key: str) -> Tensor:
            with safe_open(model / index["weight_map"][key], framework="pt") as f:
                return f.get_tensor(key).to(device)

        experts = sum(
            key.startswith(f"{prefix}.experts.") and key.endswith(".gate_proj.weight")
            for key in index["weight_map"]
        )

        def stack(name: str) -> Tensor:
            return torch.stack(
                [read(f"{prefix}.experts.{e}.{name}.weight") for e in range(experts)]
            )

        self.gate, self.up = stack("gate_proj"), stack("up_proj")  # [E, I, H] BF16
        down = stack("down_proj")  # [E, H, I]
        self.down = down.float()
        self.norm = self.down.norm(dim=1)  # [E, I]
        self.x = x.to(device).float()
        router = read(f"{prefix}.gate.weight").float()
        # OLMoE does not renormalize the natural top-k weights.
        self.w, self.ids = (self.x @ router.T).softmax(-1).topk(top_k, dim=-1)
        self.pairs = [(self.ids == e).nonzero(as_tuple=True) for e in range(experts)]
        self.signs = block_hadamard_signs(self.x.shape[1], device)
        self.a = F.silu(self.project(self.x, self.gate)) * self.project(self.x, self.up)
        self.weight2 = (self.w[..., None] * self.norm[self.ids]).square()  # [T, K, I]

    def project(self, x: Tensor, rows: Tensor) -> Tensor:
        """Project [T, H] inputs onto each token's natural-candidate rows: [T, K, I]."""
        out = x.new_empty(*self.ids.shape, rows.shape[1])
        for e, (t, k) in enumerate(self.pairs):
            if len(t):
                out[t, k] = x[t] @ rows[e].float().T
        return out

    def variance(self, norms: Tensor, residual: Tensor) -> Tensor:
        """Return sum_b r_b ||x_b||^2 per candidate row: [T, K, I]."""
        out = norms.new_empty(*self.ids.shape, residual.shape[1])
        for e, (t, k) in enumerate(self.pairs):
            if len(t):
                out[t, k] = norms[t] @ residual[e].T
        return out

    def scores(self, screen: Screen) -> tuple[dict[str, Tensor], float | None]:
        """Return selection scores and the mean feature error of the screen.

        Scores, with w the router weight and n the down-column norm: `point`
        w^2 n^2 (silu(g_hat) u_hat)^2 (gate-only screens drop u_hat); `plain`
        w^2 n^2 E[silu(g)^2] (u_hat^2 + s_u^2), the reader's five-point
        Gauss-Hermite score, for screens with residual variances; `prox`
        |silu(g_hat) u_hat|, Prox's unweighted proxy. The feature error is
        sum (w n (a - p))^2 / sum (w n a)^2, with p the BF16 screen activation.
        """
        x = rotate_blocks(self.x, self.signs) if screen.rotated else self.x
        if screen.input_keep < 1:
            keep = round(screen.input_keep * x.shape[1])
            top = x.abs().topk(keep, dim=1).indices
            x = torch.zeros_like(x).scatter(1, top, x.gather(1, top))
        norms = x.unflatten(-1, (-1, BLOCK)).square().sum(-1)
        estimates, variances = [], []
        for rows in (self.gate,) if screen.gate_only else (self.gate, self.up):
            dequantized, residual = [], []
            for expert in rows:  # one expert at a time bounds FP32 scratch
                screened = (
                    rotate_blocks(expert, self.signs) if screen.rotated else expert
                )
                q, r = screen.quantize(screened)
                dequantized.append(q)
                residual.append(r)
            estimates.append(self.project(x, torch.stack(dequantized)))
            if residual[0] is not None and screen.input_keep == 1:
                variances.append(self.variance(norms, torch.stack(residual)))
        # The reader casts FP32 screen projections to BF16 before scoring.
        g = estimates[0].bfloat16().float()
        if screen.gate_only:
            return {"point": self.weight2 * F.silu(g).square()}, None
        u = estimates[1].bfloat16().float()
        out = {"point": self.weight2 * (F.silu(g) * u).square()}
        # Recovery reads each skipped channel's BF16 sketch activation p.
        sketch_activation = (F.silu(g) * u).bfloat16().float()
        feature = (self.weight2 * (self.a - sketch_activation).square()).sum(
            (-1, -2)
        ) / (self.weight2 * self.a.square()).sum((-1, -2))
        if variances:
            gate_variance, up_variance = variances
            out["plain"] = (
                self.weight2
                * silu_moments(g, gate_variance)
                * (u.square() + up_variance)
            )
        if screen.name.startswith("int4-row") and not screen.gate_only:
            out["prox"] = (F.silu(g) * u).abs()
        return out, feature.mean().item()

    def select(self, score: Tensor, fraction: float) -> Tensor:
        """Return a [T, K, I] mask of each token's top `fraction` of candidates."""
        _, top_k, width = score.shape
        count = round(fraction * top_k * width)
        flat = score.flatten(1)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        return mask.scatter_(1, flat.topk(count, dim=1).indices, True).view_as(score)

    def energies(self, masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return total, skipped and selected diagonal routed-output energies.

        The total is ||y||^2 per token [T]. For masks [M, T, K, I], the skipped
        energy is ||y - y_F||^2 and the selected diagonal energy is
        sum_(j in F) ||d_j a_j||^2, both [M, T], where d_j = w_j D_j is the
        routed down column.
        """
        weighted = self.a * self.w[..., None]
        y = torch.zeros_like(self.x)
        omitted = self.x.new_zeros(masks.shape[0], *self.x.shape)
        for e, (t, k) in enumerate(self.pairs):
            if len(t):
                contribution = weighted[t, k]  # [n, I]
                y.index_add_(0, t, contribution @ self.down[e].T)
                dropped = contribution * ~masks[:, t, k]  # [M, n, I]
                omitted.index_add_(1, t, dropped @ self.down[e].T)
        energy = self.weight2 * self.a.square()
        selected = (energy * masks).sum((-1, -2))
        return y.square().sum(-1), omitted.square().sum(-1), selected

    def evaluate(self, masks: Tensor) -> tuple[Tensor, Tensor]:
        """Return per-token captured energy and diagonal share: [M, T] each."""
        total, omitted, selected = self.energies(masks)
        energy = self.weight2 * self.a.square()
        return 1 - omitted / total, selected / energy.sum((-1, -2))


def tune_states(states: Path, layer: int, tokens: int, role: str = "tune") -> Tensor:
    """Return `tokens` MoE inputs [T, H] of one layer from documents of one role."""
    roles = [
        d["split_role"].lower()
        for d in json.loads((states / "collection.json").read_text())["documents"]
    ]
    payload = torch.load(
        states / f"layer-{layer:02d}.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    tune = torch.tensor([roles[int(d)] == role for d in payload["document"]]).nonzero()[
        :, 0
    ]
    # A fixed sample: the seed-0 permutation of the role's positions.
    order = torch.randperm(len(tune), generator=torch.Generator().manual_seed(0))
    return payload["states"][tune[order[:tokens]]]


def summarize(values: list[float]) -> dict[str, float]:
    """Return the mean, min, max and population standard deviation."""
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return {"mean": mean, "min": min(values), "max": max(values), "std": spread}


def tabulate(report: dict[str, Any], catalog: list[Screen], hidden: int) -> None:
    """Summarize every screen/score over layers, with its byte shares."""
    layers, feature_errors = report["layers"], report["feature_error"]
    table: list[dict[str, object]] = []
    for name, by_score in next(iter(layers.values())).items():
        screen = next((s for s in catalog if s.name == name), None)
        for score, by_budget in by_score.items():
            row: dict[str, object] = {"screen": name, "score": score}
            if screen is not None:
                row |= byte_shares(screen, score, hidden)
            if name in next(iter(feature_errors.values())):
                row["feature_error"] = summarize(
                    [feature_errors[k][name] for k in feature_errors]
                )
            for budget in by_budget:
                for metric in ("captured", "diagonal"):
                    row[f"{budget}_{metric}"] = summarize(
                        [layers[k][name][score][budget][metric] for k in layers]
                    )
            table.append(row)
    report["table"] = table


def main() -> None:
    """Evaluate the screens layer by layer and write the report after each layer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--tokens", type=int, default=3000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--screens", nargs="+", help="Evaluate only these screens (no oracle)"
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    parser.add_argument(
        "--thresholds",
        action="store_true",
        help="Also select by one per-layer score threshold fitted on FIT positions to "
        "keep each fraction of FIT candidates (a calibrated threshold, not top-F)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Add results to the existing --output (same layers, tokens and states)",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only rebuild the summary table and byte shares of the existing --output",
    )
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    config = json.loads((args.model / "config.json").read_text())
    top_k, hidden = config["num_experts_per_tok"], config["hidden_size"]
    catalog = screens(hidden)
    selected = catalog
    if args.screens:
        selected = [s for s in catalog if s.name in args.screens]
        if len(selected) != len(set(args.screens)):
            parser.error(f"Unknown screen in {args.screens}")
    run = {
        "command": sys.argv,
        "host": socket.gethostname(),
        "device": args.device,
        "source_revision": args.revision,
    }
    report: dict[str, Any]
    if args.merge or args.table_only:
        report = json.loads(args.output.read_text())
        if report["tokens_per_layer"] != args.tokens or report["states"] != str(
            args.states
        ):
            parser.error("Merged results must use the same states and token sample")
        report.setdefault("merged_runs", []).append(run)
    else:
        report = {
            "model": str(args.model),
            "states": str(args.states),
            "tokens_per_layer": args.tokens,
            "fractions": FRACTIONS,
            "top_k": top_k,
            **run,
            "layers": {},
            "feature_error": {},
        }
    layers: dict[str, dict[str, Any]] = report["layers"]
    feature_errors: dict[str, dict[str, float]] = report["feature_error"]
    for layer_id in [] if args.table_only else args.layers:
        start = time.perf_counter()
        layer = Layer(
            args.model,
            layer_id,
            tune_states(args.states, layer_id, args.tokens),
            top_k,
            args.device,
        )
        fit = (
            Layer(
                args.model,
                layer_id,
                tune_states(args.states, layer_id, args.tokens, "fit"),
                top_k,
                args.device,
            )
            if args.thresholds
            else None
        )
        keys, masks = [], []
        if not args.screens:
            for fraction in args.fractions:
                keys.append(("oracle", "exact", fraction))
                masks.append(layer.select(layer.weight2 * layer.a.square(), fraction))
        errors = feature_errors.setdefault(str(layer_id), {})
        for screen in selected:
            values, feature = layer.scores(screen)
            fitted = fit.scores(screen)[0] if fit is not None else {}
            if feature is not None:
                errors[screen.name] = feature
            for score, value in values.items():
                for fraction in args.fractions:
                    keys.append((screen.name, score, fraction))
                    masks.append(layer.select(value, fraction))
                    if score in fitted:
                        # One layer threshold keeping `fraction` of FIT candidates.
                        flat = fitted[score].flatten()
                        rank = max(1, round((1 - fraction) * flat.numel()))
                        cut = flat.kthvalue(rank).values
                        keys.append((screen.name, f"{score}-threshold", fraction))
                        masks.append(value > cut)
        results = layers.setdefault(str(layer_id), {})
        for offset in range(0, len(masks), 16):
            captured, diagonal = layer.evaluate(
                torch.stack(masks[offset : offset + 16])
            )
            for j, (name, score, fraction) in enumerate(keys[offset : offset + 16]):
                kept = masks[offset + j].float().mean((-1, -2))
                results.setdefault(name, {}).setdefault(score, {})[
                    f"P{round(fraction * 100):02d}"
                    if fraction in FRACTIONS
                    else f"f{fraction:g}"
                ] = {
                    "captured": captured[j].mean().item(),
                    "diagonal": diagonal[j].mean().item(),
                    "kept": kept.mean().item(),
                    "kept_std": kept.std().item(),
                }
        summary = {
            f"{name}/{score}": round(results[name][score]["P10"]["captured"], 4)
            for name, score, fraction in keys
            if fraction == 0.10
        }
        print(
            f"layer {layer_id} ({time.perf_counter() - start:.0f}s): "
            f"10% selected {summary}",
            flush=True,
        )
        del layer, masks
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    tabulate(report, catalog, hidden)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
