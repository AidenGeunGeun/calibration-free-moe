"""Fit CATS, TEAL or WINA baseline thresholds at `--sparsity` on FIT-only MoE states.

Reads a `collect_moe_inputs.py` state collection and the BF16 checkpoint on one
CUDA device, and writes a torch file of per-layer thresholds (per-projection
sparsity levels for `wina-greedy-topk`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import Tensor

from calibration_free_moe.activation import (
    GatedActivation,
    activate,
    model_activation,
)
from calibration_free_moe.model_config import model_dimensions


def empirical_threshold(values: Tensor, sparsity: float) -> float:
    """Return the `sparsity` quantile of `values`; -1 keeps everything at 0.

    Matches torch.quantile(..., interpolation="higher") without its 2**24-element
    limit.
    """
    if sparsity == 0:
        return -1.0
    flat = values.float().reshape(-1)
    rank = math.ceil(sparsity * (flat.numel() - 1)) + 1
    return float(flat.kthvalue(rank).values)


def greedy_mlp_thresholds(
    x: Tensor,
    route: Tensor,
    gate_rows: Tensor,
    up_rows: Tensor,
    down_rows: Tensor,
    route_weights: Tensor,
    sparsity: float,
    *,
    weighted: bool = False,
    activation: GatedActivation = "silu",
    return_levels: bool = False,
) -> dict[str, float]:
    """Run TEAL's greedy sparsity search (TEAL, Sec. 4.2) on one layer's states.

    Each step raises the sparsity of the gate, up or down projection by one
    increment, choosing the projection whose step least increases the routed-output
    error, until the mean sparsity reaches `sparsity`. `x` is [tokens, H], `route`
    and `route_weights` are [tokens, K]. Returns thresholds, or with
    `return_levels` the per-projection sparsity levels.
    """
    # Error is the routed-expert output error per collected FIT token; the official
    # search compares whole decoder-block outputs over the final 25% of positions.
    n = route.shape[0]
    gate_norm: Tensor | None = (
        cast(Tensor, gate_rows.float().norm(dim=1).bfloat16()) if weighted else None
    )
    raw: list[list[Tensor]] = [[], [], []]
    for start in range(0, n, 8):
        batch, ids = x[start : start + 8], route[start : start + 8]
        g = torch.einsum("bh,bkih->bki", batch, gate_rows[ids]).bfloat16()
        u = torch.einsum("bh,bkih->bki", batch, up_rows[ids]).bfloat16()
        feature = (activate(g, activation).bfloat16() * u).bfloat16()
        if weighted:
            assert gate_norm is not None
            raw[0].append(
                (batch.float()[:, None, :].abs() * gate_norm[ids].float()).cpu()
            )
        else:
            raw[0].append(batch.abs().cpu())
        raw[1].append(batch.abs().cpu())
        raw[2].append(feature.abs().cpu())
    scores = [torch.cat(part) for part in raw]

    def output(levels: list[float]) -> Tensor:
        thresholds = [
            empirical_threshold(s, p) for s, p in zip(scores, levels, strict=True)
        ]
        chunks = []
        for start in range(0, n, 8):
            batch, ids = x[start : start + 8], route[start : start + 8]
            if weighted:
                assert gate_norm is not None
                gate_mask = (
                    batch.float()[:, None, :].abs() * gate_norm[ids].float()
                    > thresholds[0]
                )
            else:
                gate_mask = batch.abs()[:, None, :] > thresholds[0]
            up_mask = batch.abs()[:, None, :] > thresholds[1]
            g = torch.einsum(
                "bkh,bkih->bki", batch[:, None, :] * gate_mask, gate_rows[ids]
            ).bfloat16()
            u = torch.einsum(
                "bkh,bkih->bki", batch[:, None, :] * up_mask, up_rows[ids]
            ).bfloat16()
            feature = (activate(g, activation).bfloat16() * u).bfloat16()
            feature = feature * (feature.abs() > thresholds[2])
            routed = torch.einsum("bki,bkhi->bkh", feature, down_rows[ids])
            chunks.append(
                (routed.float() * route_weights[start : start + 8, :, None]).sum(dim=1)
            )
        return torch.cat(chunks)

    target = output([0.0, 0.0, 0.0])
    levels = [0.0, 0.0, 0.0]
    # Official step = .025 / matrix_weight (~.007 for Llama MLP projections).
    # Routed gate/up/down have equal I*H coefficients, so allocation steps are
    # uniformly .01; stop at the first allocation reaching the target mean sparsity.
    step = 0.01 if return_levels else 0.05
    # Top-K inference can omit an entire projection; the threshold variant stops
    # at .95.
    maximum = 1.0 - 1e-9 if return_levels else 0.95
    while sum(levels) / 3 < sparsity - 1e-9:
        candidates: list[tuple[float, int]] = []
        for proj in range(3):
            if levels[proj] >= maximum:
                continue
            trial = levels.copy()
            trial[proj] = min(1.0 if return_levels else 0.99, trial[proj] + step)
            error = cast(float, (output(trial) - target).norm(dim=1).mean().item())
            candidates.append((error, proj))
        if not candidates:
            break
        selected = min(candidates)[1]
        levels[selected] = (
            min(1.0, levels[selected] + step)
            if return_levels
            else levels[selected] + step
        )
    if return_levels:
        return dict(
            zip(("gate_sparsity", "up_sparsity", "down_sparsity"), levels, strict=True)
        )
    thresholds = [
        empirical_threshold(s, p) for s, p in zip(scores, levels, strict=True)
    ]
    return dict(
        zip(
            ("gate_threshold", "input_threshold", "down_threshold"),
            thresholds,
            strict=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=(
            "cats",
            "teal",
            "teal-greedy",
            "wina",
            "wina-greedy-topk",
        ),
        required=True,
    )
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument(
        "--max-states",
        type=int,
        default=512,
        help="Deterministically spread FIT states over this many positions per layer",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.sparsity < 1:
        parser.error("--sparsity must lie in [0,1)")
    if args.max_states < 1:
        parser.error("--max-states must be positive")
    collection = json.loads((args.states / "collection.json").read_text())
    if collection.get("complete") is not True:
        raise ValueError("MoE state collection is incomplete")
    documents = collection["documents"]
    fit = [
        idx for idx, doc in enumerate(documents) if doc["split_role"].lower() == "fit"
    ]
    if not fit:
        raise ValueError("No FIT documents in collection")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    config = model_dimensions(args.model)
    gemma = config["architecture"].startswith("Gemma4")
    activation = model_activation(SimpleNamespace(**config))
    layers = config["num_hidden_layers"]
    result: dict[str, object] = {
        "format": "activation-baselines-v1",
        "policy": args.policy,
        "sparsity": args.sparsity,
        "model_revision": args.model.name,
        "calibration": {
            "states": str(args.states),
            "fit_documents": len(fit),
            "max_states_per_layer": args.max_states,
        },
        "thresholds": [],
    }
    thresholds: list[dict[str, object]] = []
    opened: dict[str, Any] = {}

    def weight(key: str) -> Tensor:
        name = index[key]
        if name not in opened:
            opened[name] = safe_open(args.model / name, framework="pt", device="cpu")
        return opened[name].get_tensor(key)

    def expert_weight(base: str, projection: str) -> Tensor:
        return weight(f"{base}.{projection}.weight")

    for layer in range(layers):
        states = torch.load(
            args.states / f"layer-{layer:02d}.pt", map_location="cpu", weights_only=True
        )
        fit_positions = (
            torch.isin(states["document"].long(), torch.tensor(fit)).nonzero().flatten()
        )
        if len(fit_positions) > args.max_states:
            selected = torch.linspace(0, len(fit_positions) - 1, args.max_states).long()
            fit_positions = fit_positions[selected]
        x = states["states"][fit_positions].cuda()
        if x.numel() == 0:
            raise ValueError(f"Layer {layer} has no FIT states")
        qwen35 = config["architecture"] == "Qwen3_5MoeForConditionalGeneration"
        prefix = (
            f"model.language_model.layers.{layer}"
            if gemma
            else f"model.language_model.layers.{layer}.mlp"
            if qwen35
            else f"model.layers.{layer}.mlp"
        )
        if gemma:
            route = states["route_ids"][fit_positions].cuda().long()
            route_weights = states["route_weights"][fit_positions].cuda()
        else:
            gate = weight(f"{prefix}.gate.weight").cuda()
            route_weights, route = (
                (x @ gate.T)
                .float()
                .softmax(-1)
                .topk(config["num_experts_per_tok"], dim=-1)
            )
            if config.get("norm_topk_prob", False):
                route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)
        packed = f"{prefix}.experts.gate_up_proj"
        if packed in index:
            gu = weight(packed)
            gate_rows, up_rows = gu.chunk(2, dim=1)
        else:
            gate_rows, up_rows = [], []
            for expert in range(config["num_experts"]):
                base = f"{prefix}.experts.{expert}"
                gate_rows.append(expert_weight(base, "gate_proj"))
                up_rows.append(expert_weight(base, "up_proj"))
            gate_rows, up_rows = torch.stack(gate_rows), torch.stack(up_rows)
        gate_rows, up_rows = gate_rows.cuda(), up_rows.cuda()
        activations, inputs = [], []
        for start in range(0, len(x), 16):
            batch = x[start : start + 16]
            ids = route[start : start + 16]
            gate_out = torch.einsum("bh,bkih->bki", batch, gate_rows[ids])
            activations.append(activate(gate_out, activation).bfloat16().cpu())
            if args.policy == "teal":
                inputs.append(batch.cpu())
        a = torch.cat(activations)
        row: dict[str, object] = {}
        if args.policy == "cats":
            row["threshold"] = empirical_threshold(a.abs(), args.sparsity)
        elif args.policy in {"teal-greedy", "wina", "wina-greedy-topk"}:
            down_rows = (
                weight(f"{prefix}.experts.down_proj")
                if f"{prefix}.experts.down_proj" in index
                else torch.stack(
                    [
                        expert_weight(f"{prefix}.experts.{expert}", "down_proj")
                        for expert in range(config["num_experts"])
                    ]
                )
            ).cuda()
            # Uniform WINA follows WINA's Sec. 3.2; greedy TEAL follows the official
            # block-wise search but measures MLP output error only.
            if args.policy == "wina":
                norms = (
                    gate_rows.float().norm(dim=1).bfloat16(),
                    up_rows.float().norm(dim=1).bfloat16(),
                    down_rows.float().norm(dim=1).bfloat16(),
                )
                for key, score in zip(
                    ("gate_threshold", "input_threshold"),
                    (
                        (x.float()[:, None, :].abs() * norm[route].float()).cpu()
                        for norm in norms[:2]
                    ),
                    strict=True,
                ):
                    row[key] = empirical_threshold(score, args.sparsity)
                # Fit the down threshold from dense gated activations.
                values = []
                for start in range(0, len(x), 8):
                    batch, ids = x[start : start + 8], route[start : start + 8]
                    g = torch.einsum("bh,bkih->bki", batch, gate_rows[ids]).bfloat16()
                    u = torch.einsum("bh,bkih->bki", batch, up_rows[ids]).bfloat16()
                    values.append(
                        (
                            (activate(g, activation).bfloat16() * u).float().abs()
                            * norms[2][ids].float()
                        ).cpu()
                    )
                row["down_threshold"] = empirical_threshold(
                    torch.cat(values), args.sparsity
                )
            else:
                row.update(
                    greedy_mlp_thresholds(
                        x,
                        route,
                        gate_rows,
                        up_rows,
                        down_rows,
                        route_weights,
                        args.sparsity,
                        weighted=args.policy == "wina-greedy-topk",
                        activation=activation,
                        return_levels=args.policy == "wina-greedy-topk",
                    )
                )
        else:
            # TEAL: an input threshold on |x|, then a down threshold on the
            # intermediate features of the input-sparsified token.
            row["input_threshold"] = empirical_threshold(
                torch.cat(inputs).abs(), args.sparsity
            )
            masked = x.cpu().clone()
            masked[masked.abs() <= row["input_threshold"]] = 0
            intermediate = []
            for start in range(0, len(masked), 16):
                batch = masked[start : start + 16].cuda()
                ids = route[start : start + 16]
                g = torch.einsum("bh,bkih->bki", batch, gate_rows[ids]).bfloat16()
                u = torch.einsum("bh,bkih->bki", batch, up_rows[ids]).bfloat16()
                intermediate.append((activate(g, activation).bfloat16() * u).cpu())
            row["down_threshold"] = empirical_threshold(
                torch.cat(intermediate).abs(), args.sparsity
            )
        thresholds.append(row)
        label = "sparsities" if args.policy == "wina-greedy-topk" else "threshold"
        summary = (
            tuple(row[key] for key in ("gate_sparsity", "up_sparsity", "down_sparsity"))
            if args.policy == "wina-greedy-topk"
            else row.get("threshold", row.get("input_threshold"))
        )
        print(
            f"layer={layer:02d} fit_states={len(x)} {label}={summary}",
            flush=True,
        )
    result["thresholds"] = thresholds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
