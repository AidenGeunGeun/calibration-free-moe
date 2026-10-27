"""Write per-layer channel counts that spend a uniform read budget where it helps.

`weight-curves` (synthetic inputs, no data) and `calibrated-curves` (KL on FIT
documents) write each layer's error against its channel count; `solve` turns those
curves into per-layer counts with the same total read cost as uniform `--retention`.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import socket
import sys
import time
from bisect import bisect_right
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from calibration_free_moe.model_config import (
    checkpoint_identity,
    model_dimensions,
)
from calibration_free_moe.reader.budget import BudgetPlan, compile_budget

DEFAULT_GRID = [
    0.03,
    0.045,
    0.06,
    0.075,
    0.09,
    0.105,
    0.125,
    0.15,
    0.18,
    0.22,
    0.27,
    0.33,
]


def convex_envelope(curve: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Return the lower convex envelope of a measured (count, error) curve."""
    points = sorted(curve)
    if (
        len(points) < 2
        or any(type(x) is not int or x < 1 or not math.isfinite(y) for x, y in points)
        or len({x for x, _ in points}) != len(points)
    ):
        raise ValueError(
            "Curves need distinct positive integer counts and finite errors"
        )
    hull: list[tuple[int, float]] = []
    for point in points:
        while len(hull) >= 2:
            a, b = hull[-2:]
            if (b[1] - a[1]) * (point[0] - b[0]) < (point[1] - b[1]) * (b[0] - a[0]):
                break
            hull.pop()
        hull.append(point)
    return hull


def interpolate(curve: list[tuple[int, float]], count: int) -> float:
    """Return the curve's linearly interpolated error at a channel count."""
    if not curve[0][0] <= count <= curve[-1][0]:
        raise ValueError("Requested count is outside the measured grid")
    j = min(bisect_right([x for x, _ in curve], count), len(curve) - 1)
    a, b = curve[j - 1], curve[j]
    return a[1] + (b[1] - a[1]) * (count - a[0]) / (b[0] - a[0])


def solve_allocation(
    curves: list[list[tuple[int, float]]], total: int, granularity: int = 16
) -> list[int]:
    """Return per-layer counts that minimize summed envelope error at an exact total.

    Counts are multiples of the granularity g, except in one layer when
    total % g != 0. For each choice of that remainder layer, greedy g-channel
    increments in order of marginal gain solve the convex separable integer problem
    exactly. The envelope is an optimistic surrogate, not the measured nonconvex
    curve.
    """
    if (
        not curves
        or type(total) is not int
        or type(granularity) is not int
        or granularity < 1
    ):
        raise ValueError("Need curves, an integer total and positive granularity")
    hulls = [convex_envelope(curve) for curve in curves]
    g = granularity
    remainder = total % g
    best: tuple[float, list[int]] | None = None

    def marginal(layer: int, f: int) -> float:
        # Error change of the next g channels; negative while the curve falls.
        return interpolate(hulls[layer], f + g) - interpolate(hulls[layer], f)

    for remainder_layer in range(len(curves)) if remainder else [-1]:
        counts: list[int] = []
        upper: list[int] = []
        for layer, hull in enumerate(hulls):
            offset = remainder if layer == remainder_layer else 0
            counts.append(-(-(hull[0][0] - offset) // g) * g + offset)
            upper.append((hull[-1][0] - offset) // g * g + offset)
        # counts and upper get one entry per layer, so strict zips cannot fail.
        if any(a > b for a, b in zip(counts, upper, strict=True)) or not sum(
            counts
        ) <= total <= sum(upper):
            continue
        heap = [
            (marginal(layer, f), layer)
            for layer, f in enumerate(counts)
            if f + g <= upper[layer]
        ]
        heapq.heapify(heap)
        for _ in range((total - sum(counts)) // g):
            _, layer = heapq.heappop(heap)
            counts[layer] += g
            if counts[layer] + g <= upper[layer]:
                heapq.heappush(heap, (marginal(layer, counts[layer]), layer))
        objective = sum(
            interpolate(hull, f) for hull, f in zip(hulls, counts, strict=True)
        )
        if best is None or objective < best[0]:
            best = objective, counts
    if best is None:
        raise ValueError("Exact total is infeasible within grid bounds and granularity")
    return best[1]


def make_plan(
    config: dict[str, Any], retention: float, screen_keep: float
) -> BudgetPlan:
    """Compile the read plan of one layer from saved model dimensions."""
    return compile_budget(
        hidden_size=config["hidden_size"],
        expert_width=config["expert_width"],
        top_k=config["num_experts_per_tok"],
        retention=retention,
        screen_keep=screen_keep,
    )


def read_cost_bytes(plan: BudgetPlan, rank: int = 32) -> int:
    """Return one layer's read cost in bytes per token at batch one, `point` score.

    Counts BF16 native weights, the sketch without residuals, the selection norms,
    the rank-`rank` recovery map and the int8 block-Hadamard signs (one byte per
    hidden coordinate).
    """
    return (
        plan.selected_weight_bytes()
        + plan.sketch_read_bytes(residual=False)
        + plan.selection_read_bytes(geometry=False)
        + plan.recovery_read_bytes(rank)
        + plan.hidden_size
    )


def solve(args: argparse.Namespace) -> None:
    """Merge layer curve shards and write the equal-read allocation (`solve`).

    The allocation must have exactly the uniform plan's total read cost; the
    predicted objectives are envelope estimates, not measured joint-layer errors.
    """
    reports = [json.loads(path.read_text()) for path in args.curves]
    first = reports[0]
    rows: dict[int, dict[str, Any]] = {}
    for report in reports:
        if not report["complete"] or any(
            report[key] != first[key]
            for key in ("model", "dimensions", "source", "recipe", "grid_counts")
        ):
            raise ValueError("Incomplete or incompatible curve shards")
        for row in report["curves"]:
            if row["layer"] in rows:
                raise ValueError("Overlapping layer shards")
            rows[row["layer"]] = row
    config = first["dimensions"]
    layers = config["num_hidden_layers"]
    if sorted(rows) != list(range(layers)):
        raise ValueError("Curves must cover every MoE layer exactly once")
    uniform = make_plan(config, args.retention, first["recipe"]["screen_keep"])
    # Not strict: the grid-length check right below reports a short curve.
    curves = [
        list(zip(first["grid_counts"], rows[l]["errors"], strict=False))
        for l in range(layers)
    ]
    if any(len(row["errors"]) != len(first["grid_counts"]) for row in rows.values()):
        raise ValueError("Incomplete grid")
    counts = solve_allocation(curves, layers * uniform.final_count, args.granularity)
    allocated = [
        make_plan(config, f / uniform.candidate_count, first["recipe"]["screen_keep"])
        for f in counts
    ]
    uniform_bytes = layers * read_cost_bytes(uniform)
    allocated_bytes = sum(read_cost_bytes(plan) for plan in allocated)
    if allocated_bytes != uniform_bytes:
        raise ValueError("Allocation does not match total read cost")
    hulls = [convex_envelope(curve) for curve in curves]
    result = {
        "format": "layer-retention-v1",
        "source_curves": [str(p) for p in args.curves],
        "model": first["model"],
        "dimensions": config,
        "uniform_retention": args.retention,
        "uniform_final_count": uniform.final_count,
        "final_counts": counts,
        "retentions": [p.final_count / p.candidate_count for p in allocated],
        "granularity": args.granularity,
        "screen_keep": first["recipe"]["screen_keep"],
        "selection_score": "point",
        "bank": first["recipe"]["bank"],
        "total_read_cost_bytes": allocated_bytes,
        "uniform_read_cost_bytes": uniform_bytes,
        "predicted_objective": {
            "uniform": sum(interpolate(h, uniform.final_count) for h in hulls),
            "allocated": sum(
                interpolate(h, f) for h, f in zip(hulls, counts, strict=True)
            ),
            "basis": "lower convex envelope; not a measured joint-layer error",
            "raw_interpolated_uniform": sum(
                interpolate(c, uniform.final_count) for c in curves
            ),
            "raw_interpolated_allocated": sum(
                interpolate(c, f) for c, f in zip(curves, counts, strict=True)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def weight_curves(
    args: argparse.Namespace, report: dict[str, Any], bank: dict[str, Any]
) -> None:
    """Measure data-free per-layer error curves on synthetic inputs (`weight-curves`).

    Each layer sees `--tokens` N(0, I) hidden states scaled by its expert-input norm
    weight and routed by the frozen router. The error at each grid count is the
    squared error of the reader's routed output relative to the dense output's
    energy. Rows are appended to `report` and saved after every layer.
    """
    import torch
    from safetensors import safe_open
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter

    from calibration_free_moe.activation import model_activation
    from calibration_free_moe.reader.progressive import ProgressiveReader
    from calibration_free_moe.reader.sketch import block_hadamard_signs, build_sketch
    from scripts.fit_plan_recovery import load_layer_weights, route

    config = report["dimensions"]
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("--device needs an explicit CUDA index")
    torch.cuda.set_device(device)
    report["devices"] = [str(device)]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(key: str) -> torch.Tensor:
        with safe_open(args.model / index[key], framework="pt") as handle:
            return handle.get_tensor(key).to(device)

    gemma = config["architecture"].startswith("Gemma4")
    activation = model_activation(SimpleNamespace(**config))
    report["recipe"]["synthetic_input"] = (
        "N(0,I) (unit expected RMS), seed+layer; "
        "multiply the expert-input norm weight directly. "
        "Gemma4 routes the unscaled Gaussian through its own "
        "norm/scale/proj/per-expert scale."
    )
    with torch.no_grad():
        for layer in args.layers:
            started = time.perf_counter()
            gu, down, gate = load_layer_weights(args.model, layer, str(device))
            prefix = (
                f"model.language_model.layers.{layer}"
                if gemma
                else f"model.layers.{layer}"
            )
            norm = (
                "pre_feedforward_layernorm_2" if gemma else "post_attention_layernorm"
            )
            gamma = read(f"{prefix}.{norm}.weight").float()
            generator = torch.Generator(device=device).manual_seed(args.seed + layer)
            z = torch.randn(
                args.tokens, config["hidden_size"], device=device, generator=generator
            )
            x = (z * gamma).bfloat16().contiguous()
            if gemma:
                router = Gemma4TextRouter(Gemma4TextConfig(**config)).to(
                    device=device, dtype=torch.bfloat16
                )
                router.load_state_dict(
                    {key: read(f"{prefix}.router.{key}") for key in router.state_dict()}
                )
                _, weights, ids = router(z.bfloat16())
                ids, weights = ids.contiguous(), weights.contiguous()
                del router
            else:
                ids, weights = route(
                    x,
                    gate,
                    config["num_experts_per_tok"],
                    bool(config.get("norm_topk_prob", False)),
                )
            sketch = build_sketch(
                gu, signs=block_hadamard_signs(config["hidden_size"], device)
            )
            dense_reader = ProgressiveReader(
                gu,
                down,
                None,
                None,
                make_plan(config, 1.0, 1.0),
                args.batch_size,
                activation=activation,
                sketch=sketch,
            )
            dense = torch.cat(
                [
                    dense_reader(
                        x[j : j + args.batch_size],
                        ids[j : j + args.batch_size],
                        weights[j : j + args.batch_size],
                    )
                    for j in range(0, args.tokens, args.batch_size)
                ]
            ).float()
            del dense_reader
            denominator = dense.square().sum().item()
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("Dense routed output has zero or non-finite energy")
            u, b = bank["decoder"][layer].to(device), bank["encoder"][layer].to(device)
            row: dict[str, Any] = {"layer": layer, "errors": [], "point_seconds": []}
            for f in report["grid_counts"]:
                point_start = time.perf_counter()
                reader = ProgressiveReader(
                    gu,
                    down,
                    b,
                    u,
                    make_plan(
                        config,
                        f / (config["num_experts_per_tok"] * config["expert_width"]),
                        args.screen_keep,
                    ),
                    args.batch_size,
                    sketch=sketch,
                    activation=activation,
                    selection_score="point",
                )
                squared = torch.zeros((), device=device)
                for j in range(0, args.tokens, args.batch_size):
                    output = reader(
                        x[j : j + args.batch_size],
                        ids[j : j + args.batch_size],
                        weights[j : j + args.batch_size],
                    )
                    squared += (
                        (output.float() - dense[j : j + args.batch_size]).square().sum()
                    )
                error = squared.item() / denominator
                if not math.isfinite(error):
                    raise ValueError("Non-finite layer error")
                row["errors"].append(error)
                row["point_seconds"].append(time.perf_counter() - point_start)
                del reader
            row["seconds"] = time.perf_counter() - started
            report["curves"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"layer {layer}: {row['seconds']:.1f}s errors={row['errors']}",
                flush=True,
            )
            del gu, down, gate, sketch, u, b, dense, x, z
    report["peak_memory_bytes"] = {str(device): torch.cuda.max_memory_allocated(device)}


def calibrated_curves(
    args: argparse.Namespace, report: dict[str, Any], bank: dict[str, Any]
) -> None:
    """Measure per-layer KL curves on FIT documents (`calibrated-curves`).

    Only the measured layer is sparse; every other layer and the prefix through
    the panel's context length stay stock. The error at each grid count is the mean
    document KL(stock || sparse) over the first `--documents` FIT documents, in
    panel order. The partial curve is saved after every grid point.
    """
    import torch

    from scripts.fit_kl_bank import Calibration, load_calibration_model

    config = report["dimensions"]
    devices = [torch.device(value.strip()) for value in args.devices.split(",")]
    if any(d.type != "cuda" or d.index is None for d in devices):
        raise ValueError("--devices needs explicit CUDA indices")
    report["devices"] = [str(device) for device in devices]
    panel = json.loads(args.panel.read_text())
    fit = [d for d in panel["documents"] if d["split_role"].lower() == "fit"]
    if len(fit) < args.documents:
        raise ValueError(
            "Insufficient FIT documents; "
            "TUNE/evaluation documents are never substituted"
        )
    # Fixed panel order, identical across layer/grid points and layer shards.
    documents = fit[: args.documents]
    report["recipe"].update(
        panel=str(args.panel),
        fit_document_ids=[d["id"] for d in documents],
        window=args.window,
        prefix="stock through context_length",
        objective="mean document KL(stock || sparse)",
    )
    load_start = time.perf_counter()
    model = load_calibration_model(args.model, devices)
    cal = Calibration(
        model,
        make_plan(config, args.grid[0], args.screen_keep),
        args.window,
        args.model,
        devices=devices,
        sparse_layers=set(),
        reader_output=True,
        selection_score="point",
    )
    cal.load_bank(bank["decoder"].float(), bank["encoder"].float())
    report["load_seconds"] = time.perf_counter() - load_start
    with torch.no_grad():
        for layer in args.layers:
            started = time.perf_counter()
            cal.sparse_layers = {layer}
            # Only one layer's sketch is needed; teacher probabilities persist.
            cal.sketches.clear()
            row: dict[str, Any] = {"layer": layer, "errors": [], "point_seconds": []}
            for f in report["grid_counts"]:
                point_start = time.perf_counter()
                cal.plan = make_plan(
                    config,
                    f / (config["num_experts_per_tok"] * config["expert_width"]),
                    args.screen_keep,
                )
                cal.refresh()
                error = sum(cal.kl(doc).item() for doc in documents) / len(documents)
                if not math.isfinite(error):
                    raise ValueError("Non-finite calibration KL")
                row["errors"].append(error)
                row["point_seconds"].append(time.perf_counter() - point_start)
                print(
                    f"layer {layer} F={f}: KL={error:.6g} "
                    f"{row['point_seconds'][-1]:.1f}s",
                    flush=True,
                )
                # Save after every grid point; `solve` rejects an unfinished file.
                args.output.write_text(
                    json.dumps(report | {"current_layer": row}, indent=2) + "\n"
                )
            row["seconds"] = time.perf_counter() - started
            report["curves"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["peak_memory_bytes"] = {
        str(device): torch.cuda.max_memory_allocated(device) for device in devices
    }


def main() -> None:
    """Parse the command line, check the checkpoint and bank, and run a subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solver = commands.add_parser("solve")
    solver.add_argument("--curves", type=Path, nargs="+", required=True)
    solver.add_argument("--retention", type=float, required=True)
    solver.add_argument("--granularity", type=int, default=16)
    solver.add_argument("--output", type=Path, required=True)
    for name in ("weight-curves", "calibrated-curves"):
        sub = commands.add_parser(name)
        for key in ("model", "bank", "output"):
            sub.add_argument(f"--{key}", type=Path, required=True)
        sub.add_argument("--grid", nargs="+", type=float, default=DEFAULT_GRID)
        sub.add_argument("--screen-keep", type=float, default=0.625)
        sub.add_argument("--seed", type=int, default=353)
        sub.add_argument("--layers", nargs="+", type=int)
        sub.add_argument("--source-revision", required=True)
        if name == "weight-curves":
            sub.add_argument("--tokens", type=int, default=4096)
            sub.add_argument("--batch-size", type=int, default=16)
            sub.add_argument("--device", default="cuda:0")
        else:
            sub.add_argument("--panel", type=Path, required=True)
            sub.add_argument("--documents", type=int, default=64)
            sub.add_argument("--window", type=int, default=64)
            sub.add_argument("--devices", default="cuda:0")
    args = parser.parse_args()
    if args.command == "solve":
        solve(args)
        return
    import torch

    from calibration_free_moe.reader.sketch import require_sketch_acquisition

    config = model_dimensions(args.model)
    if config.get("quantization_config") or config["architecture"] not in {
        "OlmoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
    }:
        parser.error(
            "Layer allocation curves currently cover BF16 OLMoE, Qwen3 and Gemma4"
        )
    if config["architecture"].startswith("Gemma4") and not config.get(
        "enable_moe_block"
    ):
        parser.error("Gemma must have routed experts in every layer")
    args.layers = sorted(
        args.layers if args.layers is not None else range(config["num_hidden_layers"])
    )
    if (
        not args.layers
        or len(set(args.layers)) != len(args.layers)
        or any(l not in range(config["num_hidden_layers"]) for l in args.layers)
    ):
        parser.error("--layers must be distinct model layer indices")
    if args.command == "weight-curves" and min(args.tokens, args.batch_size) < 1:
        parser.error("Positive token and batch counts required")
    if args.command == "calibrated-curves" and min(args.documents, args.window) < 1:
        parser.error("Positive document count and window required")
    plans = [make_plan(config, r, args.screen_keep) for r in sorted(args.grid)]
    grid = [p.final_count for p in plans]
    if (
        len(grid) < 2
        or len(set(grid)) != len(grid)
        or any(p.final_count == p.candidate_count for p in plans)
    ):
        parser.error("Grid needs distinct screened channel counts")
    identity = checkpoint_identity(args.model)
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    if bank["format"] != "weight-subspace-recovery-v1" or bank["model"] != identity:
        raise ValueError("Need the model's data-free weight-only recovery bank")
    require_sketch_acquisition(bank)
    expected = (
        config["num_hidden_layers"],
        config["num_experts"],
        32,
        config["expert_width"],
    )
    if bank["encoder"].shape != expected or bank["decoder"].shape != (
        expected[0],
        config["hidden_size"],
        32,
    ):
        raise ValueError("Bank shape does not match model")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    report: dict[str, Any] = {
        "format": "layer-error-curves-v1",
        "complete": False,
        "source": args.command,
        "model": identity,
        "dimensions": config,
        "grid_counts": grid,
        "grid_retentions": [p.final_count / p.candidate_count for p in plans],
        "recipe": {
            "screen_keep": args.screen_keep,
            "selection_score": "point",
            "bank": str(args.bank),
            "seed": args.seed,
        },
        "curves": [],
        "command": sys.argv,
        "source_revision": args.source_revision,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "pid": os.getpid(),
        "torch": torch.__version__,
    }
    if args.command == "weight-curves":
        report["recipe"].update(tokens=args.tokens, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    if args.command == "weight-curves":
        weight_curves(args, report, bank)
    else:
        calibrated_curves(args, report, bank)
    report.update(complete=True, seconds=time.perf_counter() - started)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
