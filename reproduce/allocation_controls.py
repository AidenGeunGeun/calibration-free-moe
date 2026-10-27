"""Print the Section 4.1-4.2 channel-count controls and the Appendix E tables.

Inputs: allocation-study ``deep/``, ``reference/`` and ``sweep/`` directories (records
need torch, CPU is enough); ``--pair`` takes a policy and a base generation run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reproduce.analysis_common import mean_se, paired_difference, spearman
from reproduce.common import load_json, read_percent
from scripts.allocation_study import record_tokens, report_tokens

DEPTHS = ("d00000", "d02000", "d06000")
ARMS = ("uniform", "shuffled-s1", "shuffled-s2", "shuffled-s3", "constant", "inverse")
RUNS = ("b13-chat", "b18-chat", "b13-trace", "b18-trace")

# Per-trace KL, reads (percent) and read cost in bytes of one arm in one window.
Arm = tuple[dict[str, float], float, float]


def load_arm(path: Path) -> Arm | None:
    """Load one arm's window, or None if it is missing, unreadable or incomplete."""
    try:
        scores = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not scores.get("complete") or "baseline_work" not in scores:
        return None
    work = scores["baseline_work"]
    kl = {d["id"].rsplit("/", 1)[0]: d["mean_kl"] for d in scores["documents"]}
    return kl, read_percent(scores), work["charged_bytes"]


def paired(arm: dict[str, float], base: dict[str, float]) -> tuple[float, float]:
    """Return the mean arm-minus-base difference over traces and its SE."""
    return mean_se([arm[key] - base[key] for key in base])


def relative(arm: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    """Arm minus base over the base mean, in percent, with one paired SE."""
    mean, se = paired(arm, base)
    reference = sum(base.values()) / len(base)
    return {"change": 100 * mean / reference, "se": 100 * se / reference}


def pooled_total(parts: list[Arm]) -> tuple[dict[str, float], float]:
    """Per-trace KL summed over the depth windows, and the total read cost in bytes."""
    kl = {key: sum(part[0][key] for part in parts) for key in parts[0][0]}
    return kl, sum(part[2] for part in parts)


def deep_windows(deep: Path) -> dict[str, Any]:
    """Compare each arm with the threshold per window and pooled over depths.

    Windows sit at answer positions 0, 2,000 and 6,000 of the model's own generated
    states. Per window, an arm's per-trace mean KL is compared with the threshold's
    (``relative``). Pooled, each trace's KL is summed over the three windows; the
    per-layer fixed counts (``uniform``, the per-depth matched decay) and the
    ``constant`` and ``inverse`` schedules are compared with the threshold, and the two
    schedules with the decay, each with its read-cost ratio.
    """
    out: dict[str, Any] = {}
    for run in RUNS:
        pooled: dict[str, list[Arm]] = {}
        depths: dict[str, Any] = {}
        for depth in DEPTHS:
            base = load_arm(deep / run / depth / "threshold" / "scores.json")
            if base is None:
                continue
            pooled.setdefault("threshold", []).append(base)
            row: dict[str, Any] = {
                "threshold_kl": sum(base[0].values()) / len(base[0]),
                "threshold_reads": base[1],
            }
            for name in ARMS:
                arm = load_arm(deep / run / depth / name / "scores.json")
                if arm is None:
                    continue
                pooled.setdefault(name, []).append(arm)
                row[name] = relative(arm[0], base[0]) | {"reads": arm[1]}
            depths[depth] = row
        if not depths:
            continue
        out[run] = {"depths": depths}
        if len(pooled.get("threshold", [])) != len(DEPTHS):
            continue

        threshold_kl, threshold_bytes = pooled_total(pooled["threshold"])
        summary: dict[str, Any] = {}
        for name in ("uniform", "constant", "inverse"):
            if len(pooled.get(name, [])) == len(DEPTHS):
                kl, charged = pooled_total(pooled[name])
                summary[name] = relative(kl, threshold_kl) | {
                    "bytes_ratio": charged / threshold_bytes
                }
        if len(pooled.get("constant", [])) == len(DEPTHS):
            decay_kl, decay_bytes = pooled_total(pooled["uniform"])
            for name in ("constant", "inverse"):
                kl, charged = pooled_total(pooled[name])
                summary[f"{name}_vs_decay"] = relative(kl, decay_kl) | {
                    "bytes_ratio": charged / decay_bytes
                }
        out[run]["pooled"] = summary
    return out


def sweep_correlations(sweep: Path) -> dict[str, Any]:
    """Correlate the threshold's per-token count with need, loss and depth (Spearman).

    The count is summed over layers. Need is the KL at half the top-k count minus the
    KL at twice it, after one sparse step; loss is the full model's next-token loss.
    """
    stock: dict[str, list[float]] = {}
    for path in sorted((sweep / "reference").glob("d*.scores.json")):
        for document in json.loads(path.read_text())["documents"]:
            stock[document["id"]] = document["logprobs"]

    def arm(budget: str, name: str) -> dict[Any, dict[str, Any]]:
        tokens: dict[Any, dict[str, Any]] = {}
        for path in sorted((sweep / budget / name).glob("d*.json")):
            tokens |= report_tokens(path)
        return tokens

    out: dict[str, Any] = {}
    for budget in ("b13", "b18"):
        threshold = arm(budget, "threshold")
        low, high = arm(budget, "m0.5"), arm(budget, "m2")
        record = record_tokens(sweep / budget / "threshold" / "rec")
        keys = sorted(threshold)
        count = [float(record[k]["counts"].sum()) for k in keys]
        need = [low[k]["kl"] - high[k]["kl"] for k in keys]
        loss = [-stock[threshold[k]["id"]][k[1]] for k in keys]
        depth = [float(threshold[k]["depth"]) for k in keys]
        out[budget] = {
            "tokens": len(keys),
            "count~need": spearman(count, need),
            "count~loss": spearman(count, loss),
            "count~depth": spearman(count, depth),
            "need~loss": spearman(need, loss),
            "need~depth": spearman(need, depth),
        }
    return out


def own_state_correlations(deep: Path, reference: Path) -> dict[str, Any]:
    """Correlate the threshold's per-token count with loss in each depth window.

    Loss is the full model's next-token loss on the sparse run's own states.
    """
    out: dict[str, Any] = {}
    for depth in DEPTHS:
        stock = {
            d["id"]: d["logprobs"]
            for d in json.loads((reference / f"{depth}.scores.json").read_text())[
                "documents"
            ]
        }
        window = deep / "b13-chat" / depth / "threshold"
        threshold = report_tokens(window / "scores.json")
        record = record_tokens(window / "rec")
        keys = sorted(threshold)
        count = [float(record[k]["counts"].sum()) for k in keys]
        loss = [-stock[threshold[k]["id"]][k[1]] for k in keys]
        out[depth] = {"tokens": len(keys), "count~loss": spearman(count, loss)}
    return out


def generation_run(directory: Path) -> dict[str, float]:
    """Summarize a generation run: accuracy, reads, length-limited share, length."""
    summary = load_json(directory / "metrics.json")["summary"]
    scores = load_json(directory / "scores.json")
    return {
        "accuracy": 100 * summary["accuracy"],
        "reads": read_percent(scores),
        "capped": 100 * summary["truncation_rate"],
        "length": summary["mean_generated_tokens"],
    }


def main() -> None:
    """Print the requested window, correlation and generation comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--references",
        type=Path,
        nargs="*",
        default=[],
        help="Stock reference directory of each --deep directory, "
        "for loss correlations",
    )
    parser.add_argument("--sweep", type=Path, help="One-step sweep directory")
    parser.add_argument(
        "--pair", type=Path, nargs=2, action="append", default=[], metavar="RUN"
    )
    parser.add_argument(
        "--runs", type=Path, nargs="*", default=[], help="Unpaired generation runs"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.references and len(args.references) != len(args.deep):
        parser.error("--references needs one directory per --deep directory")
    result: dict[str, Any] = {}
    for deep in args.deep:
        windows = deep_windows(deep)
        result[f"deep/{deep}"] = windows
        print(f"Own-trajectory windows, {deep}")
        for run, entry in windows.items():
            for depth, row in entry["depths"].items():
                arms = "  ".join(
                    f"{name} {row[name]['change']:+.1f}+-{row[name]['se']:.1f}"
                    for name in ARMS
                    if name in row
                )
                print(
                    f"  {run} {depth} threshold KL {row['threshold_kl']:.4f} "
                    f"reads {row['threshold_reads']:.2f}  {arms}"
                )
            if "pooled" in entry:
                pooled = "  ".join(
                    f"{name} {row['change']:+.1f}+-{row['se']:.1f}%"
                    f" (bytes {row['bytes_ratio']:.4f})"
                    for name, row in entry["pooled"].items()
                )
                print(f"  {run} pooled  {pooled}")
    if args.sweep is not None:
        sweep = sweep_correlations(args.sweep)
        result["sweep"] = sweep
        for budget, row in sweep.items():
            values = "  ".join(
                f"{name} {value:+.2f}" for name, value in row.items() if "~" in name
            )
            print(f"One-step sweep {budget} (n={row['tokens']}): {values}")
    for deep, reference in zip(args.deep, args.references, strict=False):
        own = own_state_correlations(deep, reference)
        result[f"own-state/{deep}"] = own
        values = "  ".join(
            f"{depth} {row['count~loss']:+.2f} (n={row['tokens']})"
            for depth, row in own.items()
        )
        print(f"Own-state count~loss, {deep}: {values}")
    losses = [row["count~loss"] for row in result.get("sweep", {}).values()] + [
        row["count~loss"]
        for key, own in result.items()
        if key.startswith("own-state/")
        for row in own.values()
    ]
    if losses:
        print(f"count~loss range: {min(losses):+.2f} to {max(losses):+.2f}")
    for path in args.runs:
        run = generation_run(path)
        result[f"run/{path}"] = run
        print(
            f"{path}: accuracy {run['accuracy']:.2f} reads {run['reads']:.2f}% "
            f"capped {run['capped']:.1f}% length {run['length']:.0f}"
        )
    for policy, base in args.pair:
        runs = {str(path): generation_run(path) for path in (policy, base)}
        pair = paired_difference(policy / "metrics.json", base / "metrics.json")
        result[f"pair/{policy}|{base}"] = {"runs": runs, "paired": pair}
        for path, run in runs.items():
            print(
                f"{path}: accuracy {run['accuracy']:.2f} reads {run['reads']:.2f}% "
                f"capped {run['capped']:.1f}% length {run['length']:.0f}"
            )
        print(
            f"  paired {pair['diff']:+.2f} +- {pair['se']:.2f} "
            f"over {pair['questions']} questions"
        )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
