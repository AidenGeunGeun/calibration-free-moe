"""Print the Section 5 score, keep-rule and component results and their Appendix tables.

Inputs: the runs root's ``fidelity``, ``component-arms-olmoe/arms``, ``energy-kl`` and
``standard-protocols/arms`` directories, passed to the four flags in that order.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import pairwise
from pathlib import Path
from typing import Any

from reproduce.analysis_common import paired_difference, spearman
from reproduce.common import load_json, read_percent

PLANS = ("p05", "p10", "p20", "p30")
# Top-k Prox keeps the top 30% of inputs at 13% and 18% of reads, and 50% above.
PROX_FREE = {"p05": "k30", "p10": "k30", "p20": "k50", "p30": "k50"}
CHAT_MODELS = ("gemma4", "qwen3-thinking", "qwen35")
# Energy arms: held-out run prefix -> (screen, score) in screen_bits.py output.
ENERGY_ARMS = {
    "core": ("ternary-rot", "plain"),
    "hire-gate": ("int4-row-gate", "point"),
    "prox-free-k30": ("int4-row-x30", "prox"),
    "prox-free-k50": ("int4-row-x50", "prox"),
}


def fidelity(directory: Path) -> dict[str, Any]:
    """Load a KL run: token micro-mean KL, read fraction and per-document KL."""
    scores = load_json(directory / "scores.json")
    if not scores["complete"]:
        raise ValueError(f"{directory} is incomplete")
    return {
        "kl": scores["fidelity"]["micro_mean_kl"],
        "reads": read_percent(scores) / 100,
        "documents": {doc["id"]: doc["kl"] for doc in scores["documents"]},
        "command": scores["command"],
    }


def is_calibrated_prox(name: str) -> bool:
    """Calibrated-threshold Prox runs: prox-r*, prox-s*-r*, prox-read-*."""
    if name.endswith("-keepall") or name.startswith(("prox-free", "prox-topk")):
        return False
    return name.startswith(("prox-r", "prox-s", "prox-read-"))


def lower_envelope(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points not dominated by another with no more reads and no more KL."""
    front: list[tuple[float, float]] = []
    for reads, kl in sorted(points):
        if not front or kl < front[-1][1]:
            front.append((reads, kl))
    return front


def log_interpolate(
    curve: list[tuple[float, float]], reads: float
) -> tuple[float, bool]:
    """Interpolate KL log-linearly at ``reads``; also return whether it extrapolated."""
    for (x0, y0), (x1, y1) in pairwise(curve):
        if x0 <= reads <= x1:
            t = (reads - x0) / (x1 - x0)
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0))), False
    (x0, y0), (x1, y1) = curve[:2] if reads < curve[0][0] else curve[-2:]
    t = (reads - x0) / (x1 - x0)
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0))), True


def keep_rule(model_dir: Path, plans: dict[str, str]) -> dict[str, Any]:
    """Compare top-k Prox (budget -> run) with the calibrated Prox curve of one model.

    The curve is the lower envelope of every calibrated-threshold Prox run of the
    model. The calibrated KL at the top-k run's reads is interpolated log-linearly in
    KL between the bracketing curve points; reads outside the curve are extrapolated
    from its two nearest points and marked, and the printed range leaves them out.
    """
    calibrated = {
        run.name: fidelity(run)
        for run in sorted(model_dir.iterdir())
        if run.is_dir() and is_calibrated_prox(run.name)
    }
    curve = lower_envelope([(arm["reads"], arm["kl"]) for arm in calibrated.values()])
    rows: dict[str, Any] = {
        "curve": {
            name: (arm["reads"], arm["kl"])
            for name, arm in calibrated.items()
            if (arm["reads"], arm["kl"]) in curve
        }
    }
    for plan, run in plans.items():
        topk = fidelity(model_dir / run)
        reference, extrapolated = log_interpolate(curve, topk["reads"])
        rows[plan] = {
            "run": run,
            "reads": topk["reads"],
            "kl": topk["kl"],
            "calibrated_kl": reference,
            "change": 100 * (topk["kl"] / reference - 1),
            "extrapolated": extrapolated,
        }
    return rows


def bootstrap_difference(
    a: dict[str, list[float]], b: dict[str, list[float]], samples: int = 2000
) -> tuple[float, float]:
    """Token-micro KL difference a - b with its seed-0 document-bootstrap SE."""
    ids = sorted(a)
    if set(ids) != set(b):
        raise ValueError("The two runs score different documents")
    sum_a = [sum(a[i]) for i in ids]
    sum_b = [sum(b[i]) for i in ids]
    count = [len(a[i]) for i in ids]
    diff = (sum(sum_a) - sum(sum_b)) / sum(count)
    rng = random.Random(0)
    boots = []
    for _ in range(samples):
        index = [rng.randrange(len(ids)) for _ in ids]
        boots.append(
            (sum(sum_a[i] for i in index) - sum(sum_b[i] for i in index))
            / sum(count[i] for i in index)
        )
    mean = sum(boots) / samples
    return diff, (sum((x - mean) ** 2 for x in boots) / (samples - 1)) ** 0.5


def components(confirm: Path, component_arms: Path) -> dict[str, Any]:
    """Compare each component arm with our reader per budget, at equal reads.

    On held-out OLMoE, more exact channels instead of recovery (``core-eq``) and the
    partial screen (``stack-eq``) are each compared with our reader with the full
    screen (``wsub``), with a seed-0 paired document-bootstrap SE.
    """
    out: dict[str, Any] = {}
    for plan in PLANS:
        ours = fidelity(confirm / f"wsub-{plan}")
        arms = {
            "ours": ours,
            "ours-without-recovery": fidelity(confirm / f"core-{plan}"),
            "exact-channels-equal-reads": fidelity(component_arms / f"core-eq-{plan}"),
            "partial-screen": fidelity(component_arms / f"stack-eq-{plan}"),
            "prox-4bit-output-scaled-recovery": fidelity(
                component_arms / f"pf-ou-bank-{plan}"
            ),
        }
        rows: dict[str, Any] = {}
        for name, arm in arms.items():
            row = {"kl": arm["kl"], "reads": arm["reads"]}
            if name != "ours":
                diff, se = bootstrap_difference(arm["documents"], ours["documents"])
                row |= {"vs_ours": diff, "se": se}
            rows[name] = row
        rows["recovery_reads_pp"] = 100 * (
            ours["reads"] - arms["ours-without-recovery"]["reads"]
        )
        out[plan] = rows
    return out


def score_effect(
    confirm: Path, component_arms: Path, development: Path
) -> dict[str, Any]:
    """Compare Prox's score with output scaling, and published scores with our reader.

    The first comparison uses Prox's 4-bit copy with either score (``pf-ou`` is
    output-scaled), both top-k without recovery. The second uses OLMoE development runs
    at 17.8% of reads, all without recovery: HiRE's INT4 gate screen and TEAL's input
    magnitudes against our reader.
    """
    scaling = {}
    for plan in PLANS:
        prox = fidelity(confirm / f"prox-free-{PROX_FREE[plan]}-{plan}")
        scaled = fidelity(component_arms / f"pf-ou-{plan}")
        scaling[plan] = {
            "prox_kl": prox["kl"],
            "output_scaled_kl": scaled["kl"],
            "ratio": prox["kl"] / scaled["kl"],
        }
    reader = fidelity(development / "reader-default-p10-pro5000")
    published = {}
    for name in ("hire-gate-p10-pro5000", "teal-topk-p10-pro5000"):
        arm = fidelity(development / name)
        published[name] = {
            "kl": arm["kl"],
            "reads": arm["reads"],
            "ratio": arm["kl"] / reader["kl"],
        }
    return {
        "output_scaling": scaling,
        "published_scores": {
            "reader_without_recovery": {"kl": reader["kl"], "reads": reader["reads"]},
            **published,
        },
    }


def retention(command: list[str]) -> float:
    """Return the ``--retention`` value of a run's recorded command line."""
    return float(command[command.index("--retention") + 1])


def label_fraction(label: str) -> float:
    """Return the kept fraction of an energy-table label (``P13`` or ``f0.13``)."""
    return int(label[1:]) / 100 if label.startswith("P") else float(label[1:])


def energy_correlation(energy: Path, confirm: Path) -> dict[str, Any]:
    """Correlate each arm's captured output energy with its held-out KL (Spearman).

    The energy is the share of routed-output energy the kept channels carry, captured
    on 3,000 TUNE positions per layer and averaged over the 16 OLMoE layers. Each arm
    is matched to the energy-table row nearest its retention.
    """
    layers: dict[str, Any] = {}
    for path in sorted(energy.glob("match-g*.json")):
        layers |= json.loads(path.read_text())["layers"]
    points = []
    for prefix, (screen, score) in ENERGY_ARMS.items():
        for plan in PLANS:
            arm = fidelity(confirm / f"{prefix}-{plan}")
            kept = retention(arm["command"])
            table = layers[next(iter(layers))][screen][score]
            label = min(table, key=lambda key: abs(label_fraction(key) - kept))
            captured = [
                layers[layer][screen][score][label]["captured"] for layer in layers
            ]
            points.append(
                {
                    "arm": f"{prefix}-{plan}",
                    "retention": kept,
                    "energy_label": label,
                    "energy": sum(captured) / len(captured),
                    "kl": arm["kl"],
                }
            )
    rho = spearman([p["energy"] for p in points], [p["kl"] for p in points])
    return {"layers": len(layers), "points": points, "spearman": rho}


def main() -> None:
    """Print the keep-rule, score, energy, component and Gemma 4 results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fidelity", type=Path, required=True, help="fidelity directory"
    )
    parser.add_argument(
        "--component-arms", type=Path, required=True, help="component-arms-olmoe arms"
    )
    parser.add_argument(
        "--energy", type=Path, required=True, help="energy-kl directory"
    )
    parser.add_argument(
        "--arms", type=Path, required=True, help="standard-protocols arms directory"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    confirm = args.fidelity / "olmoe-confirm"
    result: dict[str, Any] = {}

    olmoe_plans = {plan: f"prox-free-{PROX_FREE[plan]}-{plan}" for plan in PLANS}
    chat_plans = {plan: f"prox-free-k30-{plan}" for plan in ("p05", "p10")}
    keep = {"olmoe": keep_rule(confirm, olmoe_plans)}
    for model in CHAT_MODELS:
        keep[model] = keep_rule(args.fidelity / model, chat_plans)
    result["keep_rule"] = keep
    print("Keep rule: top-k Prox KL against calibrated Prox interpolated at its reads")
    for model, rows in keep.items():
        curve = ", ".join(
            f"{name} {100 * r:.2f}/{kl:.4f}" for name, (r, kl) in rows["curve"].items()
        )
        print(f"  {model} curve: {curve}")
        for plan in (p for p in PLANS if p in rows):
            row = rows[plan]
            mark = " (extrapolated)" if row["extrapolated"] else ""
            print(
                f"    {plan} reads {100 * row['reads']:.2f} KL {row['kl']:.4f} vs "
                f"{row['calibrated_kl']:.4f}: {row['change']:+.1f}%{mark}"
            )
    for group, models in (("OLMoE held-out", ("olmoe",)), ("chat", CHAT_MODELS)):
        changes = [
            keep[model][plan]["change"]
            for model in models
            for plan in PLANS
            if plan in keep[model] and not keep[model][plan]["extrapolated"]
        ]
        print(f"  {group} range: {min(changes):+.1f}% to {max(changes):+.1f}%")

    effect = score_effect(confirm, args.component_arms, args.fidelity / "olmoe")
    result["score"] = effect
    ratios = ", ".join(
        f"{plan} {row['prox_kl']:.4f}/{row['output_scaled_kl']:.4f} "
        f"= {row['ratio']:.2f}"
        for plan, row in effect["output_scaling"].items()
    )
    print(f"Output scaling divides KL: {ratios}")
    reader = effect["published_scores"]["reader_without_recovery"]
    for name, row in effect["published_scores"].items():
        if name != "reader_without_recovery":
            print(
                f"{name}: KL {row['kl']:.4f} at {100 * row['reads']:.2f}% = "
                f"{row['ratio']:.2f}x reader {reader['kl']:.4f}"
            )

    energy = energy_correlation(args.energy, confirm)
    result["energy"] = energy
    print(
        f"Energy vs KL over {len(energy['points'])} arms, {energy['layers']} layers: "
        f"Spearman {energy['spearman']:+.4f}"
    )

    table = components(confirm, args.component_arms)
    result["components"] = table
    print(
        "Components at equal reads (KL; difference versus our reader +- bootstrap SE)"
    )
    for plan, rows in table.items():
        print(f"  {plan}: recovery costs {rows['recovery_reads_pp']:.2f} pp of reads")
        for name, row in rows.items():
            if not isinstance(row, dict):
                continue
            extra = f"  {row['vs_ours']:+.4f} +- {row['se']:.4f}" if "se" in row else ""
            print(f"    {name:34s} {100 * row['reads']:.2f}% KL {row['kl']:.4f}{extra}")

    # Gemma 4, paired by question: recovery with the full screen at about 18% of reads
    # (IFEval), and the partial against the full screen at about 13% (MMLU-Pro).
    gemma = args.arms / "gem"
    benchmarks = {
        "ifeval_recovery_p10": (gemma / "ifeval/wsub-p10", gemma / "ifeval/zero-p10"),
        "mmlu_partial_p05": (
            gemma / "mmlu-pro-1000/isp-p05",
            gemma / "mmlu-pro-1000/wsub-p05",
        ),
    }
    for name, (policy, base) in benchmarks.items():
        pair = paired_difference(policy / "metrics.json", base / "metrics.json")
        capped = [
            100 * load_json(run / "metrics.json")["summary"]["truncation_rate"]
            for run in (policy, base)
        ]
        result[f"gemma/{name}"] = pair | {"capped": capped}
        print(
            f"Gemma 4 {name}: {pair['policy']:.2f} vs {pair['base']:.2f}, "
            f"{pair['diff']:+.2f} +- {pair['se']:.2f} "
            f"over {pair['questions']} questions "
            f"(length-limited {capped[0]:.1f}% vs {capped[1]:.1f}%)"
        )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
