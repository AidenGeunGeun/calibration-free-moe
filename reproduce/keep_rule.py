"""Print the Section 5 keep-rule counts and write the Figure 5 and keep-rule table data.

Inputs: `score-calibration`, `keep-rule-study`, `prox-reconcile`, `standard-protocols`
and `benchmark-suite` under the runs root `RUNS`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Hashable
from pathlib import Path
from typing import Any

from reproduce.common import (
    paired,
    question_means,
    read_percent,
    report,
    verdict,
)

SC = "score-calibration/arms"
KR = "keep-rule-study/arms"
PR = "prox-reconcile/arms"
SP = "standard-protocols/arms"
OLMES_STOCK = "benchmark-suite/generation/stock"

# (model, task, budget, score) of every generation comparison, in figure-data order.
GENERATION = [
    ("gem", "gpqa-diamond-x1", "p13", "ou"),
    ("gem", "gpqa-diamond-x1", "p13", "prox"),
    ("gem", "ifeval", "p13", "ou"),
    ("gem", "ifeval", "p13", "prox"),
    ("gem", "ifeval", "p18", "teal"),
    *[
        ("olmoe", task, budget, score)
        for task in ("bbh-cot", "gsm8k")
        for budget in ("p13", "p18")
        for score in ("ou", "prox", "teal")
    ],
    ("q35", "ifeval", "p13", "ou"),
    ("q35", "ifeval", "p13", "prox"),
    ("q35", "ifeval", "p18", "teal"),
    ("thk", "gpqa-diamond-x1-32k", "p13", "ou"),
    ("thk", "gpqa-diamond-x1-32k", "p13", "prox"),
    ("thk", "ifeval", "p13", "ou"),
    ("thk", "ifeval", "p13", "prox"),
    ("thk", "ifeval", "p18", "teal"),
]
# Full-model run of each generation task (GPQA-Diamond: the four-sample headline run).
STOCK = {
    ("olmoe", "gsm8k"): f"{OLMES_STOCK}/gsm8k",
    ("olmoe", "bbh-cot"): f"{OLMES_STOCK}/bbh-cot",
    **{(m, "ifeval"): f"{SP}/{m}/ifeval/stock" for m in ("gem", "thk", "q35")},
    ("gem", "gpqa-diamond-x1"): f"{SP}/gem/gpqa-diamond-x4/stock",
    ("thk", "gpqa-diamond-x1-32k"): f"{SP}/thk/gpqa-diamond-x4/stock",
}
COLLAPSE = 0.1  # both rules below this share of the full model's accuracy

REGIMES = ("sp", "ctx", "sink")
BUDGETS = ("b13", "b18", "b38")
LIKELIHOOD = ("mmlu", "hellaswag", "arc_easy", "arc_challenge", "piqa")
# Prox's lm-eval 0.4.3 metric per task; MMLU pools its subjects' questions.
METRIC = {
    "mmlu": "acc",
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "piqa": "acc",
}
MATCH = 1.0  # reads (points) within which the top-k run at the budget is the comparison


def questions(metrics: dict[str, Any]) -> dict[Hashable, float]:
    """Return per-question accuracy of a generation report (thinking or OLMES)."""
    if metrics.get("role") == "development-task":
        return {
            (task, doc["doc_id"]): float(
                doc["metrics"][result["task_config"]["primary_metric"]]
            )
            for task, result in metrics["tasks"].items()
            for doc in result["documents"]
        }
    samples = {
        (doc["item_id"], doc["sample_index"]): float(doc["correct"])
        for doc in metrics["documents"]
    }
    return question_means(samples)


def accuracy(scores: dict[Hashable, float]) -> float:
    """Return the mean per-question score in percent."""
    return 100 * sum(scores.values()) / len(scores)


def scores(root: Path, run: str) -> dict[Hashable, float]:
    """Load a generation run's per-question accuracy."""
    return questions(report(root / run, "metrics"))


def reads(root: Path, run: str) -> float:
    """Load a run's read cost in percent of dense routed reads."""
    return read_percent(report(root / run, "scores"))


def comparison(
    threshold: dict[Hashable, float], topk: dict[Hashable, float]
) -> dict[str, Any]:
    """Pair a generation threshold run with a top-k run (accuracies, diff, verdict).

    The difference is threshold minus top-k accuracy, paired per question; the
    verdict is a detected difference beyond two SE.
    """
    diff, se = paired(threshold, topk)
    return {
        "threshold": accuracy(threshold),
        "topk": accuracy(topk),
        "diff": diff,
        "se": se,
        "verdict": verdict(diff, se),
    }


def generation_arms(
    model: str, task: str, budget: str, score: str
) -> tuple[str, str, str]:
    """Name the threshold, top-k at budget and top-k at realized-reads runs.

    Prox's score at 13% on Gemma 4 and Qwen3-Thinking comes from `keep-rule-study`,
    which keeps Prox's recipe and input rule.
    """
    if score == "prox" and model in ("gem", "thk") and budget == "p13":
        threshold, topk = f"{KR}/{model}/{task}/kt", f"{KR}/{model}/{task}/kk"
    else:
        threshold = f"{SC}/{model}/{task}/{score}-thr-{budget}"
        topk = f"{SC}/{model}/{task}/{score}-topk-{budget}"
    return threshold, topk, f"{SC}/{model}/{task}/{score}-topk-m{budget}"


def generation(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the generation comparisons at the target budget and at matched reads.

    The score x keep-rule factorial covers the Prox, TEAL and output-scaled Prox
    scores. Target compares with the top-k run at the budget the threshold was
    calibrated for; matched uses the top-k run at the threshold's realized reads when
    the two differ by more than `MATCH`. A comparison collapses when both rules score
    below `COLLAPSE` of the full model.
    """
    target, matched = [], []
    for model, task, budget, score in GENERATION:
        names = generation_arms(model, task, budget, score)
        threshold, topk = scores(root, names[0]), scores(root, names[1])
        stock = accuracy(scores(root, STOCK[(model, task)]))
        rerun = abs(reads(root, names[0]) - reads(root, names[1])) > MATCH
        head = {"model": model, "task": task, "budget": budget, "score": score}
        for out, other in (
            (target, topk),
            (matched, scores(root, names[2]) if rerun else topk),
        ):
            c = comparison(threshold, other)
            collapsed = max(c["threshold"], c["topk"]) < COLLAPSE * stock
            out.append(
                {
                    **head,
                    **c,
                    "n": len(threshold),
                    "matched": rerun,
                    "collapsed": collapsed,
                }
            )
    return target, matched


class LikelihoodRun:
    """Per-question metric of one likelihood task and the run's realized reads.

    The likelihood reduction embeds the evaluation's report, work report included, as
    `score_metadata`, so the reads come from the metrics file.
    """

    def __init__(self, root: Path, run: str, task: str) -> None:
        metrics = report(root / PR / run, f"{task}-metrics")
        self.scores: dict[Hashable, float] = {
            (name, doc["doc_id"]): float(doc[METRIC[task]])
            for name, result in metrics["tasks"].items()
            for doc in result["documents"]
        }
        self.reads = read_percent(metrics["score_metadata"])


def likelihood_run(root: Path, run: str, task: str) -> LikelihoodRun | None:
    """Load a likelihood run, or None if it was not run."""
    try:
        return LikelihoodRun(root, run, task)
    except FileNotFoundError:
        return None


# The likelihood reducer reports verdicts in upper case for a detected difference.
LIKELIHOOD_VERDICT = {"thr": "THR", "topk": "TOPK", "level": "lvl"}


def likelihood(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the likelihood comparisons at the target budget and at matched reads.

    These are Prox's five likelihood tasks on OLMoE under three prompt regimes and
    three budgets; target and matched are defined as in `generation`.
    """
    target, matched = [], []
    for regime in REGIMES:
        for budget in BUDGETS:
            for task in LIKELIHOOD:
                threshold = LikelihoodRun(root, f"{regime}/prox-{budget}", task)
                topk = LikelihoodRun(root, f"{regime}/topk-{budget}", task)
                head = {"regime": regime, "budget": budget, "task": task}
                target.append({**head, **pair(threshold.scores, topk.scores)})
                if abs(threshold.reads - topk.reads) > MATCH:
                    rerun = likelihood_run(
                        root, f"{regime}/topk-{budget}-matched-{task}", task
                    )
                    if rerun is None:
                        continue  # no top-k run at the threshold's reads: not compared
                    topk = rerun
                matched.append({**head, **pair(threshold.scores, topk.scores)})
    return target, matched


def pair(
    threshold: dict[Hashable, float], topk: dict[Hashable, float]
) -> dict[str, Any]:
    """Pair a likelihood threshold run with a top-k run, in the likelihood spelling."""
    diff, se = paired(threshold, topk)
    return {
        "diff": diff,
        "se": se,
        "threshold": accuracy(threshold),
        "topk": accuracy(topk),
        "verdict": LIKELIHOOD_VERDICT[verdict(diff, se)],
    }


def counted(
    gen: list[dict[str, Any]], lik: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select Figure 5's comparisons.

    These are the generation comparisons that did not collapse and the likelihood
    comparisons with a sparse or full prompt below 38% of reads.
    """
    return [c for c in gen if not c["collapsed"]] + [
        c for c in lik if c["regime"] in ("sp", "ctx") and c["budget"] != "b38"
    ]


def outcome(c: dict[str, Any]) -> str:
    """'thr', 'topk' or 'level' for either file family's verdict spelling."""
    v: str = c["verdict"]
    return "level" if v == "lvl" else v.lower()


def label(c: dict[str, Any]) -> str:
    """Name a comparison by model or regime, task, budget and score."""
    return " ".join(v for k, v in c.items() if isinstance(v, str) and k != "verdict")


def tally(cells: list[dict[str, Any]]) -> dict[str, int]:
    """Count the comparisons by outcome."""
    out = {"topk": 0, "level": 0, "thr": 0}
    for c in cells:
        out[outcome(c)] += 1
    return out


def pooled(cells: list[dict[str, Any]]) -> tuple[int, float, float]:
    """Return the count, inverse-variance pooled difference and its SE."""
    w = [1 / c["se"] ** 2 for c in cells]
    mean = sum(wi * c["diff"] for wi, c in zip(w, cells, strict=True)) / sum(w)
    return len(cells), mean, math.sqrt(1 / sum(w))


def summary(
    gen: dict[str, list[dict[str, Any]]], lik: dict[str, list[dict[str, Any]]]
) -> None:
    """Print the Section 5 counts and the pooled per-model table."""
    for mode in ("target", "matched"):
        cells = counted(gen[mode], lik[mode])
        t = tally(cells)
        print(
            f"{mode}: {len(cells)} comparisons, top-k ahead {t['topk']}, "
            f"within 2 SE {t['level']}, threshold ahead {t['thr']}; "
            f"top-k matches or beats in {t['topk'] + t['level']} of {len(cells)}; "
            f"generation {len([c for c in gen[mode] if not c['collapsed']])}"
        )
    prox = [c for c in lik["target"] if c["regime"] == "sp" and c["budget"] != "b38"]
    print("OLMoE Prox likelihood, sparse prompt, 13% and 18%:", len(prox), tally(prox))
    post = [c for c in gen["target"] if c["model"] != "olmoe" and not c["collapsed"]]
    print("post-trained models at the calibrated budget:", len(post), tally(post))
    leads = [c for c in counted(gen["matched"], lik["matched"]) if outcome(c) == "thr"]
    best = max(leads, key=lambda c: c["diff"])
    at_budget = next(
        c for c in counted(gen["target"], lik["target"]) if label(c) == label(best)
    )
    print(
        f"largest threshold lead at realized reads: {label(best)} "
        f"{best['diff']:+.1f} +- {best['se']:.1f}; at the calibrated budget top-k "
        f"{at_budget['topk']:.1f}, threshold {at_budget['threshold']:.1f}"
    )
    print(
        "pooled per model (tab-keep-pooled): comparisons, tau - k +- SE; "
        "target | realized"
    )
    for model in ("thk", "q35", "olmoe", "gem"):
        row = []
        for mode in ("target", "matched"):
            cells = [c for c in gen[mode] if c["model"] == model and not c["collapsed"]]
            if model == "olmoe":
                cells += list(counted([], lik[mode]))
            n, mean, se = pooled(cells)
            row.append(f"{n} {mean:+.1f} +- {se:.1f}")
        print(f"  {model}: " + " | ".join(row))


def main() -> None:
    """Print the counts and optionally write the generation and likelihood files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Runs root")
    parser.add_argument("--output", type=Path, help="Directory for the four JSON files")
    args = parser.parse_args()
    gen_target, gen_matched = generation(args.runs)
    lik_target, lik_matched = likelihood(args.runs)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("scal-target", gen_target),
            ("scal-matched", gen_matched),
            ("rec-target", lik_target),
            ("rec-matched", lik_matched),
        ):
            (args.output / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")
    summary(
        {"target": gen_target, "matched": gen_matched},
        {"target": lik_target, "matched": lik_matched},
    )


if __name__ == "__main__":
    main()
