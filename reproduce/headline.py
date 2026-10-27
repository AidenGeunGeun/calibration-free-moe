"""Print Table 1, the Figure 1a points and the Section 5 threshold-versus-top-k counts.

Inputs: `headline-matrix`, `standard-protocols`, `gpqa-x8`, `gemma-repro` and
`wina-published` under the runs root `RUNS`; `--output` writes `headline_data.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reproduce.common import paired, question_means, read_percent, report, verdict

MODELS = ("gem", "thk", "q35")
TASKS = ("gpqa", "ifeval", "mmlu")
CELLS = [f"{m}|{t}" for m in MODELS for t in TASKS]
GPQA = {
    "gem": "gpqa-diamond-x4",
    "thk": "gpqa-diamond-x4-32k",
    "q35": "gpqa-diamond-x4",
}
GPQA_X1 = {"gem": "gpqa-diamond-x1", "thk": "gpqa-diamond-x1-32k"}
GPQA_SAMPLES = 4
HM = "headline-matrix"
SP = "standard-protocols/arms"
X8 = "gpqa-x8/arms"
WP = "wina-published/arms"

# Selector lines in the order of the figure data; each line is one method at one budget.
LINES = (
    [f"ours-{p}" for p in ("p05", "p10", "p20", "p30")]
    + [f"prox-topk-{p}" for p in ("p05", "p10", "p20", "p30")]
    + [f"prox-cal-{t}" for t in (13, 18, 28, 38)]
    + [f"{s}-x{x}" for x in ("085", "080", "070", "060") for s in ("wina", "teal")]
    + [f"ours-cal-{p}" for p in ("p05", "p10", "p20", "p30")]
    + [
        "ours-p50",
        "ours-p70",
        "prox-topk-p50",
        "prox-topk-p70",
        "prox-cal-58",
        "prox-cal-78",
    ]
    + ["wina-x050", "wina-x040", "teal-x050", "teal-x040"]
    + ["cats-x050", "cats-topk-x050", "cats-x070", "cats-topk-x070"]
    + [f"wina-pub-x{x}" for x in ("040", "050", "060", "070", "080", "085")]
)


def task_dir(m: str, task: str) -> str:
    """Return the standard-protocols directory name of a task for model `m`."""
    return {"gpqa": GPQA[m], "ifeval": "ifeval", "mmlu": "mmlu-pro-1000"}[task]


# Cells measured before the matrix with the same requests; the matrix reuses these runs.
REUSE: dict[tuple[str, str, str], list[str]] = {
    **{
        ("ours-p05", m, "gpqa"): [
            f"{SP}/{m}/{GPQA_X1[m]}/stack-p05",
            f"{X8}/{m}/gpqa-rest7/isp-p05",
        ]
        for m in ("gem", "thk")
    },
    ("ours-p05", "gem", "ifeval"): [f"{SP}/gem/ifeval/stack-p05"],
    ("ours-p05", "thk", "ifeval"): [f"{SP}/thk/ifeval/stack-p05"],
    ("ours-p05", "q35", "ifeval"): [f"{SP}/q35/ifeval/isp-p05"],
    **{("ours-p05", m, "mmlu"): [f"{SP}/{m}/mmlu-pro-1000/isp-p05"] for m in MODELS},
    **{
        ("ours-p10", m, t): [f"{SP}/{m}/{task_dir(m, t)}/isp-p10"]
        for m in MODELS
        for t in TASKS
    },
    **{
        ("prox-cal-18", m, t): [f"{SP}/{m}/{task_dir(m, t)}/prox-read-18"]
        for m in MODELS
        for t in TASKS
        if (m, t) != ("gem", "mmlu")
    },
}
STOCK = {
    **{(m, t): [f"{SP}/{m}/{task_dir(m, t)}/stock"] for m in MODELS for t in TASKS},
    # Qwen3-Thinking's full model runs at its 81,920-token cap on the four-sample
    # requests.
    ("thk", "gpqa"): [f"{SP}/thk/gpqa-diamond-x4/stock"],
    ("gem", "mmlu"): ["gemma-repro/arms/mmlu-pro-1000-stock"],
}


def runs_of(line: str, m: str, task: str) -> list[str]:
    """Return the run directories (relative to the runs root) of one line's cell."""
    if (line, m, task) in REUSE:
        return REUSE[(line, m, task)]
    root = WP if line.startswith("wina-pub-") else f"{HM}/arms"
    return [f"{root}/{m}/{task}/{line}"]


class Cell:
    """One cell's per-answer correctness (by question and sample) and its reads.

    Reads are the first run's routed-expert read cost as a percentage of dense routed
    reads. A reused GPQA-Diamond cell can combine sample 0 of a one-sample run with
    samples 1-3 of a seven-sample run.
    """

    def __init__(
        self, root: Path, runs: list[str], task: str, with_reads: bool
    ) -> None:
        self.task = task
        self.samples: dict[tuple[str, int], float] = {}
        for run in runs:
            for doc in report(root / run, "metrics")["documents"]:
                if task != "gpqa" or doc["sample_index"] < GPQA_SAMPLES:
                    key = (doc["item_id"], doc["sample_index"])
                    self.samples.setdefault(key, float(doc["correct"]))
        self.reads = (
            read_percent(report(root / runs[0], "scores")) if with_reads else None
        )

    def entry(self) -> dict[str, Any]:
        """Return the cell's figure-data entry: score, reads, answers, sample count.

        The score is percent correct over scored answers; IFEval has one sample per
        prompt and no sample count.
        """
        out: dict[str, Any] = {
            "score": 100 * sum(self.samples.values()) / len(self.samples),
            "reads": self.reads,
            "n": len(self.samples),
        }
        if self.task != "ifeval":
            out["samples"] = len({s for _, s in self.samples})
        return out


def collect(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Cell]]]:
    """Return the figure data and the loaded cells of every line.

    The nine cells are {Gemma 4, Qwen3-Thinking, Qwen3.5} x {GPQA-Diamond with four
    samples, IFEval prompt-level strict, MMLU-Pro 1,000 questions}. A line's aggregate
    is the mean over cells of its score as a percentage of the full model's, and its
    mean reads. The cells keep per-answer correctness for the paired comparisons.
    """
    stock = {
        f"{m}|{t}": Cell(root, STOCK[(m, t)], t, with_reads=False).entry()
        for m in MODELS
        for t in TASKS
    }
    cells = {
        line: {
            f"{m}|{t}": Cell(root, runs_of(line, m, t), t, True)
            for m in MODELS
            for t in TASKS
        }
        for line in LINES
    }
    lines = {
        line: {k: c.entry() for k, c in row.items()} for line, row in cells.items()
    }
    aggregate = {
        line: {
            "cells": len(row),
            "relative": sum(
                100 * c["score"] / stock[k]["score"] for k, c in row.items()
            )
            / len(row),
            "reads": sum(c["reads"] for c in row.values()) / len(row),
        }
        for line, row in lines.items()
    }
    return {"stock": stock, "lines": lines, "aggregate": aggregate}, cells


# Table 1 rows: label, line at about 13% and at about 18% of reads, and the calibrated
# targets (None for top-k lines).
TABLE = [
    ("Our reader", "ours-p05", "ours-p10", None),
    ("Prox score, top-k", "prox-topk-p05", "prox-topk-p10", None),
    ("Prox as published, calibrated", "prox-cal-13", "prox-cal-18", (13, 18)),
    ("WINA as published, top-k", "wina-pub-x085", "wina-pub-x080", None),
    ("WINA score, calibrated", "wina-x085", "wina-x080", (15, 20)),
    ("TEAL as published, calibrated", "teal-x085", "teal-x080", (15, 20)),
]


def relative(data: dict[str, Any], line: str) -> list[float]:
    """Return each cell's score of `line` as a percentage of the full model's."""
    stock = data["stock"]
    return [
        100 * c["score"] / stock[k]["score"] for k, c in data["lines"][line].items()
    ]


def print_table(data: dict[str, Any]) -> None:
    """Print Table 1 rows as LaTeX cells."""
    print(
        "Table 1 (tab:accuracy): target, realized reads, mean and worst % "
        "of the full model"
    )
    for label, low, high, targets in TABLE:
        row = [label]
        for i, line in enumerate((low, high)):
            agg = data["aggregate"][line]
            # WINA as published meets its budget to the precision shown (whole percent).
            digits = 0 if line.startswith("wina-pub-") else 1
            row += [
                "--" if targets is None else str(targets[i]),
                f"{agg['reads']:.{digits}f}",
                f"{agg['relative']:.1f}",
                f"{min(relative(data, line)):.1f}",
            ]
        print(" & ".join(row))


def target(line: str) -> float | None:
    """Return the reads a calibrated line targets, or None for a top-k line.

    Prox targets its read percentage; TEAL and WINA target 100 minus their sparsity.
    """
    if line.startswith("prox-cal-"):
        return float(line.rsplit("-", 1)[1])
    if line.startswith(("teal-x", "wina-x", "wina-pub-x")):
        return 100.0 - int(line[-3:])
    return None


def print_figure(data: dict[str, Any]) -> None:
    """Print the Figure 1a points: each line's worst cell as % of the full model.

    A calibrated line is plotted at its target reads, any other line at the worst
    cell's realized reads.
    """
    print(
        "Figure 1a: line, plotted reads, worst cell, its realized reads, "
        "% of the full model"
    )
    for line in LINES:
        values = relative(data, line)
        worst = min(range(len(values)), key=values.__getitem__)
        cell = list(data["lines"][line])[worst]
        reads = data["lines"][line][cell]["reads"]
        aim = target(line)
        print(
            f"{line:16s} {reads if aim is None else aim:6.2f} {cell:10s} {reads:6.2f} "
            f"{values[worst]:6.1f}"
        )


# Threshold versus top-k on the same score and budget (Section 5, "Other scores and
# calibrated parts").
KEEP_RULE_PAIRS = {
    "Our reader": [
        (f"ours-cal-{p}", f"ours-{p}") for p in ("p05", "p10", "p20", "p30")
    ],
    "CATS": [(f"cats-x{s}", f"cats-topk-x{s}") for s in ("070", "050")],
}


def compare(threshold: Cell, topk: Cell) -> tuple[float, float]:
    """Threshold minus top-k accuracy (points) and its question-clustered paired SE."""
    return paired(question_means(threshold.samples), question_means(topk.samples))


def print_keep_rule(data: dict[str, Any], cells: dict[str, dict[str, Cell]]) -> None:
    """Print the Section 5 threshold-versus-top-k verdicts on the headline cells."""
    print(
        "Threshold minus top-k at the same budget, paired per question (2 SE verdict)"
    )
    for name, pairs in KEEP_RULE_PAIRS.items():
        counts = {"thr": 0, "level": 0, "topk": 0}
        for thr, topk in pairs:
            for key in CELLS:
                diff, se = compare(cells[thr][key], cells[topk][key])
                counts[verdict(diff, se)] += 1
                if verdict(diff, se) != "level":
                    t, k = data["lines"][thr][key], data["lines"][topk][key]
                    print(
                        f"  {name} {thr} {key}: {t['score']:.1f} at "
                        f"{t['reads']:.1f}% vs {k['score']:.1f} at "
                        f"{k['reads']:.1f}%, {diff:+.1f} +- {se:.1f}"
                    )
        print(f"  {name}: {sum(counts.values())} comparisons, {counts}")
    key = "q35|gpqa"
    diff, se = compare(cells["prox-cal-13"][key], cells["prox-topk-p05"][key])
    t, k = data["lines"]["prox-cal-13"][key], data["lines"]["prox-topk-p05"][key]
    print(
        f"  Prox Qwen3.5 GPQA-D: threshold (13%) {t['score']:.1f} at "
        f"{t['reads']:.1f}% vs top-k {k['score']:.1f} at {k['reads']:.1f}%, "
        f"{diff:+.1f} +- {se:.1f} ({verdict(diff, se)})"
    )


def main() -> None:
    """Collect the matrix, optionally write the figure data, and print the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Runs root")
    parser.add_argument("--output", type=Path, help="Write headline_data.json here")
    args = parser.parse_args()
    data, cells = collect(args.runs)
    if args.output is not None:
        args.output.write_text(json.dumps(data, indent=1) + "\n")
    print_table(data)
    print_figure(data)
    print_keep_rule(data, cells)


if __name__ == "__main__":
    main()
