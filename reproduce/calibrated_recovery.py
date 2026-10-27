"""Print the Section 4.2 calibrated-recovery count and the Appendix recovery table.

Inputs: ``ARMS/{gem,thk,q35}/<task>/<run>/metrics.json``, ``ARMS`` being the runs root's
``standard-protocols/arms``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reproduce.analysis_common import paired_difference

MODELS = ("gem", "thk", "q35")
TASKS = {
    "gpqa": {
        "gem": "gpqa-diamond-x4",
        "thk": "gpqa-diamond-x4-32k",
        "q35": "gpqa-diamond-x4",
    },
    "ifeval": dict.fromkeys(MODELS, "ifeval"),
    "mmlu-pro": dict.fromkeys(MODELS, "mmlu-pro-1000"),
    "gsm8k": {
        "gem": "gsm8k-cot-chat",
        "thk": "gsm8k-thinking",
        "q35": "gsm8k-cot-chat",
    },
}
CELLS = [(task, "p10") for task in TASKS] + [("ifeval", "p05"), ("mmlu-pro", "p05")]
# These two models have no isp-p05 IFEval run; stack-p05 is the weight-only run
# with the same retention, score and screen as calrec-p05.
WEIGHT_ONLY_RUN = {
    ("gem", "ifeval", "p05"): "stack-p05",
    ("thk", "ifeval", "p05"): "stack-p05",
}


def cells(arms: Path) -> list[dict[str, Any]]:
    """Pair the calibrated and weight-only recovery maps in every cell.

    A cell is one model, task and read budget (``p05`` about 13% of reads, ``p10``
    about 18%), running our reader with only the recovery map (``bank``) changed:
    fitted to calibration text (``calrec-*``) or weight-only as deployed (``isp-*``).
    Accuracy is paired by question; GSM8K is lm-eval ``gsm8k_cot`` strict-match on
    Gemma 4 and Qwen3.5 and the thinking scorer on Qwen3-Thinking. A map wins a cell
    when it leads by more than two SE.
    """
    rows = []
    for model in MODELS:
        for task, plan in CELLS:
            directory = arms / model / TASKS[task][model]
            weight_only = WEIGHT_ONLY_RUN.get((model, task, plan), f"isp-{plan}")
            pair = paired_difference(
                directory / f"calrec-{plan}" / "metrics.json",
                directory / weight_only / "metrics.json",
            )
            if pair["diff"] > 2 * pair["se"]:
                verdict = "calibrated"
            elif pair["diff"] < -2 * pair["se"]:
                verdict = "weight-only"
            else:
                verdict = "tie"
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "plan": plan,
                    "weight_only_run": weight_only,
                    **pair,
                    "verdict": verdict,
                }
            )
    return rows


def main() -> None:
    """Print every cell's paired difference and the win counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", type=Path, required=True, help="standard-protocols arms directory"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = cells(args.arms)
    for row in rows:
        print(
            f"{row['model']} {row['task']:8s} {row['plan']}: calibrated "
            f"{row['policy']:.2f} vs weight-only {row['base']:.2f} "
            f"({row['weight_only_run']}), {row['diff']:+.2f} +- {row['se']:.2f} "
            f"[{row['questions']}] {row['verdict']}"
        )
    wins = sum(row["verdict"] == "calibrated" for row in rows)
    losses = sum(row["verdict"] == "weight-only" for row in rows)
    print(f"Calibrated map wins {wins} of {len(rows)}; weight-only map wins {losses}")
    if args.output is not None:
        args.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
