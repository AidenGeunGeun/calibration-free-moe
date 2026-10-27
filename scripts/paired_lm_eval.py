"""Write a paired comparison of two arms' lm-eval benchmark reductions.

For every task, the sample-weighted MMLU group and every metric of two
`olmes_requests.py reduce` outputs, writes the difference right - left, a paired
bootstrap 95% interval over documents (stratified by MMLU subject) and, for
binary metrics, an exact two-sided McNemar test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def aligned(
    left: dict[str, Any], right: dict[str, Any], task: str, metric: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return both arms' per-document values of one metric, in document order."""
    rows = [
        {row["doc_id"]: row[metric] for row in arm["tasks"][task]["documents"]}
        for arm in (left, right)
    ]
    if rows[0].keys() != rows[1].keys():
        raise ValueError(f"{task} arms score different documents")
    keys = sorted(rows[0])
    first, second = (np.array([r[k] for k in keys], dtype=np.float64) for r in rows)
    return first, second


def mcnemar(left: NDArray[np.float64], right: NDArray[np.float64]) -> dict[str, Any]:
    """Return the discordant counts and the exact two-sided McNemar p-value."""
    left_only = int(np.sum((left == 1) & (right == 0)))
    right_only = int(np.sum((left == 0) & (right == 1)))
    n = left_only + right_only
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1))
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "p_value": min(1.0, 2 * tail / 2**n) if n else 1.0,
    }


def compare(
    strata: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    resamples: int,
    rng: np.random.Generator,
    block: int = 200,
) -> dict[str, Any]:
    """Return means, their difference and a stratified paired bootstrap 95% CI.

    Each stratum is resampled within itself, so pooled strata keep their sample
    weights. Resamples are drawn `block` at a time to bound memory.
    """
    total = sum(len(left) for left, _ in strata)
    left_all = np.concatenate([left for left, _ in strata])
    right_all = np.concatenate([right for _, right in strata])
    deltas = np.zeros(resamples)
    for start in range(0, resamples, block):
        size = min(block, resamples - start)
        for left, right in strata:
            draws = rng.integers(0, len(left), size=(size, len(left)))
            deltas[start : start + size] += (right - left)[draws].sum(axis=1) / total
    low, high = np.percentile(deltas, [2.5, 97.5])
    result: dict[str, Any] = {
        "count": total,
        "left": float(left_all.mean()),
        "right": float(right_all.mean()),
        "delta": float(right_all.mean() - left_all.mean()),
        "ci95": [float(low), float(high)],
    }
    if np.isin(np.concatenate([left_all, right_all]), (0.0, 1.0)).all():
        result["mcnemar"] = mcnemar(left_all, right_all)
    return result


def main() -> None:
    """Check that both arms share their preparation, then compare every metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Reference arm")
    parser.add_argument("--right", type=Path, required=True, help="Compared arm")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    left, right = (json.loads(path.read_text()) for path in (args.left, args.right))
    if left["role"] != right["role"] or left["tasks"].keys() != right["tasks"].keys():
        raise ValueError("Arms differ in protocol or tasks")
    # A field absent from a prepared panel means None.
    if any(
        left["data"].get(key) != right["data"].get(key)
        for key in (
            "task_spec",
            "num_fewshot",
            "eval_split",
            "limit",
            "model",
            "dataset_locators",
            "chat_template",
            "chat_template_kwargs",
        )
    ):
        raise ValueError("Arms use different task preparation")
    rng = np.random.default_rng(args.seed)
    results: dict[str, dict[str, Any]] = {}
    subjects = sorted(name for name in left["tasks"] if name.startswith("mmlu_"))
    for task in sorted(set(left["tasks"]) - set(subjects)):
        results[task] = {
            metric: compare([aligned(left, right, task, metric)], args.resamples, rng)
            for metric in left["tasks"][task]["metrics"]
        }
    if subjects:
        results["mmlu"] = {
            metric: compare(
                [aligned(left, right, task, metric) for task in subjects],
                args.resamples,
                rng,
            )
            for metric in left["tasks"][subjects[0]]["metrics"]
        }
        results["mmlu"]["subjects"] = len(subjects)
    payload = {
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "interval": f"paired document bootstrap (MMLU stratified by subject), "
        f"{args.resamples} resamples, seed {args.seed}, percentile 95%",
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for task, metrics in results.items():
        for metric, row in metrics.items():
            if isinstance(row, dict):
                print(
                    f"{task} {metric}: "
                    f"{100 * row['left']:.2f} -> {100 * row['right']:.2f} "
                    f"({100 * row['delta']:+.2f}, CI [{100 * row['ci95'][0]:+.2f}, "
                    f"{100 * row['ci95'][1]:+.2f}])"
                )


if __name__ == "__main__":
    main()
