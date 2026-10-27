"""Write the per-question answer flips of a sparse policy against stock.

Pairs two reductions of the same panel (`olmes_requests.py reduce` or
`thinking_requests.py reduce`) by question and writes, per task and pooled group
(MMLU subjects, BBH subtasks), the flips of binary metrics and the mean change of
continuous ones.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

# The metric each results table reports; other per-question metrics are extra.
HEADLINE = {
    "arc_challenge": "acc_norm",
    "arc_easy": "acc",
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "gsm8k_cot": "exact_match,strict-match",
}

# Per-question correctness metrics; everything else (F1, token counts) is continuous
# even when a value happens to be 0 or 1.
BINARY = {
    "acc",
    "acc_norm",
    "exact_match",
    "exact_match_simple",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "correct",
    "flexible_correct",
    "truncated",
}

Rows = dict[str, dict[Any, dict[str, float]]]


def panel(report: dict[str, Any]) -> dict[str, Any]:
    """Return the panel-defining preparation fields (split, shots, sampling, ...)."""
    return {
        key: value
        for key, value in report.get("data", {}).items()
        if key not in ("command", "packages") and not key.endswith("path")
    }


def lm_eval_rows(report: dict[str, Any]) -> tuple[Rows, dict[str, str]]:
    """Return per-question numeric metrics and headline metrics of an lm-eval run."""
    rows: Rows = {}
    headline = {}
    for task, result in report["tasks"].items():
        rows[task] = {
            doc["doc_id"]: {
                key: float(value)
                for key, value in doc.items()
                if key != "doc_id" and isinstance(value, (bool, int, float))
            }
            for doc in result["documents"]
        }
        headline[task] = HEADLINE.get(task, "acc")
    return rows, headline


def olmes_rows(report: dict[str, Any]) -> tuple[Rows, dict[str, str]]:
    """Return per-question numeric metrics and primary metrics of an OLMES run."""
    rows: Rows = {}
    headline = {}
    for task, result in report["tasks"].items():
        rows[task] = {
            doc["doc_id"]: {
                key: float(value)
                for key, value in doc["metrics"].items()
                if isinstance(value, (bool, int, float))
            }
            for doc in result["documents"]
        }
        headline[task] = result["task_config"]["primary_metric"]
    return rows, headline


def thinking_rows(report: dict[str, Any]) -> tuple[Rows, dict[str, str]]:
    """Return per-sample metrics of a thinking run, keyed by (question, sample)."""
    task = report["data"]["task"]
    samples: dict[Any, dict[str, float]] = {}
    for doc in report["documents"]:
        values = {"correct": float(bool(doc["correct"]))}
        extra = doc["metrics"] or {}
        if "flexible_correct" in extra:
            values["flexible_correct"] = float(bool(extra["flexible_correct"]))
        values["generated_tokens"] = float(doc["generated_tokens"])
        values["truncated"] = float(doc["truncated"])
        samples[(doc["item_id"], doc["sample_index"])] = values
    return {task: samples}, {task: "correct"}


def load(path: Path) -> tuple[str, dict[str, Any], Rows, dict[str, str]]:
    """Return the protocol, panel fields, rows and headline metrics of a reduction."""
    report = json.loads(path.read_text())
    if not report.get("complete"):
        raise ValueError(f"{path} is not a complete reduction")
    role = report.get("role")
    if role == "prox-lm-eval":
        return role, panel(report), *lm_eval_rows(report)
    if role == "development-task":
        return role, panel(report), *olmes_rows(report)
    if report.get("data", {}).get("protocol") == "provider-thinking":
        return "provider-thinking", panel(report), *thinking_rows(report)
    raise ValueError(f"{path} is not a recognized reduction")


def compare(stock: list[float], policy: list[float], binary: bool) -> dict[str, Any]:
    """Return means, their change and flips (binary) or mean |change| (continuous).

    Flips are counted as correct->incorrect and incorrect->correct, as in Dutta et
    al. (NeurIPS 2024).

    `stock` and `policy` hold one value per question in the same order.
    """
    result: dict[str, Any] = {
        "count": len(stock),
        "stock": mean(stock),
        "policy": mean(policy),
        "delta": mean(policy) - mean(stock),
    }
    if binary:
        if any(value not in (0.0, 1.0) for value in stock + policy):
            raise ValueError("A correctness metric has a non-binary value")
        lost = sum(s == 1.0 and p == 0.0 for s, p in zip(stock, policy, strict=True))
        gained = sum(s == 0.0 and p == 1.0 for s, p in zip(stock, policy, strict=True))
        result.update(
            correct_to_incorrect=lost,
            incorrect_to_correct=gained,
            flips=lost + gained,
            flip_rate=(lost + gained) / len(stock),
        )
    else:
        result["mean_abs_change"] = mean(
            abs(p - s) for s, p in zip(stock, policy, strict=True)
        )
    return result


def group_of(task: str) -> str | None:
    """Return the pooled group of a subtask (MMLU subject, BBH subtask) or None."""
    if task.startswith("mmlu_"):
        return "mmlu"
    if task.startswith("bbh_"):
        return "bbh"
    return None


def question_means(rows: dict[Any, dict[str, float]]) -> dict[Any, float]:
    """Return the mean correctness of each question over its samples."""
    by_item: dict[Any, list[float]] = defaultdict(list)
    for (item, _), values in rows.items():
        by_item[item].append(values["correct"])
    return {item: mean(values) for item, values in by_item.items()}


def flips(stock_path: Path, policy_path: Path) -> dict[str, Any]:
    """Return per-task and pooled-group answer changes of a policy against stock.

    Both reductions must share the protocol, tasks, panel fields and questions.
    Thinking runs also report how each question's mean correctness moved.
    """
    kind, stock_panel, stock, headline = load(stock_path)
    policy_kind, policy_panel, policy, _ = load(policy_path)
    if policy_kind != kind or stock.keys() != policy.keys():
        raise ValueError("Stock and policy reduce different protocols or tasks")
    differing = sorted(
        key
        for key in stock_panel.keys() | policy_panel.keys()
        if stock_panel.get(key) != policy_panel.get(key)
    )
    if differing:
        raise ValueError(f"Stock and policy panels differ in {differing}")
    tasks: dict[str, Any] = {}
    pooled: dict[str, dict[str, tuple[list[float], list[float]]]] = defaultdict(dict)
    for task, stock_rows in stock.items():
        policy_rows = policy[task]
        if stock_rows.keys() != policy_rows.keys():
            raise ValueError(f"{task}: stock and policy score different questions")
        keys = sorted(stock_rows, key=str)
        metrics = {}
        for metric in stock_rows[keys[0]]:
            left = [stock_rows[k][metric] for k in keys]
            right = [policy_rows[k][metric] for k in keys]
            binary = metric in BINARY
            metrics[metric] = compare(left, right, binary)
            group = group_of(task)
            if group is not None and metric == headline[task]:
                pair = pooled[group].setdefault(metric, ([], []))
                pair[0].extend(left)
                pair[1].extend(right)
        tasks[task] = {"headline": headline[task], "metrics": metrics}
        if kind == "provider-thinking":
            left_means, right_means = (
                question_means(r) for r in (stock_rows, policy_rows)
            )
            changes = [
                right_means[i] - left_means[i] for i in sorted(left_means, key=str)
            ]
            tasks[task]["questions"] = {
                "count": len(changes),
                "samples_per_question": len(keys) / len(changes),
                "mean_correctness_decreased": sum(c < 0 for c in changes),
                "mean_correctness_increased": sum(c > 0 for c in changes),
                "mean_correctness_unchanged": sum(c == 0 for c in changes),
                "mean_abs_change": mean(abs(c) for c in changes),
            }
    groups = {
        group: {
            "headline": metric,
            "tasks": sum(group_of(task) == group for task in tasks),
            "metrics": {metric: compare(*pair, metric in BINARY)},
        }
        for group, by_metric in pooled.items()
        for metric, pair in by_metric.items()
    }
    return {
        "scope": "Per-question answer changes of the policy against stock on the same "
        "panel (Dutta et al., NeurIPS 2024); binary metrics count flips, continuous "
        "ones report mean change; groups pool their questions",
        "protocol": kind,
        "stock": str(stock_path),
        "policy": str(policy_path),
        "tasks": tasks,
        "groups": groups,
    }


def main() -> None:
    """Compare the two reductions, write the JSON and print a summary per row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True, help="Stock reduction")
    parser.add_argument("--policy", type=Path, required=True, help="Policy reduction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = flips(args.stock, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name, entry in {**result["tasks"], **result["groups"]}.items():
        row = entry["metrics"][entry["headline"]]
        if "flips" in row:
            print(
                f"{name} {entry['headline']}: "
                f"{row['stock']:.4f} -> {row['policy']:.4f}, "
                f"flips {row['flips']}/{row['count']} "
                f"(-{row['correct_to_incorrect']} +{row['incorrect_to_correct']})"
            )
        else:
            print(
                f"{name} {entry['headline']}: "
                f"{row['stock']:.4f} -> {row['policy']:.4f}, "
                f"mean |change| {row['mean_abs_change']:.4f}"
            )


if __name__ == "__main__":
    main()
