"""Statistics and score loading shared by the analysis reducers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reproduce.common import load_json, paired, question_means
from scripts.answer_flips import HEADLINE


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean and its standard error (sample deviation over sqrt(n))."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Rank values from 0, giving tied values their mean rank."""
    x = np.asarray(values, dtype=np.float64)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x), dtype=np.float64)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Return the Spearman rank correlation: Pearson correlation of average ranks."""
    a, b = average_ranks(x), average_ranks(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def question_scores(report: dict[str, Any]) -> dict[Any, float]:
    """Return the per-question correctness of one reduced benchmark run.

    Thinking-generation reductions (``thinking_requests.py reduce``) average each
    question's samples; lm-eval reductions (``olmes_requests.py reduce``, one task) take
    the task's headline metric per document, e.g. GSM8K strict-match.
    """
    if not report.get("complete"):
        raise ValueError("The reduction is incomplete")
    if report.get("data", {}).get("protocol") == "provider-thinking":
        samples = {
            (doc["item_id"], doc["sample_index"]): float(bool(doc["correct"]))
            for doc in report["documents"]
        }
        return question_means(samples)
    (task, result), *rest = report["tasks"].items()
    if rest:
        raise ValueError("Expected a single-task lm-eval reduction")
    metric = HEADLINE.get(task, "acc")
    return {doc["doc_id"]: float(doc[metric]) for doc in result["documents"]}


def paired_difference(policy: Path, base: Path) -> dict[str, float]:
    """Return policy minus base accuracy in points, paired by question, with one SE.

    A question's samples are averaged first, so the SE is clustered by question.
    """
    left, right = question_scores(load_json(policy)), question_scores(load_json(base))
    diff, se = paired(left, right)
    return {
        "policy": 100 * sum(left.values()) / len(left),
        "base": 100 * sum(right.values()) / len(right),
        "diff": diff,
        "se": se,
        "questions": len(left),
    }
