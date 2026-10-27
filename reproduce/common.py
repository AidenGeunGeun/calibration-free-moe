"""Load run reports and compute the paired statistics shared by the reducers.

A run writes `metrics.json` (per-question scores) and `scores.json` (generations and
read statistics), each possibly as zstd-compressed `.json.zst`. A run is a directory
holding these files, or a file prefix `<run>-metrics.json`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import zstandard


def load_json(path: Path) -> Any:
    """Read `path`, or its compressed twin `path.zst` when only that is retained."""
    if path.exists():
        return json.loads(path.read_text())
    with path.with_name(path.name + ".zst").open("rb") as handle:
        return json.loads(zstandard.ZstdDecompressor().stream_reader(handle).read())


def exists(path: Path) -> bool:
    """Return whether `path` or its compressed twin exists."""
    return path.exists() or path.with_name(path.name + ".zst").exists()


def report_path(run: Path, kind: str) -> Path:
    """Locate the `kind` report ('metrics' or 'scores') of a run directory or prefix."""
    for path in (run / f"{kind}.json", run.with_name(f"{run.name}-{kind}.json")):
        if exists(path):
            return path
    raise FileNotFoundError(f"no {kind}.json[.zst] for run {run}")


def report(run: Path, kind: str) -> dict[str, Any]:
    """Load a report of a run; raise if the run did not finish."""
    path = report_path(run, kind)
    data = load_json(path)
    if not data.get("complete", True):
        raise ValueError(f"{path} is not complete")
    return data


def work(scores: Mapping[str, Any]) -> dict[str, Any]:
    """Return a run's routed-expert read statistics, our reader's or a baseline's."""
    for key in ("reader_work", "baseline_work"):
        if scores.get(key) is not None:
            return scores[key]
    raise KeyError("run has no reader_work or baseline_work")


def proxy_macs(w: Mapping[str, Any]) -> float:
    """Sum the proxy multiply-accumulates of Prox's read statistics.

    These hold either the total or per-layer entries: a dict, or a list of dicts,
    whose MACs sit at the top level or one dict deeper.
    """
    if "proxy_matrix_macs" in w:
        return w["proxy_matrix_macs"]
    total = 0.0
    for layer in w.get("layers", []):
        for part in layer if isinstance(layer, list) else [layer]:
            if not isinstance(part, dict):
                continue
            if "proxy_matrix_macs" in part:
                total += part["proxy_matrix_macs"]
            else:
                total += sum(
                    p["proxy_matrix_macs"]
                    for p in part.values()
                    if isinstance(p, dict) and "proxy_matrix_macs" in p
                )
    return total


def read_percent(scores: Mapping[str, Any]) -> float:
    """Return the routed-expert read cost as a percentage of the dense routed reads.

    Read cost (`charged_bytes`) is the bytes a method must read, priced in 32-byte
    sectors. Prox is priced on its best single layout, a coordinate-major INT4 proxy
    with two weights per byte, so its proxy costs (proxy MACs / 2) bytes; this replaces
    a larger `proxy_bytes` that some Prox runs recorded from another layout.
    """
    w = work(scores)
    charged = w["charged_bytes"]
    if w.get("proxy_bytes"):
        charged = charged - w["proxy_bytes"] + proxy_macs(w) / 2
    return 100 * charged / w["reference_bytes"]


def question_means[K: Hashable](
    samples: Mapping[tuple[K, int], float],
) -> dict[K, float]:
    """Average each question's score over its samples, keyed by question."""
    by_item: dict[K, list[float]] = defaultdict(list)
    for (item, _), value in samples.items():
        by_item[item].append(value)
    return {item: sum(values) / len(values) for item, values in by_item.items()}


def paired[K: Hashable](
    a: Mapping[K, float], b: Mapping[K, float]
) -> tuple[float, float]:
    """Return the mean of a - b over questions and its standard error, in points.

    Inputs hold one value per question (per-question means when a question has
    several samples), so the standard error is clustered by question. Both must score
    the same questions. Values are fractions; the results are scaled by 100.
    """
    if a.keys() != b.keys():
        raise ValueError("paired runs score different questions")
    keys = sorted(a, key=str)
    d = [a[k] - b[k] for k in keys]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return 100 * m, 100 * se


def verdict(diff: float, se: float) -> str:
    """Classify threshold minus top-k: 'thr' or 'topk' beyond two SE, else 'level'."""
    if diff > 2 * se:
        return "thr"
    if diff < -2 * se:
        return "topk"
    return "level"
