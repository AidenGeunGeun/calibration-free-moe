"""Summarize MoE-block and decode speed of the reader against the best stock vLLM.

Per batch size and metric, optimized stock is the fastest process median among all
stock candidates measured at that batch. Writes one JSON summary and prints a table
per batch.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
STOCK_ARMS = ("stock_auto", "stock_triton_untuned", "stock_triton_tuned")
PROCESSES = (1, 2, 3)
REPORT_NAME = re.compile(
    rf"(?P<arm>{'|'.join((*STOCK_ARMS, 'ours'))})-(?P<label>.+)-p(?P<process>\d+)\.json"
)
DECODE_METRICS = (
    ("tpot_seconds", 1000),
    ("ttft_seconds", 1000),
    ("decode_tokens_per_second", 1),
)


@dataclass
class Candidate:
    """One separately launched arm: three fresh processes with one config."""

    id: str
    arm: str
    label: str
    scope: str
    directory: Path
    reports: list[dict[str, Any]]
    batches: list[str]
    tuned_config_file: str | None
    tuned_config_batches: dict[str, int] | None

    def record(self) -> dict[str, Any]:
        """Return the candidate's identity, batches, tuning and commands."""
        return {
            "id": self.id,
            "arm": self.arm,
            "directory": str(self.directory),
            "label": self.label,
            "batches": [int(batch) for batch in self.batches],
            "tuned_config_file": self.tuned_config_file,
            "tuned_config_batches": self.tuned_config_batches,
            "packages": self.reports[0].get("packages"),
            "commands": [report["command"] for report in self.reports],
        }


def process_values(values: list[float]) -> dict[str, Any]:
    """Return the median, range and samples of one value per process."""
    return {
        "median": statistics.median(values),
        "range": [min(values), max(values)],
        "samples": values,
    }


def unique(values: list[Any], what: str, path: Path) -> Any:
    """Return the one value all processes recorded, or fail on a conflict."""
    distinct = {json.dumps(value, sort_keys=True) for value in values}
    if len(distinct) != 1:
        raise ValueError(f"Conflicting {what} within one candidate: {path}")
    return values[0]


def validate(
    directory: Path,
    arm: str,
    label: str,
    processes: dict[int, dict[str, Any]],
    revision: str,
) -> Candidate:
    """Check one arm's three process reports and return them as a candidate."""
    path = directory / f"{arm}-{label}-p*.json"
    if set(processes) != set(PROCESSES):
        raise ValueError(f"Expected processes {PROCESSES}: {path}")
    reports = [processes[number] for number in PROCESSES]
    for report in reports:
        if not report.get("complete") or report["revision"] != revision:
            raise ValueError(f"Incomplete or wrong-revision result: {path}")
        if not report["source"].endswith(f"source-{revision}"):
            raise ValueError(f"Inconsistent source snapshot: {path}")
    decode = "expert_backend" in reports[0]
    # Reports from runs with profiler trace markers are not clean timings.
    if decode and any(
        report.get("mode") != "time" or report.get("trace_markers")
        for report in reports
    ):
        raise ValueError(f"Not an unmarked end-to-end timing run: {path}")
    backends = [
        report["expert_backend"][0] if decode else report["backend"]
        for report in reports
    ]
    batches = unique(
        [sorted(report["panels"], key=int) for report in reports], "batches", path
    )
    if not batches or not {int(batch) for batch in batches} <= set(BATCHES):
        raise ValueError(f"Batch grid outside {BATCHES}: {path}")
    for report in reports:
        for key, panel in report["panels"].items():
            if decode:
                recorded = panel.get("batch")
                if any(
                    len(sample["requests"]) != int(key) for sample in panel["samples"]
                ):
                    raise ValueError(
                        f"Decode sample size differs from batch {key}: {path}"
                    )
            else:
                recorded = panel["measurement"].get("batch")
            if recorded is not None and recorded != int(key):
                raise ValueError(f"Recorded batch {recorded} labelled {key}: {path}")
    readers = {
        layer["decode_reader"] for backend in backends for layer in backend["layers"]
    }
    if readers != ({"ProgressiveReader"} if arm == "ours" else {None}):
        raise ValueError(f"Wrong reader for {arm}: {path}")
    kernels = {
        layer["experts_kernel"] for backend in backends for layer in backend["layers"]
    }
    tuned = [
        report["stock_config_json"] if decode else report["stock_tuned_config"]
        for report in reports
    ]
    if arm == "stock_triton_tuned":
        if not all(tuned) or kernels != {"TritonExperts"}:
            raise ValueError(f"Stock tuner config/backend not loaded: {path}")
        tuned_file = unique(
            [item if decode else item["file"] for item in tuned], "tuned config", path
        )
        tuned_batches = unique(
            [
                report["stock_tuned_batch_lookup"]
                if decode
                else report["stock_tuned_config"]["batch_to_config_batch"]
                for report in reports
            ],
            "tuned batch lookup",
            path,
        )
    elif arm != "ours" and any(item is not None for item in tuned):
        raise ValueError(f"{arm} inherited tuning: {path}")
    else:
        tuned_file = tuned_batches = None
    unique(
        [backend.get("triton_config_values") for backend in backends],
        "Triton configs",
        path,
    )
    unique([report.get("packages") for report in reports], "packages", path)
    return Candidate(
        id=f"{directory.name}/{arm}-{label}",
        arm=arm,
        label=label,
        scope="decode" if decode else "block",
        directory=directory,
        reports=reports,
        batches=batches,
        tuned_config_file=tuned_file,
        tuned_config_batches=tuned_batches,
    )


def load_candidates(directory: Path, revision: str, method: bool) -> list[Candidate]:
    """Return every candidate in a directory; `method` also loads our reader's runs."""
    groups: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*-p*.json")):
        match = REPORT_NAME.fullmatch(path.name)
        if match is None or (match["arm"] == "ours" and not method):
            continue
        group = groups.setdefault((match["arm"], match["label"]), {})
        group[int(match["process"])] = json.loads(path.read_text())
    return [
        validate(directory, arm, label, processes, revision)
        for (arm, label), processes in groups.items()
    ]


def block_values(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Return per-process median MoE-block step times (ms) at one batch."""
    records = [report["panels"][key]["measurement"] for report in reports]
    values = process_values([record["per_step_median_ms"] for record in records])
    values["within_process_ranges_ms"] = [
        record["per_step_range_ms"] for record in records
    ]
    return values


def decode_values(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Return per-process TPOT and TTFT (ms) and decode tokens/s at one batch."""
    return {
        metric: process_values(
            [
                report["panels"][key]["summary"][metric]["median"] * scale
                for report in reports
            ]
        )
        for metric, scale in DECODE_METRICS
    }


def measured(stock: list[Candidate], key: str) -> list[Candidate]:
    """Return the stock candidates measured at a batch; auto stock must be one."""
    candidates = [candidate for candidate in stock if key in candidate.batches]
    if not any(candidate.arm == "stock_auto" for candidate in candidates):
        raise ValueError(f"Unrestricted auto stock was not measured at batch {key}")
    return candidates


def winner(candidate: Candidate, key: str) -> dict[str, Any]:
    """Return the identity and tuning of the stock candidate that won a batch."""
    return {
        "candidate": candidate.id,
        "arm": candidate.arm,
        "tuned_config_file": candidate.tuned_config_file,
        "tuned_config_batch": (
            candidate.tuned_config_batches[key]
            if candidate.tuned_config_batches
            else None
        ),
    }


def summarize_block(
    stock: list[Candidate], ours: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return per-batch MoE-block time of the fastest stock and of our reader."""
    result = {}
    for batch in BATCHES:
        key = str(batch)
        candidates = measured(stock, key)
        arms = {
            candidate.id: block_values(candidate.reports, key)
            for candidate in candidates
        }
        best = min(candidates, key=lambda candidate: arms[candidate.id]["median"])
        ours_values = block_values(ours, key)
        result[key] = {
            "optimized_stock": winner(best, key),
            "optimized_stock_ms": arms[best.id]["median"],
            "ours_ms": ours_values["median"],
            "stock_over_ours": arms[best.id]["median"] / ours_values["median"],
            "stock_candidates": arms,
            "ours": ours_values,
        }
    return result


def summarize_decode(
    stock: list[Candidate], ours: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return per-batch decode TPOT, TTFT and throughput, best stock per metric."""
    result = {}
    for batch in BATCHES:
        key = str(batch)
        candidates = measured(stock, key)
        arms = {
            candidate.id: decode_values(candidate.reports, key)
            for candidate in candidates
        }
        ours_values = decode_values(ours, key)
        tpot, ttft, throughput = (
            min(candidates, key=lambda c: arms[c.id]["tpot_seconds"]["median"]),
            min(candidates, key=lambda c: arms[c.id]["ttft_seconds"]["median"]),
            max(
                candidates,
                key=lambda c: arms[c.id]["decode_tokens_per_second"]["median"],
            ),
        )
        stock_tpot = arms[tpot.id]["tpot_seconds"]["median"]
        ours_tpot = ours_values["tpot_seconds"]["median"]
        result[key] = {
            "optimized_stock": winner(tpot, key),
            "stock_tpot_ms": stock_tpot,
            "ours_tpot_ms": ours_tpot,
            "stock_over_ours": stock_tpot / ours_tpot,
            "optimized_stock_ttft": winner(ttft, key),
            "stock_ttft_ms": arms[ttft.id]["ttft_seconds"]["median"],
            "ours_ttft_ms": ours_values["ttft_seconds"]["median"],
            "optimized_stock_throughput": winner(throughput, key),
            "stock_decode_tokens_per_second": arms[throughput.id][
                "decode_tokens_per_second"
            ]["median"],
            "ours_decode_tokens_per_second": ours_values["decode_tokens_per_second"][
                "median"
            ],
            "stock_candidates": arms,
            "ours": ours_values,
        }
    return result


def recorded_once(values: list[Any], what: str) -> Any:
    """Return the one recorded value across reports, ignoring unrecorded ones."""
    distinct = {
        json.dumps(value, sort_keys=True) for value in values if value is not None
    }
    if len(distinct) > 1:
        raise ValueError(f"Compared workers recorded different {what}: {distinct}")
    return json.loads(next(iter(distinct))) if distinct else None


def gpus(candidate: Candidate) -> list[dict[str, str]]:
    """Return every GPU observation a candidate's reports recorded."""
    if candidate.scope == "block":
        return [report["gpu_before"] for report in candidate.reports]
    return [
        sample["gpu_before"]
        for report in candidate.reports
        for panel in report["panels"].values()
        for sample in panel["samples"]
    ]


def reduce(
    primary: Path, stock_directories: list[Path], model: str, revision: str
) -> dict[str, Any]:
    """Load and check every candidate, then return the speed summary.

    A candidate is one arm's three `{arm}-{label}-p{1,2,3}.json` reports. The
    reader's runs (arm `ours`, block and, for OLMoE and Thinking, end-to-end) come
    only from `primary` and must cover every batch; stock candidates come from all
    directories, may cover a subset of batches, and need not record versions. Fails
    on reports that disagree on revision, batch, config, model path, GPU, driver or
    packages.
    """
    directories = list(dict.fromkeys([primary, *stock_directories]))
    scopes = ("block", "decode") if model in ("olmoe", "thinking") else ("block",)
    candidates: list[Candidate] = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"Result directory does not exist: {directory}")
        loaded = load_candidates(directory, revision, directory == primary)
        if directory != primary and not any(
            candidate.scope in scopes for candidate in loaded
        ):
            raise ValueError(f"Stock directory has no stock candidate: {directory}")
        candidates.extend(loaded)
    if len({candidate.id for candidate in candidates}) != len(candidates):
        raise ValueError("Result directories must have distinct names")
    ours = {
        candidate.scope: candidate
        for candidate in candidates
        if candidate.arm == "ours"
    }
    if len(ours) != len([c for c in candidates if c.arm == "ours"]):
        raise ValueError("Primary directory has more than one method run per scope")
    stock = [candidate for candidate in candidates if candidate.arm != "ours"]
    for scope in scopes:
        if scope not in ours or ours[scope].batches != [
            str(batch) for batch in BATCHES
        ]:
            raise ValueError(f"Primary directory lacks a full-grid ours {scope} run")
    reports = [report for candidate in candidates for report in candidate.reports]
    # Block workers observe the GPU once per process; end-to-end workers
    # observe it before every timed sample.
    observations = [
        observation for candidate in candidates for observation in gpus(candidate)
    ]
    uuids = {observation["uuid"] for observation in observations}
    if len(uuids) != 1:
        raise ValueError(f"Compared runs did not use one physical GPU: {uuids}")
    by_scope = {
        scope: [candidate for candidate in stock if candidate.scope == scope]
        for scope in scopes
    }
    return {
        "model": model,
        "revision": revision,
        "model_path": recorded_once(
            [report["config"].get("model") for report in reports], "models"
        ),
        "gpu_uuid": next(iter(uuids)),
        "gpu_driver_version": recorded_once(
            [observation["driver_version"] for observation in observations],
            "driver versions",
        ),
        "runtime_packages": recorded_once(
            [report.get("packages") for report in reports], "packages"
        ),
        "primary_directory": str(primary),
        "stock_directories": [str(directory) for directory in directories[1:]],
        "stock_candidates": {
            scope: [candidate.record() for candidate in by_scope[scope]]
            for scope in scopes
        },
        "ours": {scope: ours[scope].record() for scope in scopes},
        "block": summarize_block(by_scope["block"], ours["block"].reports),
        "decode": (
            summarize_decode(by_scope["decode"], ours["decode"].reports)
            if "decode" in scopes
            else None
        ),
    }


def main() -> None:
    """Reduce the directories, write the summary and print a table per batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--stock-directory", type=Path, action="append", default=[])
    parser.add_argument(
        "--model", choices=("olmoe", "thinking", "q35", "gemma"), required=True
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(
        args.directory.resolve(),
        [directory.resolve() for directory in args.stock_directory],
        args.model,
        args.revision,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for scope, candidates in result["stock_candidates"].items():
        print(scope, "stock candidates:", [item["id"] for item in candidates])
    print(
        "B  optimized stock  FFN stock/reader ms  stock/reader  "
        "decode stock/reader TPOT ms"
    )
    for batch in BATCHES:
        b = result["block"][str(batch)]
        d = result["decode"][str(batch)] if result["decode"] else None
        print(
            batch,
            b["optimized_stock"]["candidate"],
            round(b["optimized_stock_ms"], 4),
            round(b["ours_ms"], 4),
            round(b["stock_over_ours"], 3),
            (round(d["stock_tpot_ms"], 4), round(d["ours_tpot_ms"], 4)) if d else "-",
        )


if __name__ == "__main__":
    main()
