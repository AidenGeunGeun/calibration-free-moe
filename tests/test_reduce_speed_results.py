import json
from pathlib import Path
from typing import Any

import pytest

from scripts.reduce_speed_results import BATCHES, reduce

TUNED = "/configs/E=64.json"


def block_report(arm: str, ms: float, batches: tuple[int, ...], config: str) -> Any:
    return {
        "complete": True,
        "revision": "r",
        "source": "/runs/source-r",
        "command": [arm, config],
        "config": {"model": "/models/q35"},
        "gpu_before": {"uuid": "GPU-5", "driver_version": "580"},
        "backend": {
            "layers": [
                {
                    "decode_reader": "ProgressiveReader" if arm == "ours" else None,
                    "experts_kernel": "TritonExperts",
                }
            ]
        },
        "stock_tuned_config": (
            {"file": config, "batch_to_config_batch": {str(b): 1 for b in batches}}
            if arm == "stock_triton_tuned"
            else None
        ),
        "panels": {
            str(batch): {
                "measurement": {
                    "batch": batch,
                    "per_step_median_ms": ms,
                    "per_step_range_ms": [ms - 0.1, ms + 0.1],
                }
            }
            for batch in batches
        },
    }


def decode_report(block: Any, tpot: float) -> Any:
    """Convert a block fixture; TTFT and throughput move against TPOT."""
    tuned = block.pop("stock_tuned_config")
    backend = block.pop("backend")
    # The end-to-end harness observes the GPU per timed sample, not per report.
    gpu = block.pop("gpu_before")
    return {
        **block,
        "mode": "time",
        "packages": {"vllm": "0.29.0"},
        "expert_backend": [backend],
        "stock_config_json": tuned["file"] if tuned else None,
        "stock_tuned_batch_lookup": tuned["batch_to_config_batch"] if tuned else None,
        "panels": {
            key: {
                "batch": int(key),
                "samples": [{"requests": [{}] * int(key), "gpu_before": gpu}],
                "summary": {
                    "tpot_seconds": {"median": tpot / 1000},
                    "ttft_seconds": {"median": 1 / tpot},
                    "decode_tokens_per_second": {"median": 1000 / tpot},
                },
            }
            for key in block["panels"]
        },
    }


def write_run(
    directory: Path,
    arm: str,
    label: str,
    values: list[float],
    batches: tuple[int, ...] = BATCHES,
    config: str = TUNED,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for process, ms in enumerate(values, start=1):
        report = block_report(arm, ms, batches, config)
        if label == "e2e":
            report = decode_report(report, ms)
        path = directory / f"{arm}-{label}-p{process}.json"
        path.write_text(json.dumps(report))


def test_stock_uses_process_medians_and_keeps_every_tuned_run(tmp_path: Path):
    before = tmp_path / "final-core-q35-before-tuned"
    rerun = tmp_path / "final-core-q35"
    # One slow auto sample must not decide the choice: median 10, not 100.
    write_run(before, "stock_auto", "block", [10.0, 10.0, 100.0])
    write_run(before, "stock_triton_tuned", "block", [5.0, 5.0, 5.0])
    write_run(before, "ours", "block", [1.0, 1.0, 1.0])
    # A refined tuned config that loses must not replace the faster original.
    write_run(rerun, "stock_triton_tuned", "block", [8.0, 8.0, 8.0], (1, 2), "/new")
    write_run(rerun, "ours", "block", [4.0, 4.0, 4.0])
    result = reduce(rerun, [before], "q35", "r")
    for batch in ("1", "8"):
        row = result["block"][batch]
        assert row["optimized_stock"]["candidate"] == (
            "final-core-q35-before-tuned/stock_triton_tuned-block"
        )
        assert row["optimized_stock"]["tuned_config_file"] == "/configs/E=64.json"
        assert row["stock_over_ours"] == 1.25
    assert set(result["block"]["1"]["stock_candidates"]) == {
        "final-core-q35-before-tuned/stock_auto-block",
        "final-core-q35-before-tuned/stock_triton_tuned-block",
        "final-core-q35/stock_triton_tuned-block",
    }
    assert result["block"]["1"]["stock_candidates"][
        "final-core-q35-before-tuned/stock_auto-block"
    ]["range"] == [10.0, 100.0]
    assert result["decode"] is None


def test_auto_only_model_is_reported_as_auto_only(tmp_path: Path):
    write_run(tmp_path / "gemma", "stock_auto", "block", [10.0, 10.0, 10.0])
    write_run(tmp_path / "gemma", "ours", "block", [8.0, 8.0, 8.0])
    result = reduce(tmp_path / "gemma", [], "gemma", "r")
    row = result["block"]["128"]
    assert row["optimized_stock"] == {
        "candidate": "gemma/stock_auto-block",
        "arm": "stock_auto",
        "tuned_config_file": None,
        "tuned_config_batch": None,
    }
    assert row["stock_over_ours"] == 1.25


def test_recorded_batch_contradicting_its_label_fails(tmp_path: Path):
    write_run(tmp_path, "stock_auto", "block", [10.0, 10.0, 10.0])
    write_run(tmp_path, "ours", "block", [8.0, 8.0, 8.0])
    path = tmp_path / "ours-block-p2.json"
    report = json.loads(path.read_text())
    report["panels"]["4"]["measurement"]["batch"] = 2
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Recorded batch 2 labelled 4"):
        reduce(tmp_path, [], "gemma", "r")


def test_decode_selects_ttft_independently_of_tpot(tmp_path: Path):
    for arm, tpot in (("stock_auto", 10.0), ("stock_triton_tuned", 9.0)):
        write_run(tmp_path, arm, "block", [tpot] * 3)
        write_run(tmp_path, arm, "e2e", [tpot] * 3)
    write_run(tmp_path, "ours", "block", [8.0] * 3)
    write_run(tmp_path, "ours", "e2e", [8.0] * 3)
    result = reduce(tmp_path, [], "olmoe", "r")
    assert result["gpu_driver_version"] == "580"
    assert result["runtime_packages"] == {"vllm": "0.29.0"}
    row = result["decode"]["1"]
    name = tmp_path.name
    assert row["optimized_stock"]["candidate"] == f"{name}/stock_triton_tuned-e2e"
    assert row["stock_tpot_ms"] == 9.0
    assert row["optimized_stock_ttft"]["candidate"] == f"{name}/stock_auto-e2e"
    assert row["stock_ttft_ms"] == 100.0
    assert row["optimized_stock_throughput"]["arm"] == "stock_triton_tuned"


def test_misspelled_or_empty_stock_directory_fails(tmp_path: Path):
    write_run(tmp_path / "run", "stock_auto", "block", [10.0] * 3)
    write_run(tmp_path / "run", "ours", "block", [4.0] * 3)
    with pytest.raises(ValueError, match="does not exist"):
        reduce(tmp_path / "run", [tmp_path / "typo"], "q35", "r")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no stock candidate"):
        reduce(tmp_path / "run", [tmp_path / "empty"], "q35", "r")


def test_stock_from_another_model_fails(tmp_path: Path):
    write_run(tmp_path / "run", "stock_auto", "block", [10.0] * 3)
    write_run(tmp_path / "run", "ours", "block", [4.0] * 3)
    write_run(tmp_path / "other", "stock_triton_tuned", "block", [5.0] * 3)
    for path in (tmp_path / "other").iterdir():
        report = json.loads(path.read_text())
        report["config"]["model"] = "/models/gemma"
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different models"):
        reduce(tmp_path / "run", [tmp_path / "other"], "q35", "r")
