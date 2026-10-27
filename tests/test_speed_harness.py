import argparse
import json
import os
import sys
import sysconfig
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.speed_harness import (
    associate_prompts,
    configure_source,
    prompt_panel,
    stock_union_bytes,
    timing_sample,
    validate_stock_vllm_import,
    validate_tuned_stock,
)


def test_speed_panel_keeps_one_real_source_at_every_batch_size(tmp_path):
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "role": "development",
                "generation": [{"id": "generated", "tokens": [9]}],
                "documents": [
                    {"id": f"document-{index}", "tokens": [index, 1, 2]}
                    for index in range(128)
                ],
            }
        )
    )
    first = prompt_panel(panel, 1, 2)
    largest = prompt_panel(panel, 128, 2)
    assert first == largest[:1] == [{"id": "document-0", "tokens": [0, 1]}]
    with pytest.raises(ValueError, match="requested token count"):
        prompt_panel(panel, 1, 4)


def test_stock_reads_are_the_per_step_union_of_full_expert_rows():
    prompts = [
        {"id": "a", "request_id": "req-42", "tokens": [1, 2]},
        {"id": "b", "request_id": "req-17", "tokens": [3, 4, 5]},
    ]
    a = np.array([[[4]], [[4]], [[1]], [[2]]])
    b = np.array([[[4]], [[4]], [[4]], [[4]], [[3]]])
    outputs = [
        SimpleNamespace(
            request_id="req-17", outputs=[SimpleNamespace(routed_experts=b)]
        ),
        SimpleNamespace(
            request_id="req-42", outputs=[SimpleNamespace(routed_experts=a)]
        ),
    ]
    assert stock_union_bytes(outputs, prompts, 1, 1, 96, 3) == [192, 192]
    outputs[0].outputs[0].routed_experts = b[:-1]
    with pytest.raises(ValueError, match="Incomplete routed-expert capture"):
        stock_union_bytes(outputs, prompts, 1, 1, 96, 3)


def test_completed_prompts_use_external_ids_not_randomized_engine_ids():
    prompts = [{"id": "a", "tokens": [1, 2]}, {"id": "b", "tokens": [3, 4]}]
    outputs = [
        SimpleNamespace(request_id="1", prompt_token_ids=[3, 4]),
        SimpleNamespace(request_id="0", prompt_token_ids=[1, 2]),
    ]
    associate_prompts(outputs, prompts)
    assert [item["request_id"] for item in prompts] == ["0", "1"]
    outputs[0].prompt_token_ids = [8, 9]
    with pytest.raises(RuntimeError, match="Returned prompts differ"):
        associate_prompts(outputs, prompts)


def test_timing_records_stable_prompt_id_independent_of_worker_request_id():
    prompts = [{"id": "source-text-0", "request_id": "worker-923"}]
    metrics = SimpleNamespace(
        is_corrupted=False,
        arrival_time=1_700_000_000.0,
        scheduled_ts=1.1,
        first_token_ts=1.2,
        last_token_ts=1.8,
        first_token_latency=0.2,
    )
    outputs = [
        SimpleNamespace(
            request_id="worker-923",
            metrics=metrics,
            outputs=[SimpleNamespace(token_ids=[2, 3, 4])],
        )
    ]
    record = timing_sample(outputs, prompts, 1.0, 3)
    assert record["requests"][0]["prompt_id"] == "source-text-0"
    assert record["requests"][0]["token_ids"] == [2, 3, 4]
    assert record["ttft_seconds"] == pytest.approx(0.2)
    assert record["decode_tokens_per_second"] == pytest.approx(2 / 0.6)


def test_stock_tuning_must_match_worker_loaded_config(tmp_path):
    file = tmp_path / "E=64,N=1024,dtype=bf16.json"
    file.write_text(json.dumps({"triton_version": "3.7.1", "1": {"BLOCK_SIZE_M": 16}}))
    args = argparse.Namespace(
        arm="stock_triton_tuned", tuned_config_dir=tmp_path, batches=[1, 2]
    )
    backend: dict[str, Any] = {
        "layers": [{"experts_kernel": "TritonExperts"}],
        "triton_config_filename": file.name,
        "triton_config_values": {1: {"BLOCK_SIZE_M": 16}},
    }
    assert validate_tuned_stock(args, backend) == {
        "file": str(file.resolve()),
        "batch_to_config_batch": {"1": 1, "2": 1},
    }
    backend["triton_config_values"][1]["BLOCK_SIZE_M"] = 32
    with pytest.raises(RuntimeError, match="Worker did not load"):
        validate_tuned_stock(args, backend)


def test_method_worker_inherits_vllm_overlay_before_installed_package(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    package = source / "calibration_free_moe"
    package.mkdir(parents=True)
    (package / "model_config.py").write_text("")
    overlay = tmp_path / "vllm-method"
    overlay.mkdir()
    args = argparse.Namespace(
        source=source,
        overlay=overlay,
        arm="ours",
        mode="time",
        revision="rev",
        cache_root=tmp_path / "cache",
        tuned_config_dir=None,
    )
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    original_path = sys.path[:]
    try:
        configure_source(args)
        assert sys.path[:2] == [str(source), str(overlay)]
        assert str(overlay) in os.environ["PYTHONPATH"]
    finally:
        sys.path[:] = original_path


def test_stock_import_must_resolve_inside_venv_package(tmp_path, monkeypatch):
    installed = tmp_path / "venv" / "site-packages" / "vllm"
    installed.mkdir(parents=True)
    injected = tmp_path / "unrelated-name" / "vllm" / "__init__.py"
    injected.parent.mkdir(parents=True)
    monkeypatch.setattr(
        sysconfig, "get_paths", lambda: {"platlib": str(installed.parent)}
    )
    validate_stock_vllm_import("stock_auto", str(installed / "__init__.py"))
    with pytest.raises(RuntimeError, match="outside its venv site-packages"):
        validate_stock_vllm_import("stock_auto", str(injected))
    (installed / "escaped.py").symlink_to(injected)
    with pytest.raises(RuntimeError, match="outside its venv site-packages"):
        validate_stock_vllm_import("stock_auto", str(installed / "escaped.py"))
    validate_stock_vllm_import("ours", str(injected))
