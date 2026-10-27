"""Measure end-to-end vLLM decode speed and expert reads, one arm per fresh process.

`time` writes TPOT, TTFT and decode throughput per batch of panel prompts, `bytes`
reruns a timed panel to count the expert bytes read per decode step, and `tune` runs
vLLM's unmodified MoE tuner for the tuned stock arm. The reader's arm needs
`--overlay`, a vLLM 0.29.0 tree with `patches/` applied.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

import numpy as np

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
STOCK_ARMS = ("stock_auto", "stock_triton_untuned", "stock_triton_tuned")


def validate_stock_vllm_import(arm: str, imported_file: str | None) -> None:
    """Fail if a stock arm imported vLLM from outside the venv's site-packages."""
    if arm not in STOCK_ARMS:
        return
    installed = (Path(sysconfig.get_paths()["platlib"]) / "vllm").resolve()
    if imported_file is None or not Path(imported_file).resolve().is_relative_to(
        installed
    ):
        raise RuntimeError(
            f"Stock {arm} imported vLLM outside its venv site-packages: "
            f"{imported_file!r} (expected under {installed})"
        )


class SpeedWorkerExtension:
    """Identify the instantiated expert implementation, not merely its request."""

    def speed_backend_report(
        self: Any, experts: int, width: int, arm: str
    ) -> dict[str, Any]:
        """Return each MoE layer's reader and expert kernel, and Triton configs."""
        import torch
        import vllm
        from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            get_config_file_name,
            get_moe_configs,
        )
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        worker_vllm_file = vllm.__file__
        if worker_vllm_file is None:
            raise RuntimeError("Cannot identify the worker vLLM package")
        validate_stock_vllm_import(arm, worker_vllm_file)
        result = []
        for name, module in self.model_runner.model.named_modules():
            if not isinstance(module, RoutedExperts):
                continue
            method = module.quant_method
            kernel = getattr(method, "moe_kernel", None)
            result.append(
                {
                    "layer": name,
                    "quant_method": type(method).__name__,
                    "decode_reader": type(module.reader).__name__
                    if getattr(module, "reader", None) is not None
                    else None,
                    "experts_kernel": (
                        type(kernel.fused_experts).__name__
                        if kernel is not None
                        else "not_instantiated"
                    ),
                }
            )
        report: dict[str, Any] = {
            "layers": result,
            "worker_vllm_file": str(Path(worker_vllm_file).resolve()),
        }
        if not any(item["experts_kernel"] == "TritonExperts" for item in result):
            return report
        dtype = _get_config_dtype_str(torch.bfloat16)
        report["triton_config_filename"] = get_config_file_name(experts, width, dtype)
        report["triton_config_values"] = get_moe_configs(experts, width, dtype)
        return report


def prompt_panel(path: Path, count: int, maximum: int) -> list[dict[str, Any]]:
    """Return the first `count` panel prompts, each cut to `maximum` tokens."""
    panel = json.loads(path.read_text())
    if panel.get("role") != "development":
        raise ValueError("Speed prompts must come from a fixed development panel")
    candidates = (
        panel.get("requests") or panel.get("documents") or panel.get("generation", [])
    )
    if len(candidates) < count:
        raise ValueError(f"Panel has {len(candidates)} prompts, needs {count}")
    prompts = []
    for item in candidates[:count]:
        tokens = item.get("prompt", item.get("tokens"))
        if (
            not isinstance(tokens, list)
            or len(tokens) < maximum
            or any(not isinstance(token, int) or token < 0 for token in tokens)
        ):
            raise ValueError("Panel lacks a real prompt with the requested token count")
        prompts.append({"id": item["id"], "tokens": tokens[:maximum]})
    if len({item["id"] for item in prompts}) != count:
        raise ValueError("Speed prompt IDs must be distinct")
    return prompts


def stock_union_bytes(
    outputs: list[Any],
    prompts: list[dict[str, Any]],
    layers: int,
    top_k: int,
    bytes_per_expert: int,
    output_tokens: int,
) -> list[int]:
    """Return the bytes of the routed-expert union over requests per decode step.

    Each layer's routed experts are united across requests and counted at
    `bytes_per_expert` each; the result has one total per decode step.
    """
    by_id = {item["request_id"]: item for item in prompts}
    if {output.request_id for output in outputs} != set(by_id):
        raise ValueError("Returned requests differ from the fixed speed panel")
    routed = []
    for output in outputs:
        prompt = by_id[output.request_id]
        routes = output.outputs[0].routed_experts
        expected = len(prompt["tokens"]) + output_tokens - 1
        if routes is None or routes.shape != (expected, layers, top_k):
            raise ValueError(
                f"Incomplete routed-expert capture for {output.request_id}: "
                f"{None if routes is None else routes.shape}, expected "
                f"{(expected, layers, top_k)}"
            )
        routed.append(routes[len(prompt["tokens"]) :])
    return [
        sum(
            len(np.unique(np.concatenate([route[step, layer] for route in routed])))
            * bytes_per_expert
            for layer in range(layers)
        )
        for step in range(output_tokens - 1)
    ]


def distribution(values: list[float]) -> dict[str, float]:
    """Return the median, mean, min, max and median absolute deviation."""
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mad": statistics.median(abs(x - statistics.median(ordered)) for x in ordered),
    }


def gpu_state(index: int) -> dict[str, str]:
    """Return the target GPU's identity, clocks, power, temperature and load."""
    fields = (
        "index,uuid,name,driver_version,clocks.sm,clocks.mem,power.draw,"
        "power.limit,temperature.gpu,utilization.gpu,memory.used"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        raise RuntimeError("Incomplete target GPU clock/power observation")
    return dict(zip(names, values, strict=True))


def host_card_occupancy() -> dict[str, Any]:
    """Return memory use and utilization of every GPU on the host."""
    fields = "index,memory.used,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    rows = [
        [part.strip() for part in line.split(",")]
        for line in result.stdout.splitlines()
    ]
    if not rows or any(len(row) != 3 for row in rows):
        raise RuntimeError("Cannot observe other cards on the measurement host")
    partial = result.returncode != 0 or bool(result.stderr.strip())
    return {
        "cards": [dict(zip(fields.split(","), row, strict=True)) for row in rows],
        "partial": partial,
        "diagnostic": result.stderr.strip() if partial else None,
    }


def write(report: dict[str, Any], path: Path) -> None:
    """Write the report as indented JSON, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def configure_source(args: argparse.Namespace) -> None:
    """Put the source (and our reader's vLLM overlay) on the path and set vLLM env.

    Requires exactly one visible GPU in PCI bus order. Each revision, arm and mode
    gets its own vLLM cache; only the tuned stock arm sees the tuned config.
    """
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if (
        os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
        or not devices
        or len(devices.split(",")) != 1
    ):
        raise ValueError("TP1 measurements require one PCI_BUS_ID-ordered visible GPU")
    source = args.source.resolve()
    if not (source / "calibration_free_moe/model_config.py").is_file():
        raise ValueError("--source must be a complete model-integration snapshot")
    harness_dir = Path(__file__).resolve().parent
    existing = [part for part in os.environ.get("PYTHONPATH", "").split(":") if part]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(harness_dir),
            str(source),
            *([str(args.overlay.resolve())] if args.arm == "ours" else []),
            *existing,
        ]
    )
    # Multiprocess workers inherit the frontend's sys.path. Setting only
    # PYTHONPATH would load stock vLLM in this process despite the overlay.
    if args.arm == "ours":
        sys.path.insert(0, str(args.overlay.resolve()))
    sys.path.insert(0, str(source))
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_CACHE_ROOT"] = str(
        args.cache_root.resolve() / f"{args.revision}-{args.arm}-{args.mode}"
    )
    Path(os.environ["VLLM_CACHE_ROOT"]).mkdir(parents=True, exist_ok=True)
    if args.arm == "stock_triton_tuned":
        if args.tuned_config_dir is None or not args.tuned_config_dir.is_dir():
            raise ValueError("Tuned stock requires an existing vLLM config directory")
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(args.tuned_config_dir.resolve())
    else:
        os.environ.pop("VLLM_TUNED_CONFIG_FOLDER", None)


def validate_tuned_stock(
    args: argparse.Namespace, backend: dict[str, Any]
) -> dict[str, Any] | None:
    """Check that tuned stock loaded the requested Triton configs; map batches.

    Returns None for every other arm.
    """
    if args.arm != "stock_triton_tuned":
        return None
    layers = backend["layers"]
    if not layers or any(part["experts_kernel"] != "TritonExperts" for part in layers):
        raise RuntimeError("Tuned stock did not instantiate Triton experts")
    path = args.tuned_config_dir / backend["triton_config_filename"]
    if not path.is_file():
        raise RuntimeError(f"vLLM stock tuning file was not found: {path}")
    configs = json.loads(path.read_text())
    active = backend["triton_config_values"]
    loaded = {
        int(key): value for key, value in configs.items() if key != "triton_version"
    }
    if not loaded or active != loaded:
        raise RuntimeError("Worker did not load the requested stock tuning values")
    # vLLM chooses the nearest tuned batch size for each invocation; report
    # this mapping instead of pretending an untuned shape has an exact entry.
    return {
        "file": str(path.resolve()),
        "batch_to_config_batch": {
            str(batch): min(loaded, key=lambda key: abs(key - batch))
            for batch in args.batches
        },
    }


def tuned_model_config(model: Path, folder: Path) -> Path:
    """Return a model directory with the config aliases vLLM's tuner needs.

    OLMoE and Gemma configs get a copy with the aliases in `folder`; other
    models are returned unchanged.
    """
    from calibration_free_moe.model_config import model_dimensions

    dimensions = model_dimensions(model)
    architecture = dimensions["architecture"]
    if architecture not in ("OlmoeForCausalLM", "Gemma4ForConditionalGeneration"):
        return model
    config = json.loads((model / "config.json").read_text())
    target = config["text_config"] if architecture.startswith("Gemma4") else config
    target["num_local_experts"] = dimensions["num_experts"]
    if architecture.startswith("Gemma4"):
        target["num_experts_per_tok"] = dimensions["num_experts_per_tok"]
        target["intermediate_size"] = dimensions["expert_width"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return folder


def tune(args: argparse.Namespace) -> None:
    """Run vLLM's MoE tuner for the requested batches and record the result."""
    configure_source(args)
    import vllm

    validate_stock_vllm_import(args.arm, vllm.__file__)
    if not args.upstream_benchmark.is_file():
        raise ValueError("Point --upstream-benchmark to vLLM v0.29.0 benchmark_moe.py")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    model = tuned_model_config(args.model, args.save_dir / "tuning-model")
    command = [
        sys.executable,
        str(args.upstream_benchmark.resolve()),
        "--model",
        str(model),
        "--tp-size",
        "1",
        "--dtype",
        "auto",
        "--seed",
        "353",
        "--tune",
        "--save-dir",
        str(args.save_dir.resolve()),
        "--batch-size",
        *[str(batch) for batch in args.batches],
    ]
    report: dict[str, Any] = {
        "complete": False,
        "command": command,
        "model": str(args.model),
        "tuner_model": str(model),
        "revision": args.revision,
        "host": socket.gethostname(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "PYTHONPATH": os.environ["PYTHONPATH"],
        },
    }
    write(report, args.output)
    start = time.perf_counter()
    try:
        with args.output.with_suffix(".log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        candidates = list(args.save_dir.glob("E=*.json"))
        if len(candidates) != 1:
            raise RuntimeError("Expected one vLLM MoE tuning JSON")
        config = json.loads(candidates[0].read_text())
        if not all(str(batch) in config for batch in args.batches):
            raise RuntimeError("vLLM tuning did not complete every batch shape")
        report["stock_config_json"] = str(candidates[0])
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        write(report, args.output)


def sample(
    llm: Any, prompts: list[dict[str, Any]], params: Any, expected: int
) -> tuple[list[Any], float]:
    """Generate one batch from a paused engine; return outputs and wall seconds.

    All prompts are enqueued before scheduling resumes, so the batch prefills
    together.
    """
    from vllm import RequestOutput

    llm.sleep(level=0, mode="keep")
    llm.enqueue(
        [{"prompt_token_ids": item["tokens"]} for item in prompts],
        params,
        use_tqdm=False,
    )
    start = time.perf_counter()
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(output_type=RequestOutput, use_tqdm=False)
    elapsed = time.perf_counter() - start
    if len(outputs) != len(prompts) or any(
        len(output.outputs[0].token_ids) != expected for output in outputs
    ):
        raise RuntimeError("Incomplete native generation")
    associate_prompts(outputs, prompts)
    return outputs, elapsed


def associate_prompts(outputs: list[Any], prompts: list[dict[str, Any]]) -> None:
    """Match finished outputs by executed input, not internal randomized IDs."""
    by_tokens = {tuple(item["tokens"]): item for item in prompts}
    if len(by_tokens) != len(prompts):
        raise ValueError("Distinct speed prompts must have distinct token sequences")
    if {tuple(output.prompt_token_ids) for output in outputs} != set(by_tokens):
        raise RuntimeError("Returned prompts differ from the fixed speed panel")
    for output in outputs:
        by_tokens[tuple(output.prompt_token_ids)]["request_id"] = output.request_id


def timing_sample(
    outputs: list[Any], prompts: list[dict[str, Any]], elapsed: float, tokens: int
) -> dict[str, Any]:
    """Return per-request and median TTFT and TPOT (s) and token throughputs."""
    prompt_ids = {item["request_id"]: item["id"] for item in prompts}
    records = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.is_corrupted:
            raise RuntimeError("Missing native request metrics")
        # vLLM records arrival in wall time, but scheduled/first/last in
        # monotonic time. Its own frontend latency has the correct clock base.
        ttft = metrics.first_token_latency
        prefill = metrics.first_token_ts - metrics.scheduled_ts
        decode = metrics.last_token_ts - metrics.first_token_ts
        if any(not math.isfinite(x) or x <= 0 for x in (ttft, prefill, decode)):
            raise RuntimeError("Invalid request phase intervals")
        records.append(
            {
                "request_id": output.request_id,
                "prompt_id": prompt_ids[output.request_id],
                "ttft_seconds": ttft,
                "scheduled_to_first_seconds": prefill,
                "tpot_seconds": decode / (tokens - 1),
                "first_token_ts": metrics.first_token_ts,
                "last_token_ts": metrics.last_token_ts,
                "token_ids": list(output.outputs[0].token_ids),
            }
        )
    decode_window = max(r["last_token_ts"] for r in records) - min(
        r["first_token_ts"] for r in records
    )
    return {
        "requests": records,
        "batch_completion_seconds": elapsed,
        "tpot_seconds": statistics.median(r["tpot_seconds"] for r in records),
        "ttft_seconds": statistics.median(r["ttft_seconds"] for r in records),
        "scheduled_to_first_seconds": statistics.median(
            r["scheduled_to_first_seconds"] for r in records
        ),
        "decode_tokens_per_second": len(records) * (tokens - 1) / decode_window,
        "output_tokens_per_second": len(records) * tokens / elapsed,
    }


def run(args: argparse.Namespace) -> None:
    """Start one vLLM engine and time or count reads for every context and batch."""
    configure_source(args)
    from vllm import LLM, SamplingParams

    validate_stock_vllm_import(args.arm, sys.modules["vllm"].__file__)
    from calibration_free_moe.model_config import (
        model_dimensions,
        model_override,
    )

    dimensions = model_dimensions(args.model)
    # Each prompt's context prefixes share one panel; all fit one unchunked prefill.
    contexts: list[int] = args.max_prompt_tokens
    prompts = prompt_panel(args.panel, max(args.batches), max(contexts))
    if (
        max(len(item["tokens"]) for item in prompts) + args.output_tokens
        > args.max_model_len
    ):
        raise ValueError("Prompt and output exceed max model length")
    if sum(len(item["tokens"]) for item in prompts) > args.max_num_batched_tokens:
        raise ValueError("Full prompt batch must prefill together before decode")
    config: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(args.batches),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": 353,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "worker_extension_cls": "speed_harness.SpeedWorkerExtension",
    }
    if args.arm in ("stock_triton_untuned", "stock_triton_tuned", "ours"):
        config["moe_backend"] = "triton"
    if args.arm == "ours":
        # Without --bank the method arm is the training-free reader.
        if args.retention is None:
            raise ValueError("Method arm requires --retention")
        config.update(
            model_class_overrides=model_override(str(args.model)),
            additional_config={
                "tokenwise_phase_metadata": True,
                "tokenwise": {
                    "bank": str(args.bank) if args.bank is not None else None,
                    "retention": args.retention,
                    "selection_score": args.selection_score,
                    "screen_keep": args.screen_keep,
                },
            },
        )
    if args.mode == "bytes" and args.arm in STOCK_ARMS:
        config["enable_return_routed_experts"] = True
    frontend_vllm_file = sys.modules["vllm"].__file__
    if not frontend_vllm_file:
        raise RuntimeError("Cannot identify the frontend vLLM package")
    report: dict[str, Any] = {
        "complete": False,
        "mode": args.mode,
        "arm": args.arm,
        "scope": (
            "Native offline vLLM TP1 generation; frontend TTFT includes the "
            "batch-enqueue scheduling pause. Not HTTP latency or DRAM counters."
        ),
        "read_accounting": (
            "Stock: unique full BF16 expert rows per layer/decode step. Method: "
            "sector charge and summed group-block union issues per decode step, "
            "not a global distinct-sector union or measured DRAM traffic. "
            "Switched stock rows use a dense per-token price without cache reuse. "
            "Bytes use their own greedy worker trajectory, which can differ "
            "numerically from a separate timing worker."
        ),
        "command": sys.argv,
        "revision": args.revision,
        "source": str(args.source),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "config": config,
        "frontend_vllm_file": str(Path(frontend_vllm_file).resolve()),
        "panel": str(args.panel.resolve()),
        "prompts": [
            {"id": item["id"], "length": len(item["tokens"])} for item in prompts
        ],
        "context_tokens": contexts,
        "sampling": {
            "temperature": 0,
            "ignore_eos": True,
            "max_tokens": args.output_tokens,
            "seed": 353,
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton")
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "NCCL_P2P_DISABLE",
                "PYTHONPATH",
                "VLLM_CACHE_ROOT",
                "VLLM_TUNED_CONFIG_FOLDER",
                "FLASHINFER_WORKSPACE_BASE",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
        "panels": {},
    }
    write(report, args.output)
    start = time.perf_counter()
    try:
        llm = LLM(**config)
        report["setup_seconds"] = time.perf_counter() - start
        backend = llm.collective_rpc(
            "speed_backend_report",
            args=(
                int(dimensions["num_experts"]),
                int(dimensions["expert_width"]),
                args.arm,
            ),
        )
        report["expert_backend"] = backend
        tuned = validate_tuned_stock(args, backend[0])
        report["stock_config_json"] = tuned["file"] if tuned else None
        report["stock_tuned_batch_lookup"] = (
            tuned["batch_to_config_batch"] if tuned else None
        )
        params = SamplingParams(**report["sampling"])
        reference = (
            json.loads(args.timing_report.read_text()) if args.timing_report else None
        )
        for context, batch in (
            (context, batch) for context in contexts for batch in args.batches
        ):
            selected = [
                {"id": str(item["id"]), "tokens": item["tokens"][:context]}
                for item in prompts[:batch]
            ]
            panel: dict[str, Any] = {
                "batch": batch,
                "context_tokens": context,
                "samples": [],
            }
            # A single-context report keys its panels by batch alone.
            key = str(batch) if len(contexts) == 1 else f"{context}/{batch}"
            report["panels"][key] = panel
            for _ in range(args.warmup):
                sample(llm, selected, params, args.output_tokens)
            if args.mode == "bytes":
                if reference is None or not reference.get("complete"):
                    raise ValueError("Byte capture requires a complete --timing-report")
                if args.arm == "ours":
                    llm.collective_rpc("tokenwise_reader_reset_work")
                outputs, _ = sample(llm, selected, params, args.output_tokens)
                generated = {
                    output.request_id: list(output.outputs[0].token_ids)
                    for output in outputs
                }
                reference_panel = reference["panels"][key]
                reference_tokens = {
                    item["prompt_id"]: item["token_ids"]
                    for item in reference_panel["samples"][0]["requests"]
                }
                generated_by_prompt = {
                    item["id"]: generated[item["request_id"]] for item in selected
                }
                if generated_by_prompt.keys() != reference_tokens.keys():
                    raise RuntimeError("Byte-capture prompts differ from timed panel")
                panel["timing_trajectory_difference_requests"] = sum(
                    generated_by_prompt[prompt_id] != reference_tokens[prompt_id]
                    for prompt_id in reference_tokens
                )
                steps = args.output_tokens - 1
                if args.arm in STOCK_ARMS:
                    values = stock_union_bytes(
                        outputs,
                        selected,
                        int(dimensions["num_hidden_layers"]),
                        int(dimensions["num_experts_per_tok"]),
                        6
                        * int(dimensions["hidden_size"])
                        * int(dimensions["expert_width"]),
                        args.output_tokens,
                    )
                    panel["routed_expert_sector_union_bytes_per_step"] = distribution(
                        [float(value) for value in values]
                    )
                else:
                    work = llm.collective_rpc("tokenwise_reader_work_report")[0]
                    if not work["reference_bytes"] or not work["issued_gate_up_bytes"]:
                        raise RuntimeError("Missing method decode read-work report")
                    panel["reader_work"] = work
                    panel["method_group_block_issued_bytes_per_step"] = {
                        "sketch": work["issued_sketch_bytes"] / steps,
                        "gate_up": work["issued_gate_up_bytes"] / steps,
                        "down": work["issued_down_bytes"] / steps,
                        "recovery_encoder": work["issued_recovery_encoder_bytes"]
                        / steps,
                        "batch_sector_charge": work["charged_bytes"] / steps,
                    }
            else:
                for _ in range(args.repeats):
                    before = gpu_state(args.gpu_index)
                    host_before = host_card_occupancy()
                    outputs, elapsed = sample(llm, selected, params, args.output_tokens)
                    record = timing_sample(
                        outputs, selected, elapsed, args.output_tokens
                    )
                    record["gpu_before"] = before
                    record["gpu_after"] = gpu_state(args.gpu_index)
                    record["host_cards_before"] = host_before
                    record["host_cards_after"] = host_card_occupancy()
                    panel["samples"].append(record)
                panel["summary"] = {
                    key: distribution([float(s[key]) for s in panel["samples"]])
                    for key in (
                        "tpot_seconds",
                        "ttft_seconds",
                        "scheduled_to_first_seconds",
                        "decode_tokens_per_second",
                        "output_tokens_per_second",
                    )
                }
            write(report, args.output)
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        write(report, args.output)


def main() -> None:
    """Parse and check the arguments, then tune, time or count reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tune", "time", "bytes"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=(*STOCK_ARMS, "ours"), default="stock_auto")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--retention", type=float)
    parser.add_argument(
        "--selection-score",
        choices=("recovery-aware", "point"),
        default="recovery-aware",
    )
    parser.add_argument("--screen-keep", type=float, default=1.0)
    parser.add_argument("--tuned-config-dir", type=Path)
    parser.add_argument("--timing-report", type=Path)
    parser.add_argument("--upstream-benchmark", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        nargs="+",
        default=[128],
        help="Prompt prefix lengths; several time each context in one engine",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if not args.batches or any(b not in BATCHES for b in args.batches):
        parser.error(f"--batches must be selected from {BATCHES}")
    if args.warmup < 1 or args.repeats < 3 or args.output_tokens < 2:
        parser.error("Require warmup, three timing repeats and two output tokens")
    if args.mode == "tune":
        if args.upstream_benchmark is None or args.save_dir is None:
            parser.error("Tuning requires --upstream-benchmark and --save-dir")
        tune(args)
    else:
        if args.panel is None or args.gpu_index is None:
            parser.error("Measurements require --panel and physical --gpu-index")
        if args.arm == "ours" and args.overlay is None:
            parser.error("Method arm requires --overlay")
        if args.mode == "bytes" and args.timing_report is None:
            parser.error("Byte capture requires --timing-report")
        run(args)


if __name__ == "__main__":
    main()
