"""Generate free-running answers to prepared requests under stock or a sparse policy.

Prefill is stock and every generated token runs the policy, including tokens a
preempted request recomputes in a prefill. Writes one JSON report with each
request's continuation, finish reason and token count and the policy's read
statistics, for `olmes_requests.py reduce` or `thinking_requests.py reduce`.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, cast

from vllm import LLM, RequestOutput, SamplingParams

from calibration_free_moe.analysis import allocation
from calibration_free_moe.model_config import (
    CALIBRATION_FREE_POLICIES,
    model_dimensions,
    model_override,
    validate_prox_keep_options,
)
from calibration_free_moe.runtime.work_reports import (
    aggregate_work_reports,
    dense_mlp_work_report,
    shared_expert_work_report,
)


def gdn_first_chunk_is_prefill() -> bool:
    """Return whether the loaded vLLM carries the fix of vllm-project/vllm#51565.

    The release applies it with `patches/vllm-0.29.0-gdn-first-chunk.patch`. With
    it, a gated-delta-net request's stateless one-token first chunk is a prefill,
    so Qwen3.5 may chunk prompts; stock 0.29 lacks it.
    """
    from vllm.v1.attention.backends import gdn_attn

    return "no_prior_state" in inspect.getsource(gdn_attn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--bank")
    parser.add_argument(
        "--policy",
        choices=(
            "progressive",
            "cats",
            "teal",
            "teal-greedy",
            "wina",
            "wina-greedy-topk",
            "prox",
            *CALIBRATION_FREE_POLICIES,
        ),
        default="progressive",
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument(
        "--prox-recipe",
        choices=("legacy", "study"),
        default="legacy",
        help="Prox calibration recipe and top-k tie convention",
    )
    parser.add_argument("--retention", type=float)
    parser.add_argument(
        "--layer-retention", help="JSON exact-channel allocation per layer"
    )
    parser.add_argument(
        "--input-keep",
        type=float,
        help="prox/prox-free: per-token input top-k fraction instead of a threshold",
    )
    parser.add_argument(
        "--channel-keep",
        type=float,
        help="prox: per-token joint channel top-k fraction instead of a threshold",
    )
    parser.add_argument(
        "--channel-floor",
        type=float,
        help="prox: minimum fraction of K*I channels under a calibrated threshold",
    )
    parser.add_argument(
        "--proxy-score",
        choices=("prox", "output-unit", "exact"),
        default="prox",
        help="Prox score: |proxy|, output-unit |w| n |proxy| (router weight w, "
        "down-column norm n; prox-free or study prox with --input-keep), or exact "
        "|w| n |a| from native rows (analysis with --allocation; --thresholds is "
        "then its JSON)",
    )
    allocation.add_arguments(parser)
    parser.add_argument(
        "--selection-score",
        choices=("recovery-aware", "point"),
        default="recovery-aware",
        help="Reader ablation: point ignores the sketch residual variances",
    )
    parser.add_argument(
        "--screen-keep",
        type=float,
        default=1.0,
        help="Fraction of each 128-coordinate rotated input block the screen reads "
        "(largest magnitudes, rounded up)",
    )
    parser.add_argument(
        "--selection-thresholds",
        type=Path,
        help="Calibrated control: keep every candidate whose screen score reaches the "
        "layer threshold (scripts/calibrate_selection_thresholds.py), at most the "
        "--retention count, instead of the global top-F",
    )
    parser.add_argument(
        "--sparse-prefill",
        action="store_true",
        help="Run the policy on prompt tokens too, with unchunked prefill (without a "
        "policy: the stock scheduling control)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        help="Scheduler token budget (default: the longest running prompts, chunked); "
        "with --sparse-prefill it also sizes the reader workspace and is raised to "
        "the longest request",
    )
    # Requests enqueued together; above --max-num-seqs vLLM batches them continuously.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        help="Fixed KV cache size (default: from --gpu-memory-utilization); a small "
        "cache forces preemption",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--moe-backend", choices=("triton",))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--token-ids",
        action="store_true",
        help="Append each request's prompt IDs as vLLM consumed them and its generated "
        "IDs to OUTPUT's sibling <stem>.token-ids.jsonl, one JSON line per request",
    )
    args = parser.parse_args()
    if min(args.batch_size, args.max_num_seqs) < 1 or (
        args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1
    ):
        parser.error("Require positive --batch-size, --max-num-seqs and token budget")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.layer_retention and args.policy != "progressive":
        parser.error("--layer-retention requires progressive")
    if args.policy == "progressive":
        # --retention alone is the training-free reader (no recovery bank).
        if (
            args.bank and args.retention is None and not args.layer_retention
        ) or args.thresholds:
            parser.error(
                "Progressive policy requires --retention or --layer-retention "
                "(optionally --bank), or "
                "neither for stock"
            )
    elif args.policy in CALIBRATION_FREE_POLICIES:
        if args.retention is None or args.bank or args.thresholds:
            parser.error(
                "Calibration-free baselines take --retention (prox-free also "
                "--input-keep and --proxy-score) only"
            )
    elif args.allocation is not None:
        # Prox with an allocation: validate_allocation_options checks the rest.
        pass
    elif args.bank or args.retention is not None or not args.thresholds:
        parser.error("Baseline policy requires --thresholds, not --bank/--retention")
    if args.allocation_record and (
        args.allocation is None or args.allocation_record.exists()
    ):
        parser.error("--allocation-record is a new directory of an allocation run")
    if args.allocation is not None and args.sparse_prefill:
        parser.error("Allocation runs decode sparse after a stock prefill")
    if args.allocation_record and args.batch_size > args.max_num_seqs:
        # Queued requests would reuse a recorded request's slot before collection.
        parser.error("Recording an allocation needs --batch-size <= --max-num-seqs")
    progressive = args.retention is not None or args.layer_retention is not None
    # The worker binding's options (additional_config), validated as bound.
    options: dict[str, object] = {
        "bank": args.bank,
        "retention": args.retention,
        "layer_retention": args.layer_retention,
        "input_keep": args.input_keep,
        "channel_keep": args.channel_keep,
        "channel_floor": args.channel_floor,
        "prox_recipe": args.prox_recipe,
        "proxy_score": args.proxy_score,
        "policy": args.policy,
        "baseline": str(args.thresholds) if args.thresholds else None,
        "sparse_prefill": args.sparse_prefill,
        "selection_score": args.selection_score,
        "screen_keep": args.screen_keep,
        "selection_thresholds": str(args.selection_thresholds)
        if args.selection_thresholds
        else None,
        **allocation.cli_options(args),
    }
    try:
        validate_prox_keep_options(
            args.policy,
            args.input_keep,
            args.channel_keep,
            args.channel_floor,
            args.prox_recipe,
            cast(str, args.proxy_score),
        )
        allocation.validate_allocation_options(options)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.selection_score != "recovery-aware" or args.screen_keep != 1
    ) and not progressive:
        parser.error("Selection options apply to the progressive reader")
    if args.selection_thresholds and (args.retention is None or args.layer_retention):
        parser.error(
            "--selection-thresholds requires the reader's uniform --retention cap"
        )
    if (
        args.tensor_parallel_size > 1
        and (progressive or args.thresholds or args.policy != "progressive")
        and not args.enable_expert_parallel
    ):
        parser.error("Sparse TP requires --enable-expert-parallel")
    baseline = args.policy != "progressive"
    token_ids_path = (
        args.output.with_name(f"{args.output.stem}.token-ids.jsonl")
        if args.token_ids
        else None
    )
    if args.output.exists() or (token_ids_path and token_ids_path.exists()):
        parser.error(f"Refusing to overwrite {args.output} or its token IDs")
    panel = json.loads(args.requests.read_text())
    if panel["data"]["kind"] != "generation" or not panel["requests"]:
        parser.error("Expected prepared generation requests")
    documents = panel["requests"]
    running = min(args.batch_size, args.max_num_seqs)
    thinking = panel["data"].get("protocol") == "provider-thinking"
    max_model_len = panel["data"]["max_model_len"]
    config: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "max_model_len": max_model_len,
        "max_num_seqs": args.max_num_seqs,
        # Any running set of prompts prefills whole in one step.
        "max_num_batched_tokens": max(
            2048,
            sum(sorted((len(d["prompt"]) for d in documents), reverse=True)[:running]),
        ),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": 353,
        "enable_prefix_caching": False,
        # The preemption counter below needs stats.
        "disable_log_stats": False,
    }
    qwen35 = (
        model_dimensions(Path(args.model))["architecture"]
        == "Qwen3_5MoeForConditionalGeneration"
    )
    if qwen35:
        config["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if args.max_num_batched_tokens is not None:
        config["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.sparse_prefill or (qwen35 and not gdn_first_chunk_is_prefill()):
        # Stock vLLM 0.29 treats a new request's one-token chunk as decode, which
        # reuses a stale gated-delta-net state; admit whole prompts only (budget >=
        # model length, so a preempted request's prompt plus generated text also
        # recomputes whole). Sparse prefill follows eval_vllm_continuation.py: whole
        # prompts, the policy on every row of a step.
        config["enable_chunked_prefill"] = False
        config["max_model_len"] = max(
            len(d["prompt"]) + d["max_tokens"] for d in documents
        )
        config["max_num_batched_tokens"] = max(
            config["max_num_batched_tokens"], config["max_model_len"]
        )
    if args.moe_backend:
        config["moe_backend"] = args.moe_backend
    if args.kv_cache_memory_bytes is not None:
        config["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if progressive or baseline:
        config.update(
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            model_class_overrides=model_override(args.model),
            additional_config={"tokenwise_phase_metadata": True, "tokenwise": options},
        )
    report: dict[str, Any] = {
        "complete": False,
        "scope": "Free-running generation; policy decode with own KV after stock "
        "prefill, or (sparse_prefill, evaluation only) after policy prefill. A "
        "preempted request recomputes its generated tokens with the policy and its "
        "prompt with stock; read statistics include those recomputed tokens",
        "command": sys.argv,
        "revision": args.revision,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "config": config,
        "batch_size": args.batch_size,
        "sparse_prefill": args.sparse_prefill,
        "allocation_record": str(args.allocation_record)
        if args.allocation_record
        else None,
        "panel": str(args.requests),
        "sampling": panel["data"].get("sampling", {"temperature": 0}),
        "protocol": panel["data"].get("protocol", "olmes"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton")
        },
        "documents": [],
    }
    if token_ids_path:
        # Appended once per batch: the throttled report rewrite stays small.
        report["token_ids_file"] = token_ids_path.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    token_ids = token_ids_path.open("x") if token_ids_path else None
    start = time.perf_counter()
    try:
        llm = LLM(**config)
        scheduler = llm.llm_engine.vllm_config.scheduler_config
        # The resolved prefill budget and chunking a recompute is scheduled under.
        report["prefill_token_budget"] = scheduler.max_num_batched_tokens
        report["chunked_prefill"] = scheduler.enable_chunked_prefill
        if args.allocation is not None:
            recording = llm.collective_rpc(
                "tokenwise_allocation_start",
                args=(
                    str(args.allocation_record) if args.allocation_record else None,
                    max(d["max_tokens"] for d in documents),
                ),
            )
            if args.allocation_record and sum(recording) != 1:
                raise RuntimeError("Allocation records must come from exactly one rank")
        if baseline:
            llm.collective_rpc("tokenwise_baseline_reset_work")
        elif progressive:
            llm.collective_rpc("tokenwise_reader_reset_work")
        evaluation_start = written = time.perf_counter()
        for offset in range(0, len(documents), args.batch_size):
            batch = documents[offset : offset + args.batch_size]
            # Allocation sources address requests by panel and document.
            extra = [
                {allocation.KEY: allocation.request_key(args.requests, d["id"])}
                if args.allocation is not None
                else None
                for d in batch
            ]
            params = [
                SamplingParams(
                    **panel["data"]["sampling"],
                    seed=d["seed"],
                    max_tokens=d["max_tokens"],
                    skip_special_tokens=False,
                    extra_args=extra_args,
                )
                if thinking
                else SamplingParams(
                    # Greedy; lm-eval panels also carry the harness's detokenization.
                    **panel["data"].get("sampling", {"temperature": 0}),
                    max_tokens=d["max_tokens"],
                    stop=d["stop"],
                    extra_args=extra_args,
                )
                for d, extra_args in zip(batch, extra, strict=True)
            ]
            llm.sleep(level=0, mode="keep")
            llm.enqueue(
                [{"prompt_token_ids": d["prompt"]} for d in batch],
                params,
                use_tqdm=False,
            )
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(output_type=RequestOutput, use_tqdm=False)
            for document, param, output in zip(batch, params, outputs, strict=True):
                result = output.outputs[0]
                if not result.token_ids:
                    raise RuntimeError(f"Empty generation for {document['id']}")
                report["documents"].append(
                    {
                        "id": document["id"],
                        "source": document["source"],
                        "prompt": document["prompt"],
                        "continuation": result.text,
                        "num_tokens": len(result.token_ids),
                        "finish_reason": result.finish_reason,
                        **(
                            {
                                "sample_index": document["sample_index"],
                                "seed": document["seed"],
                                "max_tokens": param.max_tokens,
                                "sampling": {
                                    key: getattr(param, key)
                                    for key in panel["data"]["sampling"]
                                },
                                "truncated": result.finish_reason == "length",
                                "prompt_tokens": len(
                                    output.prompt_token_ids or document["prompt"]
                                ),
                            }
                            if thinking
                            else {}
                        ),
                    }
                )
                if token_ids:
                    if output.prompt_token_ids is None:
                        raise RuntimeError(f"No prompt IDs for {document['id']}")
                    record = {
                        "id": document["id"],
                        "prompt_token_ids": list(output.prompt_token_ids),
                        "token_ids": list(result.token_ids),
                    }
                    token_ids.write(json.dumps(record) + "\n")
            if token_ids:
                token_ids.flush()
            if args.allocation_record:
                # Generated decode steps only; a request may stop before max_tokens.
                llm.collective_rpc("tokenwise_allocation_collect")
            # Throttled progress snapshot: rewriting the growing report after every
            # batch is quadratic I/O and dominated long all-token runs.
            if time.perf_counter() - written > 300:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                written = time.perf_counter()
        if args.allocation is not None:
            llm.collective_rpc("tokenwise_allocation_finish")
        if baseline:
            report["baseline_work"] = aggregate_work_reports(
                llm.collective_rpc("tokenwise_baseline_work_report")
            )
            if report["baseline_work"]["read_fraction"] is None:
                raise RuntimeError("No measured baseline decode reads")
        elif progressive:
            report["reader_work"] = aggregate_work_reports(
                llm.collective_rpc("tokenwise_reader_work_report")
            )
        architectures = llm.llm_engine.model_config.hf_config.architectures or []
        if "Qwen3_5MoeForConditionalGeneration" in architectures:
            # The first generated token follows prefill; later tokens are
            # decode inputs. Do not multiply shared MLP work by TP rank count.
            report["shared_expert_work"] = shared_expert_work_report(
                model_dimensions(Path(args.model)),
                sum(len(d["prompt"]) for d in report["documents"]),
                sum(d["num_tokens"] - 1 for d in report["documents"]),
                args.tensor_parallel_size,
            )
        elif any(name.startswith("Gemma4") for name in architectures):
            report["dense_mlp_work"] = dense_mlp_work_report(
                model_dimensions(Path(args.model)),
                sum(len(d["prompt"]) for d in report["documents"]),
                sum(d["num_tokens"] - 1 for d in report["documents"]),
            )
        report["evaluation_seconds"] = time.perf_counter() - evaluation_start
        report["preemptions"] = int(
            sum(
                getattr(metric, "value", 0)
                for metric in llm.get_metrics()
                if metric.name == "vllm:num_preemptions"
            )
        )
        if report["preemptions"]:
            # Routed read statistics count every policy row, recomputation included;
            # shared-expert and dense-MLP totals above count each token once.
            for key in ("shared_expert_work", "dense_mlp_work"):
                if key in report:
                    report[key]["token_counts"] = (
                        "unique tokens; excludes recomputation"
                    )
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if token_ids:
            token_ids.close()
        report["elapsed_seconds"] = time.perf_counter() - start
        report["generated_tokens"] = sum(d["num_tokens"] for d in report["documents"])
        if report.get("evaluation_seconds"):
            report["tokens_per_second"] = (
                report["generated_tokens"] / report["evaluation_seconds"]
            )
            report["requests_per_second"] = (
                len(report["documents"]) / report["evaluation_seconds"]
            )
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
