"""Run a short sparse-decode smoke test in vLLM and record how it executed.

Four small batches of fixed prompts, one mixing a one-token prefill tail with a
decode step, generate 16 greedy tokens each under the reader. Writes the tokens,
their log probabilities, the CUDA graph mode, captured and replayed graph sizes and
the scheduled batches to one JSON file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

from vllm import LLM, RequestOutput, SamplingParams

from calibration_free_moe.model_config import model_dimensions, model_override


class ProbeWorkerExtension:
    """Worker RPCs that record CUDA graph replays and scheduled batches.

    Named methods avoid enabling arbitrary callable serialization in
    `collective_rpc`.
    """

    tokenwise_probe_trace: dict[str, Any] | None = None

    def observe_execution(self: Any) -> None:
        """Wrap graph replay and batch gathering to record every execution step."""
        runner = self.model_runner
        manager = runner.cudagraph_manager
        trace: dict[str, Any] = {
            "resolved_graph_mode": str(manager.cudagraph_mode),
            "captured_buckets": sorted({desc.num_tokens for desc in manager.graphs}),
            "replay_buckets": [],
            "batches": [],
        }
        self.tokenwise_probe_trace = trace
        replay = manager.run_fullgraph
        gather = runner.gather_batch_req_state

        def record_replay(desc: Any) -> Any:
            trace["replay_buckets"].append(desc.num_tokens)
            return replay(desc)

        def record_batch(*args: Any, **kwargs: Any) -> Any:
            batch, uniform = gather(*args, **kwargs)
            if batch is not None:
                trace["batches"].append(
                    {
                        "request_ids": batch.req_ids,
                        "tokens": batch.num_tokens,
                        "decode_tokens": batch.num_decode_tokens,
                        "request_tokens": batch.num_scheduled_tokens.tolist(),
                        "computed_prefill_tokens": (
                            batch.num_computed_prefill_tokens_np.tolist()
                        ),
                        "is_prefilling": batch.is_prefilling_np.tolist(),
                    }
                )
            return batch, uniform

        manager.run_fullgraph = record_replay
        runner.gather_batch_req_state = record_batch

    def probe_trace(self: Any) -> dict[str, Any]:
        """Return the recorded execution trace."""
        return self.tokenwise_probe_trace


def main() -> None:
    """Build the sparse engine, run the probe batches and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--retention", type=float, required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--moe-backend", choices=("auto", "triton"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 128,
        "gpu_memory_utilization": 0.6,
        "seed": 353,
        "enable_prefix_caching": False,
        "enforce_eager": args.eager,
        "worker_extension_cls": "scripts.probe_vllm.ProbeWorkerExtension",
        "compilation_config": {
            "cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY"
        },
        "model_class_overrides": model_override(args.model),
        "additional_config": {
            "tokenwise_phase_metadata": True,
            "tokenwise": {
                "bank": args.bank,
                "retention": args.retention,
            },
        },
    }
    if args.moe_backend != "auto":
        config["moe_backend"] = args.moe_backend
    if (
        model_dimensions(Path(args.model))["architecture"]
        == "Qwen3_5MoeForConditionalGeneration"
    ):
        config["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    report: dict[str, object] = {
        "complete": False,
        "scope": (
            "Hybrid multi-token execution smoke; "
            "not task quality or a matched timing benchmark"
        ),
        "command": sys.argv,
        "revision": args.revision,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "config": config,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton")
        },
        "batches": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    try:
        llm = LLM(**config)
        report["setup_seconds"] = time.perf_counter() - start
        llm.collective_rpc("observe_execution")
        tokenizer = llm.get_tokenizer()
        texts = [
            "The capital of France is",
            "One plus one equals",
            "The largest planet in our solar system is",
            "Water freezes at a temperature of",
        ]
        encoded = [tokenizer.encode(text) for text in texts]
        # Scheduling the short prompt first leaves 127 of 128 tokens for the long
        # prompt, then a one-token prefill tail alongside the first decode request.
        panels = [
            [encoded[0][:1], (encoded[1] * 128)[:128]],
            encoded[:3],
            encoded,
            encoded[:3],
        ]
        params = SamplingParams(
            temperature=0, max_tokens=16, ignore_eos=True, logprobs=5
        )
        batches = []
        for panel in panels:
            before = time.perf_counter()
            llm.sleep(level=0, mode="keep")
            request_ids = llm.enqueue(
                [{"prompt_token_ids": ids} for ids in panel], params, use_tqdm=False
            )
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(output_type=RequestOutput, use_tqdm=False)
            records = []
            for output in outputs:
                result = output.outputs[0]
                if len(result.token_ids) != 16 or result.logprobs is None:
                    raise RuntimeError("Incomplete continuation")
                chosen = []
                for token, scores in zip(
                    result.token_ids, result.logprobs, strict=True
                ):
                    if scores is None:
                        raise RuntimeError("Missing generated-token log probabilities")
                    chosen.append(scores[token].logprob)
                if not all(math.isfinite(value) for value in chosen):
                    raise RuntimeError("Non-finite generated-token log probability")
                records.append(
                    {
                        "token_ids": list(result.token_ids),
                        "logprobs": chosen,
                        "top_logprobs": [
                            {
                                str(token): value.logprob
                                for token, value in scores.items()
                            }
                            for scores in result.logprobs
                            if scores is not None
                        ],
                        "text": result.text,
                    }
                )
            batches.append(
                {
                    "prompts": panel,
                    "request_ids": request_ids,
                    "seconds": time.perf_counter() - before,
                    "outputs": records,
                }
            )
            report["batches"] = batches
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["execution"] = llm.collective_rpc("probe_trace")
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
