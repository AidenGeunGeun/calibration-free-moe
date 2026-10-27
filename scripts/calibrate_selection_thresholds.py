"""Fit per-layer thresholds on the reader's own screen score (calibrated control).

The control keeps every candidate channel whose score reaches its layer's threshold,
fitted on FIT text with Prox's chat recipe so that the mean kept count is as close
as attainable to the reader's F at the read budget. Writes the thresholds JSON to
`--output`, with `<stem>-calibration.json` and `<stem>-initial.json` beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from calibration_free_moe.model_config import model_dimensions, model_override
from calibration_free_moe.reader.budget import compile_budget

FORMAT = "selection-thresholds-v1"


def fit_selection_threshold(scores: Tensor, target: float, cap: int) -> float:
    """Return the attainable threshold whose mean capped kept count is nearest `target`.

    `scores` is [positions, candidates] in FP32. A token keeps
    min(cap, #{scores >= tau}). Nonnegative FP32 bit patterns order as their
    values, so the search runs over integer keys without sorting.
    """
    if scores.dtype != torch.float32 or scores.ndim != 2 or bool((scores < 0).any()):
        raise ValueError("Expected nonnegative FP32 scores [positions, candidates]")
    keys = (scores + 0.0).view(torch.int32)

    def kept(key: int) -> float:
        return float((keys >= key).sum(-1).clamp_max(cap).float().mean())

    low, high = 0, int(keys.max()) + 1
    # Smallest key keeping at most target: kept decreases as the key grows.
    while low < high:
        middle = (low + high) // 2
        if kept(middle) <= target:
            high = middle
        else:
            low = middle + 1
    # kept(low - 1) > target means a score sits at low - 1: the next attainable
    # count above target. Take it only when strictly nearer.
    if low > 0 and abs(kept(low - 1) - target) < abs(kept(low) - target):
        low -= 1
    # The cap can hold the mean on a plateau of keys; take its largest, which
    # keeps no channel below the cap that a higher key would drop.
    mean, high = kept(low), int(keys.max())
    while low < high:
        middle = (low + high + 1) // 2
        if kept(middle) == mean:
            low = middle
        else:
            high = middle - 1
    threshold = float(torch.tensor([low], dtype=torch.int32).view(torch.float32)[0])
    if not math.isfinite(threshold):
        raise ValueError("Target count cannot be met below the largest finite score")
    return threshold


class SelectionCalibrationWorkerExtension:
    """Replay FIT prefill with fitted earlier layers, collecting one native layer.

    vLLM worker extension: the driver calls these methods on every rank through
    `collective_rpc`.
    """

    selection_modules: ClassVar[list[Any]] = []
    selection_target: int = -1
    selection_states: ClassVar[list[Tensor]] = []
    selection_routes: ClassVar[list[Tensor]] = []
    selection_weights: ClassVar[list[Tensor]] = []

    def prepare_selection_calibration(self: Any) -> int:
        """Wrap every MoE layer's forward and return the number of MoE layers.

        Layers before the target run the thresholded reader, the target layer runs
        native while its inputs and router outputs are recorded, and later layers
        run native.
        """
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        from calibration_free_moe.runtime.experts import ProgressiveExperts

        modules = {
            int(name.split(".layers.")[1].split(".")[0]): module
            for name, module in self.model_runner.model.named_modules()
            if isinstance(module, ProgressiveExperts)
        }
        self.selection_modules = [modules[i] for i in range(len(modules))]
        self.selection_target = -1
        for layer, module in enumerate(self.selection_modules):
            sparse = module.forward_modular

            def forward(
                x: Tensor,
                topk_weights: Tensor,
                topk_ids: Tensor,
                *extra: Any,
                _layer: int = layer,
                _module: Any = module,
                _sparse: Any = sparse,
                **kwargs: Any,
            ) -> Tensor:
                if _layer < self.selection_target:
                    # Fitted layers run the thresholded reader on every row.
                    return _sparse(x, topk_weights, topk_ids, *extra, **kwargs)
                if _layer == self.selection_target:
                    self.selection_states.append(x.detach().cpu())
                    self.selection_routes.append(topk_ids.detach().cpu())
                    self.selection_weights.append(topk_weights.detach().cpu())
                return RoutedExperts.forward_modular(
                    _module, x, topk_weights, topk_ids, *extra, **kwargs
                )

            cast(Any, module).forward_modular = forward
        return len(self.selection_modules)

    def begin_selection_layer(self: Any, layer: int) -> None:
        """Make `layer` the native layer whose inputs the next prefill records."""
        self.selection_target = layer
        self.selection_states = []
        self.selection_routes = []
        self.selection_weights = []

    @torch.no_grad()
    def finalize_selection_layer(
        self: Any, target_count: int, max_states: int
    ) -> dict[str, float | int]:
        """Fit the target layer's threshold, install it, and return fit statistics.

        Uses at most `max_states` evenly spaced recorded tokens and returns the
        threshold with the mean and standard deviation of the kept count.
        """
        module = self.selection_modules[self.selection_target]
        reader = module.reader
        states = torch.cat(self.selection_states)
        if not len(states):
            raise RuntimeError(f"No FIT inputs in layer {self.selection_target}")
        positions = torch.linspace(
            0, len(states) - 1, min(max_states, len(states))
        ).long()
        device = reader.native_gu.device
        x = states[positions].to(device).contiguous()
        ids = torch.cat(self.selection_routes)[positions].to(device).contiguous()
        weights = torch.cat(self.selection_weights)[positions].to(device).contiguous()
        # Score with nothing kept: the screen is unchanged and no native channel is
        # computed.
        reader.selection_threshold.fill_(torch.finfo(torch.float32).max)
        scores = []
        for start in range(0, len(x), reader.max_tokens):
            end = start + reader.max_tokens
            reader(x[start:end], ids[start:end], weights[start:end])
            scores.append(reader.scores[0][: len(x[start:end])].cpu())
        # Fit on the host: the search's [positions, N] temporaries would exceed
        # the GPU memory vLLM leaves outside its budget.
        values = torch.cat(scores)
        cap = reader.plan.final_count
        threshold = fit_selection_threshold(values, target_count, cap)
        reader.selection_threshold.fill_(threshold)
        kept = (values >= threshold).sum(-1).clamp_max(cap).float()
        self.selection_states = []
        self.selection_routes = []
        self.selection_weights = []
        return {
            "threshold": threshold,
            "fit_states": len(x),
            "fit_mean_count": float(kept.mean()),
            "fit_count_sd": float(kept.std()),
        }


def main() -> None:
    """Fit every layer in order in one engine and write the thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--target-retention",
        type=float,
        required=True,
        help="Our reader's retention at the read budget; thresholds keep its F "
        "channels on average",
    )
    parser.add_argument("--screen-keep", type=float, default=1.0)
    parser.add_argument(
        "--selection-score",
        choices=("recovery-aware", "point"),
        default="recovery-aware",
    )
    parser.add_argument("--max-states", type=int, default=20_480)
    parser.add_argument("--fit-documents", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    args = parser.parse_args()
    if args.tensor_parallel_size > 1 and not args.enable_expert_parallel:
        parser.error("The reader under TP requires --enable-expert-parallel")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    dimensions = model_dimensions(args.model)
    hidden, width = int(dimensions["hidden_size"]), int(dimensions["expert_width"])
    top_k = int(dimensions["num_experts_per_tok"])
    n = top_k * width
    # The reader needs a screened plan, so N - 1 channels per token is the largest
    # cap; the fitted mean counts it.
    cap_retention = (n - 1) / n
    cap = compile_budget(
        hidden_size=hidden,
        expert_width=width,
        top_k=top_k,
        retention=cap_retention,
        screen_keep=args.screen_keep,
    )
    target = compile_budget(
        hidden_size=hidden,
        expert_width=width,
        top_k=top_k,
        retention=args.target_retention,
        screen_keep=args.screen_keep,
    ).final_count
    if cap.final_count != n - 1 or not target < n - 1:
        parser.error("The cap must round to N - 1 channels, above the target count")
    panel = json.loads(args.panel.read_text())
    documents = [
        doc for doc in panel["documents"] if doc["split_role"].lower() == "fit"
    ][: args.fit_documents]
    if (
        len(documents) != args.fit_documents
        or sum(len(doc["tokens"]) for doc in documents) < args.max_states
    ):
        parser.error("Insufficient FIT-only tokens in panel")
    fitted: dict[str, Any] = {
        "format": FORMAT,
        "model_revision": args.model.name,
        "selection_score": args.selection_score,
        "screen_coordinates": cap.screen_coordinates,
        "cap": cap.final_count,
        "bank": args.bank,
        "thresholds": [0.0] * int(dimensions["num_hidden_layers"]),
    }
    # The engine loads with zero thresholds; each layer's is set once it is fitted.
    initial = args.output.with_name(f"{args.output.stem}-initial.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial.write_text(json.dumps(fitted, indent=2) + "\n")
    report: dict[str, Any] = {
        "complete": False,
        "command": sys.argv,
        "revision": args.revision,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "panel": str(args.panel),
        "fit_document_ids": [doc["id"] for doc in documents],
        "target_retention": args.target_retention,
        "target_count": target,
        "layers": [],
    }
    report_path = args.output.with_name(f"{args.output.stem}-calibration.json")
    start = time.perf_counter()
    try:
        from vllm import LLM, SamplingParams

        longest = max(len(doc["tokens"]) for doc in documents) + 1
        config: dict[str, Any] = {
            "model": str(args.model),
            "tokenizer": str(args.model),
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_expert_parallel": args.enable_expert_parallel,
            "max_model_len": longest,
            "max_num_seqs": args.batch_size,
            # Whole prompts per step: every row of a step takes the fitted layers.
            # The step budget sizes the sparse-prefill reader workspace.
            "enable_chunked_prefill": False,
            "max_num_batched_tokens": max(2048, longest),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
            "enable_prefix_caching": False,
            "moe_backend": "triton",
            "model_class_overrides": model_override(str(args.model)),
            "additional_config": {
                "tokenwise_phase_metadata": True,
                "tokenwise": {
                    "policy": "progressive",
                    "bank": args.bank,
                    "retention": cap_retention,
                    "screen_keep": args.screen_keep,
                    "selection_score": args.selection_score,
                    "selection_thresholds": str(initial),
                    "sparse_prefill": True,
                },
            },
            "worker_extension_cls": "scripts.calibrate_selection_thresholds."
            "SelectionCalibrationWorkerExtension",
        }
        if dimensions["architecture"].startswith("Gemma4"):
            # Same Gemma scheduling floor as the evaluation runner.
            config["max_num_batched_tokens"] = max(
                config["max_num_batched_tokens"], 3072
            )
        if dimensions["architecture"] == "Qwen3_5MoeForConditionalGeneration":
            config["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm = LLM(**config)
        try:
            layers = llm.collective_rpc("prepare_selection_calibration")[0]
            for layer in range(layers):
                llm.collective_rpc("begin_selection_layer", args=(layer,))
                for offset in range(0, len(documents), args.batch_size):
                    batch = documents[offset : offset + args.batch_size]
                    llm.generate(
                        [{"prompt_token_ids": doc["tokens"]} for doc in batch],
                        SamplingParams(temperature=0, max_tokens=1),
                        use_tqdm=False,
                    )
                ranks = llm.collective_rpc(
                    "finalize_selection_layer", args=(target, args.max_states)
                )
                # EP ranks reduce the same global scores before fitting.
                if any(rank != ranks[0] for rank in ranks):
                    raise RuntimeError(f"EP ranks disagree on thresholds: {ranks}")
                fitted["thresholds"][layer] = ranks[0]["threshold"]
                report["layers"].append(ranks[0])
                print(f"layer={layer} {ranks[0]}", flush=True)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            llm.llm_engine.engine_core.shutdown()
        fitted["calibration"] = {
            "trajectory": "sequential-sparse",
            "panel": str(args.panel),
            "fit_documents": len(documents),
            "max_states_per_layer": args.max_states,
            "target_retention": args.target_retention,
            "target_count": target,
            "threshold_ties": "inclusive; nearest attainable capped mean count",
        }
        args.output.write_text(json.dumps(fitted, indent=2) + "\n")
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
