"""Time the FFN block of every decoder layer in one CUDA graph on real decode states.

`capture` (stock, eager) decodes from a fixed panel and saves every layer's FFN
inputs and outputs of selected decode steps to `<output>-b<batch>.activations.pt`;
`bench` loads them into native vLLM blocks and times all layers in one CUDA graph.
Each run is one fresh process on one GPU writing one JSON report; the reader's arm
needs `--overlay`, a vLLM 0.29.0 tree with `patches/` applied.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.speed_harness import (
    SpeedWorkerExtension,
    associate_prompts,
    configure_source,
    gpu_state,
    host_card_occupancy,
    prompt_panel,
    validate_stock_vllm_import,
    validate_tuned_stock,
    write,
)

# Some prompt batches finish prefill at different scheduler iterations, so
# 64 generated tokens need not produce 63 all-B decode forward invocations.
CAPTURE_STEPS = (0, 1, 7, 15, 31, 47, 55)


@dataclass
class CaptureState:
    """Per-layer FFN inputs, outputs and Gemma residuals of selected decode steps."""

    inputs: list[list[Any]]
    outputs: list[list[Any]]
    residuals: list[list[Any]] | None
    counts: list[int]
    hooks: list[Any]


def decoder_layers(model: Any, architecture: str) -> list[Any]:
    """Return the model's decoder layers in order; fail if there are none."""
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
    from vllm.model_executor.models.olmoe import OlmoeDecoderLayer
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    classes = {
        "OlmoeForCausalLM": OlmoeDecoderLayer,
        "Qwen3MoeForCausalLM": Qwen3MoeDecoderLayer,
        "Qwen3_5MoeForConditionalGeneration": Qwen3_5DecoderLayer,
        "Gemma4ForConditionalGeneration": Gemma4DecoderLayer,
    }
    layer_type = classes[architecture]
    layers = [module for module in model.modules() if isinstance(module, layer_type)]
    if not layers:
        raise RuntimeError(f"No {architecture} decoder layers to measure")
    return layers


def ffn_output(
    layer: Any, state: Any, residual: Any | None, streams: Any | None = None
) -> Any:
    """Run one decoder layer's FFN block on its captured input.

    `state` is the FFN input [B, H]. Gemma layers take the post-attention
    `residual` instead; `streams` selects the stock Gemma reference that overlaps
    the dense and routed branches on two CUDA streams.
    """
    if residual is None:
        return layer.mlp(state)
    if hasattr(layer, "feedforward"):
        # Our Gemma layer's production FFN, with its dense branch overlapped
        # on the aux stream. Stock workers never import it.
        return layer.feedforward(residual)
    # Gemma's dense MLP and routed experts are parallel branches. Both arms
    # start from the post-attention residual and include every branch norm,
    # router, merge and common output norm, but not the following layer
    # residual, optional PLE or layer scalar.
    if streams is not None:
        # Stock reference with our layer's fork order: branch norms first, the
        # router projection on the routed stream before the dense fork, stock
        # routed experts there too, both joined before the norms. Unlike our
        # layer, stock's dense down projection is not gated on routing.
        import torch

        dense_stream, routed_stream = streams
        router = layer.router
        router_input = router.norm(residual)
        router_input = router_input * router.root_size.to(router_input.dtype)
        router_input = router_input * router.scale.to(router_input.dtype)
        dense_input = layer.pre_feedforward_layernorm(residual)
        routed_input = layer.pre_feedforward_layernorm_2(residual)
        main = torch.cuda.current_stream()
        routed_stream.wait_stream(main)
        with torch.cuda.stream(routed_stream):
            router_logits, _ = router.proj(router_input)
        dense_stream.wait_stream(main)
        with torch.cuda.stream(dense_stream):
            dense = layer.mlp(dense_input)
        with torch.cuda.stream(routed_stream):
            routed = layer.moe(routed_input, router_logits)
        main.wait_stream(routed_stream)
        main.wait_stream(dense_stream)
        dense = layer.post_feedforward_layernorm_1(dense)
        routed = layer.post_feedforward_layernorm_2(routed)
        return layer.post_feedforward_layernorm(dense + routed)
    dense = layer.mlp(layer.pre_feedforward_layernorm(residual))
    if layer.enable_moe_block:
        dense = layer.post_feedforward_layernorm_1(dense)
        routed_input = layer.pre_feedforward_layernorm_2(residual)
        routed = layer.moe(routed_input, layer.router(residual))
        dense = dense + layer.post_feedforward_layernorm_2(routed)
    return layer.post_feedforward_layernorm(dense)


class SpeedBlockWorkerExtension(SpeedWorkerExtension):
    """vLLM worker methods that capture FFN states and time the FFN blocks."""

    speed_block_capture: CaptureState | None = None

    def block_capture_begin(self: Any, batch: int, architecture: str) -> None:
        """Start recording FFN states of full-batch decode steps in every layer."""
        layers = decoder_layers(self.model_runner.model, architecture)
        gemma = architecture == "Gemma4ForConditionalGeneration"
        captured = CaptureState(
            inputs=[[] for _ in layers],
            outputs=[[] for _ in layers],
            residuals=[[] for _ in layers] if gemma else None,
            counts=[0 for _ in layers],
            hooks=[],
        )
        self.speed_block_capture = captured

        for index, layer in enumerate(layers):

            def capture_input(
                _module: Any, _args: Any, result: Any, layer_index: int = index
            ) -> None:
                state = result[0] if isinstance(result, tuple) else result
                if state.shape[0] != batch:
                    return
                step = captured.counts[layer_index]
                captured.counts[layer_index] += 1
                if step in CAPTURE_STEPS:
                    captured.inputs[layer_index].append(state.detach().clone())
                    if captured.residuals is not None:
                        captured.residuals[layer_index].append(
                            _args[0].detach().clone()
                        )

            def capture_output(
                _module: Any, _args: Any, result: Any, layer_index: int = index
            ) -> None:
                state = result[0] if isinstance(result, tuple) else result
                if state.shape[0] != batch:
                    return
                if captured.counts[layer_index] - 1 in CAPTURE_STEPS:
                    captured.outputs[layer_index].append(state.detach().clone())

            entry = (
                layer.pre_feedforward_layernorm
                if gemma
                else layer.post_attention_layernorm
            )
            exit_module = layer.post_feedforward_layernorm if gemma else layer.mlp
            captured.hooks.extend(
                (
                    entry.register_forward_hook(capture_input),
                    exit_module.register_forward_hook(capture_output),
                )
            )

    def block_capture_end(self: Any, path: str) -> dict[str, Any]:
        """Remove the hooks, save the captured states to `path` and describe them."""
        import torch

        captured = self.speed_block_capture
        if captured is None:
            raise RuntimeError("Activation capture was not started")
        for hook in captured.hooks:
            hook.remove()
        counts = captured.counts
        if len(set(counts)) != 1 or counts[0] < CAPTURE_STEPS[-1] + 1:
            raise RuntimeError(f"Incomplete decode activation capture: {counts}")
        if any(
            len(states) != len(CAPTURE_STEPS)
            for states in (
                *captured.inputs,
                *captured.outputs,
                *(captured.residuals or []),
            )
        ):
            raise RuntimeError("Incomplete FFN input/output layers")
        # [selected decode step, decoder layer, batch, hidden]. The same real
        # states feed both independently instantiated timing workers.
        inputs = torch.stack(
            [torch.stack(states) for states in captured.inputs], dim=1
        ).cpu()
        outputs = torch.stack(
            [torch.stack(states) for states in captured.outputs], dim=1
        ).cpu()
        residuals = (
            torch.stack(
                [torch.stack(states) for states in captured.residuals], dim=1
            ).cpu()
            if captured.residuals is not None
            else None
        )
        torch.save(
            {
                "inputs": inputs,
                "outputs": outputs,
                "residuals": residuals,
                "steps": CAPTURE_STEPS,
            },
            path,
        )
        return {
            "steps_seen": counts[0],
            "steps_selected": CAPTURE_STEPS,
            "shape": list(inputs.shape),
            "path": path,
        }

    def block_bench(
        self: Any,
        path: str,
        batch: int,
        method: bool,
        repeats: int,
        target_seconds: float,
        architecture: str,
        dense_overlap: bool = False,
    ) -> dict[str, Any]:
        """Time all decoder layers' FFN blocks on saved states in one CUDA graph.

        Each of `repeats` samples replays the graph for about `target_seconds`.
        Times are per decode step over all layers, in ms. Stock arms also report
        their error against the captured outputs; our reader's arm reports the
        allocator footprint of one warmed eager FFN call.
        """
        import torch
        import vllm.kernels.vllm_c  # noqa: F401  (registers the fused norm kernels)
        from vllm import ir
        from vllm.config import set_current_vllm_config
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        captured = torch.load(path, map_location="cpu", weights_only=True)
        inputs = captured["inputs"].to(device="cuda", non_blocking=False)
        targets = captured["outputs"].to(device="cuda", non_blocking=False)
        residuals = captured["residuals"]
        if residuals is not None:
            residuals = residuals.to(device="cuda", non_blocking=False)
        steps, layer_count, captured_batch, hidden = inputs.shape
        if captured_batch != batch:
            raise RuntimeError(f"Captured batch {captured_batch} requested as {batch}")
        layers = decoder_layers(self.model_runner.model, architecture)
        expected_hidden = self.vllm_config.model_config.hf_text_config.hidden_size
        if layer_count != len(layers) or hidden != expected_hidden:
            raise RuntimeError("Captured states do not match loaded decoder blocks")
        if (residuals is not None) != (
            architecture == "Gemma4ForConditionalGeneration"
        ):
            raise RuntimeError("Captured FFN branch inputs do not match the model")
        # `num_decode_tokens` exists only in the phase-metadata patch that our
        # reader's overlay applies to vLLM; stock vLLM takes no such keyword.
        kwargs: dict[str, Any] = {"num_decode_tokens": batch} if method else {}
        if dense_overlap and (method or residuals is None):
            raise RuntimeError("Dense-branch overlap is a stock Gemma reference")
        streams = (torch.cuda.Stream(), torch.cuda.Stream()) if dense_overlap else None
        if residuals is not None and not method:
            # Deployed stock fuses Gemma's disabled GeluAndMul in its graph, and
            # our layer compiles it inside its op; compile stock's here as well
            # rather than run its eager native form as two bf16 kernels.
            with set_current_vllm_config(self.vllm_config):
                for layer in layers:
                    if not layer.mlp.act_fn.enabled():
                        layer.mlp.act_fn.compile(dynamic=True)

        def execute(step: int, index: int) -> Any:
            residual = residuals[step, index] if residuals is not None else None
            return ffn_output(layers[index], inputs[step, index], residual, streams)

        # The blocks run outside the compiled model, where vLLM IR norms would
        # dispatch to their native decomposition (eight kernels each). Deployed
        # inductor graphs fuse each norm, so both arms use vLLM's fused kernels.
        fused_norms = ["vllm_c", "native"]
        with (
            torch.inference_mode(),
            ir.ops.rms_norm.set_priority(fused_norms),
            ir.ops.fused_add_rms_norm.set_priority(fused_norms),
            set_forward_context(
                None,
                self.vllm_config,
                num_tokens=batch,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                **kwargs,
            ),
        ):
            if not method:
                errors = []
                squared_errors = []
                for step in range(steps):
                    for index in range(layer_count):
                        difference = (
                            execute(step, index).float() - targets[step, index].float()
                        )
                        errors.append(difference.abs().amax())
                        squared_errors.append(difference.square().sum())
                error = torch.stack(errors).amax()
                target_max = targets.float().abs().amax()
                relative_l2 = (
                    torch.stack(squared_errors).sum() / targets.float().square().sum()
                ).sqrt()
                if not torch.isfinite(error) or not torch.isfinite(relative_l2):
                    raise RuntimeError("Non-finite native FFN output error")
            else:
                for index in range(layer_count):
                    execute(0, index)
                error = target_max = relative_l2 = None

            block_call_allocator = None
            if method:
                # Warmed eager diagnostic for our reader. Measure the entire FFN
                # call, not just the explicitly allocated kernel scratch. No
                # selection tracing is requested.
                allocator_samples = []
                for step in range(steps):
                    for index in range(layer_count):
                        torch.cuda.synchronize()
                        allocated_before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        measured_output = execute(step, index)
                        torch.cuda.synchronize()
                        peak_delta = (
                            torch.cuda.max_memory_allocated() - allocated_before
                        )
                        live_delta = torch.cuda.memory_allocated() - allocated_before
                        del measured_output
                        torch.cuda.synchronize()
                        allocator_samples.append(
                            {
                                "step": step,
                                "layer": index,
                                "allocated_before_bytes": allocated_before,
                                "peak_allocated_delta_bytes": peak_delta,
                                "live_delta_with_output_bytes": live_delta,
                                "live_delta_after_output_release_bytes": (
                                    torch.cuda.memory_allocated() - allocated_before
                                ),
                            }
                        )
                block_call_allocator = {
                    "basis": "warmed_eager_full_ffn_call_pytorch_cuda_allocator",
                    "scope": (
                        "All allocator-visible temporaries and outputs, including "
                        "selection, routing and shared branches; excludes already "
                        "resident/preallocated buffers and non-PyTorch allocations. "
                        "Not CUDA graph pool residency or DRAM traffic."
                    ),
                    "max_peak_allocated_delta_bytes": max(
                        sample["peak_allocated_delta_bytes"]
                        for sample in allocator_samples
                    ),
                    "samples": allocator_samples,
                }

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for step in range(steps):
                    for index in range(layer_count):
                        execute(step, index)
            torch.cuda.synchronize()
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(4):
                graph.replay()
            end.record()
            end.synchronize()
            pilot_seconds = start.elapsed_time(end) / 1000 / 4
            iterations = max(12, min(256, round(target_seconds / pilot_seconds)))
            samples_ms = []
            for _ in range(repeats):
                start.record()
                for _ in range(iterations):
                    graph.replay()
                end.record()
                end.synchronize()
                samples_ms.append(start.elapsed_time(end) / (iterations * steps))

        return {
            "batch": batch,
            "steps": steps,
            "layers": layer_count,
            "iterations": iterations,
            "per_step_ms": samples_ms,
            "per_step_median_ms": statistics.median(samples_ms),
            "per_step_range_ms": [min(samples_ms), max(samples_ms)],
            "per_layer_mean_us": statistics.median(samples_ms) * 1000 / layer_count,
            "capture_input_dtype": str(inputs.dtype),
            "stock_output_max_abs_error": error.item() if error is not None else None,
            "stock_output_max_abs_target": (
                target_max.item() if target_max is not None else None
            ),
            "stock_output_rel_l2_error": (
                relative_l2.item() if relative_l2 is not None else None
            ),
            "block_call_allocator": block_call_allocator,
            "norm_priority": fused_norms,
        }


def run(args: argparse.Namespace) -> None:
    """Start one vLLM engine and capture or time every requested batch."""
    configure_source(args)
    from vllm import LLM, SamplingParams

    from calibration_free_moe.model_config import (
        model_dimensions,
        model_override,
    )

    frontend_vllm_file = sys.modules["vllm"].__file__
    if frontend_vllm_file is None:
        raise RuntimeError("Cannot identify the frontend vLLM package")
    validate_stock_vllm_import(args.arm, frontend_vllm_file)
    dimensions = model_dimensions(args.model)
    architecture = dimensions["architecture"]
    if architecture not in (
        "OlmoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
    ):
        raise ValueError(f"MoE block benchmark does not support {architecture}")
    config: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
        "max_num_batched_tokens": 32768,
        "max_num_seqs": max(args.batches),
        "gpu_memory_utilization": 0.9,
        "seed": 353,
        "enable_prefix_caching": False,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"},
        "worker_extension_cls": "speed_block.SpeedBlockWorkerExtension",
    }
    if args.mode == "capture":
        config["enforce_eager"] = True
        config["compilation_config"] = {"cudagraph_mode": "NONE"}
    if args.arm == "stock_triton_tuned":
        config["moe_backend"] = "triton"
    if args.arm == "ours":
        config["moe_backend"] = "triton"
        config["model_class_overrides"] = model_override(str(args.model))
        config["additional_config"] = {
            "tokenwise_phase_metadata": True,
            "tokenwise": {
                "bank": str(args.bank) if args.bank is not None else None,
                "retention": args.retention,
                "selection_score": args.selection_score,
                "screen_keep": args.screen_keep,
            },
        }
    report: dict[str, Any] = {
        "complete": False,
        "revision": args.revision,
        "command": sys.argv,
        "host": socket.gethostname(),
        "gpu_before": gpu_state(args.gpu_index),
        "other_cards_before": host_card_occupancy(),
        "source": str(args.source.resolve()),
        "config": config,
        "front_vllm_file": str(Path(frontend_vllm_file).resolve()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton")
        },
        "panels": {},
    }
    write(report, args.output)
    start = time.monotonic()
    try:
        llm = LLM(**config)
        report["backend"] = llm.collective_rpc(
            "speed_backend_report",
            args=(
                int(dimensions["num_experts"]),
                int(dimensions["expert_width"]),
                args.arm,
            ),
        )[0]
        report["stock_tuned_config"] = validate_tuned_stock(args, report["backend"])
        for batch in args.batches:
            panel: dict[str, Any] = {}
            report["panels"][str(batch)] = panel
            if args.mode == "capture":
                prompts = prompt_panel(args.panel, batch, 128)
                llm.collective_rpc("block_capture_begin", args=(batch, architecture))
                params = SamplingParams(
                    temperature=0, ignore_eos=True, max_tokens=64, seed=353
                )
                outputs = llm.generate(
                    [{"prompt_token_ids": item["tokens"]} for item in prompts],
                    params,
                    use_tqdm=False,
                )
                if len(outputs) != batch or any(
                    len(output.outputs[0].token_ids) != 64 for output in outputs
                ):
                    raise RuntimeError("Incomplete capture generation")
                associate_prompts(outputs, prompts)
                by_request = {output.request_id: output for output in outputs}
                captured = args.output.with_name(
                    f"{args.output.stem}-b{batch}.activations.pt"
                )
                panel["capture"] = llm.collective_rpc(
                    "block_capture_end", args=(str(captured),)
                )[0]
                panel["generation"] = {
                    prompt["id"]: list(
                        by_request[prompt["request_id"]].outputs[0].token_ids
                    )
                    for prompt in prompts
                }
            else:
                panel["measurement"] = llm.collective_rpc(
                    "block_bench",
                    args=(
                        args.activations.format(batch=batch),
                        batch,
                        args.arm == "ours",
                        7,
                        0.5,
                        architecture,
                        args.stock_dense_overlap,
                    ),
                )[0]
            write(report, args.output)
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report["gpu_after"] = gpu_state(args.gpu_index)
        report["other_cards_after"] = host_card_occupancy()
        write(report, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "bench"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=("stock_auto", "stock_triton_tuned", "ours"),
        required=True,
    )
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--activations")
    parser.add_argument("--tuned-config-dir", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--retention", type=float)
    parser.add_argument(
        "--selection-score",
        choices=("recovery-aware", "point"),
        default="recovery-aware",
    )
    parser.add_argument("--screen-keep", type=float, default=1.0)
    parser.add_argument("--overlay", type=Path)
    batch_group = parser.add_mutually_exclusive_group(required=True)
    batch_group.add_argument("--batch", type=int)
    batch_group.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--stock-dense-overlap",
        action="store_true",
        help="Gemma stock reference with the dense branch beside the routed one",
    )
    args = parser.parse_args()
    if args.stock_dense_overlap and (
        args.mode != "bench" or not args.arm.startswith("stock")
    ):
        parser.error("--stock-dense-overlap times a stock arm")
    if args.mode == "capture" and (args.arm != "stock_auto" or args.panel is None):
        parser.error("Activation capture requires stock and --panel")
    if args.mode == "bench" and args.activations is None:
        parser.error("Block timing requires --activations")
    if args.arm == "ours" and (args.retention is None or args.overlay is None):
        parser.error("Method timing requires --retention and --overlay")
    args.batches = args.batches or [args.batch]
    if any(batch < 1 or batch > 128 for batch in args.batches):
        parser.error("Batches must be within the planned B1-B128 range")
    if (
        args.mode == "bench"
        and len(args.batches) > 1
        and "{batch}" not in args.activations
    ):
        parser.error("Multi-batch timing requires a {batch} activation path")
    run(args)


if __name__ == "__main__":
    main()
