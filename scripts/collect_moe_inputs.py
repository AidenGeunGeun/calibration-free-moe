"""Save stock-vLLM prefill MoE inputs of a FIT/TUNE panel, one file per layer.

Writes `collection.json` and one `layer-NN.pt` per MoE layer to a new `--output`
directory: the selected rows' routed-expert inputs (`states`, `document`,
`position`), optionally with routing (`--record-routing`, always on Gemma 4) and
residual-stream norms (`--record-norms`).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, RequestOutput, SamplingParams

from calibration_free_moe.model_config import (
    checkpoint_identity,
    model_dimensions,
)


class MoEInputWorkerExtension:
    """Worker RPCs that capture MoE inputs of selected prefill rows.

    vLLM worker extension: the driver calls these methods through
    `collective_rpc`. Hooks copy states on device and never synchronize per layer;
    `flush_moe_inputs` moves each batch to the host. Only TP rank 0 records.
    """

    # install_moe_hooks assigns fresh per-worker containers before any use.
    moe_requests: dict[str, int] | None = None
    moe_rows: list[tuple[int, int, int]] | None = None
    moe_index: torch.Tensor | None = None
    # Per layer and key: device tensors of the current batch, then CPU chunks.
    moe_pending: dict[int, dict[str, list[torch.Tensor]]] | None = None
    moe_saved: dict[int, dict[str, list[torch.Tensor]]] | None = None
    moe_shared: dict[int, torch.Tensor] | None = None
    moe_seen: dict[int, list[tuple[int, int]]] | None = None
    moe_keys: dict[int, set[str]] | None = None
    moe_selected: set[tuple[int, int]] | None = None

    def install_moe_hooks(
        self: Any,
        selected: list[tuple[int, int]],
        layers_count: int,
        record_routing: bool = False,
        record_norms: bool = False,
    ) -> dict[str, Any] | None:
        """Install capture hooks for the selected (document, position) rows.

        `record_routing` saves `router_logits` and the `route_ids` and
        `route_weights` the router applied (logical IDs; weights after top-k
        normalization and, on Gemma 4, its per-expert scale). `record_norms` saves
        FP32 L2 norms per row: `residual_in_norm` (stream entering the layer),
        `attention_out_norm` (what attention adds; Gemma 4: after its post-norm),
        `residual_mid_norm` (stream the MoE input normalizes), `moe_out_norm`
        (routed-expert output), `ffn_out_norm` (all the feed-forward part adds;
        Gemma 4: before the layer scalar), Qwen3.5 `shared_out_norm` and Gemma 4
        `dense_out_norm`, `dense_branch_norm` and `moe_branch_norm` (dense output
        before and after its post-norm, MoE output after its post-norm). Returns
        the hooked layers and model class on TP rank 0, None elsewhere.
        """
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.layers.fused_moe import MoERunner
        from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
        from vllm.model_executor.models.olmoe import OlmoeDecoderLayer
        from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

        # Attention reduces to the same full-width MoE input on every TP rank.
        # Capture once rather than writing duplicate (and colliding) shards.
        if get_tensor_model_parallel_rank() != 0:
            return None
        layers = {
            int(name.rsplit(".", 1)[-1]): module
            for name, module in self.model_runner.model.named_modules()
            if isinstance(
                module,
                (
                    OlmoeDecoderLayer,
                    Qwen3MoeDecoderLayer,
                    Qwen3_5DecoderLayer,
                    Gemma4DecoderLayer,
                ),
            )
        }
        if sorted(layers) != list(range(layers_count)):
            raise RuntimeError(f"Incomplete native MoE layers: {sorted(layers)}")
        self.moe_selected = set(map(tuple, selected))
        self.moe_requests = {}
        self.moe_rows = []
        self.moe_index = None
        self.moe_pending = {layer: {} for layer in layers}
        self.moe_saved = {layer: {} for layer in layers}
        self.moe_shared = {}
        self.moe_seen = {layer: [] for layer in layers}
        self.moe_keys = {}
        prepare = self.model_runner.prepare_inputs

        def prepare_rows(*args: Any, **kwargs: Any) -> Any:
            batch = prepare(*args, **kwargs)
            self.moe_rows = []
            self.moe_index = None
            if batch is not None and self.moe_requests:
                for i, request_id in enumerate(batch.req_ids):
                    document = self.moe_requests[request_id]
                    if not bool(batch.is_prefilling_np[i]):
                        continue
                    start = int(batch.num_computed_tokens_np[i])
                    count = int(batch.num_scheduled_tokens[i])
                    row = int(batch.query_start_loc_np[i])
                    self.moe_rows.extend(
                        (row + j, document, start + j)
                        for j in range(count)
                        if (document, start + j) in self.moe_selected
                    )
            return batch

        self.model_runner.prepare_inputs = prepare_rows

        def rows(device: torch.device) -> torch.Tensor | None:
            """Return the step's selected rows as one device index, built once."""
            if not self.moe_rows:
                return None
            if self.moe_index is None:
                self.moe_index = torch.tensor(
                    [row for row, _, _ in self.moe_rows], device=device
                )
            return self.moe_index

        def keep(layer_id: int, key: str, value: torch.Tensor) -> None:
            self.moe_pending[layer_id].setdefault(key, []).append(value)

        def norm_hook(layer_id: int, key: str, pre: bool = False) -> Any:
            """Record the L2 norm of a module output (or, if pre, of its first input).

            A fused add-norm returns (normed, residual sum); the sum is the stream.
            Norms are taken at once: fused norms later update these buffers in place.
            """

            def hook(_module: Any, args: Any, output: Any = None) -> None:
                value = args[0] if pre else output
                if isinstance(value, tuple):
                    value = value[1]
                index = rows(value.device)
                if index is not None:
                    keep(layer_id, key, value[index].float().norm(dim=-1))

            return hook

        def stream_hook(layer_id: int) -> Any:
            """Record the norm of the residual stream entering a layer's input norm."""

            def hook(_module: Any, args: Any, output: Any) -> None:
                stream = output[1] if isinstance(output, tuple) else args[0]
                index = rows(stream.device)
                if index is not None:
                    keep(
                        layer_id, "residual_in_norm", stream[index].float().norm(dim=-1)
                    )

            return hook

        def shared_hook(layer_id: int) -> Any:
            def hook(module: Any, args: Any) -> None:
                index = rows(args[0].device)
                if index is not None:
                    shared = module.shared_expert(args[0][index]).float()
                    self.moe_shared[layer_id] = shared
                    keep(layer_id, "shared_out_norm", shared.norm(dim=-1))

            return hook

        def block_hook(layer_id: int, shared: bool) -> Any:
            def hook(_module: Any, _args: Any, output: torch.Tensor) -> None:
                index = rows(output.device)
                if index is not None:
                    total = output[index].float()
                    keep(layer_id, "ffn_out_norm", total.norm(dim=-1))
                    routed = total - self.moe_shared.pop(layer_id) if shared else total
                    keep(layer_id, "moe_out_norm", routed.norm(dim=-1))

            return hook

        def routing_wrapper(layer_id: int, select: Any) -> Any:
            """Record the router's own output as applied by the expert kernel."""

            def select_experts(
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                *args: Any,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                weights, ids = select(hidden_states, router_logits, *args, **kwargs)
                index = rows(weights.device)
                if index is not None:
                    keep(layer_id, "router_logits", router_logits[index].clone())
                    keep(layer_id, "route_ids", ids[index].to(torch.int32))
                    keep(layer_id, "route_weights", weights[index].float())
                return weights, ids

            return select_experts

        tensor_parallel = get_tensor_model_parallel_world_size()
        for layer_id, layer in layers.items():
            gemma = isinstance(layer, Gemma4DecoderLayer)

            def capture(
                _module: Any, args: Any, layer_id: int = layer_id, gemma: bool = gemma
            ) -> None:
                index = rows(args[0].device)
                if index is not None:
                    keep(layer_id, "states", args[0][index].clone())
                    if gemma and not record_routing:
                        from vllm.model_executor.models.gemma4 import (
                            gemma4_fused_routing_kernel_triton,
                        )

                        weights, ids = gemma4_fused_routing_kernel_triton(
                            args[1][index],
                            _module.experts.routed_experts.top_k,
                            _module.per_expert_scale,
                        )
                        keep(layer_id, "route_ids", ids)
                        keep(layer_id, "route_weights", weights)
                    self.moe_seen[layer_id].extend(
                        (document, position) for _, document, position in self.moe_rows
                    )

            routed = layer.moe if gemma else layer.mlp
            if routed is None:
                raise ValueError(f"Layer {layer_id} has no routed expert block")
            routed.register_forward_pre_hook(capture)
            keys = self.moe_keys[layer_id] = {"states"}
            if gemma or record_routing:
                keys |= {"route_ids", "route_weights"}
            if record_routing:
                keys.add("router_logits")
                runner = routed.experts
                if not isinstance(runner, MoERunner):
                    raise TypeError(f"Layer {layer_id} has no vLLM MoE runner")
                router = runner.router
                router.select_experts = routing_wrapper(layer_id, router.select_experts)
            if not record_norms:
                continue
            layer.input_layernorm.register_forward_hook(stream_hook(layer_id))
            keys |= {
                "residual_in_norm",
                "attention_out_norm",
                "residual_mid_norm",
                "moe_out_norm",
                "ffn_out_norm",
            }
            if gemma:
                keys |= {"dense_out_norm", "dense_branch_norm", "moe_branch_norm"}
                for module, key in (
                    (layer.post_attention_layernorm, "attention_out_norm"),
                    (layer.mlp, "dense_out_norm"),
                    (layer.post_feedforward_layernorm_1, "dense_branch_norm"),
                    # routed is layer.moe here, already checked to exist.
                    (routed, "moe_out_norm"),
                    (layer.post_feedforward_layernorm_2, "moe_branch_norm"),
                    (layer.post_feedforward_layernorm, "ffn_out_norm"),
                ):
                    # The branch norms exist exactly when the MoE block does.
                    if module is None:
                        raise ValueError(f"Layer {layer_id} has no {key} module")
                    module.register_forward_hook(norm_hook(layer_id, key))
                layer.pre_feedforward_layernorm.register_forward_pre_hook(
                    norm_hook(layer_id, "residual_mid_norm", pre=True)
                )
                continue
            # Qwen3.5 linear-attention layers have linear_attn instead of self_attn.
            attention = (
                layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
            )
            if not isinstance(attention, torch.nn.Module):
                raise TypeError(f"Layer {layer_id} has no attention module")
            attention.register_forward_hook(norm_hook(layer_id, "attention_out_norm"))
            layer.post_attention_layernorm.register_forward_hook(
                norm_hook(layer_id, "residual_mid_norm")
            )
            shared = getattr(routed, "shared_expert", None) is not None
            if shared:
                # The shared expert's output is a partial sum on each TP rank.
                if tensor_parallel != 1:
                    raise ValueError("Shared-expert norms require tensor parallel 1")
                routed.register_forward_pre_hook(shared_hook(layer_id))
                keys.add("shared_out_norm")
            routed.register_forward_hook(block_hook(layer_id, shared))
        return {
            "layers": sorted(layers),
            "model_class": type(self.model_runner.model).__name__,
        }

    def arm_moe_inputs(self: Any, requests: dict[str, int]) -> None:
        """Map the next batch's request IDs to panel document indices."""
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return
        self.moe_requests = requests
        self.moe_rows = []
        self.moe_index = None

    def flush_moe_inputs(self: Any) -> None:
        """Move the finished batch's captured tensors to the host and disarm."""
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return
        for layer, values in self.moe_pending.items():
            for key, chunks in values.items():
                self.moe_saved[layer].setdefault(key, []).append(
                    torch.cat(chunks).cpu()
                )
            self.moe_pending[layer] = {}
        self.moe_requests = {}
        self.moe_rows = []
        self.moe_index = None

    def save_moe_inputs(self: Any, output: str) -> list[dict[str, Any]]:
        """Check that every selected row was seen once, then save `layer-NN.pt` files.

        Returns one shard record (layer, path, rows, bytes) per layer on TP rank 0.
        """
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return []
        expected = self.moe_selected
        shards = []
        for layer in range(len(self.moe_seen)):
            observed = self.moe_seen[layer]
            if len(observed) != len(expected) or set(observed) != expected:
                raise RuntimeError(
                    f"Incomplete/duplicate MoE inputs in layer {layer}: "
                    f"observed={len(observed)} expected={len(expected)} "
                    f"missing={sorted(expected - set(observed))[:8]} "
                    f"extra={sorted(set(observed) - expected)[:8]}"
                )
            values = {
                key: torch.cat(chunks) for key, chunks in self.moe_saved[layer].items()
            }
            if set(values) != self.moe_keys[layer]:
                raise RuntimeError(
                    f"Layer {layer} recorded {sorted(values)}, "
                    f"expected {sorted(self.moe_keys[layer])}"
                )
            states = values.pop("states")
            if states.dtype != torch.bfloat16 or not bool(torch.isfinite(states).all()):
                raise RuntimeError(f"Invalid MoE state in layer {layer}")
            for key, value in values.items():
                if value.shape[0] != len(observed) or (
                    value.is_floating_point() and not bool(torch.isfinite(value).all())
                ):
                    raise RuntimeError(
                        f"Invalid {key} in layer {layer}: {value.shape[0]} rows for "
                        f"{len(observed)} states"
                    )
            path = Path(output) / f"layer-{layer:02d}.pt"
            payload = {
                "states": states,
                "document": torch.tensor([d for d, _ in observed], dtype=torch.int32),
                "position": torch.tensor([p for _, p in observed], dtype=torch.int32),
            } | values
            torch.save(payload, path)
            shards.append(
                {
                    "layer": layer,
                    "path": str(path),
                    "rows": len(observed),
                    "bytes": path.stat().st_size,
                }
            )
            self.moe_saved[layer] = {}
        return shards


def main() -> None:
    """Validate the panel, select rows, run stock prefill in batches and save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--moe-backend", choices=("auto", "triton"), default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--max-states-per-layer", type=int, default=0)
    parser.add_argument(
        "--positions",
        type=Path,
        help="JSON object mapping document IDs to the token positions to collect "
        "(instead of evenly spaced states); other documents contribute no rows",
    )
    parser.add_argument("--record-routing", action="store_true")
    parser.add_argument("--record-norms", action="store_true")
    args = parser.parse_args()
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.record_routing and args.moe_backend != "triton":
        # Only the modular Triton path calls router.select_experts, where routing
        # is read.
        parser.error("--record-routing requires --moe-backend triton")
    if args.positions is not None and args.max_states_per_layer:
        parser.error("--positions and --max-states-per-layer are exclusive")
    model_config = model_dimensions(Path(args.model))
    panel = json.loads(args.panel.read_text())
    documents = panel["documents"]
    if not documents or any(
        doc["split_role"].lower() not in ("fit", "tune")
        or not 1 <= len(doc["tokens"]) <= args.max_model_len
        for doc in documents
    ):
        raise ValueError("Expected nonempty FIT/TUNE panel within max model length")
    if len({d["id"] for d in documents}) != len(documents):
        raise ValueError("Duplicate document IDs")
    if args.max_states_per_layer < 0:
        parser.error("--max-states-per-layer must be nonnegative")
    total = sum(len(d["tokens"]) for d in documents)
    cap = args.max_states_per_layer or total
    selected = [
        (d, p) for d, doc in enumerate(documents) for p in range(len(doc["tokens"]))
    ]
    if cap < total:
        selected = [selected[j * total // cap] for j in range(cap)]
    if args.positions is not None:
        requested = json.loads(args.positions.read_text())
        index = {doc["id"]: d for d, doc in enumerate(documents)}
        if not requested or set(requested) - set(index):
            raise ValueError("--positions needs known document IDs")
        selected = sorted(
            {
                (index[key], int(p))
                for key, positions in requested.items()
                for p in positions
            }
        )
        if any(not 0 <= p < len(documents[d]["tokens"]) for d, p in selected):
            raise ValueError("--positions lie outside their documents")
        cap = len(selected)
    groups: dict[tuple[str, str], str] = {}
    for document in documents:
        group = (document["source"], document["group_id"])
        role = document["split_role"].lower()
        if groups.setdefault(group, role) != role:
            raise ValueError("Source-document group crosses FIT/TUNE split")
    args.output.mkdir(parents=True, exist_ok=False)
    config: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": args.max_model_len * args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
        "enable_prefix_caching": False,
        "worker_extension_cls": ("scripts.collect_moe_inputs.MoEInputWorkerExtension"),
    }
    if model_config["architecture"] == "Qwen3_5MoeForConditionalGeneration":
        config["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if args.moe_backend != "auto":
        config["moe_backend"] = args.moe_backend
    report: dict[str, Any] = {
        "complete": False,
        "model": checkpoint_identity(Path(args.model)),
        "dimensions": {
            "layers": model_config["num_hidden_layers"],
            "experts": model_config["num_experts"],
            "top_k": model_config["num_experts_per_tok"],
            "hidden": model_config["hidden_size"],
            "expert_width": model_config["expert_width"],
        },
        "max_states_per_layer": cap,
        "positions": None if args.positions is None else str(args.positions),
        "record_routing": args.record_routing,
        "record_norms": args.record_norms,
        "model_path": args.model,
        "source_revision": args.revision,
        "documents": [
            {key: d[key] for key in ("id", "source", "group_id", "split_role")}
            | {"tokens": len(d["tokens"])}
            for d in documents
        ],
        "shards": [],
        "command": sys.argv,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "python": sys.executable,
        "config": config,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton")
        },
    }
    report_path = args.output / "collection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    llm = None
    try:
        llm = LLM(**config)
        native = llm.collective_rpc(
            "install_moe_hooks",
            args=(
                selected,
                model_config["num_hidden_layers"],
                args.record_routing,
                args.record_norms,
            ),
        )
        if len(native) != args.tensor_parallel_size or native[0] is None:
            raise RuntimeError("Collector requires TP rank zero")
        report["native_model"] = native[0]
        for offset in range(0, len(documents), args.batch_size):
            batch = documents[offset : offset + args.batch_size]
            # Enqueue can schedule before the extension RPC arms request IDs.
            llm.sleep(level=0, mode="keep")
            request_ids = llm.enqueue(
                [{"prompt_token_ids": d["tokens"]} for d in batch],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
                use_tqdm=False,
            )
            llm.collective_rpc(
                "arm_moe_inputs",
                args=(
                    dict(
                        zip(
                            request_ids, range(offset, offset + len(batch)), strict=True
                        )
                    ),
                ),
            )
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(output_type=RequestOutput, use_tqdm=False)
            if len(outputs) != len(batch) or any(
                out.prompt_token_ids != d["tokens"]
                for out, d in zip(outputs, batch, strict=True)
            ):
                raise RuntimeError("Native prefill did not preserve prompt tokens")
            llm.collective_rpc("flush_moe_inputs")
            print(
                f"collected {offset + len(batch)}/{len(documents)} documents",
                flush=True,
            )
        report["shards"] = llm.collective_rpc(
            "save_moe_inputs",
            args=(str(args.output.resolve()),),
        )[0]
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        finally:
            report["elapsed_seconds"] = time.perf_counter() - start
            report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
