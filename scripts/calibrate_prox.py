"""Fit Prox thresholds layer by layer on FIT text in vLLM, one file per target read.

Each layer is fitted while earlier layers run their fitted sparse rules. Writes
`calibration.json` and one `prox-read-NN.pt` per `--target-reads` value (NN in
percent of dense routed-expert reads).
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
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor

from calibration_free_moe.activation import model_activation
from calibration_free_moe.model_config import (
    model_dimensions,
    model_override,
    validate_prox_keep_options,
)


def quantile(values: Tensor, sparsity: float) -> float:
    """Return the inclusive `sparsity` quantile of `values`; -1 keeps everything."""
    if sparsity == 0:
        return -1.0
    flat = values.float().reshape(-1)
    return float(flat.kthvalue(math.ceil(sparsity * (flat.numel() - 1)) + 1).values)


def fit_keep_threshold(values: Tensor, kept: float) -> float:
    """Return the >= threshold whose kept count is nearest `kept`.

    Equal-valued scores are never split: the threshold either keeps or drops a
    whole tie at the boundary. The input comparison is in BF16, the channel
    comparison in FP32. Step in the original dtype when excluding a tied boundary,
    otherwise BF16 rounds it back.
    """
    if kept == 0:
        return math.inf
    if kept == values.numel():
        return -1.0
    flat = values.float().flatten()
    boundary = flat.kthvalue(flat.numel() - math.ceil(kept) + 1).values
    above = int((flat > boundary).sum())
    inclusive = int((flat >= boundary).sum())
    if abs(above - kept) < abs(inclusive - kept):
        return float(
            torch.nextafter(
                boundary.to(values.dtype), torch.full((), math.inf, dtype=values.dtype)
            )
        )
    return float(boundary)


def reservoir_positions(length: int) -> Tensor:
    """Return at most 200,000 evenly spaced integer positions in [0, length).

    Integer arithmetic, because a floating-point linspace can land past the end.
    """
    count = min(200_000, length)
    if count == 1:
        return torch.zeros(1, dtype=torch.int64)
    return torch.arange(count, dtype=torch.int64) * (length - 1) // (count - 1)


def allocation(
    target_read: float, input_sparsity: float | None = None
) -> tuple[float, float]:
    """Return the Stage-1 (input) and Stage-2 (channel) sparsities for a read target.

    Without `input_sparsity` this is the Prox paper's latency-model split; with it,
    Stage 2 gets the read budget left after the input proxy.
    """
    if input_sparsity is None:
        # Prox paper, Sec. 4.2: alpha=1/3 is a latency model, not the byte ratio.
        e = 1 - target_read
        s1 = max(0.0, min(0.7, 1 - 4.5 * (0.7 - e)))
        return s1, e + (2 / 9) * (1 - s1)
    # Byte allocation for a fixed Stage-1 sparsity: the coordinate-major INT4 proxy
    # reads (1 - s1) / 6 of a slot's BF16 gate/up/down bytes, native rows 1 - s2.
    # Row scales (4 / (3H), 0.07 points at H = 2048) are left unfunded, so the
    # target is approximate; results are reported at their measured reads.
    s2 = 1 - (target_read - (1 - input_sparsity) / 6)
    if not 0 <= s2 < 1:
        raise ValueError("Target read cannot fund this input sparsity's proxy")
    return input_sparsity, s2


class ProxCalibrationWorkerExtension:
    """Replay FIT prefill with finalized sparse MoEs, collecting one dense layer.

    vLLM worker extension: the driver calls these methods on every rank through
    `collective_rpc`. Under expert parallelism every rank scores its resident
    experts at the same sample positions, and a collective assembles the
    single-GPU sample.
    """

    prox_modules: ClassVar[list[Any]] = []
    prox_target: int = -1
    prox_states: ClassVar[list[Tensor]] = []
    prox_routes: ClassVar[list[Tensor]] = []
    prox_weights: ClassVar[list[Tensor]] = []
    prox_limit: int = 0

    def prepare_prox_calibration(
        self: Any,
        max_states: int,
        proxy_score: Literal["prox", "output-unit"] = "prox",
    ) -> int:
        """Wrap every MoE layer's forward and build its INT4 proxy; return the count.

        Layers before the target run their fitted Prox reader, the target layer runs
        dense while its inputs and router outputs are recorded, and later layers
        run dense.
        """
        from calibration_free_moe.baselines.prox import quantize_proxy
        from calibration_free_moe.runtime.experts import ProgressiveExperts

        modules = {
            int(name.split(".layers.")[1].split(".")[0]): module
            for name, module in self.model_runner.model.named_modules()
            if isinstance(module, ProgressiveExperts)
        }
        self.prox_modules = [modules[i] for i in range(len(modules))]
        self.prox_target = -1
        self.prox_states = []
        self.prox_routes = []
        worker = cast(ProxCalibrationWorkerExtension, self)
        self.prox_weights = []
        self.prox_limit = max_states
        for layer, module in enumerate(self.prox_modules):
            native_forward = module.forward_modular

            def forward(
                x: Tensor,
                topk_weights: Tensor,
                topk_ids: Tensor,
                *extra: Any,
                _layer: int = layer,
                _module: Any = module,
                _native: Any = native_forward,
                **kwargs: Any,
            ) -> Tensor:
                if _layer == self.prox_target:
                    # Collect the same on-trajectory input and natural router IDs.
                    self.prox_states.append(x.detach().cpu())
                    self.prox_routes.append(topk_ids.detach().cpu())
                    if proxy_score == "output-unit":
                        worker.prox_weights.append(topk_weights.detach().cpu())
                if _layer < self.prox_target:
                    return _module.reader(x, topk_ids, topk_weights)
                return _native(x, topk_weights, topk_ids, *extra, **kwargs)

            cast(Any, module).forward_modular = forward
            # The same deterministic INT4 proxy the evaluation binding builds.
            module.prox_packed, module.prox_scales = quantize_proxy(
                cast(Tensor, module.w13_weight).detach()
            )
        return len(self.prox_modules)

    def reset_prox_calibration(self: Any) -> None:
        """Start a read level from the all-dense model on the same engine."""
        for module in self.prox_modules:
            module.reader = None
        self.prox_target = -1
        self.prox_states = []
        self.prox_routes = []
        self.prox_weights = []

    def begin_prox_layer(self: Any, layer: int) -> None:
        """Make `layer` the dense layer whose inputs the next prefill records."""
        self.prox_target = layer
        self.prox_states = []
        self.prox_routes = []
        self.prox_weights = []

    def finalize_prox_layer(
        self: Any,
        s1: float,
        s2: float,
        target_read: float,
        input_keep: float | None = None,
        channel_keep: float | None = None,
        channel_floor: float | None = None,
        recipe: Literal["legacy", "study"] = "legacy",
        proxy_score: Literal["prox", "output-unit"] = "prox",
    ) -> dict[str, float | int | None]:
        """Fit the target layer's thresholds and install its Prox reader.

        Uses at most `prox_limit` evenly spaced recorded tokens. Returns the input
        and channel thresholds (None where a fixed top-k replaces one), the number
        of fitting states and, for the study recipe, the read fraction realized on
        the fitting states.
        """
        from vllm.distributed import get_ep_group

        from calibration_free_moe.baselines.activation_sparsity import (
            grouped_assignment,
            local_ids,
        )
        from calibration_free_moe.baselines.prox import (
            ProxBaseline,
            output_unit_scores,
            proxy_scores,
        )

        layer = self.prox_target
        module = self.prox_modules[layer]
        activation = model_activation(self.model_runner.model_config.hf_text_config)
        expert_map = module.expert_map
        experts = (
            expert_map.numel() if expert_map is not None else module.w13_weight.shape[0]
        )
        states = torch.cat(self.prox_states)
        ids = torch.cat(self.prox_routes)
        positions = torch.linspace(
            0, len(states) - 1, min(self.prox_limit, len(states))
        ).long()
        x = states[positions].to(module.w13_weight.device)
        routes = ids[positions].to(x.device)
        weights = (
            torch.cat(cast(ProxCalibrationWorkerExtension, self).prox_weights)[
                positions
            ].to(x.device)
            if proxy_score == "output-unit"
            else None
        )
        norms = (
            torch.stack(
                [
                    expert.float().norm(dim=0)
                    for expert in cast(Tensor, module.w2_weight)
                ]
            )
            if proxy_score == "output-unit"
            else None
        )
        if not len(x):
            raise RuntimeError(f"No FIT calibration inputs in layer {layer}")
        k, width = routes.shape[1], module.prox_scales.shape[-1] // 2
        hidden = x.shape[-1]
        n = k * width
        inputs = (
            None
            if input_keep is None
            else max(1, math.ceil(input_keep * hidden - 1e-9))
        )
        channels = (
            None if channel_keep is None else max(1, math.ceil(channel_keep * n - 1e-9))
        )
        floor = (
            0 if channel_floor is None else max(1, math.ceil(channel_floor * n - 1e-9))
        )
        study = recipe == "study"
        # Prices are fractions of dense routed BF16 reads, as the Prox operator
        # counts them: the INT4 proxy is read in 32-byte coordinate-major runs
        # (input_price per kept input coordinate) plus FP32 row scales, and
        # output-unit scores also read FP32 down-column norms. Selected native rows
        # cost F/(K*I). A matched prox-free arm at read R with J input coordinates
        # thus keeps F = the nearest integer to K*I*(R - J*input_price - scale_price).
        input_price = math.ceil(width / 32) * 32 / (6 * width * hidden)
        scale_price = 4 / (3 * hidden)
        norm_price = 2 / (3 * hidden) if proxy_score == "output-unit" else 0.0
        if channels is not None:
            input_fraction = (target_read - channels / n - scale_price) / (
                hidden * input_price
            )
            if not 0 < input_fraction <= 1:
                raise ValueError(
                    "Target read cannot fund the fixed channels "
                    "plus a sparse input proxy"
                )
            s1 = 1 - input_fraction
        # Keep the Prox paper's scalar cap (its Appendix A.7) for the legacy recipe.
        # Study cells fit the full empirical distribution so a floor reserves
        # exactly m per token.
        tx = None
        if inputs is None:
            if study:
                tx = fit_keep_threshold(x.abs().cpu(), (1 - s1) * x.numel())
            else:
                input_indices = reservoir_positions(x.numel()).to(x.device)
                tx = quantile(x.abs().flatten()[input_indices].cpu(), s1)
        samples, sampled = None, None
        if not study:
            samples = reservoir_positions(len(x) * k * width).to(x.device)
            sampled = torch.zeros(len(samples), device=x.device, dtype=torch.float32)
        tails: list[Tensor] = []
        kept_inputs = 0
        for start in range(0, len(x), 64):
            batch = routes[start : start + 64]
            if channels is not None:
                kept_inputs += int(
                    (x[start : start + 64].abs() >= cast(float, tx)).sum()
                )
                continue
            score, _ = proxy_scores(
                x[start : start + 64],
                batch,
                module.prox_packed,
                module.prox_scales,
                tx,
                assignment=grouped_assignment(batch.reshape(-1), experts, expert_map),
                activation=activation,
                expert_map=expert_map,
                input_keep=inputs,
                stable_ties=study,
            )
            slot_experts = local_ids(batch, expert_map)
            if norms is not None:
                assert weights is not None
                score = output_unit_scores(
                    score,
                    weights[start : start + 64],
                    norms,
                    slot_experts,
                )
            # Grouped assignment can leave inactive rows unwritten. Only resident
            # slots contribute to the assembled nonnegative calibration sample.
            score = torch.where(slot_experts.reshape(-1, 1) >= 0, score.abs(), 0)
            if study:
                # Assemble full tokens before a joint floor, never EP-local tails.
                if expert_map is not None:
                    torch.distributed.all_reduce(
                        score,
                        op=torch.distributed.ReduceOp.MAX,
                        group=get_ep_group().device_group,
                    )
                values = score.view(len(batch), n)
                if floor:
                    # Calibration needs only the tail's values, not identities:
                    # choosing any of several equal scores leaves the same tail.
                    values = values.topk(
                        n - floor, dim=-1, largest=False, sorted=False
                    ).values
                tails.append(values.cpu())
                kept_inputs += (
                    len(batch) * inputs
                    if inputs is not None
                    else int((x[start : start + 64].abs() >= cast(float, tx)).sum())
                )
                continue
            offset = start * k * width
            first = int(torch.searchsorted(cast(Tensor, samples), offset))
            last = int(
                torch.searchsorted(cast(Tensor, samples), offset + score.numel())
            )
            cast(Tensor, sampled)[first:last] = score.abs().flatten()[
                cast(Tensor, samples)[first:last] - offset
            ]
        fit_read: float | None = None
        ts: float | None
        if study:
            proxy_read = kept_inputs / len(x) * input_price + scale_price + norm_price
            if channels is not None:
                ts = None
                fit_read = proxy_read + channels / n
            else:
                desired = (target_read - proxy_read) * n
                if not floor <= desired <= n:
                    raise ValueError(
                        f"Target {target_read:g} cannot fund proxy {proxy_read:g} "
                        f"and floor {floor}/{n}, or exceeds dense channel reads"
                    )
                values = torch.cat(tails).flatten()
                # The top-m are always kept; only the other N-m values are
                # thresholded. The runtime union of both sets has exactly this
                # count, also when several channels tie at the boundary.
                ts = fit_keep_threshold(values, (desired - floor) * len(x))
                fit_read = proxy_read + (floor + int((values >= ts).sum()) / len(x)) / n
        else:
            group = get_ep_group()
            if group.world_size > 1:
                torch.distributed.all_reduce(
                    cast(Tensor, sampled), group=group.device_group
                )
            ts = quantile(cast(Tensor, sampled).cpu(), s2)
        module.reader = ProxBaseline(
            module.w13_weight,
            module.w2_weight,
            module.prox_packed,
            module.prox_scales,
            tx,
            ts,
            activation,
            expert_map,
            input_keep=inputs,
            channel_keep=channels,
            channel_scope="token" if study else "slot",
            channel_floor=floor,
            recipe=recipe,
            proxy_score=proxy_score,
        )
        self.prox_states = []
        self.prox_routes = []
        self.prox_weights = []
        # Current layer becomes sparse for the next pass, but was dense here.
        return {
            "input_threshold": tx,
            "channel_threshold": ts,
            "fit_states": len(x),
            "fit_read_fraction": fit_read,
        }


def main() -> None:
    """Check the read targets, then fit and save every target read in one engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-states", type=int, default=20_480)
    parser.add_argument("--fit-documents", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--target-reads", type=float, nargs="+", required=True)
    parser.add_argument(
        "--recipe",
        choices=("legacy", "study"),
        default="legacy",
        help="legacy: the published Prox recipe (capped scalar sample, inclusive "
        "quantile, prescribed stage budgets); study: full-sample nearest-attainable "
        "thresholds and realized read accounting, the same for every keep rule",
    )
    _ = parser.add_argument(
        "--proxy-score",
        choices=("prox", "output-unit"),
        default="prox",
        help="Fit thresholds on |proxy| or on router weight x down-column norm x "
        "|proxy| (output-unit; study recipe with --input-keep)",
    )
    parser.add_argument(
        "--input-sparsity",
        type=float,
        help="Fix Stage-1 sparsity s1 and give Stage 2 the remaining read budget "
        "(default: the Prox paper's latency-model split)",
    )
    parser.add_argument(
        "--input-keep",
        type=float,
        help="Fix per-token input top-k; fit only the channel threshold",
    )
    parser.add_argument(
        "--channel-keep",
        type=float,
        help="Fix joint channel top-k; fit the input threshold to the remaining reads",
    )
    parser.add_argument(
        "--channel-floor",
        type=float,
        help="Keep each token's top channels, this fraction of K*I, and fit the "
        "channel threshold on the rest",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    args = parser.parse_args()
    proxy_score = cast(Literal["prox", "output-unit"], args.proxy_score)
    try:
        validate_prox_keep_options(
            "prox",
            args.input_keep,
            args.channel_keep,
            args.channel_floor,
            args.recipe,
            proxy_score,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.input_sparsity is not None and (
        args.input_keep is not None or args.channel_keep is not None
    ):
        parser.error("--input-sparsity cannot accompany either fixed top-k rule")
    if (
        args.tensor_parallel_size > 1
        and args.recipe != "study"
        and (args.channel_keep is not None or args.channel_floor is not None)
    ):
        parser.error("Joint EP calibration requires --recipe study")
    if args.tensor_parallel_size > 1 and not args.enable_expert_parallel:
        parser.error("Prox under TP requires --enable-expert-parallel")
    if (
        args.max_states < 1
        or args.fit_documents < 1
        or args.batch_size < 1
        or args.tensor_parallel_size < 1
        or not 0 < args.gpu_memory_utilization < 1
        or any(not 0 < r < 1 for r in args.target_reads)
        or not (args.input_sparsity is None or 0 <= args.input_sparsity < 1)
    ):
        parser.error("Positive state/doc caps and target reads in (0,1) required")
    study = args.recipe == "study"
    if not study and any(
        v is not None for v in (args.input_keep, args.channel_keep, args.channel_floor)
    ):
        parser.error("Mixed keep rules and floors require --recipe study")
    if study:
        dimensions = model_dimensions(args.model)
        hidden, width = int(dimensions["hidden_size"]), int(dimensions["expert_width"])
        n = int(dimensions["num_experts_per_tok"]) * width
        scale_price = 4 / (3 * hidden)
        norm_price = 2 / (3 * hidden) if proxy_score == "output-unit" else 0.0
        full_proxy = math.ceil(width / 32) * 32 / (6 * width)
        if args.channel_keep is not None:
            fixed = max(1, math.ceil(args.channel_keep * n - 1e-9)) / n + scale_price
            if any(
                not fixed < read <= fixed + full_proxy for read in args.target_reads
            ):
                parser.error(
                    "Target reads cannot fund this channel keep "
                    "and a sparse input proxy"
                )
        else:
            minimum = scale_price + norm_price
            if args.input_keep is not None:
                minimum += (
                    max(1, math.ceil(args.input_keep * hidden - 1e-9))
                    / hidden
                    * full_proxy
                )
            if args.channel_floor is not None:
                minimum += max(1, math.ceil(args.channel_floor * n - 1e-9)) / n
            if any(read < minimum for read in args.target_reads):
                parser.error(
                    "Target reads cannot fund the input keep and channel floor"
                )
    if args.input_keep is None and args.channel_keep is None:
        for read in args.target_reads:
            allocation(read, args.input_sparsity)
    panel = json.loads(args.panel.read_text())
    documents = [
        doc for doc in panel["documents"] if doc["split_role"].lower() == "fit"
    ][: args.fit_documents]
    if (
        len(documents) != args.fit_documents
        or sum(len(doc["tokens"]) for doc in documents) < args.max_states
    ):
        parser.error("Insufficient FIT-only tokens in panel")
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "complete": False,
        "command": sys.argv,
        "revision": args.revision,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "panel": str(args.panel),
        "fit_document_ids": [doc["id"] for doc in documents],
        "max_states": args.max_states,
        "recipe": args.recipe,
        "proxy_score": proxy_score,
        "levels": {},
    }
    report_path = args.output / "calibration.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    start = time.perf_counter()
    try:
        from vllm import LLM, SamplingParams

        # The method override keeps stock vLLM attention, routing, and prefill.
        # Only this calibration worker substitutes finalized sparse MoE outputs.
        config: dict[str, Any] = {
            "model": str(args.model),
            "tokenizer": str(args.model),
            "dtype": "bfloat16",
            "tensor_parallel_size": args.tensor_parallel_size,
            "enable_expert_parallel": args.enable_expert_parallel,
            "max_model_len": max(len(doc["tokens"]) for doc in documents) + 1,
            "max_num_seqs": args.batch_size,
            "max_num_batched_tokens": max(
                2048,
                sum(
                    sorted((len(doc["tokens"]) for doc in documents), reverse=True)[
                        : args.batch_size
                    ]
                ),
            ),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
            "enable_prefix_caching": False,
            "moe_backend": "triton",
            "model_class_overrides": model_override(str(args.model)),
            "additional_config": {
                "tokenwise_phase_metadata": True,
                "tokenwise": {"policy": "dense-calibration"},
            },
            "worker_extension_cls": (
                "scripts.calibrate_prox.ProxCalibrationWorkerExtension"
            ),
        }
        if model_dimensions(args.model)["architecture"].startswith("Gemma4"):
            # Same Gemma scheduling floor as the evaluation runner.
            config["max_num_batched_tokens"] = max(
                config["max_num_batched_tokens"], 3072
            )
            config["max_model_len"] = max(
                config["max_model_len"],
                math.ceil(config["max_num_batched_tokens"] / args.batch_size),
            )
        # The weight loader's calibration policy is only used at runtime:
        # load native weights without binding a reader, then fit one at a time.
        # One engine and INT4 proxy serve every level; each level restarts from
        # the dense model and repeats the same sequential calibration.
        llm = LLM(**config)
        try:
            layers = llm.collective_rpc(
                "prepare_prox_calibration", args=(args.max_states, proxy_score)
            )[0]
            for read in args.target_reads:
                s1, s2 = (
                    allocation(read, args.input_sparsity)
                    if args.input_keep is None and args.channel_keep is None
                    else (0.0, 0.0)
                )
                llm.collective_rpc("reset_prox_calibration")
                thresholds = []
                for layer in range(layers):
                    llm.collective_rpc("begin_prox_layer", args=(layer,))
                    for offset in range(0, len(documents), args.batch_size):
                        batch = documents[offset : offset + args.batch_size]
                        llm.generate(
                            [{"prompt_token_ids": doc["tokens"]} for doc in batch],
                            SamplingParams(temperature=0, max_tokens=1),
                            use_tqdm=False,
                        )
                    ranks = llm.collective_rpc(
                        "finalize_prox_layer",
                        args=(
                            s1,
                            s2,
                            read,
                            args.input_keep,
                            args.channel_keep,
                            args.channel_floor,
                            args.recipe,
                            proxy_score,
                        ),
                    )
                    # EP ranks see replicated inputs and one summed score sample.
                    if any(rank != ranks[0] for rank in ranks):
                        raise RuntimeError(f"EP ranks disagree on thresholds: {ranks}")
                    threshold = ranks[0]
                    thresholds.append(threshold)
                    print(f"read={read:.2f} layer={layer} {threshold}", flush=True)
                entry = {
                    "format": "prox-v1",
                    "policy": "prox",
                    "model_revision": args.model.name,
                    "proxy": (
                        "symmetric per-row INT4 of native gate/up, quantized at load"
                    ),
                    "target_read": read,
                    "recipe": args.recipe,
                    "proxy_score": proxy_score,
                    "stage_sparsity": None if study else (s1, s2),
                    "input_sparsity_target": s1
                    if args.input_keep is None and args.channel_keep is None
                    else None,
                    "keep_rules": {
                        "input_keep": args.input_keep,
                        "channel_keep": args.channel_keep,
                        "channel_floor": args.channel_floor,
                    },
                    "calibration": {
                        "trajectory": "sequential-sparse",
                        "tensor_parallel_size": args.tensor_parallel_size,
                        "expert_parallel": args.enable_expert_parallel,
                        "fit_documents": len(documents),
                        "max_states_per_layer": args.max_states,
                        "scalar_sampling": "all_collected_scalars"
                        if study
                        else "deterministic_evenly_spaced_integer_positions",
                        "max_scalar_samples_per_signal": None if study else 200_000,
                        "threshold_ties": (
                            "nearest_attainable_inclusive_on_equal_distance"
                            if study
                            else "inclusive_quantile"
                        ),
                        "topk_ties": "lowest_index" if study else "torch_topk",
                        "read_allocation": (
                            "realized_proxy_sectors_scales_and_norms_then_channels"
                            if proxy_score == "output-unit"
                            else "realized_proxy_sectors_and_scales_then_channels"
                        )
                        if study
                        else "prescribed_stage_sparsities",
                    },
                    "thresholds": thresholds,
                }
                path = args.output / f"prox-read-{round(read * 100):02d}.pt"
                torch.save(entry, path)
                report["levels"][str(read)] = str(path)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            llm.llm_engine.engine_core.shutdown()
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
