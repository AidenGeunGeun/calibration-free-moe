"""Fit a calibrated recovery map (`bank`) by whole-model KL, as a reference.

Trains every layer's rank-32 decoder U and encoders B with Adam to minimize
KL(stock || sparse) over a `--window` of tokens after each FIT document's context,
with the reader's channel selection in the loop; TUNE documents pick the best
bank. Writes `decoder.pt` and `fit.json`, or `validation.json` with `--validate`.
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
from typing import Any, cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import (
    Gemma4ForCausalLM,
    OlmoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3MoeForCausalLM,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
    OlmoeSparseMoeBlock,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeSparseMoeBlock,
)

from calibration_free_moe.activation import (
    GatedActivation,
    activate,
    model_activation,
)
from calibration_free_moe.model_config import (
    checkpoint_identity,
    model_dimensions,
)
from calibration_free_moe.reader.budget import BudgetPlan, compile_budget
from calibration_free_moe.reader.progressive import (
    ProgressiveReader,
    ReaderWorkspace,
    SelectionScore,
    channel_major_down,
)
from calibration_free_moe.reader.sketch import (
    ACQUISITION,
    block_hadamard_signs,
    build_sketch,
    differentiable_sketch_feature,
    mask_rotated_blocks,
    require_sketch_acquisition,
    rotate_blocks,
    unpack_sketch,
)
from scripts.fit_plan_recovery import (
    RANK,
    candidate_activations,
    extracted_model_output,
    load_layer_weights,
    route,
)


def selected_output(
    x: Tensor,
    reader: ProgressiveReader,
    gu: Tensor,
    down: Tensor,
    u: Tensor,
    b: Tensor,
    plan: BudgetPlan,
    unpacked: Tensor | None = None,
    checkpoint_cached: bool = False,
    activation: GatedActivation = "silu",
) -> Tensor:
    """Return the routed output [tokens, H] the reader computes, differentiably in U, B.

    The selection comes from the reader's last call on `x`; selected channels are
    exact, skipped ones go through the sketch activation and the recovery map
    (`u`, `b`), so gradients reach the map.
    """
    t, h = x.shape
    i, k = plan.expert_width, plan.top_k
    ids = reader.ids[:t].clone().long().reshape(-1)
    weights = reader.weights[:t].clone().float().reshape(-1, 1)
    completed = reader.completed[:t].clone().reshape(t * k, i)
    selected_mask = reader.selected[:t].clone().reshape(t * k, i)
    result = torch.zeros((t, h), device=x.device, dtype=torch.float32)
    latent = torch.zeros((t, RANK), device=x.device, dtype=torch.float32)
    token = torch.arange(t, device=x.device).repeat_interleave(k)
    # Rotate and mask once per token and share the result across its routed experts.
    rotated = mask_rotated_blocks(
        rotate_blocks(x, reader.sketch_signs), plan.screen_coordinates
    )
    for expert in ids.unique().tolist():
        selected = (ids == expert).nonzero().flatten()
        rows = token[selected]
        membership = selected_mask[selected]
        # Same math as rotating per expert; only the input-gradient rounding order
        # differs.
        sketch_feature = differentiable_sketch_feature(
            rotated[rows],
            reader.sketch_codes[expert],
            reader.sketch_statistics[expert],
            reader.sketch_signs,
            unpacked=unpacked[expert] if unpacked is not None else None,
            checkpoint_cached=checkpoint_cached,
            activation=activation,
            input_is_rotated=True,
        )
        full_g, full_u = F.linear(x[rows], gu[expert]).chunk(2, dim=-1)
        exact = (activate(full_g, activation).bfloat16() * full_u).bfloat16()
        partial = torch.where(completed[selected], exact, sketch_feature)
        selected_output = F.linear(exact * membership, down[expert]).float()
        omitted = (partial * ~membership).float()
        result = result.index_add(0, rows, weights[selected] * selected_output)
        latent = latent.index_add(
            0, rows, weights[selected] * F.linear(omitted, b[expert])
        )
    return (result + F.linear(latent.bfloat16().float(), u)).bfloat16()


def moe_block(
    model: OlmoeForCausalLM
    | Qwen3MoeForCausalLM
    | Qwen3_5MoeForConditionalGeneration
    | Gemma4ForCausalLM,
    layer: int,
) -> Any:
    """Return the module that owns one layer's routed experts.

    That is the decoder layer for Gemma 4 and the sparse MoE block otherwise.
    """
    if isinstance(model, Gemma4ForCausalLM):
        return cast(Gemma4TextDecoderLayer, model.model.layers[layer])
    if isinstance(model, Qwen3_5MoeForConditionalGeneration):
        return cast(
            Qwen3_5MoeSparseMoeBlock,
            model.model.language_model.layers[layer].mlp,
        )
    return cast(
        OlmoeSparseMoeBlock | Qwen3MoeSparseMoeBlock,
        cast(OlmoeDecoderLayer | Qwen3MoeDecoderLayer, model.model.layers[layer]).mlp,
    )


def place_model(
    model: OlmoeForCausalLM
    | Qwen3MoeForCausalLM
    | Qwen3_5MoeForConditionalGeneration
    | Gemma4ForCausalLM,
    devices: list[torch.device],
) -> None:
    """Place contiguous decoder-layer ranges on the devices, without Accelerate.

    Hooks move each layer's inputs to its device and the logits back to the first.
    """
    if len(devices) == 1:
        torch.nn.Module.to(model, devices[0])
        return

    def move(value: Any, device: torch.device) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move(item, device) for item in value)
        if isinstance(value, list):
            return [move(item, device) for item in value]
        if isinstance(value, dict):
            return {key: move(item, device) for key, item in value.items()}
        return value

    def transfer(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        device: torch.device,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return move(args, device), move(kwargs, device)

    base = cast(Any, model).model
    if isinstance(model, Qwen3_5MoeForConditionalGeneration):
        # Text-only calibration never enters the vision tower; leave it on CPU.
        text = base.language_model
    else:
        text = base
    tied_head = text.embed_tokens.weight is model.lm_head.weight
    torch.nn.Module.to(text.embed_tokens, devices[-1] if tied_head else devices[0])
    if tied_head:
        # Gemma shares the input embedding and output head. Keep that one
        # parameter beside the head, moving only token embeddings back to the
        # first decoder device; never manufacture a second full vocab matrix.
        text.embed_tokens.register_forward_pre_hook(
            lambda module, args, kwargs: transfer(
                module, args, kwargs, device=devices[-1]
            ),
            with_kwargs=True,
        )
        text.embed_tokens.register_forward_hook(
            lambda _module, _args, output: move(output, devices[0])
        )
    torch.nn.Module.to(text.rotary_emb, devices[0])
    for layer, block in enumerate(text.layers):
        device = devices[layer * len(devices) // len(text.layers)]
        torch.nn.Module.to(block, device)
        # HF passes the original rotary/mask tensors to every layer, not the
        # copies moved at the preceding decoder boundary.
        if device != devices[0]:
            block.register_forward_pre_hook(
                lambda module, args, kwargs, device=device: transfer(
                    module, args, kwargs, device=device
                ),
                with_kwargs=True,
            )
            # Hybrid attention can launch its own Triton kernels before MoE.
            decoder_forward = block.forward

            def forward_on_device(
                *args: Any,
                device: torch.device = device,
                original: Any = decoder_forward,
                **kwargs: Any,
            ) -> Any:
                with torch.cuda.device(device):
                    return original(*args, **kwargs)

            cast(Any, block).forward = forward_on_device
    torch.nn.Module.to(text.norm, devices[-1])
    # Keep the large output head beside the final decoder layer, then return
    # only the scored logits to the first device for the KL objective.
    torch.nn.Module.to(model.lm_head, devices[-1])
    model.lm_head.register_forward_pre_hook(
        lambda module, args, kwargs: transfer(module, args, kwargs, device=devices[-1]),
        with_kwargs=True,
    )
    model.lm_head.register_forward_hook(
        lambda _module, _args, output: move(output, devices[0])
    )


class Calibration:
    """A frozen HF model whose MoE layers can run the reader with a trainable bank.

    Patching each layer's routed-expert forward lets one model serve as the stock
    teacher (disabled) and the sparse student (enabled, from `sparse_from` on).
    `sparse_layers` limits the student to some layers; `reader_output` returns the
    reader's own output instead of the differentiable one.
    """

    def __init__(
        self,
        model: OlmoeForCausalLM
        | Qwen3MoeForCausalLM
        | Qwen3_5MoeForConditionalGeneration
        | Gemma4ForCausalLM,
        plan: BudgetPlan,
        window: int,
        model_path: Path,
        devices: list[torch.device] | None = None,
        cache_sketch: bool = False,
        teacher_on_gpu: bool = False,
        sparse_prefix: bool = False,
        capacity: int | None = None,
        selection_score: SelectionScore = "recovery-aware",
        sparse_layers: set[int] | None = None,
        reader_output: bool = False,
    ):
        self.model = model
        self.devices = devices or [torch.device("cuda:0")]
        layer_count = model_dimensions(model_path)["num_hidden_layers"]
        self.layer_devices = [
            self.devices[layer * len(self.devices) // layer_count]
            for layer in range(layer_count)
        ]
        self.cache_sketch = cache_sketch
        self.teacher_on_gpu = teacher_on_gpu
        # A sparse-prefix student runs the policy on every token, so every layer
        # sees sparse-trajectory states; its reader must hold whole sequences.
        self.sparse_prefix = sparse_prefix
        self.capacity = capacity or window
        self.unpacked: dict[int, Tensor] = {}
        self.sketch_cache_bytes: dict[str, int] = {
            str(device): 0 for device in self.devices
        }
        self.plan = plan
        self.selection_score: SelectionScore = selection_score
        self.sparse_layers = sparse_layers
        self.reader_output = reader_output
        self.window = window
        self.sparse_from = 0
        self.enabled = False
        self.readers: dict[int, ProgressiveReader] = {}
        self.sketches: dict[int, tuple[Tensor, Tensor]] = {}
        self.model_path = model_path
        self.refresh_seconds = 0.0
        # Frozen teacher distributions are reused at every update and monitor.
        self.teacher: dict[str, tuple[Tensor, Tensor]] = {}
        config = cast(
            Any,
            (
                model.config.text_config
                if isinstance(model, Qwen3_5MoeForConditionalGeneration)
                else model.config
            ),
        )
        self.layers = config.num_hidden_layers
        self.gemma = isinstance(model, Gemma4ForCausalLM)
        self.activation: GatedActivation = model_activation(config)
        self.renormalize = bool(
            getattr(
                config,
                "norm_topk_prob",
                isinstance(model, Qwen3_5MoeForConditionalGeneration),
            )
        )
        self.u: Tensor | list[Tensor]
        self.b: Tensor | list[Tensor]
        if len(self.devices) == 1:
            self.u = torch.zeros(
                (self.layers, plan.hidden_size, RANK),
                device=self.devices[0],
                dtype=torch.float32,
            )
            self.b = torch.zeros(
                (self.layers, config.num_experts, RANK, plan.expert_width),
                device=self.devices[0],
                dtype=torch.float32,
            )
        else:
            self.u = [
                torch.zeros(
                    (plan.hidden_size, RANK), device=device, dtype=torch.float32
                )
                for device in self.layer_devices
            ]
            self.b = [
                torch.zeros(
                    (config.num_experts, RANK, plan.expert_width),
                    device=device,
                    dtype=torch.float32,
                )
                for device in self.layer_devices
            ]
        for layer in range(self.layers):
            block = moe_block(model, layer)
            # The offline teacher and differentiable output share this one
            # frozen storage with every reader refresh, not two resident banks.
            down = cast(Tensor, block.experts.down_proj)
            down.data = channel_major_down(down)
            target = block.experts if self.gemma else block
            original = target.forward

            def forward(
                hidden: Tensor,
                *routing: Tensor,
                layer: int = layer,
                original: Any = original,
            ) -> Tensor:
                # Reader Triton launches use the current CUDA device, not the
                # device inferred from their tensor arguments.
                with torch.cuda.device(self.layer_devices[layer]):
                    if not self.enabled or (
                        self.sparse_layers is not None
                        and layer not in self.sparse_layers
                    ):
                        return original(hidden, *routing)
                    # A causal stock prefix cannot depend on any sparse suffix
                    # state. Cut this dead gradient path before the frozen prefill
                    # MLP.
                    prefix = (
                        hidden[: self.sparse_from]
                        if self.gemma
                        else hidden[:, : self.sparse_from]
                    ).detach()
                    sparse = (
                        hidden[self.sparse_from :]
                        if self.gemma
                        else hidden[:, self.sparse_from :].reshape(-1, hidden.shape[-1])
                    )
                    if sparse.shape[0] == 0:
                        return original(hidden, *routing)
                    current = moe_block(self.model, layer)
                    with torch.no_grad():
                        if self.gemma:
                            ids = routing[0][self.sparse_from :]
                            weights = routing[1][self.sparse_from :]
                        else:
                            gate = current.gate.weight
                            ids, weights = route(
                                sparse.detach(), gate, self.plan.top_k, self.renormalize
                            )
                        reader = self.readers[layer]
                        output = reader(
                            sparse.detach().contiguous(),
                            ids.contiguous(),
                            weights.contiguous(),
                        )
                    experts = current.experts
                    answer = (
                        output
                        if self.reader_output
                        else selected_output(
                            sparse,
                            reader,
                            cast(Tensor, experts.gate_up_proj),
                            cast(Tensor, experts.down_proj),
                            self.u[layer],
                            self.b[layer],
                            self.plan,
                            self.unpacked.get(layer),
                            checkpoint_cached=len(self.devices) > 1,
                            activation=self.activation,
                        )
                    )
                    if not self.gemma:
                        answer = answer.reshape(hidden.shape[0], -1, hidden.shape[-1])
                    if isinstance(current, Qwen3_5MoeSparseMoeBlock):
                        shared = current.shared_expert(sparse)
                        shared = shared * torch.sigmoid(
                            current.shared_expert_gate(sparse)
                        )
                        answer = answer + shared.reshape_as(answer)
                    if self.sparse_from:
                        routed_prefix = (
                            (r[: self.sparse_from] for r in routing)
                            if self.gemma
                            else (r[:, : self.sparse_from] for r in routing)
                        )
                        answer = torch.cat(
                            (original(prefix, *routed_prefix), answer),
                            dim=0 if self.gemma else 1,
                        )
                    return answer

            cast(Any, target).forward = forward

    def refresh(self) -> None:
        """Rebuild every sparse layer's reader from the current bank.

        Sketches are built once and kept; decoded sketches are cached when memory
        allows.
        """
        started = time.perf_counter()
        self.readers.clear()
        # Layers run sequentially and selected_output copies reader state before
        # the next layer, so each device's readers share one scratch workspace.
        workspaces: dict[str, ReaderWorkspace] = {}
        # Keep room for sketches/readers not yet built and the backward graph.
        cache_budget = {
            str(device): max(0, torch.cuda.mem_get_info(device)[0] - 14 * 1024**3)
            for device in self.devices
        }
        with torch.no_grad():
            for layer in range(self.layers):
                if self.sparse_layers is not None and layer not in self.sparse_layers:
                    continue
                device = self.layer_devices[layer]
                with torch.cuda.device(device):
                    experts = moe_block(self.model, layer).experts
                    gu = cast(Tensor, experts.gate_up_proj).detach().contiguous()
                    if layer not in self.sketches:
                        self.sketches[layer] = build_sketch(
                            gu,
                            signs=block_hadamard_signs(
                                self.plan.hidden_size, gu.device
                            ),
                        )
                        codes = self.sketches[layer][0]
                        decoded_bytes = (
                            codes.shape[0]
                            * 2
                            * self.plan.expert_width
                            * self.plan.hidden_size
                        )
                        key = str(device)
                        if (
                            self.cache_sketch
                            and self.sketch_cache_bytes[key] + decoded_bytes
                            <= cache_budget[key]
                        ):
                            self.unpacked[layer] = unpack_sketch(
                                codes, 2 * self.plan.expert_width, self.plan.hidden_size
                            )
                            self.sketch_cache_bytes[key] += decoded_bytes
                    self.readers[layer] = ProgressiveReader(
                        gu,
                        cast(Tensor, experts.down_proj).detach(),
                        self.b[layer].detach().bfloat16(),
                        self.u[layer].detach().bfloat16(),
                        self.plan,
                        self.capacity,
                        sketch=self.sketches[layer],
                        activation=self.activation,
                        workspace=workspaces.get(str(device)),
                        selection_score=self.selection_score,
                    )
                    workspaces.setdefault(str(device), self.readers[layer].workspace)
        for device in self.devices:
            torch.cuda.synchronize(device)
        self.refresh_seconds += time.perf_counter() - started

    def logits(self, tokens: list[int], sparse_from: int, keep: int = 0) -> Tensor:
        """Return FP32 logits, sparse from position `sparse_from` on.

        `keep` > 0 returns only the last `keep` positions; 0 returns all.
        """
        self.sparse_from = sparse_from
        self.enabled = True
        try:
            return (
                self.model(
                    torch.tensor(tokens, device=self.devices[0])[None],
                    use_cache=False,
                    logits_to_keep=keep,
                )
                .logits[0]
                .float()
            )
        finally:
            self.enabled = False

    def kl(self, document: dict[str, Any]) -> Tensor:
        """Return the mean KL(stock || sparse) over the window after the context.

        Teacher distributions are computed once per document and cached.
        """
        context = document["context_length"]
        end = min(len(document["tokens"]), context + self.window)
        tokens = document["tokens"][:end]
        if end <= context:
            raise ValueError("Document has no decode window")
        key = document["id"]
        if key not in self.teacher:
            with torch.no_grad():
                stock = self.logits(tokens, len(tokens), end - context).float()
                log_prob = F.log_softmax(stock, -1)
                target = log_prob.exp()
                entropy_term = (target * log_prob).sum(-1)
                self.teacher[key] = (
                    target.detach() if self.teacher_on_gpu else target.cpu(),
                    entropy_term.detach()
                    if self.teacher_on_gpu
                    else entropy_term.cpu(),
                )
        target, entropy_term = self.teacher[key]
        target = target.to(self.devices[0], non_blocking=True)
        entropy_term = entropy_term.to(self.devices[0], non_blocking=True)
        sparse = self.logits(
            tokens, 0 if self.sparse_prefix else context, end - context
        )
        return (entropy_term - (target * F.log_softmax(sparse, -1)).sum(-1)).mean()

    def bank_cpu(self) -> tuple[Tensor, Tensor]:
        """Return CPU copies of the decoder U [layers, H, r] and encoder B."""
        if isinstance(self.u, Tensor) and isinstance(self.b, Tensor):
            return self.u.detach().cpu(), self.b.detach().cpu()
        assert isinstance(self.u, list) and isinstance(self.b, list)
        return (
            torch.stack([value.detach().cpu() for value in self.u]),
            torch.stack([value.detach().cpu() for value in self.b]),
        )

    def load_bank(self, u: Tensor, b: Tensor) -> None:
        """Copy a bank into the trainable factors on their devices."""
        if isinstance(self.u, Tensor) and isinstance(self.b, Tensor):
            self.u.copy_(u.to(self.devices[0]))
            self.b.copy_(b.to(self.devices[0]))
            return
        for layer in range(self.layers):
            self.u[layer].copy_(u[layer].to(self.layer_devices[layer]))
            self.b[layer].copy_(b[layer].to(self.layer_devices[layer]))

    def bank_parameters(self) -> list[Tensor]:
        """Return the trainable bank tensors."""
        if isinstance(self.u, Tensor) and isinstance(self.b, Tensor):
            return [self.u, self.b]
        assert isinstance(self.u, list) and isinstance(self.b, list)
        return self.u + self.b


def pca_init(
    cal: Calibration,
    states: Path,
    fit_indices: list[int],
    count: int,
    batch_size: int,
) -> list[float]:
    """Initialize each layer's bank from the PCA of its skipped-channel output.

    U holds the top-r eigenvectors of that output's second moment on `count`
    random FIT states, and B_e = U^T D_e. Returns the seconds per layer.
    """
    from scripts.fit_plan_recovery import extract

    torch.manual_seed(353)
    layer_seconds = []
    for layer in range(cal.layers):
        with torch.cuda.device(cal.layer_devices[layer]):
            started = time.perf_counter()
            shard = torch.load(
                states / f"layer-{layer:02d}.pt", map_location="cpu", weights_only=True
            )
            candidates = (
                torch.isin(shard["document"], torch.tensor(fit_indices))
                .nonzero()
                .flatten()
            )
            if candidates.numel() == 0:
                raise ValueError("No FIT states for PCA")
            selected = candidates[torch.randperm(len(candidates))[:count]]
            block = moe_block(cal.model, layer)
            gu = cast(Tensor, block.experts.gate_up_proj).detach().contiguous()
            down = cast(Tensor, block.experts.down_proj).detach()
            gate = (block.router.proj if cal.gemma else block.gate).weight.detach()
            _, _, residual = extract(
                shard["states"][selected],
                gu,
                down,
                gate,
                cal.b[layer].bfloat16(),
                cal.u[layer].bfloat16(),
                cal.plan,
                batch_size,
                cal.renormalize,
                cal.sketches[layer],
                cal.activation,
                shard["route_ids"][selected] if cal.gemma else None,
                shard["route_weights"][selected] if cal.gemma else None,
                selection_score=cal.selection_score,
            )
            moment = residual.double().T @ residual.double()
            eigval, eigvec = torch.linalg.eigh(moment)
            if not bool(torch.isfinite(eigval).all()):
                raise ValueError("Non-finite PCA second moment")
            u = eigvec[:, -RANK:].float().contiguous()
            cal.u[layer].copy_(u)
            cal.b[layer].copy_((down.float().transpose(1, 2) @ u).transpose(1, 2))
            torch.cuda.synchronize(cal.layer_devices[layer])
            layer_seconds.append(time.perf_counter() - started)
            print(f"PCA layer {layer}: {layer_seconds[-1]:.1f}s", flush=True)
            del shard, residual, moment
    return layer_seconds


def validate(cal: Calibration, args: argparse.Namespace, plan: BudgetPlan) -> None:
    """Check a saved bank's reader parity and decode NLLs; write `validation.json`.

    Parity compares the sketch partials and outputs with the reader's own on FIT
    states; decode NLLs are compared with reference runs.
    """
    assert args.states is not None
    if not (args.bank and args.reference and args.stock_reference):
        raise ValueError("Validation requires --bank, --reference, --stock-reference")
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    if bank["decoder"].shape != (cal.layers, *cal.u[0].shape) or bank[
        "encoder"
    ].shape != (cal.layers, *cal.b[0].shape):
        raise ValueError("Bank shapes do not match model")
    require_sketch_acquisition(bank)
    fitted = bank.get("plan", {})
    if (
        bank.get("format") != "weight-subspace-recovery-v1"
        and fitted
        and (
            fitted.get("retention") != args.retention
            or math.ceil(128 * fitted.get("screen_keep", 1.0))
            != plan.screen_coordinates
            or fitted.get("selection_score", "recovery-aware") != cal.selection_score
        )
    ):
        raise ValueError("Validation bank/plan mismatch")
    cal.load_bank(bank["decoder"].float(), bank["encoder"].float())
    cal.refresh()
    parity = []
    for layer in (0, cal.layers // 2):
        device = cal.layer_devices[layer]
        with torch.cuda.device(device):
            shard = torch.load(
                args.states / f"layer-{layer:02d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            collection = json.loads((args.states / "collection.json").read_text())
            fit_indices = [
                j
                for j, d in enumerate(collection["documents"])
                if d["split_role"].lower() == "fit"
            ]
            indices = (
                torch.isin(shard["document"], torch.tensor(fit_indices))
                .nonzero()
                .flatten()[:2048]
            )
            if not len(indices):
                raise ValueError("Insufficient FIT states for parity")
            gu, down, gate = load_layer_weights(args.model, layer, str(device))
            diffs, scales, squared, norms = [], [], [], []
            reader = cal.readers[layer]
            for chunk in indices.split(8):
                x = shard["states"][chunk].to(device)
                if cal.gemma:
                    ids = shard["route_ids"][chunk].to(device).contiguous()
                    weights = shard["route_weights"][chunk].to(device).contiguous()
                else:
                    ids, weights = route(x, gate, plan.top_k, cal.renormalize)
                reader(x.contiguous(), ids, weights)
                t = len(x)
                signed = unpack_sketch(
                    reader.sketch_codes, 2 * plan.expert_width, plan.hidden_size
                )
                scale = (
                    reader.sketch_statistics[..., 0].float().repeat_interleave(128, -1)
                )
                reconstructed = signed.float() * scale.reshape_as(signed)
                rotated = mask_rotated_blocks(
                    rotate_blocks(x, reader.sketch_signs), plan.screen_coordinates
                )
                partial = torch.stack(
                    [
                        torch.bmm(
                            reconstructed[reader.ids[:t, slot].long()],
                            rotated.unsqueeze(-1),
                        )
                        .squeeze(-1)
                        .reshape(t, 2, plan.expert_width)
                        for slot in range(plan.top_k)
                    ],
                    dim=2,
                ).reshape(t, 2, -1)
                omitted = ~reader.completed[:t]
                delta = (partial - reader.partial[:t]).abs() * omitted[:, None]
                diffs.append(delta.amax().item())
                scales.append(
                    (
                        delta.square().sum().item(),
                        (reader.partial[:t] * omitted[:, None]).square().sum().item(),
                    )
                )
                actual = selected_output(
                    x,
                    reader,
                    gu,
                    down,
                    cal.u[layer],
                    cal.b[layer],
                    plan,
                    activation=cal.activation,
                ).float()
                target = extracted_model_output(
                    reader,
                    down,
                    reader.activations[:t],
                    candidate_activations(reader, t),
                    cal.b[layer],
                    cal.u[layer],
                )
                squared.append((actual - target).square().sum().item())
                norms.append(target.square().sum().item())
            parity.append(
                {
                    "layer": layer,
                    "states": len(indices),
                    "partial_max_abs": max(diffs),
                    "partial_relative_l2": math.sqrt(
                        sum(x for x, _ in scales) / sum(y for _, y in scales)
                    ),
                    "output_relative_l2": math.sqrt(sum(squared) / sum(norms)),
                }
            )
            del gu, down, gate, shard
    reference = {
        d["id"]: d for d in json.loads(args.reference.read_text())["documents"]
    }
    stock_reference = {
        d["id"]: d for d in json.loads(args.stock_reference.read_text())["documents"]
    }
    sparse_diff, stock_diff = [], []
    for doc in list(reference.values())[:16]:
        tokens = doc["prompt"] + doc["target"]
        prefix = len(doc["prompt"])
        with torch.no_grad():
            stock_logits = cal.logits(tokens, len(tokens))
            sparse_logits = cal.logits(tokens, prefix)
        target = torch.tensor(doc["target"][1:], device=cal.devices[0])
        sparse_nll = (
            -F.log_softmax(sparse_logits[prefix:-1], -1)
            .gather(1, target[:, None])
            .mean()
            .item()
        )
        stock_nll = (
            -F.log_softmax(stock_logits[prefix:-1], -1)
            .gather(1, target[:, None])
            .mean()
            .item()
        )
        sparse_diff.append(abs(sparse_nll - doc["decode_nll"]))
        stock_diff.append(abs(stock_nll - stock_reference[doc["id"]]["decode_nll"]))
    report = {
        "screen_coordinates": plan.screen_coordinates,
        "selection_score": cal.selection_score,
        "decision_feature": parity,
        "sparse_nll_mae": sum(sparse_diff) / len(sparse_diff),
        "sparse_nll_max": max(sparse_diff),
        "stock_nll_mae": sum(stock_diff) / len(stock_diff),
        "stock_nll_max": max(stock_diff),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def load_calibration_model(
    model_path: Path, devices: list[torch.device]
) -> (
    OlmoeForCausalLM
    | Qwen3MoeForCausalLM
    | Qwen3_5MoeForConditionalGeneration
    | Gemma4ForCausalLM
):
    """Load the frozen BF16 calibration teacher, including Gemma's text-only mapping."""
    config = model_dimensions(model_path)
    model_class = {
        "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
        "OlmoeForCausalLM": OlmoeForCausalLM,
        "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeForConditionalGeneration,
        "Gemma4ForConditionalGeneration": Gemma4ForCausalLM,
        "Gemma4ForCausalLM": Gemma4ForCausalLM,
    }[config["architecture"]]
    load_kwargs: dict[str, Any] = {}
    if config["architecture"].startswith("Gemma4"):
        load_kwargs["key_mapping"] = {r"^model\.language_model\.": "model."}
        load_kwargs["output_loading_info"] = True
    loaded = cast(
        Any,
        model_class.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            **load_kwargs,
        ),
    )
    if config["architecture"].startswith("Gemma4"):
        loaded, loading = loaded
        unexpected = {
            key
            for key in loading["unexpected_keys"]
            if not key.startswith(("model.vision_tower.", "model.embed_vision."))
        }
        if loading["missing_keys"] or unexpected:
            raise ValueError(f"Gemma text weights were not fully loaded: {loading}")
    model = cast(
        OlmoeForCausalLM
        | Qwen3MoeForCausalLM
        | Qwen3_5MoeForConditionalGeneration
        | Gemma4ForCausalLM,
        loaded.eval(),
    )
    model.requires_grad_(False)
    place_model(model, devices)
    return model


def main() -> None:
    """Initialize, fit and save the bank (or validate one with `--validate`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "panel", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--states",
        type=Path,
        help="Required for PCA initialization and parity validation",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--retention", type=float, default=0.10)
    parser.add_argument("--screen-keep", type=float, default=1.0)
    parser.add_argument(
        "--selection-score", choices=get_args(SelectionScore), default="recovery-aware"
    )
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=353)
    parser.add_argument("--docs-per-update", type=int, default=8)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--refresh-every", type=int, default=4)
    parser.add_argument("--pca-states", type=int, default=256)
    parser.add_argument("--pca-batch", type=int, default=16)
    parser.add_argument("--tune-every", type=int, default=8)
    parser.add_argument(
        "--devices", default="cuda:0", help="Comma-separated CUDA devices"
    )
    parser.add_argument(
        "--cache-sketch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache decoded int8 trits per layer (disable if device memory is tight)",
    )
    parser.add_argument(
        "--sparse-prefix",
        action="store_true",
        help="Student runs the sparse policy on every token (prefix and window); "
        "the teacher and KL window are unchanged",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--reference-bank", type=Path)
    parser.add_argument(
        "--init",
        choices=("pca", "reference"),
        default="pca",
        help="Start from the PCA of skipped-channel output or from --reference-bank",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--stock-reference", type=Path)
    args = parser.parse_args()
    devices = [torch.device(value.strip()) for value in args.devices.split(",")]
    if any(device.type != "cuda" or device.index is None for device in devices):
        parser.error("--devices requires explicit CUDA indices")
    config = model_dimensions(args.model)
    if min(args.window, args.docs_per_update, args.refresh_every, args.tune_every) < 1:
        parser.error("Invalid length or cadence")
    if args.init == "reference" and args.reference_bank is None:
        parser.error("--init reference requires --reference-bank")
    if (args.init == "pca" or args.validate) and args.states is None:
        parser.error("--states is required for PCA and validation")
    plan = compile_budget(
        hidden_size=config["hidden_size"],
        expert_width=config["expert_width"],
        top_k=config["num_experts_per_tok"],
        retention=args.retention,
        screen_keep=args.screen_keep,
    )
    panel = json.loads(args.panel.read_text())
    documents = panel["documents"]
    fit = [d for d in documents if d["split_role"].lower() == "fit"]
    tune = [d for d in documents if d["split_role"].lower() == "tune"]
    identity = checkpoint_identity(args.model)
    collection: dict[str, Any] = (
        json.loads((args.states / "collection.json").read_text())
        if args.states is not None
        else {"model": identity, "documents": documents}
    )
    if args.states is not None:
        if not collection["complete"] or collection["model"] != identity:
            raise ValueError("Incomplete state collection or model mismatch")
        if [d["id"] for d in collection["documents"]] != [d["id"] for d in documents]:
            raise ValueError("State/panel document mismatch")
    if not fit or not tune or len(fit) + len(tune) != len(documents):
        raise ValueError("Expected disjoint FIT/TUNE documents")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "complete": False,
        "command": sys.argv,
        "source_revision": args.source_revision,
        "host": socket.gethostname(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tune_kl_history": [],
        "update_seconds": [],
        "reader_refresh_seconds": [],
        "fit_document_ids": [d["id"] for d in fit],
        "tune_document_ids": [d["id"] for d in tune],
    }
    if not args.validate:
        (args.output / "fit.json").write_text(json.dumps(report, indent=2))
    model = load_calibration_model(args.model, devices)
    # Calibration.kl scores at least 64 tokens; size the sparse-prefix reader for
    # that window.
    window = max(args.window, 64)
    cal = Calibration(
        model,
        plan,
        window,
        args.model,
        devices=devices,
        cache_sketch=args.cache_sketch,
        # A sparse-prefix student's whole-sequence backward graph needs that memory.
        teacher_on_gpu=(
            not args.sparse_prefix
            and len(documents) * args.window * config["vocab_size"] * 4 < 3 * 1024**3
        ),
        sparse_prefix=args.sparse_prefix,
        capacity=max(
            min(len(d["tokens"]), d["context_length"] + window) for d in documents
        )
        if args.sparse_prefix
        else None,
        selection_score=cast(SelectionScore, args.selection_score),
    )
    if args.validate:
        with torch.no_grad():
            validate(cal, args, plan)
        return
    cal.refresh()
    report["sketch_cache_bytes"] = cal.sketch_cache_bytes
    report["sketch_cached_layers"] = sorted(cal.unpacked)

    def monitor(label: str) -> float:
        started = time.perf_counter()
        with torch.no_grad():
            value = sum(cal.kl(doc).item() for doc in tune) / len(tune)
        if not math.isfinite(value):
            raise ValueError("Non-finite TUNE KL")
        report["tune_kl_history"].append(
            {"stage": label, "kl": value, "seconds": time.perf_counter() - started}
        )
        (args.output / "fit.json").write_text(json.dumps(report, indent=2))
        print(
            f"TUNE {label}: kl={value:.6g} "
            f"seconds={report['tune_kl_history'][-1]['seconds']:.1f}",
            flush=True,
        )
        return value

    best = monitor("zero")
    best_bank = cal.bank_cpu()
    if args.reference_bank is not None:
        reference = torch.load(
            args.reference_bank, map_location="cpu", weights_only=True
        )
        if (
            reference["decoder"].shape != best_bank[0].shape
            or reference["encoder"].shape != best_bank[1].shape
        ):
            raise ValueError("Reference bank dimensions do not match model")
        try:
            require_sketch_acquisition(reference)
            compatible_reference = True
        except ValueError:
            if args.init == "reference":
                raise
            # A bank for another sketch acquisition is only a diagnostic control,
            # never an eligible starting point or result.
            compatible_reference = False
        if args.init == "reference" and any(
            not bool(torch.count_nonzero(reference[name]))
            for name in ("decoder", "encoder")
        ):
            raise ValueError("Reference initialization requires nonzero bank factors")
        with torch.no_grad():
            cal.load_bank(reference["decoder"], reference["encoder"])
        cal.refresh()
        value = monitor("reference")
        if compatible_reference and value < best:
            best = value
            best_bank = cal.bank_cpu()
        if args.init != "reference":
            for tensor in cal.bank_parameters():
                tensor.zero_()
    if args.init == "pca":
        assert args.states is not None
        with torch.no_grad():
            report["pca_layer_seconds"] = pca_init(
                cal,
                args.states,
                [
                    j
                    for j, d in enumerate(collection["documents"])
                    if d["split_role"].lower() == "fit"
                ],
                args.pca_states,
                args.pca_batch,
            )
        cal.refresh()
        report["reader_refresh_seconds"].append(cal.refresh_seconds)
        value = monitor("pca")
        if value < best:
            best = value
            best_bank = cal.bank_cpu()
    for tensor in cal.bank_parameters():
        tensor.requires_grad_()
    initial_bank = cal.bank_cpu()
    optimizer = torch.optim.Adam(cal.bank_parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    last_refresh = 0
    for step in range(1, args.updates + 1):
        started = time.perf_counter()
        if step - 1 - last_refresh >= args.refresh_every:
            before = cal.refresh_seconds
            cal.refresh()
            report["reader_refresh_seconds"].append(cal.refresh_seconds - before)
            last_refresh = step - 1
        indices = torch.randperm(len(fit), generator=generator)[: args.docs_per_update]
        optimizer.zero_grad(set_to_none=True)
        fit_error = torch.zeros((), device=devices[0])
        for j in indices.tolist():
            error = cal.kl(fit[j]) / len(indices)
            error.backward()
            fit_error += error.detach()
        finite_gradients = {
            device: torch.ones((), device=device, dtype=torch.bool)
            for device in devices
        }
        for tensor in cal.bank_parameters():
            if tensor.grad is not None:
                finite_gradients[tensor.device] &= torch.isfinite(tensor.grad).all()
        if not bool(torch.isfinite(fit_error)):
            raise ValueError("Non-finite FIT KL")
        if not all(bool(finite) for finite in finite_gradients.values()):
            raise ValueError("Non-finite recovery bank gradient")
        optimizer.step()
        for device in devices:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        report["update_seconds"].append(elapsed)
        if step % args.tune_every == 0 or step == args.updates:
            before = cal.refresh_seconds
            cal.refresh()
            report["reader_refresh_seconds"].append(cal.refresh_seconds - before)
            last_refresh = step
            value = monitor(str(step))
            if value < best:
                best = value
                best_bank = cal.bank_cpu()
        report["peak_memory_bytes"] = {
            str(device): torch.cuda.max_memory_allocated(device) for device in devices
        }
        report["last_update"] = step
        (args.output / "fit.json").write_text(json.dumps(report, indent=2))
        print(
            f"update {step}: fit_kl={fit_error.item():.6g} seconds={elapsed:.2f} "
            f"best_tune={best:.6g}",
            flush=True,
        )
    # Banks are (decoder, encoder) pairs, so the strict zips cannot fail.
    report["bank_relative_change_from_init"] = {
        name: ((current - initial).norm() / initial.norm()).item()
        for name, initial, current in zip(
            ("decoder", "encoder"), initial_bank, cal.bank_cpu(), strict=True
        )
    }
    report["saved_bank_relative_change_from_init"] = {
        name: ((saved - initial).norm() / initial.norm()).item()
        for name, initial, saved in zip(
            ("decoder", "encoder"), initial_bank, best_bank, strict=True
        )
    }
    recipe = {
        "objective": (
            "whole-model forward KL(stock || sparse), equal document weighting"
        ),
        "updates": args.updates,
        "lr": args.lr,
        "seed": args.seed,
        "docs_per_update": args.docs_per_update,
        "window": args.window,
        "prefix": (
            "sparse policy on every student token; stock teacher"
            if args.sparse_prefix
            else "stock native MLPs through context_length"
        ),
        "reference_bank": str(args.reference_bank) if args.reference_bank else None,
        "refresh_every": args.refresh_every,
        "tune_every": args.tune_every,
        "init": (
            f"zero-bank {args.selection_score} omitted-output PCA, "
            f"{args.pca_states} FIT states/layer"
            if args.init == "pca"
            else f"warm start from {args.reference_bank}"
        ),
    }
    bank = {
        "format": "native-plan-recovery-v1",
        "model": identity,
        "rank": RANK,
        "layers": list(range(cal.layers)),
        "dimensions": {
            "layers": cal.layers,
            "experts": config["num_experts"],
            "top_k": plan.top_k,
            "hidden": plan.hidden_size,
            "expert_width": plan.expert_width,
        },
        "plan": {
            "acquisition": ACQUISITION,
            "retention": args.retention,
            "screen_keep": args.screen_keep,
            "selection_score": args.selection_score,
            "final_count": plan.final_count,
        },
        "recipe": recipe,
        "fit_document_ids": report["fit_document_ids"],
        "tune_document_ids": report["tune_document_ids"],
        "source_revision": args.source_revision,
        "decoder": best_bank[0].bfloat16().cpu(),
        "encoder": best_bank[1].bfloat16().cpu(),
    }
    torch.save(bank, args.output / "decoder.pt")
    report["complete"] = True
    report["best_tune_kl"] = best
    report["recipe"] = recipe
    (args.output / "fit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
