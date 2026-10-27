"""Run the reader on saved MoE inputs and return what its recovery map sees.

Helper module for `fit_kl_bank.py` and `allocate_layers.py`; it has no command line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor

from calibration_free_moe.activation import GatedActivation, activate
from calibration_free_moe.model_config import model_dimensions
from calibration_free_moe.reader.progressive import (
    ProgressiveReader,
    SelectionScore,
    channel_major_down,
)

RANK = 32


def candidate_activations(reader: ProgressiveReader, tokens: int) -> Tensor:
    """Return the activation p of every candidate channel, [tokens, K * I].

    Matches the reader's `_encode` kernel: the exact BF16 activation for channels
    whose native gate and up rows were read, else the BF16 product of the
    activated sketch projections.
    """
    g, u = reader.partial[:tokens].unbind(1)
    g, u = g.bfloat16().float(), u.bfloat16().float()
    partial = (activate(g, reader.activation) * u).bfloat16()
    return torch.where(reader.completed[:tokens], reader.activations[:tokens], partial)


def extracted_model_output(
    reader: ProgressiveReader,
    down: Tensor,
    activation: Tensor,
    features: Tensor,
    encoder: Tensor,
    decoder: Tensor,
) -> Tensor:
    """Return the routed output sum_k w_k (D a + U B p), [tokens, H], in FP32.

    Selected channels contribute their exact activation through D, skipped ones
    their sketch activation p through the recovery map (`bank`, U and B). The
    selection comes from the reader's last call.
    """
    t, k = reader.ids[: activation.shape[0]].shape
    i = down.shape[-1]
    ids = reader.ids[:t].long()
    weights = reader.weights[:t]
    keep = reader.selected[:t].view(t, k, i)
    a = activation.view(t, k, i)
    p = features.view(t, k, i).float()
    selected = torch.zeros((t, down.shape[1]), device=down.device)
    z = torch.zeros((t, encoder.shape[1]), device=down.device)
    for slot in range(k):
        e = ids[:, slot]
        w = weights[:, slot, None]
        # The reader's exact-activation buffer is uninitialized outside the
        # selected channels, so select rather than multiply by a mask.
        kept = torch.where(keep[:, slot], a[:, slot].float(), 0.0)
        selected += w * torch.bmm(down[e].float(), kept.unsqueeze(-1)).squeeze(-1)
        z += w * torch.bmm(
            encoder[e].float(), (p[:, slot] * ~keep[:, slot]).unsqueeze(-1)
        ).squeeze(-1)
    return selected + z @ decoder.float().T


def load_layer_weights(
    model: Path, layer: int, device: str
) -> tuple[Tensor, Tensor, Tensor]:
    """Return one layer's gate/up rows [E, 2I, H], channel-major down and router.

    Reads fused or per-expert checkpoint layouts; the router weight is [E, H].
    """
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(key: str) -> Tensor:
        with safe_open(model / index[key], framework="pt") as handle:
            return handle.get_tensor(key)

    config = model_dimensions(model)
    if config["architecture"].startswith("Gemma4"):
        prefix = f"model.language_model.layers.{layer}"
        return (
            read(f"{prefix}.experts.gate_up_proj").to(device),
            channel_major_down(read(f"{prefix}.experts.down_proj").to(device)),
            read(f"{prefix}.router.proj.weight").to(device),
        )
    prefix = (
        f"model.language_model.layers.{layer}.mlp"
        if config["architecture"] == "Qwen3_5MoeForConditionalGeneration"
        else f"model.layers.{layer}.mlp"
    )
    if f"{prefix}.experts.gate_up_proj" in index:
        return (
            read(f"{prefix}.experts.gate_up_proj").to(device),
            channel_major_down(read(f"{prefix}.experts.down_proj").to(device)),
            read(f"{prefix}.gate.weight").to(device),
        )
    experts = []
    down = []
    for e in range(config["num_experts"]):
        p = f"{prefix}.experts.{e}"
        experts.append(
            torch.cat(
                (
                    read(f"{p}.gate_proj.weight"),
                    read(f"{p}.up_proj.weight"),
                )
            )
        )
        down.append(read(f"{p}.down_proj.weight"))
    gate = read(f"{prefix}.gate.weight").to(device)
    return (
        torch.stack(experts).to(device),
        channel_major_down(torch.stack(down).to(device)),
        gate,
    )


def route(
    x: Tensor, gate: Tensor, k: int, renormalize: bool = False
) -> tuple[Tensor, Tensor]:
    """Return the natural top-k expert IDs and weights with vLLM's fused router."""
    # Native OLMoE: BF16 gate logits, FP32 softmax, natural top-8, no renorm.
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    weights, ids, _ = fused_topk(x, x @ gate.T, k, renormalize=renormalize)
    return ids.contiguous(), weights.contiguous()


@torch.no_grad()
def extract(
    states: Tensor,
    gate_up: Tensor,
    down: Tensor,
    gate: Tensor,
    encoder: Tensor,
    decoder: Tensor,
    plan: Any,
    batch_size: int,
    renormalize: bool = False,
    sketch: tuple[Tensor, Tensor] | None = None,
    activation: GatedActivation = "silu",
    route_ids: Tensor | None = None,
    route_weights: Tensor | None = None,
    selection_score: SelectionScore = "recovery-aware",
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the reader on saved states and return its skipped-channel quantities.

    Returns device tensors (expert ids [T, K], router-weighted sketch activations p
    of the skipped channels [T, K, I] in FP32, exact output of the skipped
    channels [T, H]). Gemma 4 passes its recorded routing; other models route
    with `gate`.
    """
    reader = ProgressiveReader(
        gate_up,
        down,
        encoder,
        decoder,
        plan,
        batch_size,
        sketch=sketch,
        activation=activation,
        selection_score=selection_score,
    )
    all_ids, all_p, all_r = [], [], []
    k, i = plan.top_k, plan.expert_width
    for start in range(0, len(states), batch_size):
        x = states[start : start + batch_size].to(gate_up.device, non_blocking=True)
        t = len(x)
        if route_ids is not None and route_weights is not None:
            ids = route_ids[start : start + t].to(gate_up.device).contiguous()
            weights = route_weights[start : start + t].to(gate_up.device).contiguous()
        else:
            ids, weights = route(x, gate, k, renormalize)
        reader(x.contiguous(), ids, weights)
        features = candidate_activations(reader, t).view(t, k, i).float()
        omitted = ~reader.selected[:t].view(t, k, i)
        exact = torch.empty_like(features)
        for slot in range(k):
            projections = torch.bmm(
                gate_up[reader.ids[:t, slot].long()].float(), x.float().unsqueeze(-1)
            ).squeeze(-1)
            g, u = (
                projections[:, :i].bfloat16().float(),
                projections[:, i:].bfloat16().float(),
            )
            exact[:, slot] = (
                activate(g, activation).bfloat16().float().mul(u).bfloat16().float()
            )
        residual = torch.zeros((t, x.shape[1]), device=x.device)
        for slot in range(k):
            residual += reader.weights[:t, slot, None] * torch.bmm(
                down[reader.ids[:t, slot].long()].float(),
                (exact[:, slot] * omitted[:, slot]).unsqueeze(-1),
            ).squeeze(-1)
        weighted_p = features * omitted * reader.weights[:t, :, None]
        if not bool(
            torch.isfinite(residual).all() and torch.isfinite(weighted_p).all()
        ):
            raise RuntimeError("Non-finite omitted output or candidate activations")
        all_ids.append(reader.ids[:t].clone())
        all_p.append(weighted_p)
        all_r.append(residual)
    return torch.cat(all_ids), torch.cat(all_p), torch.cat(all_r)
