"""Build the data-free rank-32 recovery map (`bank`) from the down projections.

Per layer, U holds the top-r left singular vectors of D = [D_0 ... D_{E-1}] (every
expert's down columns) and B_e = shrinkage * U^T D_e; no calibration data enter.
Writes `weight-subspace-s1.pt` with the BF16 decoder U [layers, H, r] and encoder
B [layers, E, r, I].
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from calibration_free_moe.model_config import (
    checkpoint_identity,
    model_dimensions,
)
from calibration_free_moe.reader.sketch import ACQUISITION

RANK = 32
SHRINKAGE = 1.0


def load_down(model: Path, layer: int, config: dict[str, Any]) -> torch.Tensor:
    """Return one layer's native routed-expert down projections [E, H, I].

    Reads only down weights, opening each shard once (per-expert checkpoints have
    hundreds of keys per layer). Shared experts and dense MLPs are not routed
    experts and are skipped.
    """
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    architecture = config["architecture"]
    if architecture.startswith("Gemma4"):
        keys = [f"model.language_model.layers.{layer}.experts.down_proj"]
    else:
        prefix = (
            f"model.language_model.layers.{layer}.mlp"
            if architecture == "Qwen3_5MoeForConditionalGeneration"
            else f"model.layers.{layer}.mlp"
        )
        fused = f"{prefix}.experts.down_proj"
        keys = (
            [fused]
            if fused in index
            else [
                f"{prefix}.experts.{e}.down_proj.weight"
                for e in range(config["num_experts"])
            ]
        )
    with ExitStack() as stack:
        handles: dict[str, Any] = {}

        def read(key: str) -> torch.Tensor:
            file = index[key]
            if file not in handles:
                handles[file] = stack.enter_context(
                    safe_open(model / file, framework="pt")
                )
            return handles[file].get_tensor(key)

        weights = []
        for key in keys:
            weights.append(read(key))
    down = weights[0] if len(keys) == 1 else torch.stack(weights)
    expected = (config["num_experts"], config["hidden_size"], config["expert_width"])
    if tuple(down.shape) != expected:
        raise ValueError(f"Down projections {tuple(down.shape)} != {expected}")
    return down


def subspace_bank(
    down: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (U [H,r], U^T D_e [E,r,I], captured energy fraction) for down [E,H,I].

    The reader adds U B_e p, the rank-r projection of the skipped channels'
    sketch-estimated output sum_j w_j d_j p_j.
    """
    columns = down.double().permute(1, 0, 2).reshape(down.shape[1], -1)
    # Left singular vectors of D are the eigenvectors of D D^T (H x H, FP64).
    eigenvalues, eigenvectors = torch.linalg.eigh(columns @ columns.T)
    decoder = eigenvectors[:, -rank:].flip(-1)
    captured = float(eigenvalues[-rank:].sum() / eigenvalues.sum())
    encoder = torch.einsum("hr,ehi->eri", decoder, down.double())
    return decoder, encoder, captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    config = model_dimensions(args.model)
    layers = config["num_hidden_layers"]
    started = time.perf_counter()
    decoders, encoders, captured = [], [], []
    for layer in range(layers):
        down = load_down(args.model, layer, config).to(args.device)
        u, b, energy = subspace_bank(down, RANK)
        decoders.append(u.cpu())
        encoders.append(b.cpu())
        captured.append(energy)
        print(
            f"layer {layer}: rank-{RANK} captured down energy {energy:.4f}", flush=True
        )
    args.output.mkdir(parents=True, exist_ok=True)
    for shrinkage in (SHRINKAGE,):
        path = args.output / f"weight-subspace-s{shrinkage:g}.pt"
        torch.save(
            {
                "format": "weight-subspace-recovery-v1",
                "model": checkpoint_identity(args.model),
                "rank": RANK,
                "layers": list(range(layers)),
                "dimensions": {
                    "layers": layers,
                    "experts": config["num_experts"],
                    "top_k": config["num_experts_per_tok"],
                    "hidden": config["hidden_size"],
                    "expert_width": config["expert_width"],
                },
                # Independent of the read budget; only the sketch acquisition that
                # defines p is fixed.
                "plan": {"acquisition": ACQUISITION},
                "recipe": {
                    "construction": "per-layer top-r left singular vectors of all "
                    "experts' native down columns; B_e = shrinkage * U^T D_e",
                    "shrinkage": shrinkage,
                    "data": None,
                },
                "captured_down_energy": captured,
                "decoder": torch.stack(decoders).bfloat16(),
                "encoder": torch.stack([shrinkage * b for b in encoders]).bfloat16(),
                "build_seconds": time.perf_counter() - started,
                "source_revision": args.source_revision,
            },
            path,
        )
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
