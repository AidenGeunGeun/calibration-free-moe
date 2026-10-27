"""Gated expert activations shared by calibration, the reader and the baselines."""

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

GatedActivation = Literal["silu", "gelu_tanh"]


def model_activation(config: object) -> GatedActivation:
    """Map a Hugging Face config's ``hidden_activation`` to a gated activation.

    Configs without the field (OLMoE, Qwen) use SiLU.
    """
    name = getattr(config, "hidden_activation", "silu")
    if name in ("gelu_pytorch_tanh", "gelu_tanh"):
        return "gelu_tanh"
    if name in ("silu", "swiglu"):
        return "silu"
    raise ValueError(f"Unsupported routed expert activation: {name}")


def activate(gate: Tensor, activation: GatedActivation) -> Tensor:
    """Apply the gate nonlinearity phi (SiLU or tanh-approximated GELU)."""
    if activation == "gelu_tanh":
        return F.gelu(gate, approximate="tanh")
    return F.silu(gate)
