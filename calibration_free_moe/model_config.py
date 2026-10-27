"""Read and validate the frozen MoE architectures this package supports.

Covers OLMoE, Qwen3-MoE, Qwen3.5-MoE and Gemma 4 checkpoints, the policy-option
checks that need no GPU, and the vLLM model-class overrides.
"""

import json
from pathlib import Path
from typing import Any

# Baselines without calibration data or thresholds; ``retention`` sets the fraction
# (``prox-free`` also takes the Stage-1 fraction ``input_keep``).
CALIBRATION_FREE_POLICIES = (
    "hire-gate",
    "cats-topk",
    "teal-topk",
    "wina-topk",
    "prox-free",
)


def validate_prox_keep_options(
    policy: object,
    input_keep: object,
    channel_keep: object,
    channel_floor: object,
    recipe: object = "legacy",
    proxy_score: object = "prox",
) -> None:
    """Validate the Prox keep fractions without importing the GPU implementation."""
    if recipe not in {"legacy", "study"} or (
        recipe != "legacy" and policy not in {"prox", "prox-free"}
    ):
        raise ValueError("The study recipe belongs to prox or prox-free")
    for name, value in (
        ("input_keep", input_keep),
        ("channel_keep", channel_keep),
        ("channel_floor", channel_floor),
    ):
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < value <= 1
        ):
            raise ValueError(f"{name} must be a fraction in (0,1]")
    if (policy == "prox-free" and input_keep is None) or (
        input_keep is not None and policy not in {"prox", "prox-free"}
    ):
        raise ValueError("input_keep is optional for prox and required for prox-free")
    if (channel_keep is not None or channel_floor is not None) and policy != "prox":
        raise ValueError("channel_keep and channel_floor are prox options")
    if channel_keep is not None and channel_floor is not None:
        raise ValueError("A channel floor requires a threshold, not channel_keep")
    if input_keep is not None and channel_keep is not None:
        raise ValueError(
            "For two top-k rules use prox-free with input_keep and retention"
        )
    if proxy_score == "exact":
        # The allocation study's analysis score; allocation options validate the rest.
        if policy != "prox" or any(
            value is not None for value in (input_keep, channel_keep, channel_floor)
        ):
            raise ValueError("The exact score is a prox allocation score without keeps")
        return
    if proxy_score not in {"prox", "output-unit"} or (
        proxy_score != "prox"
        and policy != "prox-free"
        and not (
            policy == "prox"
            and recipe == "study"
            and input_keep is not None
            and channel_keep is None
        )
    ):
        raise ValueError(
            "output-unit requires prox-free or prox with study input "
            "top-k/channel threshold"
        )


def model_dimensions(model: Path) -> dict[str, Any]:
    """Read a checkpoint's ``config.json`` as one flat text config.

    Multimodal configs are merged with their ``text_config``. Adds ``expert_width``
    (the routed expert's intermediate size I) and ``architecture``. Rejects shared
    experts (except Qwen3.5's, which stay dense) and interleaved dense MoE layers.
    """
    config = json.loads((model / "config.json").read_text())
    architecture = config["architectures"][0]
    if architecture not in {
        "OlmoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
    }:
        raise ValueError(f"Unsupported routed architecture: {architecture}")
    if architecture == "Qwen3_5MoeForConditionalGeneration":
        config = {**config, **config["text_config"]}
        config["norm_topk_prob"] = config.get("norm_topk_prob", True)
    elif architecture.startswith("Gemma4"):
        config = {**config, **config.get("text_config", {})}
        config["num_experts_per_tok"] = config["top_k_experts"]
    elif config.get("shared_expert_intermediate_size", 0):
        raise ValueError("Shared expert layers are not supported")
    if config.get("decoder_sparse_step", 1) != 1 or config.get("mlp_only_layers"):
        raise ValueError("Interleaved dense MoE layers are not supported")
    config["expert_width"] = config.get(
        "moe_intermediate_size", config.get("intermediate_size")
    )
    config["architecture"] = architecture
    return config


def checkpoint_identity(model: Path) -> dict[str, str]:
    """Read a local Hub snapshot's repository and immutable revision."""
    config = model_dimensions(model)
    directory = model.parent.parent.name
    if model.parent.name == "snapshots" and directory.startswith("models--"):
        namespace, name = directory.removeprefix("models--").split("--", 1)
        repository = f"{namespace}/{name}"
    elif config["architecture"] == "OlmoeForCausalLM":
        # Persisted OLMoE REAP checkpoints retain their source model identity.
        repository = "allenai/OLMoE-1B-7B-0924"
    else:
        repository = config.get("_name_or_path") or str(model.resolve())
    return {"id": repository, "revision": model.name}


def model_override(model: str) -> dict[str, str]:
    """Return vLLM ``model_class_overrides`` that select this package's classes."""
    architecture = model_dimensions(Path(model))["architecture"]
    if architecture.startswith("Gemma4"):
        overrides = {
            "Gemma4ForCausalLM": (
                "calibration_free_moe.runtime.gemma4:ProgressiveGemma4ForCausalLM"
            ),
        }
        if architecture == "Gemma4ForConditionalGeneration":
            overrides["Gemma4ForConditionalGeneration"] = (
                "calibration_free_moe.runtime.gemma4:"
                "ProgressiveGemma4ForConditionalGeneration"
            )
        return overrides
    name = {
        "OlmoeForCausalLM": "ProgressiveOlmoeForCausalLM",
        "Qwen3MoeForCausalLM": "ProgressiveQwen3MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration": (
            "ProgressiveQwen35ForConditionalGeneration"
        ),
    }[architecture]
    module = (
        "runtime.qwen35"
        if architecture == "Qwen3_5MoeForConditionalGeneration"
        else "runtime.experts"
    )
    return {architecture: f"calibration_free_moe.{module}:{name}"}
