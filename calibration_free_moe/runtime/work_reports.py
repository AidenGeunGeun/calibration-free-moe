"""Combine per-worker work reports and price the dense MLPs beside the MoE.

Byte counts add across workers; read fractions are recomputed from the sums rather
than averaged over ranks.
"""

from __future__ import annotations

from typing import Any, cast


def shared_expert_work_report(
    dimensions: dict[str, Any],
    prefill_tokens: int,
    decode_tokens: int,
    tensor_parallel_size: int,
) -> dict[str, object]:
    """Price Qwen3.5's dense shared expert in logical BF16 weight bytes.

    The tensor-parallel shared MLP is charged once per token and its replicated
    sigmoid gate once per rank. Routed reads are reported separately.
    """
    hidden = int(dimensions["hidden_size"])
    width = int(dimensions["shared_expert_intermediate_size"])
    layers = int(dimensions["num_hidden_layers"])
    mlp_per_layer = 3 * hidden * width * 2
    mlp_per_token = layers * mlp_per_layer
    gate_per_token = layers * tensor_parallel_size * 2 * hidden
    per_token = mlp_per_token + gate_per_token
    return {
        "scope": "Native dense shared MLP sharded over TP plus replicated BF16 "
        "sigmoid gate (excludes controls); separate from routed reads",
        "tensor_parallel_size": tensor_parallel_size,
        "shared_mlp_logical_weight_bytes_per_token": mlp_per_token,
        "shared_gate_logical_weight_bytes_per_token": gate_per_token,
        "logical_weight_bytes_per_token": per_token,
        "decode_tokens": decode_tokens,
        "decode_logical_weight_bytes": decode_tokens * per_token,
        "prefill_tokens": prefill_tokens,
        "prefill_logical_weight_bytes": prefill_tokens * per_token,
    }


def dense_mlp_work_report(
    dimensions: dict[str, Any], prefill_tokens: int, decode_tokens: int
) -> dict[str, Any]:
    """Price Gemma 4's dense MLP in logical BF16 weight bytes.

    The dense gated MLP runs at every layer beside the routed experts; routed reads
    are reported separately.
    """
    per_token = (
        int(dimensions["num_hidden_layers"])
        * 3
        * int(dimensions["hidden_size"])
        * int(dimensions["intermediate_size"])
        * 2
    )
    return {
        "scope": "native BF16 dense gated MLP; logical bytes separate from "
        "routed reads",
        "logical_weight_bytes_per_token": per_token,
        "decode_tokens": decode_tokens,
        "decode_logical_weight_bytes": decode_tokens * per_token,
        "prefill_tokens": prefill_tokens,
        "prefill_logical_weight_bytes": prefill_tokens * per_token,
    }


def aggregate_work_reports(parts: list[dict[str, object]]) -> dict[str, object]:
    """Sum numeric fields of per-worker reports and recompute the read fractions.

    Layer lists are concatenated; descriptive fields are taken from the first
    worker, since every worker runs the same configuration.
    """
    if not parts:
        raise ValueError("No MoE worker reports")
    result: dict[str, object] = {
        key: sum(float(cast(float, part[key])) for part in parts)
        for key, value in parts[0].items()
        if key not in {"read_fraction", "logical_read_fraction", "layers"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    reference = cast(float, result["reference_bytes"])
    result["read_fraction"] = (
        cast(float, result["charged_bytes"]) / reference if reference else None
    )
    if "logical_charged_bytes" in result:
        result["logical_read_fraction"] = (
            cast(float, result["logical_charged_bytes"]) / reference
            if reference
            else None
        )
    if "layers" in parts[0]:
        result["layers"] = [
            layer for part in parts for layer in cast(list[object], part["layers"])
        ]
    for key in (
        "accounting_basis",
        "priced_proxy_layout",
        "executed_proxy_layout",
        "screen_coordinates_per_block",
        "priced_sketch_layout",
        "executed_sketch_layout",
        "sketch_read_accounting",
        "screen_residual_scope",
        "native_execution",
        "actual_traffic_measured",
        "allocation_source",
    ):
        if key in parts[0]:
            result[key] = parts[0][key]
    return result
