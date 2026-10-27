import pytest
import torch

from scripts.calibrate_baselines import greedy_mlp_thresholds


@pytest.mark.parametrize("sparsity", [0.0, 0.2, 0.98])
def test_wina_greedy_allocation_exports_projection_levels_at_target(
    sparsity: float,
) -> None:
    x = torch.tensor([[0.1, 0.2, 0.3, 1.0]], dtype=torch.bfloat16)
    gate = torch.eye(4, dtype=torch.bfloat16)[None]
    # Only the largest input coordinate contributes to the up projection.
    up = torch.zeros_like(gate)
    up[:, :, -1] = 1
    allocation = greedy_mlp_thresholds(
        x,
        torch.tensor([[0]]),
        gate,
        up,
        gate,
        torch.ones(1, 1),
        sparsity,
        weighted=True,
        return_levels=True,
    )
    levels = [
        allocation[key] for key in ("gate_sparsity", "up_sparsity", "down_sparsity")
    ]
    assert all(0 <= level <= 1 for level in levels)
    assert sum(levels) / 3 >= sparsity - 1e-9
    assert sum(levels) / 3 < sparsity + 0.01 / 3 + 1e-9
    if sparsity == 0.2:
        # Joint output-error search spends the budget on redundant up inputs,
        # rather than applying the target sparsity uniformly to all projections.
        assert levels == pytest.approx([0.0, 0.6, 0.0])


def test_wina_calibration_keeps_up_scores_unweighted() -> None:
    x = torch.tensor([[0.1, 0.2, 0.3, 1.0]], dtype=torch.bfloat16)
    gate = torch.eye(4, dtype=torch.bfloat16)[None]
    up = torch.zeros_like(gate)
    # Norm weighting would rescue this small but essential up input and make
    # redundant up coordinates free to prune; official WINA ranks raw |x| instead.
    up[:, :, 0] = 10
    allocation = greedy_mlp_thresholds(
        x,
        torch.tensor([[0]]),
        gate,
        up,
        gate,
        torch.ones(1, 1),
        0.1,
        weighted=True,
        return_levels=True,
    )
    assert allocation == pytest.approx(
        {"gate_sparsity": 0.3, "up_sparsity": 0.0, "down_sparsity": 0.0}
    )
