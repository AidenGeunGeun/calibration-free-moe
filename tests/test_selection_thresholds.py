import torch

from scripts.calibrate_selection_thresholds import fit_selection_threshold


def capped_mean(scores: torch.Tensor, threshold: float, cap: int) -> float:
    return float((scores >= threshold).sum(-1).clamp_max(cap).float().mean())


def test_threshold_meets_the_mean_count_nearest_attainably() -> None:
    torch.manual_seed(5)
    # Token-dependent scales: a common threshold keeps different counts per token.
    scores = torch.rand(400, 96) * torch.rand(400, 1) * 3
    cap = 95
    for target in (4.0, 12.0, 30.0):
        threshold = fit_selection_threshold(scores, target, cap)
        mean = capped_mean(scores, threshold, cap)
        counts = (scores >= threshold).sum(-1)
        assert counts.min() < counts.max()
        # No attainable threshold lands nearer the target mean count.
        values = scores.flatten().unique()
        nearest = min(abs(capped_mean(scores, float(v), cap) - target) for v in values)
        assert abs(mean - target) == nearest
        assert abs(mean - target) < 1 / len(scores) * 2


def test_ties_at_the_threshold_are_kept_together_and_the_cap_binds() -> None:
    scores = torch.tensor([[3.0, 2.0, 2.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    # Threshold 2.0 keeps 3 and min(4, cap 3) = 3: mean 3, exactly the target.
    threshold = fit_selection_threshold(scores, 3.0, cap=3)
    assert threshold == 2.0
    assert capped_mean(scores, threshold, 3) == 3.0
    # Mean 1.0 is unattainable between 0.5 (threshold 3.0) and 3.0 (threshold 2.0).
    assert fit_selection_threshold(scores, 1.0, cap=3) == 3.0
