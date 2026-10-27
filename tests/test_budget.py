import pytest

from calibration_free_moe.reader.budget import PaidRound, compile_budget


@pytest.mark.parametrize("screen_keep,coordinates", [(1.0, 128), (0.625, 80)])
def test_sketch_plan_and_paid_reads(screen_keep: float, coordinates: int):
    plan = compile_budget(
        hidden_size=2048,
        expert_width=1024,
        top_k=8,
        retention=0.15,
        screen_keep=screen_keep,
    )
    n, h, f = 8192, 2048, plan.final_count
    assert plan.rounds == (PaidRound(0, 0, n, f), PaidRound(0, h, f, f))
    assert plan.sketch_read_bytes() == 8 * 16 * (coordinates * 416 + 2 * 4096)
    assert plan.sketch_read_bytes(sectors=False) == 8 * 16 * (
        coordinates * 410 + 2 * 4096
    )
    # The point score reads no residual variances.
    assert plan.sketch_read_bytes(residual=False) == 8 * 16 * (coordinates * 416 + 4096)
    assert plan.selected_weight_bytes() == 2 * h * 3 * f
    assert plan.selection_read_bytes() == 12 * n
    assert plan.native_read_bytes() == 2 * 3 * n * h
    assert plan.recovery_read_bytes(32) == 64 * (n - f + h)


def test_dense_bypass():
    """Full retention compiles to one dense stage with no sketch or selection reads."""
    dense = compile_budget(hidden_size=128, expert_width=32, top_k=4, retention=1)
    assert dense.rounds == (PaidRound(0, 128, 128, 128),)
    assert dense.sketch_read_bytes() == 0
    assert dense.selection_read_bytes() == 0


@pytest.mark.parametrize(
    ("retention", "f"), [(2.5 / 32, 2), (3.5 / 32, 4), (3 / 32, 3)]
)
def test_nearest_even_rounding(retention: float, f: int):
    plan = compile_budget(
        hidden_size=128,
        expert_width=16,
        top_k=2,
        retention=retention,
    )
    assert (plan.final_count, plan.completion_count) == (f, f)


@pytest.mark.parametrize("screen_keep", [0.625, 1.0])
def test_unaligned_sketch_metadata_width_is_rejected(screen_keep: float):
    # At I=88, adjacent 80-channel CTAs request overlapping scale/residual sectors
    # that the read cost would count too low, in both sparse and full screens.
    with pytest.raises(ValueError, match=r"divisible by 16.*sketch metadata"):
        compile_budget(
            hidden_size=128,
            expert_width=88,
            top_k=1,
            retention=0.125,
            screen_keep=screen_keep,
        )


@pytest.mark.parametrize(
    ("hidden", "width", "retention"),
    [
        (0, 16, 0.1),
        (64, 16, 0.1),
        (128, 0, 0.1),
        (128, 16, 0),
        (128, 16, float("nan")),
        (128, 16, 0.01),
    ],
)
def test_invalid_budgets(hidden: int, width: int, retention: float):
    with pytest.raises(ValueError):
        compile_budget(
            hidden_size=hidden,
            expert_width=width,
            top_k=2,
            retention=retention,
        )
