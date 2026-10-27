from typing import cast

import pytest
import torch

from calibration_free_moe.activation import activate
from calibration_free_moe.baselines.activation_sparsity import (
    grouped_assignment,
    grouped_config,
)
from calibration_free_moe.baselines.hire import (
    HireBaseline,
    dequantize_int4_groups,
)
from calibration_free_moe.reader.progressive import channel_major_down

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _weights(experts: int, i: int, h: int):
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 20).bfloat16()
    down = (torch.randn(experts, h, i, device="cuda") / 20).bfloat16()
    return gu, down


def _reference_scores(op: HireBaseline, x, ids, weights, active):
    """Torch scores from the dequantized copy, independent of the Triton decode."""
    i, safe = op.width, ids.clamp_min(0)
    # One token's routed experts at a time bounds the FP32 copy's memory.
    estimate = torch.stack(
        [
            dequantize_int4_groups(op.packed[e], op.scales[e]) @ row.float()
            for row, e in zip(x, safe, strict=True)
        ]
    )
    gate = activate(estimate[..., :i], op.activation)
    proxy = gate
    norm = torch.stack([op.down[e].float().norm(dim=1) for e in safe])
    score = (weights[..., None] * norm * proxy).abs()
    return torch.where(active[..., None], score, -1.0).flatten(1)


@pytest.mark.parametrize("count", [1, 16])
def test_top_f_mask_matches_reference_at_model_dimensions(count):
    torch.manual_seed(5 + count)
    h, i, experts, k = 2048, 1024, 64, 8
    gu, down = _weights(experts, i, h)
    op = HireBaseline(gu, channel_major_down(down), 0.1)
    # Group-wise INT4 is within half a quantization step of the native weights.
    copy = dequantize_int4_groups(op.packed[:4], op.scales[:4])
    step = op.scales[:4].float().repeat_interleave(128, dim=-1)
    error = copy - gu[:4, : copy.shape[1]].float()
    assert (error.abs() <= step / 2 + 1e-6).all()
    x = torch.randn(count, h, device="cuda").bfloat16()
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:k] for _ in range(count)]
    )
    weights = torch.rand(count, k, device="cuda").softmax(-1)
    active = torch.ones(count, k, device="cuda", dtype=torch.bool)
    active[0, 3] = False
    assignment = grouped_assignment(
        ids.reshape(-1), experts, None, grouped_config(gu, down, k, count)
    )
    mask = op.select(x, ids, weights, active, assignment)
    score = _reference_scores(op, x, ids, weights, active)
    _assert_top(mask.view(count, -1), score, round(0.1 * k * i))
    assert not mask.view(count, k, i)[0, 3].any()


def _assert_top(mask, score, keep, tolerance=1e-4):
    """Exactly ``keep`` per token; no unselected score beats a selected one.

    The tolerance covers FP32 reassociation between Triton and Torch scores.
    """
    for token in range(mask.shape[0]):
        chosen = mask[token]
        assert int(chosen.sum()) == keep
        if 0 < keep < chosen.numel():
            low = score[token][chosen].min()
            assert score[token][~chosen].max() <= low + tolerance * low.abs()


def test_boundary_ties_keep_exactly_f_and_zero_f_reads_nothing():
    torch.manual_seed(8)
    h, i, experts, k = 256, 64, 2, 2
    gu, down = _weights(experts, i, h)
    # Expert 1 duplicates expert 0 with the same router weight, so every score
    # appears twice and the odd F = 33 must split a tied pair.
    gu[1], down[1] = gu[0], down[0]
    x = torch.randn(2, h, device="cuda").bfloat16()
    ids = torch.tensor([[0, 1], [1, 0]], device="cuda")
    weights = torch.full((2, k), 0.5, device="cuda")
    active = torch.ones(2, k, device="cuda", dtype=torch.bool)
    op = HireBaseline(gu, channel_major_down(down), 33 / (k * i))
    assignment = grouped_assignment(
        ids.reshape(-1), experts, None, grouped_config(gu, down, k, 2)
    )
    mask = op.select(x, ids, weights, active, assignment)
    score = _reference_scores(op, x, ids, weights, active)
    _assert_top(mask.view(2, -1), score, 33)
    # round(0.003 * 128) = 0 channels: only the screen is read.
    empty = HireBaseline(gu, channel_major_down(down), 0.003)
    assert not empty(x, ids, weights).any()
    report = empty.report()
    assert report["gate_coefficients"] == 0
    assert report["charged_bytes"] == 4 * op.screen_bytes


def test_worker_report_keeps_logical_and_sector_totals():
    """Sector-rounded scale and norm blocks exceed their logical bytes at I=1."""
    from calibration_free_moe.runtime import experts as vllm_model

    torch.manual_seed(4)
    gu, down = _weights(2, 1, 128)
    op = HireBaseline(gu, channel_major_down(down), 1.0)
    op(
        torch.randn(1, 128, device="cuda").bfloat16(),
        torch.tensor([[0, 1]], device="cuda"),
        torch.full((1, 2), 0.5, device="cuda"),
    )
    local = op.report()
    vllm_model.BASELINE_OPERATORS[:] = [op]
    try:
        merged = vllm_model.baseline_work_report(None)
    finally:
        vllm_model.BASELINE_OPERATORS.clear()
    assert cast(float, local["logical_charged_bytes"]) < cast(
        float, local["charged_bytes"]
    )
    for key in ("charged_bytes", "logical_charged_bytes", "accounting_basis"):
        assert merged[key] == local[key]


def test_output_read_cost_padding_and_graph_replay():
    torch.manual_seed(3)
    h, i, experts, k = 256, 64, 4, 2
    gu, down = _weights(experts, i, h)
    x = torch.randn(3, h, device="cuda").bfloat16()
    ids = torch.tensor([[0, 2], [1, 3], [3, 1]], device="cuda")
    weights = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.6, 0.4]], device="cuda")
    padding = torch.tensor([False, True, False], device="cuda")
    op = HireBaseline(gu, channel_major_down(down), 0.25)
    result = op(x, ids, weights, padding)
    active = (~padding)[:, None].expand(-1, k)
    score = _reference_scores(op, x, ids, weights, active)
    keep = round(0.25 * k * i)
    order = score.argsort(dim=-1, descending=True, stable=True)[:, :keep]
    chosen = torch.zeros_like(score, dtype=torch.bool).scatter(-1, order, True)
    expected = torch.zeros(3, h, device="cuda")
    for token in (0, 2):
        for slot in range(k):
            expert = int(ids[token, slot])
            mask = chosen[token, slot * i : (slot + 1) * i]
            gate, up = (gu[expert].float() @ x[token].float()).bfloat16().chunk(2)
            feature = (activate(gate, "silu").bfloat16() * up).bfloat16() * mask
            expected[token] += (down[expert].float() @ feature.float()) * weights[
                token, slot
            ]
    torch.testing.assert_close(result.float(), expected, atol=2e-3, rtol=2e-2)
    assert not result[1].any()
    report = op.report()
    # Two real tokens, K=2 slots each; gate INT4 rows, FP16 group scales, norms.
    slots = 4
    screen = i * h // 2 + i * (h // 128) * 2 + 4 * i
    assert report["charged_bytes"] == slots * screen + 2 * keep * 3 * h * 2
    assert report["reference_bytes"] == slots * 3 * i * h * 2
    assert report["read_fraction"] == pytest.approx(
        screen / (6 * i * h) + keep / (k * i)
    )
    graphed = HireBaseline(gu, channel_major_down(down), 0.25)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = graphed(x, ids, weights, padding)
    graphed.work.zero_()
    graph.replay()
    graph.replay()
    torch.testing.assert_close(replayed, result, atol=0, rtol=0)
    torch.testing.assert_close(graphed.work, op.work * 2)
