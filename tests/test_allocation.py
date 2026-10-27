import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from calibration_free_moe.analysis.allocation import (
    FEATURES,
    RECORD_FORMAT,
    AllocationSource,
    LayerAllocation,
    Schedule,
    TokenContext,
    integer_counts,
    request_key,
    rounding_seed,
    shuffled_counts,
)
from scripts.allocation_study import histogram_threshold, layer_counts


def context(
    source: AllocationSource,
    rows: int,
    *,
    histogram: bool = False,
    recording: bool = False,
) -> TokenContext:
    return TokenContext(
        source,
        1,
        rows,
        2,
        torch.device("cpu"),
        histogram=histogram,
        recording=recording,
    )


def reference_top(score: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Independent rule: the n largest, ties to the lowest index (Prox study)."""
    index = torch.arange(score.shape[1]).expand_as(score)
    order = torch.from_numpy(np.lexsort((index.numpy(), -score.numpy()), axis=-1))
    ranks = torch.empty_like(order).scatter_(1, order, index.clone())
    return ranks < counts[:, None]


def test_counts_keep_each_tokens_top_n_and_integer_targets_unchanged() -> None:
    torch.manual_seed(3)
    rows, candidates = 64, 48
    # Rounded scores create ties; -1 marks inactive candidates.
    ranked = (torch.rand(rows, candidates) * 8).round() / 8
    ranked[:, :5] = -1.0
    valid = torch.ones(rows, dtype=torch.bool)
    valid[-3:] = False
    uniform = context("uniform", rows)
    uniform.depth.copy_(torch.arange(rows) * 7 - 20)
    mask = LayerAllocation(uniform, 0, candidates, 9, 9.0, None)(
        ranked, valid, slice(0, rows)
    )
    expected = reference_top(ranked, torch.where(valid, 9, 0))
    assert torch.equal(mask, expected)
    assert torch.equal(uniform.counts[:, 0], torch.where(valid, 9, 0).int())
    # A real-valued per-layer mean: counts vary by depth, each mask is its own top-n.
    fractional = context("uniform", rows)
    fractional.depth.copy_(torch.arange(rows))
    mask = LayerAllocation(fractional, 0, candidates, 9, 9.4, 0.5)(
        ranked, valid, slice(0, rows)
    )
    counts = fractional.counts[:, 0].long()
    assert set(counts[valid].tolist()) == {9, 10}
    assert torch.equal(mask, reference_top(ranked, counts))
    # The threshold source keeps exactly the threshold mask and records its count.
    threshold = context("threshold", rows, recording=True)
    mask = LayerAllocation(threshold, 0, candidates, 9, 9.0, 0.5)(
        ranked, valid, slice(0, rows)
    )
    assert torch.equal(mask, ranked >= 0.5)
    assert torch.equal(threshold.counts[:, 0], (ranked >= 0.5).sum(-1).int())
    above = threshold.features[:, 0, FEATURES.index("above_threshold")]
    torch.testing.assert_close(above, (ranked >= 0.5).float().mean(-1))


@pytest.mark.parametrize("target", [3.25, 17.9, 0.4, 5.0])
def test_rounded_counts_are_unbiased_across_documents_at_each_depth(
    target: float,
) -> None:
    """The panel mean is the target at every depth and layer, not only on average."""
    documents = 20000
    seeds = torch.tensor(
        [
            rounding_seed(request_key(Path("p.json"), f"doc{d}"))
            for d in range(documents)
        ]
    )
    for depth in (0, 1, 777):
        for layer in (0, 5):
            counts = integer_counts(
                torch.full((documents,), target),
                seeds,
                torch.full((documents,), depth),
                layer,
                64,
            )
            # Bernoulli(frac) mean: 4.5 standard errors at most.
            frac = target - int(target)
            bound = 4.5 * (frac * (1 - frac) / documents) ** 0.5 + 1e-12
            assert abs(counts.double().mean().item() - target) <= bound
            assert set(counts.tolist()) <= {int(target), int(target) + 1}


def test_shuffle_permutes_whole_vectors_within_each_panel() -> None:
    generator = torch.Generator().manual_seed(0)
    records = {}
    for panel in ("a.json", "b.json"):
        for doc in range(3):
            records[request_key(Path(panel), f"doc{doc}")] = {
                "steps": torch.arange(1, 6),
                "counts": torch.randint(0, 1000, (5, 4), generator=generator),
            }
    shuffled = shuffled_counts(records, seed=11)
    for panel in ("a.json", "b.json"):
        keys = [k for k in records if json.loads(k)[0].endswith(panel)]
        before = sorted(
            map(tuple, torch.cat([records[k]["counts"] for k in keys]).tolist())
        )
        after = sorted(map(tuple, torch.cat([shuffled[k][1] for k in keys]).tolist()))
        assert before == after
        assert any(not torch.equal(records[k]["counts"], shuffled[k][1]) for k in keys)


def test_exact_threshold_from_histograms_keeps_the_target_mean() -> None:
    torch.manual_seed(2)
    rows, candidates = 400, 256
    ranked = torch.rand(rows, candidates).pow(6) * torch.rand(rows, 1) * 10
    recorder = context("uniform", rows, histogram=True)
    LayerAllocation(recorder, 0, candidates, 20, 20.0, None)(
        ranked, torch.ones(rows, dtype=torch.bool), slice(0, rows)
    )
    assert recorder.histograms is not None and recorder.histogram_tokens is not None
    threshold = histogram_threshold(
        recorder.histograms[0], float(recorder.histogram_tokens[0]), 20.0, (-48.0, 16.0)
    )
    realized = (ranked >= threshold).sum(-1).double().mean().item()
    assert realized == pytest.approx(20.0, rel=0.02)


def write_record(
    directory: Path, counts: torch.Tensor, charged: torch.Tensor, channel: list[float]
) -> None:
    directory.mkdir()
    records = {
        request_key(Path("panel.json"), f"doc{t}"): {
            "steps": torch.tensor([1]),
            "depth": torch.tensor([0]),
            "counts": counts[t][None].int(),
            "features": torch.zeros(1, counts.shape[1], len(FEATURES)),
            "charged_bytes": charged[t][None],
        }
        for t in range(len(counts))
    }
    torch.save(
        {
            "format": RECORD_FORMAT,
            "channel_bytes": torch.tensor(channel),
            "records": records,
        },
        directory / "records-00000.pt",
    )


def test_two_pass_layer_counts_match_the_threshold_arms_charged_bytes(
    tmp_path: Path,
) -> None:
    # Equal counts alone miss the reads when retained inputs differ; Prox's read cost
    # is linear in rows, so one byte-residual correction per layer recovers them.
    torch.manual_seed(5)
    tokens, channel = 50, [600.0, 1200.0]
    width = torch.tensor(channel, dtype=torch.float64)

    def charged(counts: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + counts.double() * width

    threshold_counts = torch.randint(10, 40, (tokens, 2))
    write_record(
        tmp_path / "threshold",
        threshold_counts,
        charged(threshold_counts, 5000 + 100 * torch.rand(tokens, 2)),
        channel,
    )
    first = tmp_path / "pass1.json"
    layer_counts(
        Namespace(
            target_records=tmp_path / "threshold",
            realized_records=None,
            counts=None,
            output=first,
        )
    )
    spec = json.loads(first.read_text())
    target = torch.tensor(spec["layer_target_bytes_per_token"], dtype=torch.float64)

    def run(counts_file: Path, name: str) -> None:
        # The arm keeps the mean counts but retains more input coordinates.
        means = torch.tensor(json.loads(counts_file.read_text())["layer_counts"])
        counts = means.double().expand(tokens, 2)
        write_record(
            tmp_path / name,
            counts,
            charged(counts, 7000 + torch.zeros(tokens, 2)),
            channel,
        )

    run(first, "arm1")
    second = tmp_path / "pass2.json"
    layer_counts(
        Namespace(
            target_records=tmp_path / "threshold",
            realized_records=tmp_path / "arm1",
            counts=first,
            output=second,
        )
    )
    assert abs(json.loads(second.read_text())["residual_percent"]) > 2
    run(second, "arm2")
    check = tmp_path / "check.json"
    layer_counts(
        Namespace(
            target_records=tmp_path / "threshold",
            realized_records=tmp_path / "arm2",
            counts=second,
            output=check,
        )
    )
    result = json.loads(check.read_text())
    assert result["pass"] == 3
    residual = torch.tensor(result["layer_residual_bytes_per_token"])
    assert residual.abs().max().item() <= 1e-6 * target.min().item()


def test_recorded_token_bytes_carry_each_calls_parameter_reads() -> None:
    """Per-token records share each call's parameter reads and sum to the read cost."""
    rows, candidates = 6, 32
    allocation_context = context("schedule", rows, recording=True)
    schedule = Schedule(
        torch.tensor([0.0, 8.0], dtype=torch.float64),
        torch.tensor([1.2, 0.8], dtype=torch.float64),
    )
    allocation = LayerAllocation(
        allocation_context, 0, candidates, 8, 8.0, 0.5, schedule
    )
    assert allocation.parameter_bytes > 0
    expected_bytes = torch.zeros((), dtype=torch.float64)
    recorded = torch.zeros((), dtype=torch.float64)
    generator = torch.Generator().manual_seed(6)
    # Calls of different sizes and padding, including one with no valid token.
    for valid_rows in (
        [True] * 6,
        [True, False, True, False, False, False],
        [False] * 2,
    ):
        valid = torch.tensor(valid_rows)
        count = len(valid)
        tokens = slice(0, count)
        ranked = torch.rand(count, candidates, generator=generator)
        mask = allocation(ranked, valid, tokens)
        base = torch.where(valid, 1000 + mask.sum(-1) * 600, 0)
        allocation.record_bytes(tokens, allocation.token_bytes(base, valid))
        recorded += allocation_context.charged[tokens, 0].sum()
        expected_bytes += base.sum() + allocation.call_parameter_bytes(valid)
    assert recorded.item() == pytest.approx(expected_bytes.item(), rel=1e-12)
