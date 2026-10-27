from __future__ import annotations

import torch

from scripts.drift_mechanism import Grid, align


def test_unequal_row_counts_do_not_create_drift() -> None:
    # Trace 0 always keeps every candidate, trace 1 none; they are sampled 10:1 in the
    # first bin and 1:10 in the second. Row pooling would read 0.909 -> 0.091.
    counts = {(0, 0): 10, (1, 0): 1, (0, 1): 1, (1, 1): 10}
    document, position = [], []
    for (trace, column), count in counts.items():
        document += [trace] * count
        position += [10 * column + j for j in range(count)]
    document_ids = torch.tensor(document, dtype=torch.int32)
    grid = Grid(
        document_ids,
        torch.tensor(position, dtype=torch.int32),
        torch.zeros(2, dtype=torch.long),
        torch.ones(2, dtype=torch.bool),
        width=10,
        end=20,
    )
    kept = (document_ids == 0).double()
    assert torch.allclose(
        grid.pooled(kept), torch.tensor([0.5, 0.5], dtype=torch.float64)
    )

    high, low = torch.tensor([2.0, 3.0, 4.0, 5.0]), torch.tensor([0.1, 0.2, 0.3, 0.4])
    scores = torch.where(kept[:, None] > 0, high, low)
    log_scale, observed, reconstructed = align(scores, grid, tau=1.0)
    assert torch.allclose(log_scale, torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(observed, torch.full((2,), 0.5, dtype=torch.float64))
    assert torch.allclose(reconstructed, observed)
