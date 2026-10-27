import itertools
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from calibration_free_moe.reader.budget import compile_budget, load_layer_budgets
from scripts.allocate_layers import (
    convex_envelope,
    interpolate,
    read_cost_bytes,
    solve_allocation,
)


@pytest.mark.parametrize("granularity", [1, 2, 3])
def test_convex_allocation_matches_brute_force(granularity: int) -> None:
    curves = [[(f, scale / f) for f in range(2, 11)] for scale in (1.0, 2.0, 5.0)]
    for total in range(6, 31):
        candidates = [
            counts
            for counts in itertools.product(range(2, 11), repeat=3)
            if sum(counts) == total
            and sum(f % granularity != 0 for f in counts) == bool(total % granularity)
        ]
        if not candidates:
            with pytest.raises(ValueError, match="infeasible"):
                solve_allocation(curves, total, granularity)
            continue
        counts = solve_allocation(curves, total, granularity)
        assert sum(counts) == total
        assert all(2 <= f <= 10 for f in counts)
        assert sum(f % granularity != 0 for f in counts) == bool(total % granularity)

        def objective(fs: Sequence[int]) -> float:
            return sum(interpolate(c, f) for c, f in zip(curves, fs, strict=True))

        assert objective(counts) == pytest.approx(min(map(objective, candidates)))


def test_nonconvex_curves_use_lower_envelope() -> None:
    curve = [(2, 9.0), (4, 8.0), (6, 3.0), (8, 2.0)]
    assert convex_envelope(curve) == [(2, 9.0), (6, 3.0), (8, 2.0)]
    assert interpolate(convex_envelope(curve), 4) == 6.0
    counts = solve_allocation([curve, [(2, 4.0), (8, 1.0)]], 10, 2)
    assert sum(counts) == 10
    assert (
        interpolate(convex_envelope(curve), counts[0])
        + interpolate([(2, 4.0), (8, 1.0)], counts[1])
        == 6.0
    )


@pytest.mark.parametrize("hidden,width", [(2048, 1024), (2048, 768), (2816, 704)])
def test_saved_allocation_preserves_total_read_cost(
    tmp_path: Path, hidden: int, width: int
) -> None:
    config = {
        "hidden_size": hidden,
        "expert_width": width,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 3,
    }
    source = tmp_path / "curves.json"
    output = tmp_path / "allocation.json"
    source.write_text(
        json.dumps(
            {
                "complete": True,
                "model": {"id": "test", "revision": "test"},
                "dimensions": config,
                "source": "weight-curves",
                "recipe": {"screen_keep": 0.625, "bank": "weight-only.pt"},
                "grid_counts": [64, 256, 1024],
                "curves": [
                    {"layer": l, "errors": [scale / f for f in (64, 256, 1024)]}
                    for l, scale in enumerate((1, 3, 9))
                ],
            }
        )
    )
    retention = 257 / (8 * width)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.allocate_layers",
            "solve",
            "--curves",
            str(source),
            "--retention",
            str(retention),
            "--output",
            str(output),
        ],
        check=True,
    )
    plans = load_layer_budgets(
        output,
        layers=3,
        hidden_size=hidden,
        expert_width=width,
        top_k=8,
        screen_keep=0.625,
    )
    uniform = compile_budget(
        hidden_size=hidden,
        expert_width=width,
        top_k=8,
        retention=retention,
        screen_keep=0.625,
    )
    assert len({p.final_count for p in plans}) > 1
    assert sum(p.final_count for p in plans) == 3 * uniform.final_count
    assert sum(read_cost_bytes(p) for p in plans) == 3 * read_cost_bytes(uniform)
    saved = json.loads(output.read_text())
    assert (
        saved["total_read_cost_bytes"]
        == saved["uniform_read_cost_bytes"]
        == 3 * read_cost_bytes(uniform)
    )
    with pytest.raises(ValueError, match="one count"):
        load_layer_budgets(
            output,
            layers=4,
            hidden_size=hidden,
            expert_width=width,
            top_k=8,
            screen_keep=0.625,
        )


@pytest.mark.parametrize("counts", [[1.5, 32], [True, 32], [0, 32], [64, 32]])
def test_invalid_allocations_are_not_rounded(
    tmp_path: Path, counts: list[object]
) -> None:
    path = tmp_path / "allocation.json"
    path.write_text(json.dumps({"final_counts": counts}))
    with pytest.raises(ValueError, match="integers"):
        load_layer_budgets(
            path, layers=2, hidden_size=128, expert_width=32, top_k=2, screen_keep=0.625
        )
