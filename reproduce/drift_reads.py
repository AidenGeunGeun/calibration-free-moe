"""Print the Section 4.1 and Figure 4c budget misses and the appendix drift table.

Inputs: the runs root of `reproduce.headline`, or the `headline_data.json` it wrote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reproduce.headline import CELLS, collect

THRESHOLD_LINES = ("prox-cal-", "ours-cal-", "teal-x", "wina-x", "cats-x")
# Figure 4c: Prox and our reader's threshold version at targets near 13, 18, 28 and 38%.
FIGURE_4C = tuple(f"prox-cal-{t}" for t in (13, 18, 28, 38)) + tuple(
    f"ours-cal-{p}" for p in ("p05", "p10", "p20", "p30")
)


def budget(lines: dict[str, Any], line: str, cell: str) -> float:
    """Return the reads the threshold of `line` was calibrated for on `cell`.

    Prox: its read target. Our reader's threshold version: the reads of its top-k
    version at the same budget on the same cell (the threshold is fitted to keep the
    top-k count on average). TEAL and WINA's score: 100 minus their input sparsity.
    CATS: the reads of its top-k version at the same sparsity.
    """
    if line.startswith("prox-cal-"):
        return float(line.rsplit("-", 1)[1])
    if line.startswith(("teal-x", "wina-x")):
        return 100 - int(line[-3:])
    if line.startswith("ours-cal-"):
        return lines["ours-" + line.rsplit("-", 1)[1]][cell]["reads"]
    return lines[line.replace("cats-", "cats-topk-")][cell]["reads"]


def runs(
    lines: dict[str, Any], names: list[str]
) -> list[tuple[str, str, float, float]]:
    """List (line, cell, budget, realized reads) for every run of the named lines."""
    return [
        (line, cell, budget(lines, line, cell), lines[line][cell]["reads"])
        for line in names
        for cell in CELLS
    ]


def main() -> None:
    """Print the Figure 4c count, Prox at 18%, our reader's gaps and the totals."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--runs", type=Path, help="Runs root (rebuilds the headline matrix)"
    )
    source.add_argument("--headline", type=Path, help="headline_data.json")
    args = parser.parse_args()
    data = (
        json.loads(args.headline.read_text())
        if args.headline
        else collect(args.runs)[0]
    )
    lines = data["lines"]

    main_runs = runs(lines, list(FIGURE_4C))
    below = sum(reads < aim for _, _, aim, reads in main_runs)
    print(
        f"Figure 4c: {below} of {len(main_runs)} Prox and our-reader threshold runs "
        "read below budget"
    )
    prox18 = [lines["prox-cal-18"][c]["reads"] for c in CELLS]
    print(
        f"Prox calibrated for 18%: mean {sum(prox18) / 9:.1f}%, "
        f"minimum {min(prox18):.1f}%"
    )
    gaps = [
        aim - reads for line, _, aim, reads in main_runs if line.startswith("ours-cal-")
    ]
    print(
        f"Our reader, calibrated: {min(gaps):.1f} to {max(gaps):.1f} "
        "points below budget"
    )
    every = runs(lines, [line for line in lines if line.startswith(THRESHOLD_LINES)])
    settings = len({line for line, *_ in every})
    below = sum(reads < aim for _, _, aim, reads in every)
    print(
        f"All calibrated thresholds: {below} of {len(every)} runs below budget, "
        f"{settings} settings"
    )


if __name__ == "__main__":
    main()
