"""Write read-matched layer counts, score thresholds or depth schedules (CPU only).

`layer-counts` matches a comparison arm's per-layer read cost to a calibrated
threshold arm, `thresholds` fits per-layer score thresholds from a recorded
histogram, and `schedule` writes a depth schedule of token counts. Each reads
`eval_vllm_continuation.py --allocation-record` output and writes one JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from calibration_free_moe.analysis.allocation import (
    HISTOGRAM_BINS,
    load_channel_bytes,
    load_records,
)

Token = tuple[str, int]  # (allocation key, sampled step)


def report_tokens(path: Path) -> dict[Token, dict[str, Any]]:
    """Return per-token KL, top-1, logprob and answer depth of an allocation report.

    A token is (allocation key, decode step s >= 1); its answer depth is the
    request's answer offset plus s - 1.
    """
    report = json.loads(path.read_text())
    if not report.get("complete"):
        raise ValueError(f"{path} is incomplete")
    tokens: dict[Token, dict[str, Any]] = {}
    for document in report["documents"]:
        if "kl" not in document or "allocation_key" not in document:
            raise ValueError(f"{path} has no fidelity or allocation keys")
        for index, kl in enumerate(document["kl"]):
            step = index + 1
            tokens[(document["allocation_key"], step)] = {
                "kl": float(kl),
                "top1": bool(document["top1"][index]),
                "logprob": float(document["logprobs"][step]),
                "depth": int(document["answer_offset"]) + step - 1,
                "fidelity_key": document["fidelity_key"],
                "id": document["id"],
            }
    return tokens


def record_tokens(directory: Path) -> dict[Token, dict[str, Tensor]]:
    """Return per-token counts, score features, read cost and depth of a record.

    Counts and read cost (`charged_bytes`) are per-layer rows of one sampled token.
    """
    tokens: dict[Token, dict[str, Tensor]] = {}
    for key, record in load_records(directory).items():
        for row, step in enumerate(record["steps"].tolist()):
            tokens[(key, step)] = {
                "counts": record["counts"][row],
                "features": record["features"][row],
                "charged": record["charged_bytes"][row],
                "depth": record["depth"][row],
            }
    return tokens


def mean_layer_bytes(
    directory: Path,
) -> tuple[dict[Token, dict[str, Tensor]], Tensor, Tensor]:
    """Return a record's tokens and its per-layer mean counts and read cost."""
    tokens = record_tokens(directory)
    counts = torch.stack([t["counts"] for t in tokens.values()]).double()
    charged = torch.stack([t["charged"] for t in tokens.values()]).double()
    return tokens, counts.mean(0), charged.mean(0)


def layer_counts(args: argparse.Namespace) -> None:
    """Match a comparison arm's per-layer read cost to the threshold arm in two passes.

    The target is the threshold arm's realized read cost (`charged_bytes`, bytes it
    must read, priced in 32-byte sectors) per token in each layer. Without
    --realized-records this writes the threshold arm's mean counts (pass 1). With
    the pass-1 run's record and --counts, each layer's byte residual divided by its
    native bytes per kept channel corrects the counts (pass 2); rerun on the pass-2
    record to report the remaining residual. Read cost includes input coordinates
    and allocation parameters read along the arm's own trajectory, so equal counts
    are not equal reads.
    """
    target_tokens, threshold_counts, target = mean_layer_bytes(args.target_records)
    result: dict[str, Any] = {
        "format": "allocation-layer-counts-v1",
        "target_records": str(args.target_records),
        "tokens": len(target_tokens),
        "target_bytes_per_token": target.sum().item(),
        "layer_target_bytes_per_token": target.tolist(),
    }
    if args.realized_records is None:
        result |= {"pass": 1, "layer_counts": threshold_counts.tolist()}
    else:
        realized_tokens, _, realized = mean_layer_bytes(args.realized_records)
        if set(realized_tokens) != set(target_tokens):
            raise ValueError("Match reads on the threshold arm's own tokens")
        spec = json.loads(args.counts.read_text())
        counts = torch.tensor(spec["layer_counts"], dtype=torch.float64)
        channel_bytes = load_channel_bytes(args.realized_records)
        if counts.shape != target.shape or channel_bytes.shape != target.shape:
            raise ValueError("Counts, records and channel bytes cover other layers")
        residual = target - realized
        result |= {
            "pass": int(spec.get("pass", 1)) + 1,
            "realized_records": str(args.realized_records),
            "input_counts": str(args.counts),
            "realized_bytes_per_token": realized.sum().item(),
            "residual_bytes_per_token": residual.sum().item(),
            "residual_percent": 100 * residual.sum().item() / target.sum().item(),
            "layer_residual_bytes_per_token": residual.tolist(),
            "layer_residual_percent": (100 * residual / target).tolist(),
            "layer_channel_bytes": channel_bytes.tolist(),
            "layer_counts": (counts + residual / channel_bytes).clamp_min(0).tolist(),
        }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "layer_counts"}))


def histogram_threshold(
    histogram: Tensor, tokens: float, target: float, log2_range: tuple[float, float]
) -> float:
    """Return the score at or above which `target` candidates per token lie.

    Bins are uniform in log2(score) over `log2_range`; the threshold interpolates
    log-uniformly inside the bin that crosses the target.
    """
    low, high = log2_range
    width = (high - low) / HISTOGRAM_BINS
    wanted = target * tokens
    tail = histogram.flip(0).cumsum(0).flip(0)  # candidates in bins b..end
    if not 0 < wanted <= float(tail[0]):
        raise ValueError("The target count is outside the recorded scores")
    b = int((tail >= wanted).nonzero().max())
    above = float(tail[b + 1]) if b + 1 < HISTOGRAM_BINS else 0.0
    fraction = (wanted - above) / float(histogram[b])
    return 2.0 ** (low + (b + 1 - fraction) * width)


def thresholds(args: argparse.Namespace) -> None:
    """Write per-layer exact-score thresholds for a target mean count per token."""
    saved = cast(
        dict[str, Any], torch.load(args.record / "histogram.pt", weights_only=True)
    )
    if saved["format"] != "allocation-histogram-v1":
        raise ValueError("Not an allocation histogram")
    histograms, tokens = saved["histograms"], saved["tokens"]
    values = [
        histogram_threshold(
            histograms[layer],
            float(tokens[layer]),
            args.target_count,
            tuple(saved["log2_range"]),
        )
        for layer in range(len(histograms))
    ]
    result = {
        "format": "allocation-thresholds-v1",
        "score": "exact",
        "target_count": args.target_count,
        "record": str(args.record),
        "tokens_per_layer": tokens.tolist(),
        "fit": "one-shot: the recorded run's own trajectory (its allocation), "
        "log-uniform interpolation inside a 1/32-octave histogram bin",
        "thresholds": values,
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(args.output, "layers", len(values))


def schedule(args: argparse.Namespace) -> None:
    """Write depth-bin count multipliers relative to the record's mean token count.

    `--edges` are lower bin edges in answer depth, starting at 0; the weights are
    each bin's share of the record's tokens.
    """
    records = load_records(args.record)
    depth = torch.cat([r["depth"] for r in records.values()])
    value = torch.cat([r["counts"].sum(-1) for r in records.values()]).double()
    edges = torch.tensor(args.edges, dtype=torch.float64)
    if edges[0] != 0 or not bool((edges[1:] > edges[:-1]).all()):
        raise ValueError("Edges are increasing lower bin edges from 0")
    bins = torch.searchsorted(edges, depth.double(), right=True) - 1
    weights = torch.bincount(bins, minlength=len(edges)).double()
    if bool((weights == 0).any()):
        raise ValueError("Every depth bin needs tokens")
    means = torch.zeros(len(edges), dtype=torch.float64).index_add_(0, bins, value)
    means = means / weights
    result = {
        "format": "allocation-schedule-v1",
        "source": str(args.record),
        "edges": args.edges,
        "multipliers": (means / value.mean()).tolist(),
        "weights": (weights / weights.sum()).tolist(),
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


def main() -> None:
    """Parse the command line and run one subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    means = commands.add_parser("layer-counts")
    means.add_argument(
        "--target-records", type=Path, required=True, help="The threshold arm's record"
    )
    means.add_argument(
        "--realized-records", type=Path, help="The comparison arm's record (pass 1)"
    )
    means.add_argument(
        "--counts", type=Path, help="The layer-counts JSON that produced that record"
    )
    means.add_argument("--output", type=Path, required=True)
    fit_thresholds = commands.add_parser("thresholds")
    fit_thresholds.add_argument("--record", type=Path, required=True)
    fit_thresholds.add_argument("--target-count", type=int, required=True)
    fit_thresholds.add_argument("--output", type=Path, required=True)
    depth = commands.add_parser("schedule")
    depth.add_argument("--record", type=Path, required=True)
    depth.add_argument("--edges", type=int, nargs="+", required=True)
    depth.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "layer-counts" and (args.realized_records is None) != (
        args.counts is None
    ):
        parser.error("--realized-records and --counts go together")
    {
        "layer-counts": layer_counts,
        "thresholds": thresholds,
        "schedule": schedule,
    }[args.command](args)


if __name__ == "__main__":
    main()
