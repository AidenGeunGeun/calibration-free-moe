"""Print the Section 4.1 context cut (57% more channels) and the Appendix E cut table.

Inputs: one allocation-study root per model for ``--prox``; ``DEEP/cut-thk/`` for the
exact-score cut, stored (``--cut-json``) or recomputed on CPU (``--states``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from reproduce.analysis_common import mean_se
from scripts.allocation_study import record_tokens
from scripts.drift_mechanism import Checkpoint, project, score

# Contexts before the same answer window: the full answer, the question alone, the last
# 1,000 answer tokens, or another trace's 6,000-token answer.
CONDITIONS = ("full", "cut0", "cut1000", "foreign")
# 256-token windows at answer positions 0, 2,000 and 6,000 of GPQA-Diamond traces,
# under Prox's threshold calibrated for 13% of reads. The cuts re-feed the window at
# 6,000 after a shorter context and are compared with it.
PROX_ARMS = {
    "depth0": "deep/b13-chat/d00000/threshold/rec",
    "depth2000": "deep/b13-chat/d02000/threshold/rec",
    "cut0": "cut/keep0.rec",
    "cut1000": "cut/keep1000.rec",
    "full6000": "deep/b13-chat/d06000/threshold/rec",
}


def trace_counts(directory: Path) -> dict[str, float]:
    """Average each trace's kept channels per token (summed over layers) over steps."""
    steps: dict[str, dict[int, float]] = {}
    for (key, step), token in record_tokens(directory).items():
        trace = json.loads(key)[1].split("-cut")[0].rsplit("/", 1)[0]
        steps.setdefault(trace, {})[step] = float(token["counts"].sum())
    return {trace: sum(s.values()) / len(s) for trace, s in steps.items()}


def prox_cut(root: Path) -> dict[str, Any]:
    """Compare kept channels per token after each context with the full answer.

    The change is the mean over traces of the per-trace ratio minus one, with one SE.
    """
    counts = {name: trace_counts(root / path) for name, path in PROX_ARMS.items()}
    base = counts["full6000"]
    out: dict[str, Any] = {}
    for name, per_trace in counts.items():
        row: dict[str, Any] = {
            "channels_per_token": sum(per_trace.values()) / len(per_trace),
            "traces": len(per_trace),
        }
        if name.startswith("cut"):
            ratios = [per_trace[t] / base[t] - 1 for t in base if t in per_trace]
            change, se = mean_se(ratios)
            row |= {"change": change, "se": se}
        out[name] = row
    return out


def exact_kept(
    states: Path, model: Path, mechanism: Path, device: str
) -> dict[str, list[list[float]]]:
    """Return per condition the [layer][trace] kept fractions under the exact score.

    A channel is kept when its exact output contribution is strictly above the
    chat-calibrated per-layer tau. Only routed experts are projected, so CPU suffices.
    """
    taus = json.loads(mechanism.read_text())["thresholds"]["exact"]
    checkpoint = Checkpoint(model, device)
    config = checkpoint.config
    documents = json.loads((states / "collection.json").read_text())["documents"]
    condition = [d["id"].split("|")[1] for d in documents]
    kept_by: dict[str, list[list[float]]] = {c: [] for c in CONDITIONS}
    for layer in range(int(config["num_hidden_layers"])):
        payload = torch.load(states / f"layer-{layer:02d}.pt", map_location=device)
        rows = project(payload, checkpoint.layer(layer), config)
        kept = (score(rows, checkpoint.activation) > taus[layer]).float().mean(-1)
        document = payload["document"].long()
        for c in CONDITIONS:
            kept_by[c].append(
                [
                    float(kept[document == i].mean())
                    for i in range(len(documents))
                    if condition[i] == c
                ]
            )
    return kept_by


def exact_cut(kept_by: dict[str, list[list[float]]]) -> dict[str, Any]:
    """Average over traces the relative change of each trace's layer-mean kept fraction.

    The change is against the same trace after the full context (ratio minus one),
    with the SE over traces.
    """
    full = torch.tensor(kept_by["full"], dtype=torch.float32)  # [L, traces]
    out: dict[str, Any] = {}
    for c in CONDITIONS:
        values = torch.tensor(kept_by[c], dtype=torch.float32)
        ratio = values.mean(0) / full.mean(0) - 1
        out[c] = {
            "mean": float(values.mean()),
            "rel_vs_full": float(ratio.mean()),
            "se": float(ratio.std() / len(ratio) ** 0.5),
            "per_layer": values.mean(1).tolist(),
        }
    return out


def main() -> None:
    """Print the Prox cut of each root and the exact-score cut."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prox", type=Path, nargs="*", default=[])
    parser.add_argument("--cut-json", type=Path, help="Stored exact-score cut summary")
    parser.add_argument("--states", type=Path, help="Captured cut-panel MoE inputs")
    parser.add_argument("--model", type=Path, help="Checkpoint for --states")
    parser.add_argument("--mechanism", type=Path, help="mechanism.json with exact tau")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    for root in args.prox:
        cut = prox_cut(root)
        result[f"prox/{root.name}"] = cut
        print(f"Prox cut, {root}")
        for name, row in cut.items():
            line = f"  {name:9s} {row['channels_per_token']:7.0f} channels/token"
            if "change" in row:
                line += f"  {100 * row['change']:+.1f} +- {100 * row['se']:.1f}%"
            print(line)
    if args.states is not None:
        if args.model is None or args.mechanism is None:
            parser.error("--states needs --model and --mechanism")
        kept_by = exact_kept(args.states, args.model, args.mechanism, args.device)
        result["exact/kept"] = kept_by
        exact = exact_cut(kept_by)
    elif args.cut_json is not None:
        stored = json.loads(args.cut_json.read_text())
        exact = {c: stored[f"kept/{c}"] for c in CONDITIONS}
    else:
        exact = None
    if exact is not None:
        result["exact"] = exact
        print("Exact-score cut (strict score > chat-calibrated tau)")
        for c in CONDITIONS[1:]:
            row = exact[c]
            print(
                f"  {c:8s} kept {row['mean']:.4f}  "
                f"{100 * row['rel_vs_full']:+.1f} +- {100 * row['se']:.1f}%"
            )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
