"""Print the Section 4.2 calibrated layer-budget results and the Appendix layer table.

Inputs: ``DEEP/layer-oracle/`` (``eval/`` and ``panel-eval.json``); ``--pair`` takes a
calibrated-budget run and the uniform-budget run at the same reads.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from reproduce.analysis_common import paired_difference

ARMS = ("hindsight", "chatcal", "weightonly")
ROLES = ("FIT", "TUNE")
BOOTSTRAPS = 2000


def load_eval(path: Path) -> tuple[dict[str, list[float]], float]:
    """Load one budget arm: per-window token KLs and the read fraction."""
    report = json.loads(path.read_text())
    if not report["complete"]:
        raise ValueError(f"{path} is incomplete")
    kl = {doc["id"]: doc["kl"] for doc in report["documents"]}
    return kl, report["reader_work"]["read_fraction"]


def layer_oracle(directory: Path) -> dict[str, Any]:
    """Compare each layer-budget arm with uniform per split, bootstrapped by trace.

    The KL is our reader's teacher-forced KL at 17.8% of reads on windows of
    Qwen3-Thinking GPQA-Diamond answers, every layer sparse inside the window. Arms:
    hindsight (fitted on the FIT traces), chat-calibrated and weight-only budgets. Per
    split (FIT, held-out TUNE; the paper reports TUNE), the token-micro mean KL
    difference against uniform, relative to uniform's, with the SE of 2,000 seed-0
    bootstraps over traces.
    """
    requests = json.loads((directory / "panel-eval.json").read_text())["requests"]
    split = {request["id"]: request["split"] for request in requests}
    uniform, uniform_reads = load_eval(directory / "eval" / "uniform.json")
    out: dict[str, Any] = {}
    for arm in ARMS:
        kl, reads = load_eval(directory / "eval" / f"{arm}.json")
        for role in ROLES:
            ids = [i for i in uniform if split[i] == role]
            traces = sorted({i.split("|")[0] for i in ids})
            trace_of = {i: i.split("|")[0] for i in ids}
            change = {
                t: [sum(kl[i]) - sum(uniform[i]) for i in ids if trace_of[i] == t]
                for t in traces
            }
            tokens = {
                t: sum(len(uniform[i]) for i in ids if trace_of[i] == t) for t in traces
            }
            total_tokens = sum(tokens.values())
            uniform_kl = sum(sum(uniform[i]) for i in ids) / total_tokens
            diff = sum(sum(v) for v in change.values()) / total_tokens
            rng = random.Random(0)
            boots = []
            for _ in range(BOOTSTRAPS):
                sample = [rng.choice(traces) for _ in traces]
                boots.append(
                    sum(sum(change[t]) for t in sample) / sum(tokens[t] for t in sample)
                )
            mean = sum(boots) / len(boots)
            se = (sum((b - mean) ** 2 for b in boots) / (len(boots) - 1)) ** 0.5
            out[f"{arm}/{role}"] = {
                "reads": 100 * reads,
                "uniform_reads": 100 * uniform_reads,
                "uniform_kl": uniform_kl,
                "diff": diff,
                "se": se,
                "change": 100 * diff / uniform_kl,
                "change_se": 100 * se / uniform_kl,
                "traces": len(traces),
                "windows": len(ids),
            }
    return out


def main() -> None:
    """Print the layer-oracle comparison and the requested benchmark pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer-oracle", type=Path, help="Directory with eval/ and panel-eval.json"
    )
    parser.add_argument(
        "--pair", type=Path, nargs=2, action="append", default=[], metavar="RUN"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    if args.layer_oracle is not None:
        oracle = layer_oracle(args.layer_oracle)
        result["layer_oracle"] = oracle
        for name, row in oracle.items():
            print(
                f"{name:16s} reads {row['reads']:.2f}% "
                f"(uniform {row['uniform_reads']:.2f}%)"
                f"  uniform KL {row['uniform_kl']:.4f}  diff {row['diff']:+.4f} +- "
                f"{row['se']:.4f}  ({row['change']:+.1f} +- {row['change_se']:.1f}%)"
            )
    for policy, base in args.pair:
        pair = paired_difference(policy / "metrics.json", base / "metrics.json")
        result[f"pair/{policy}|{base}"] = pair
        print(
            f"{policy} vs {base}: {pair['policy']:.2f} vs {pair['base']:.2f}, "
            f"{pair['diff']:+.2f} +- {pair['se']:.2f} "
            f"over {pair['questions']} questions"
        )
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
