"""Build and summarize teacher-forced fidelity panels on a model's own traces (CPU).

`prepare` turns a stock generation report into one panel per depth: the prompt and
the first `depth` generated tokens are stock context, the next `--window` tokens are
teacher-forced. `calibration-panel` builds a FIT/TUNE panel from such traces,
`loops` builds windows around a policy's repetition loops, and `reduce` summarizes
`eval_vllm_continuation.py --fidelity-reference` reports.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def load_tokenizer(model: str) -> Any:
    """Return the model's Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def generated_documents(scores: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the report and its documents with the token IDs vLLM used.

    Decoded text can re-encode to other IDs of the same length, so traces are never
    rebuilt from text. A stopped trace drops its final stop token, as its text does.
    """
    report = json.loads(scores.read_text())
    if "token_ids_file" not in report:
        raise ValueError(
            f"{scores} has no recorded token IDs; generate with "
            "eval_vllm_generation.py --token-ids"
        )
    with (scores.parent / report["token_ids_file"]).open() as handle:
        recorded = {row["id"]: row for row in map(json.loads, handle)}
    for document in report["documents"]:
        row = recorded[document["id"]]
        generated = row["token_ids"]
        if len(generated) != document["num_tokens"]:
            raise ValueError(f"Recorded IDs differ from the report: {document['id']}")
        document["prompt"] = row["prompt_token_ids"]
        document["tokens"] = (
            generated[:-1] if document["finish_reason"] == "stop" else generated
        )
    return report, report["documents"]


def reencoded_changes(
    tokenizer: Any, documents: list[dict[str, Any]]
) -> dict[str, int]:
    """Count used traces whose decoded text re-encodes to other token IDs."""
    changed = sum(
        tokenizer(d["continuation"], add_special_tokens=False)["input_ids"]
        != d["tokens"]
        for d in documents
    )
    return {"traces": len(documents), "changed": changed}


def role_documents(args: argparse.Namespace, roles: set[str]) -> list[dict[str, Any]]:
    """Return generated documents, restricted to prepared requests of `roles`."""
    report, generated = generated_documents(args.scores)
    if args.requests is None:
        return generated
    requests = {r["id"]: r for r in json.loads(args.requests.read_text())["requests"]}
    if not report["complete"] or {d["id"] for d in generated} != set(requests):
        raise ValueError("Generation does not cover exactly the prepared requests")
    documents = []
    for document in generated:
        request = requests[document["id"]]
        if document["prompt"] != request["prompt"]:
            raise ValueError(f"Prepared prompt differs: {document['id']}")
        if request["split_role"] in roles:
            documents.append(
                document
                | {
                    "group_id": request["conversation_id"],
                    "split_role": request["split_role"],
                }
            )
    return documents


def select_traces(
    args: argparse.Namespace, tokenizer: Any, need: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one seeded trace per seeded question, and the panel metadata.

    Eligible questions have a trace that stopped on its own (not at the token cap)
    with at least `need` tokens.
    """
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in role_documents(args, {args.split_role}):
        samples[document["id"].rsplit("/", 1)[0]].append(document)
    rng = random.Random(args.seed)
    eligible: dict[str, list[dict[str, Any]]] = {}
    for question, group in sorted(samples.items()):
        kept = [
            document
            for document in group
            if document["finish_reason"] == "stop" and len(document["tokens"]) >= need
        ]
        if kept:
            eligible[question] = kept
    if len(eligible) < args.traces:
        raise ValueError(f"Only {len(eligible)} questions have {need}-token traces")
    questions = sorted(rng.sample(sorted(eligible), args.traces))
    traces = [rng.choice(eligible[question]) for question in questions]
    longest = max(len(t["prompt"]) for t in traces) + need
    data = {
        "scores": str(args.scores),
        "requests": str(args.requests) if args.requests else None,
        "split_role": args.split_role,
        "tokenizer": args.model,
        "seed": args.seed,
        "selection": "seeded questions with a completed sample of at least "
        f"{need} generated tokens (recorded IDs, less the stop token); one seeded "
        "sample per question",
        "eligible_questions": len(eligible),
        "questions": len(samples),
        "reencoded_trace_changes": reencoded_changes(tokenizer, traces),
        "max_model_len": math.ceil(longest / 1024) * 1024,
    }
    return traces, data


def prepare(args: argparse.Namespace, tokenizer: Any = None) -> None:
    """Write one teacher-forced panel per depth into `--output-dir`."""
    tokenizer = tokenizer or load_tokenizer(args.model)
    traces, data = select_traces(args, tokenizer, max(args.depths) + args.window)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for depth in args.depths:
        requests = [
            {
                "id": f"{trace['id']}/d{depth}",
                "source": trace["source"],
                "depth": depth,
                "prompt": trace["prompt"] + trace["tokens"][:depth],
                "target": trace["tokens"][depth : depth + args.window],
            }
            for trace in traces
        ]
        panel = {
            "role": "DEV",
            "data": data | {"depth": depth, "window_tokens": args.window},
            "requests": requests,
        }
        path = args.output_dir / f"d{depth:05d}.json"
        path.write_text(json.dumps(panel) + "\n")
        print(path, len(requests), "eligible questions", data["eligible_questions"])


def calibration_panel(args: argparse.Namespace, tokenizer: Any = None) -> None:
    """Write a FIT/TUNE calibration panel with one seeded window per trace.

    Each document is the prompt and generated trace, capped to `--max-length`
    tokens, as stock context, with one uniform window inside the trace. HELDOUT
    traces never enter.
    """
    tokenizer = tokenizer or load_tokenizer(args.model)
    chosen: dict[str, list[dict[str, Any]]] = {"FIT": [], "TUNE": []}
    skips: Counter[str] = Counter()
    used: list[dict[str, Any]] = []
    fit_total = 0
    for document in role_documents(args, set(chosen)):
        role = document["split_role"]
        if (
            role == "FIT"
            and args.fit_tokens is not None
            and fit_total >= args.fit_tokens
        ):
            skips["fit_tokens_reached"] += 1
            continue
        prompt = document["prompt"]
        end = min(len(document["tokens"]), args.max_length - len(prompt))
        if end < args.window:
            skips["no_window_within_cap"] += 1
            continue
        start = random.Random(f"{args.seed}/{document['id']}").randint(
            0, end - args.window
        )
        tokens = prompt + document["tokens"][: start + args.window]
        chosen[role].append(
            {
                "id": document["id"],
                "source": document["source"],
                "group_id": document["group_id"],
                "split_role": role,
                "finish_reason": document["finish_reason"],
                "generated_tokens": document["num_tokens"],
                "prompt_tokens": len(prompt),
                "trace_window_start": start,
                "tokens": tokens,
                "context_length": len(prompt) + start,
            }
        )
        used.append(document)
        if role == "FIT":
            fit_total += len(tokens)
    if not chosen["FIT"] or not chosen["TUNE"]:
        raise ValueError("No FIT or TUNE traces qualify")
    if args.fit_tokens is not None and fit_total < args.fit_tokens:
        raise ValueError(f"FIT traces reach {fit_total} of {args.fit_tokens} tokens")
    documents = chosen["FIT"] + chosen["TUNE"]
    changes = reencoded_changes(tokenizer, used)
    panel = {
        "role": "calibration",
        "data": {
            "scores": str(args.scores),
            "requests": str(args.requests),
            "tokenizer": args.model,
            "seed": args.seed,
            "selection": "FIT/TUNE traces in prepared order (FIT until --fit-tokens); "
            "recorded prompt plus generated IDs (less the stop token) capped at "
            "max_length, cut after one seeded uniform window inside the trace; "
            "context_length is the window start",
            "max_length": args.max_length,
            "window_tokens": args.window,
            "fit_tokens": fit_total,
            "fit_tokens_target": args.fit_tokens,
            "documents_by_role": {role: len(docs) for role, docs in chosen.items()},
            "length_capped_generations": sum(
                d["finish_reason"] == "length" for d in documents
            ),
            "skips": dict(skips),
            "reencoded_trace_changes": changes,
        },
        "documents": documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(panel) + "\n")
    print(args.output, json.dumps(panel["data"]))


def loop_onset(tokens: list[int], max_period: int = 1000) -> tuple[int, int] | None:
    """Return the onset and period of the shortest loop repeated three times at the end.

    Returns None when no period below `max_period` repeats three times.
    """
    for period in range(1, max_period):
        if len(tokens) < 3 * period:
            break
        if (
            tokens[-period:] == tokens[-2 * period : -period]
            and tokens[-2 * period : -period] == tokens[-3 * period : -2 * period]
        ):
            i = len(tokens) - period - 1
            while i >= 0 and tokens[i] == tokens[i + period]:
                i -= 1
            return i + 1, period
    return None


def loops(args: argparse.Namespace) -> None:
    """Write windows just before and inside a policy's own repetition loops."""
    report = json.loads(args.scores.read_text())
    tokenizer = load_tokenizer(args.model)
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in report["documents"]:
        if document["finish_reason"] != "length":
            continue
        tokens = tokenizer(document["continuation"], add_special_tokens=False)[
            "input_ids"
        ]
        found = loop_onset(tokens)
        if abs(len(tokens) - document["num_tokens"]) > 1 or found is None:
            continue
        onset, period = found
        if args.window <= onset <= args.max_onset:
            document.update(tokens=tokens, onset=onset, period=period)
            eligible[document["id"].rsplit("/", 1)[0]].append(document)
    rng = random.Random(args.seed)
    questions = sorted(rng.sample(sorted(eligible), args.traces))
    traces = [rng.choice(eligible[question]) for question in questions]
    longest = max(len(t["prompt"]) + t["onset"] for t in traces) + 2 * args.window
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, start in (("entry", -args.window), ("inside", args.window)):
        requests = [
            {
                "id": f"{t['id']}/loop-{name}",
                "source": t["source"],
                "prompt": t["prompt"] + t["tokens"][: t["onset"] + start],
                "target": t["tokens"][
                    t["onset"] + start : t["onset"] + start + args.window
                ],
            }
            for t in traces
        ]
        panel = {
            "role": "DEV",
            "data": {
                "scores": str(args.scores),
                "tokenizer": args.model,
                "seed": args.seed,
                "selection": "length-capped samples whose re-encoded tokens end in a "
                "period repeated three times, onset in [window, max_onset]; one seeded "
                f"sample per seeded question; window {name} the loop",
                "eligible_questions": len(eligible),
                "onsets": [t["onset"] for t in traces],
                "periods": [t["period"] for t in traces],
                "window_tokens": args.window,
                "max_model_len": math.ceil(longest / 1024) * 1024,
            },
            "requests": requests,
        }
        path = args.output_dir / f"loop-{name}.json"
        path.write_text(json.dumps(panel) + "\n")
        print(path, len(requests), "eligible questions", len(eligible))


def reduce(args: argparse.Namespace) -> None:
    """Print and optionally write KL, top-1 and read summaries per report.

    Each summary includes mean KL(stock || policy), top-1 agreement, realized reads,
    KL quantiles and the share of total KL carried by the top 1% of positions.
    """
    rows = []
    for path in args.scores:
        report = json.loads(path.read_text())
        if not report.get("complete"):
            raise ValueError(f"{path} is incomplete")
        work = report.get("reader_work") or report.get("baseline_work") or {}
        # Teacher-forced NLL of the forced tokens (decode steps; the stock record too).
        nll = sum(d["decode_nll"] for d in report["documents"]) / len(
            report["documents"]
        )
        if "kl" not in report["documents"][0]:
            rows.append({"scores": str(path), "decode_nll": nll})
            print(json.dumps(rows[-1]))
            continue
        kl = torch.tensor([x for d in report["documents"] for x in d["kl"]])
        top1 = torch.tensor([x for d in report["documents"] for x in d["top1"]])
        ordered = kl.sort(descending=True).values
        top = max(1, len(kl) // 100)
        row: dict[str, Any] = {
            "scores": str(path),
            "positions": len(kl),
            "decode_nll": nll,
            "read_fraction": work.get("read_fraction"),
            "mean_kl": kl.mean().item(),
            "top1": top1.float().mean().item(),
            "kl_p50": kl.quantile(0.5).item(),
            "kl_p90": kl.quantile(0.9).item(),
            "kl_p99": kl.quantile(0.99).item(),
            "share_top1pct": (ordered[:top].sum() / kl.sum()).item(),
            "share_top10pct": (
                ordered[: max(1, len(kl) // 10)].sum() / kl.sum()
            ).item(),
            "frac_kl_over_1": (kl > 1).float().mean().item(),
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "scores"}), path)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    """Parse the subcommand and its arguments, then run it."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("--scores", type=Path, required=True)
    make.add_argument("--model", required=True)
    make.add_argument("--output-dir", type=Path, required=True)
    make.add_argument("--traces", type=int, default=64)
    make.add_argument("--depths", type=int, nargs="+", default=[0, 2000, 6000])
    make.add_argument("--window", type=int, default=512)
    make.add_argument("--seed", type=int, default=353)
    make.add_argument(
        "--requests", type=Path, help="Prepared requests carrying trace roles"
    )
    make.add_argument("--split-role", help="With --requests: the eligible role")
    calibration = commands.add_parser("calibration-panel")
    calibration.add_argument("--scores", type=Path, required=True)
    calibration.add_argument("--requests", type=Path, required=True)
    calibration.add_argument("--model", required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--max-length", type=int, default=8192)
    calibration.add_argument("--window", type=int, default=128)
    calibration.add_argument(
        "--fit-tokens",
        type=int,
        help="Stop adding FIT traces once this token total is reached",
    )
    calibration.add_argument("--seed", type=int, default=353)
    loop = commands.add_parser("loops")
    loop.add_argument("--scores", type=Path, required=True)
    loop.add_argument("--model", required=True)
    loop.add_argument("--output-dir", type=Path, required=True)
    loop.add_argument("--traces", type=int, default=32)
    loop.add_argument("--window", type=int, default=512)
    loop.add_argument("--max-onset", type=int, default=7000)
    loop.add_argument("--seed", type=int, default=353)
    summary = commands.add_parser("reduce")
    summary.add_argument("--scores", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare" and (args.requests is None) != (
        args.split_role is None
    ):
        parser.error("--requests and --split-role go together")
    if args.command == "calibration-panel" and (
        args.window < 1
        or args.max_length <= args.window
        or (args.fit_tokens is not None and args.fit_tokens < 1)
    ):
        parser.error("Require 0 < window < max_length and positive --fit-tokens")
    {
        "prepare": prepare,
        "calibration-panel": calibration_panel,
        "loops": loops,
        "reduce": reduce,
    }[args.command](args)


if __name__ == "__main__":
    main()
