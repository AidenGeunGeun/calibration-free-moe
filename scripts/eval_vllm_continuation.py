"""Score teacher-forced continuations of a panel under stock or a sparse policy.

By default stock prefill predicts the first target and every later target is a
policy decode step; `--sparse-prefill` instead scores all targets from one sparse
prefill, as lm-eval's loglikelihood does. Writes one JSON report per input with
per-document target log probabilities and NLLs, the policy's read statistics and,
with `--fidelity-reference`, KL and top-1 agreement against stock.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

import torch
from vllm import LLM, RequestOutput, SamplingParams

from calibration_free_moe.analysis import allocation
from calibration_free_moe.eval.fidelity import KEY as FIDELITY_KEY
from calibration_free_moe.eval.fidelity import FidelityWorkerExtension
from calibration_free_moe.model_config import (
    CALIBRATION_FREE_POLICIES,
    model_dimensions,
    model_override,
    validate_prox_keep_options,
)
from calibration_free_moe.runtime.work_reports import (
    aggregate_work_reports,
    dense_mlp_work_report,
    shared_expert_work_report,
)


class ContinuationWorkerExtension(FidelityWorkerExtension):
    """Fidelity worker extension that also reports resident model memory."""

    def tokenwise_model_memory(self: Any) -> dict[str, int]:
        """Return this rank's model load, parameter, buffer and allocated bytes."""
        model = self.model_runner.model
        return {
            "model_load_bytes": int(self.model_runner.model_memory_usage),
            "parameter_bytes": sum(
                p.numel() * p.element_size() for p in model.parameters()
            ),
            "registered_buffer_bytes": sum(
                b.numel() * b.element_size() for b in model.buffers()
            ),
            "allocated_after_startup_bytes": torch.cuda.memory_allocated(),
        }


def load_documents(
    args: argparse.Namespace, input_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter[str]]:
    """Return the panel, its prompt/target documents and documents per source.

    Token panels give `--documents-per-source` documents per source, each cut
    into a prefix of at most `--prefix-tokens` and `--decode-tokens` targets.
    Request files are used as prepared.
    """
    panel = json.loads(input_path.read_text())
    counts: Counter[str] = Counter()
    documents = []
    for document in panel["requests"] if args.requests else panel["documents"]:
        source = document["source"]
        if args.requests:
            documents.append(
                {key: document[key] for key in ("id", "source", "prompt", "target")}
                # Trace windows: answer tokens already in the prompt.
                | {"answer_offset": document.get("depth", 0)}
            )
            counts[source] += 1
            continue
        if counts[source] >= args.documents_per_source:
            continue
        tokens = document["tokens"]
        prefix = min(args.prefix_tokens, len(tokens) - args.decode_tokens)
        if prefix < 1:
            raise ValueError(
                "A selected document has insufficient prefix/continuation tokens"
            )
        documents.append(
            {
                "id": document["id"],
                "source": source,
                "prompt": tokens[:prefix],
                "target": tokens[prefix : prefix + args.decode_tokens],
                "answer_offset": 0,
            }
        )
        counts[source] += 1
    if not documents or (
        not args.requests
        and any(count != args.documents_per_source for count in counts.values())
    ):
        raise ValueError("Incomplete source-balanced cohort")
    # Request panels without a recorded context length were prepared for 2048.
    max_model_len = (
        panel["data"].get("max_model_len") if args.requests else None
    ) or 2048
    if any(
        not d["prompt"]
        or not d["target"]
        or len(d["prompt"]) + len(d["target"]) > max_model_len
        for d in documents
    ):
        raise ValueError(
            "Prepared continuations require nonempty prompt/target within max_model_len"
        )
    panel["max_model_len"] = max_model_len
    return panel, documents, counts


def prepare_fidelity(args: argparse.Namespace, documents: list[dict[str, Any]]) -> None:
    """Key each document by panel order and bind the reference to its tokens.

    A comparison is defined only against a stock run that teacher-forced the same
    prompts and targets, so the reference records them and a comparison requires
    them unchanged.
    """
    for index, document in enumerate(documents):
        document["fidelity_key"] = f"{index:05d}"
    tokens = [
        {key: d[key] for key in ("fidelity_key", "id", "prompt", "target")}
        for d in documents
    ]
    if args.fidelity_record:
        args.fidelity_record.mkdir(parents=True, exist_ok=False)
        (args.fidelity_record / "documents.json").write_text(json.dumps(tokens))
    elif json.loads((args.fidelity_reference / "documents.json").read_text()) != tokens:
        raise ValueError("The fidelity reference used different documents or tokens")


def fidelity_summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Return micro (all decode positions) and source-macro KL and top-1 means.

    The source-macro means average per-document means within each source, as
    `macro_decode_nll` does.
    """
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        by_source[document["source"]].append(document)
    summary: dict[str, Any] = {
        "divergence": "KL(P_stock || P_policy) over the full vocabulary in float32 "
        "from raw logits; top-1 agreement is argmax equality",
        "positions": "decode steps only; the shared stock-prefill step is excluded, "
        "as in decode NLL",
        "scored_positions": sum(len(d["kl"]) for d in documents),
        "max_prefill_step_kl": max(d["prefill_step_kl"] for d in documents),
    }
    for field, per_document in (("kl", "mean_kl"), ("top1", "top1_agreement")):
        values = [float(x) for d in documents for x in d[field]]
        sources = {
            source: mean(d[per_document] for d in group)
            for source, group in by_source.items()
        }
        summary[f"micro_{per_document}"] = mean(values)
        summary[f"macro_{per_document}"] = mean(sources.values())
        summary[f"source_{per_document}"] = sources
    return summary


def run_requests(
    llm: LLM, prompts: list[list[int]], params: list[SamplingParams]
) -> list[RequestOutput]:
    """Run one batch of requests until all finish and return their outputs."""
    # Queue the whole batch before scheduling so vLLM batches it FCFS.
    llm.sleep(level=0, mode="keep")
    llm.enqueue(
        [{"prompt_token_ids": prompt} for prompt in prompts], params, use_tqdm=False
    )
    llm.wake_up(tags=["scheduling"])
    # Pinned vLLM sorts finished outputs by external numeric request ID;
    # enqueue returns different internal IDs, not RequestOutput keys.
    return llm.wait_for_completion(output_type=RequestOutput, use_tqdm=False)


def target_logprob(scores: dict[int, Any] | None, token: int) -> float:
    """Return the finite raw log probability of `token` from one position's scores."""
    if scores is None or token not in scores:
        raise RuntimeError("Missing target raw log probability")
    value = scores[token].logprob
    if not math.isfinite(value):
        raise RuntimeError("Non-finite target log probability")
    return value


def sampled_scores(output: RequestOutput) -> dict[int, Any]:
    """Return the log probabilities of a request's first generated position."""
    logprobs = output.outputs[0].logprobs
    if logprobs is None or logprobs[0] is None:
        raise RuntimeError("Missing prefill target log probabilities")
    return logprobs[0]


def score_shared_contexts(
    llm: LLM, documents: list[dict[str, Any]], batch_size: int
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Score one-prefill likelihoods, computing each shared context once.

    Choices of one question share their context. One request prefills the context
    and reads every choice's first target from its next-token distribution; each
    multi-token choice then prefills context plus target, reusing cached KV, and
    reads later targets from prompt log probabilities. Under a token-local policy a
    cached position's KV depends only on its prefix, so this equals scoring every
    choice from its own full prefill. vLLM returns prompt log probabilities only
    for computed positions: when a choice's block-aligned cache hit (from a sibling
    with the same leading target tokens) covers scored positions, a probe request
    ending at the hit scores them from its own, shorter hit.
    """
    groups: list[list[dict[str, Any]]] = []
    by_context: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for d in documents:
        group = by_context.get(key := tuple(d["prompt"]))
        # vLLM returns at most 128 requested token IDs per request.
        if group is None or len({x["target"][0] for x in [*group, d]}) > 128:
            group = by_context[key] = []
            groups.append(group)
        group.append(d)
    # A batch's contexts must stay cached until its choices run: bound its new KV
    # by half the cache (hybrid layouts hold fewer tokens; eviction only costs time).
    cache = llm.llm_engine.vllm_config.cache_config
    budget = (cache.num_gpu_blocks or 0) * cache.block_size // 2
    batch: list[list[dict[str, Any]]] = []
    batches: list[list[list[dict[str, Any]]]] = [batch]
    size = 0
    for group in groups:
        tokens = len(group[0]["prompt"]) + sum(len(d["target"]) for d in group)
        if batch and (size + tokens > budget or len(batch) == batch_size):
            batch = []
            batches.append(batch)
            size = 0
        batch.append(group)
        size += tokens
    # Token index -> log probability of that token given every earlier token.
    known: dict[str, dict[int, float]] = {d["id"]: {} for d in documents}
    reuse = {"requests": 0, "computed_tokens": 0, "cached_tokens": 0, "probes": 0}
    for batch in batches:
        # A group request scores the first targets from its context. A choice
        # request (d, n, s) prefills d's first n tokens and scores tokens s..n.
        # Contexts run first and alone: vLLM computes prompt log probabilities
        # over every token of a step that contains any requesting prompt.
        rounds: list[list[tuple[Any, int, int]]] = [
            [
                (group, len(group[0]["prompt"]), len(group[0]["prompt"]))
                for group in batch
            ],
            [
                (d, len(d["prompt"]) + len(d["target"]) - 1, len(d["prompt"]) + 1)
                for group in batch
                for d in group
                if len(d["target"]) > 1
            ],
        ]
        while rounds:
            pending = rounds.pop(0)
            if not pending:
                continue
            prompts = []
            params = []
            for item, n, s in pending:
                if isinstance(item, list):
                    prompts.append(item[0]["prompt"])
                    tokens = sorted({d["target"][0] for d in item})
                    params.append(
                        SamplingParams(
                            temperature=0, max_tokens=1, logprob_token_ids=tokens
                        )
                    )
                    continue
                full = item["prompt"] + item["target"]
                prompts.append(full[:n])
                params.append(
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        logprob_token_ids=[full[n]],
                        prompt_logprobs=0 if s < n else None,
                        skip_reading_prefix_cache=False,
                    )
                )
            outputs = run_requests(llm, prompts, params)
            probes = []
            for (item, n, s), output in zip(pending, outputs, strict=True):
                cached = output.num_cached_tokens or 0
                reuse["requests"] += 1
                reuse["computed_tokens"] += n - cached
                reuse["cached_tokens"] += cached
                scores = sampled_scores(output)
                if isinstance(item, list):
                    for d in item:
                        known[d["id"]][n] = target_logprob(scores, d["target"][0])
                    continue
                full = item["prompt"] + item["target"]
                found = known[item["id"]]
                found[n] = target_logprob(scores, full[n])
                if s < n:
                    # prompt_logprobs[j] scores token cached + j (j >= 1).
                    prompt_scores = output.prompt_logprobs
                    if prompt_scores is None:
                        raise RuntimeError("Missing prefill target log probabilities")
                    for i in range(max(s, cached + 1), n):
                        found[i] = target_logprob(prompt_scores[i - cached], full[i])
                if cached >= s:
                    probes.append((item, cached, s))
            reuse["probes"] += len(probes)
            rounds.append(probes)
    scored = {}
    for d in documents:
        context = len(d["prompt"])
        values = known[d["id"]]
        if sorted(values) != list(range(context, context + len(d["target"]))):
            raise RuntimeError("Shared-context scoring missed a target position")
        scored[d["id"]] = [values[i] for i in sorted(values)]
    return scored, reuse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--panel", type=Path)
    inputs.add_argument(
        "--requests",
        type=Path,
        nargs="+",
        help="Prepared prompt/target requests; several panels share one engine and "
        "each writes its own --output report",
    )
    parser.add_argument(
        "--all-token",
        action="store_true",
        help="Prefill only the first prompt token; replay the rest and score targets",
    )
    parser.add_argument(
        "--dense-prefix-fraction",
        type=float,
        default=0.0,
        help="With --all-token, stock-prefill this leading fraction of each request "
        "(TEAL: 0.5); every target prediction stays in replay",
    )
    parser.add_argument(
        "--dense-prefix-tokens",
        type=int,
        default=1,
        help="With --all-token, stock-prefill at least this many leading tokens",
    )
    parser.add_argument(
        "--sparse-prefill",
        action="store_true",
        help="Run the policy on prompt tokens too and score each request from one "
        "prefill over prompt+target, as lm-eval's loglikelihood does (with "
        "--all-token: the replay control with a sparse one-token lead)",
    )
    parser.add_argument(
        "--prefix-reuse",
        action="store_true",
        help="Use vLLM's prefix cache: with --sparse-prefill, prefill each shared "
        "context once for all its choices (--batch-size caps contexts; not Qwen3.5); "
        "with stock-prefill decode, run longest prompts first so reused KV is stock",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Scheduler token budget for --sparse-prefill scoring; vLLM profiles the "
        "baselines' scratch at this size, so larger budgets shrink the KV cache "
        "(16384 leaves Prox none on a 24 GB card)",
    )
    parser.add_argument(
        "--parity-check",
        action="store_true",
        help="Compare first batch to stock full-prefill prompt logprobs (stock only)",
    )
    parser.add_argument("--bank")
    parser.add_argument(
        "--policy",
        choices=(
            "progressive",
            "cats",
            "teal",
            "teal-greedy",
            "wina",
            "wina-greedy-topk",
            "prox",
            *CALIBRATION_FREE_POLICIES,
        ),
        default="progressive",
    )
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument(
        "--prox-recipe",
        choices=("legacy", "study"),
        default="legacy",
        help="Prox calibration recipe and top-k tie convention",
    )
    parser.add_argument("--retention", type=float)
    parser.add_argument(
        "--layer-retention", help="JSON exact-channel allocation per layer"
    )
    parser.add_argument(
        "--input-keep",
        type=float,
        help="prox/prox-free: per-token input top-k fraction instead of a threshold",
    )
    parser.add_argument(
        "--channel-keep",
        type=float,
        help="prox: per-token joint channel top-k fraction instead of a threshold",
    )
    parser.add_argument(
        "--channel-floor",
        type=float,
        help="prox: minimum fraction of K*I channels under a calibrated threshold",
    )
    parser.add_argument(
        "--proxy-score",
        choices=("prox", "output-unit", "exact"),
        default="prox",
        help="Prox score: |proxy|, output-unit |w| n |proxy| (router weight w, "
        "down-column norm n; prox-free or study prox with --input-keep), or exact "
        "|w| n |a| from native rows (analysis with --allocation; --thresholds is "
        "then its JSON)",
    )
    allocation.add_arguments(parser)
    parser.add_argument(
        "--selection-score",
        choices=("recovery-aware", "point"),
        default="recovery-aware",
        help="Reader ablation: point ignores the sketch residual variances",
    )
    parser.add_argument(
        "--screen-keep",
        type=float,
        default=1.0,
        help="Fraction of each 128-coordinate rotated input block the screen reads "
        "(largest magnitudes, rounded up)",
    )
    parser.add_argument("--moe-backend", choices=("auto", "triton"), default="auto")
    parser.add_argument("--documents-per-source", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        help="Fixed KV cache size (default: from --gpu-memory-utilization); a small "
        "cache forces preemption",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    fidelity = parser.add_mutually_exclusive_group()
    fidelity.add_argument(
        "--fidelity-record",
        type=Path,
        help="Stock only: store each decode step's raw logits in this directory "
        "as the fidelity reference for the same panel",
    )
    fidelity.add_argument(
        "--fidelity-reference",
        type=Path,
        help="Report exact full-vocabulary KL(stock || policy) and top-1 agreement "
        "against a --fidelity-record directory of the same panel",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--output", type=Path, nargs="+", required=True, help="One report per panel"
    )
    args = parser.parse_args()
    inputs_paths: list[Path] = args.requests or [args.panel]
    if len(args.output) != len(inputs_paths) or len(set(args.output)) != len(
        args.output
    ):
        parser.error("Give one distinct --output per input panel")
    if args.batch_size < 1 or args.max_num_seqs < args.batch_size:
        parser.error("Require 1 <= --batch-size <= --max-num-seqs")
    if (
        not 0 <= args.dense_prefix_fraction < 1
        or args.dense_prefix_tokens < 1
        or (
            (args.dense_prefix_fraction or args.dense_prefix_tokens > 1)
            and not args.all_token
        )
    ):
        parser.error(
            "Dense-prefix options need --all-token, fraction [0,1), tokens >= 1"
        )
    if args.sparse_prefill and (
        args.dense_prefix_fraction or args.dense_prefix_tokens > 1 or args.parity_check
    ):
        parser.error("--sparse-prefill has no dense prefix or parity control")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.parity_check and (
        not args.all_token
        or args.retention is not None
        or args.layer_retention
        or args.thresholds
    ):
        parser.error("--parity-check requires stock --all-token")
    # One prefill over prompt+target scores every target (no replay).
    scoring = args.sparse_prefill and not args.all_token
    if args.prefix_reuse and args.all_token:
        parser.error("--prefix-reuse needs one prefill per request (no --all-token)")
    sparse_policy = (
        args.retention is not None
        or args.layer_retention is not None
        or args.thresholds is not None
        or args.allocation is not None
    )
    if min(args.documents_per_source, args.prefix_tokens) < 1 or args.decode_tokens < 2:
        parser.error(
            "Positive cohort/prefix and at least two continuation tokens required"
        )
    if args.layer_retention and args.policy != "progressive":
        parser.error("--layer-retention requires progressive")
    if args.policy == "progressive":
        # --retention alone is the training-free reader (no recovery bank).
        if (
            args.bank is not None
            and args.retention is None
            and not args.layer_retention
        ) or args.thresholds is not None:
            parser.error(
                "Progressive sparse decode requires --retention or --layer-retention "
                "(optionally --bank), "
                "not --thresholds"
            )
    elif args.policy in CALIBRATION_FREE_POLICIES:
        if args.retention is None or args.bank is not None or args.thresholds:
            parser.error(
                "Calibration-free baselines take --retention (prox-free also "
                "--input-keep and --proxy-score) only"
            )
    elif args.allocation is not None:
        # Prox with an allocation: validate_allocation_options checks the rest.
        pass
    elif args.thresholds is None or args.bank is not None or args.retention is not None:
        parser.error("Baseline decode requires --thresholds, not --bank/--retention")
    if args.allocation is not None and (
        args.sparse_prefill or args.all_token or args.prefix_reuse
    ):
        parser.error("Allocation runs score stock-prefill decode steps")
    if args.allocation_record and (
        args.allocation is None or args.allocation_record.exists()
    ):
        parser.error("--allocation-record is a new directory of an allocation run")
    # The worker binding's options (additional_config), validated as bound.
    options: dict[str, object] = {
        "bank": args.bank,
        "retention": args.retention,
        "layer_retention": args.layer_retention,
        "input_keep": args.input_keep,
        "channel_keep": args.channel_keep,
        "channel_floor": args.channel_floor,
        "prox_recipe": args.prox_recipe,
        "proxy_score": args.proxy_score,
        "policy": args.policy,
        "baseline": str(args.thresholds) if args.thresholds else None,
        "sparse_prefill": args.sparse_prefill,
        "selection_score": args.selection_score,
        "screen_keep": args.screen_keep,
        **allocation.cli_options(args),
    }
    try:
        validate_prox_keep_options(
            args.policy,
            args.input_keep,
            args.channel_keep,
            args.channel_floor,
            args.prox_recipe,
            cast(str, args.proxy_score),
        )
        allocation.validate_allocation_options(options)
    except ValueError as error:
        parser.error(str(error))
    if (
        (args.selection_score != "recovery-aware" or args.screen_keep != 1)
        and args.retention is None
        and args.layer_retention is None
    ):
        parser.error("Selection options apply to the progressive reader")
    if (
        args.tensor_parallel_size > 1
        and (
            args.retention is not None
            or args.layer_retention is not None
            or args.thresholds
            or args.policy != "progressive"
        )
        and not args.enable_expert_parallel
    ):
        parser.error("Sparse TP requires --enable-expert-parallel")
    fidelity_mode = (
        "record"
        if args.fidelity_record
        else "compare"
        if args.fidelity_reference
        else None
    )
    if fidelity_mode and (args.all_token or args.sparse_prefill or args.parity_check):
        # Several --requests panels share one reference keyed across all of them;
        # each panel's report carries its own fidelity summary and read statistics.
        parser.error("Fidelity scores stock-prefill decode steps")
    if args.fidelity_record and sparse_policy:
        parser.error("--fidelity-record stores the stock reference")
    architecture = model_dimensions(Path(args.model))["architecture"]
    if args.prefix_reuse and architecture == "Qwen3_5MoeForConditionalGeneration":
        # Gated-delta-net prefix caching is unvalidated here (see --sparse-prefill).
        parser.error("--prefix-reuse is not supported for Qwen3.5")
    panels = [
        (input_path, output, *load_documents(args, input_path))
        for input_path, output in zip(inputs_paths, args.output, strict=True)
    ]
    if args.prefix_reuse and not scoring:
        # A cache hit covers only blocks that lie wholly inside a later request's
        # prompt. Under nonincreasing prompt length, every position a policy decode
        # step wrote (at or beyond its own prompt length) lies beyond each later
        # prompt, so reused KV is always stock-prefilled. The cache resets between
        # panels; a preempted request, recomputed after a shorter one decoded, would
        # break the order, so preemption fails the run.
        for *_, documents, _ in panels:
            documents.sort(key=lambda d: -len(d["prompt"]))
    every = [d for *_, documents, _ in panels for d in documents]
    if fidelity_mode:
        prepare_fidelity(args, every)
    if args.allocation is not None:
        for input_path, _, _, documents, _ in panels:
            for document in documents:
                document["allocation_key"] = allocation.request_key(
                    input_path, document["id"]
                )
    config: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "max_model_len": max(panel["max_model_len"] for _, _, panel, *_ in panels),
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": 353,
        "enable_prefix_caching": args.prefix_reuse,
        "enable_trace_replay": True,
        # The preemption counter needs stats; a preempted replay recomputes its
        # replayed tokens with the policy (exact) and its stock lead with stock.
        "disable_log_stats": False,
        "logprobs_mode": "raw_logprobs",
        "worker_extension_cls": (
            "scripts.eval_vllm_continuation.ContinuationWorkerExtension"
        ),
    }
    if args.requests and not args.all_token:
        # Keep first-token controls on the same full prefill cohort, even when
        # few-shot prompts are longer than the NLL panel's fixed short prefixes.
        config["max_num_batched_tokens"] = max(
            2048,
            max(
                sum(
                    len(d["prompt"])
                    for d in documents[offset : offset + args.batch_size]
                )
                for *_, documents, _ in panels
                for offset in range(0, len(documents), args.batch_size)
            ),
        )
    if args.sparse_prefill:
        # Many whole prefills per step. The policy also runs on vLLM's profiling
        # batch, so keep this budget bounded. No chunking: vLLM 0.29 classifies a
        # new request's one-token first chunk as decode, which on gated-delta-net
        # layers (Qwen3.5) reads a stale state slot instead of a zero state.
        # Unchunked scheduling needs a budget of at least max_model_len, which only
        # sizes memory, so it shrinks to the longest request.
        longest = max(len(d["prompt"]) + len(d["target"]) for d in every)
        config["enable_chunked_prefill"] = False
        config["max_model_len"] = longest
        config["max_num_batched_tokens"] = max(args.max_num_batched_tokens, longest)
    elif args.all_token:
        config["max_num_batched_tokens"] = max(
            2048,
            max(
                sum(
                    len(d["prompt"]) + len(d["target"])
                    for d in documents[i : i + args.batch_size]
                )
                for *_, documents, _ in panels
                for i in range(0, len(documents), args.batch_size)
            ),
        )
    if args.moe_backend != "auto":
        config["moe_backend"] = args.moe_backend
    if args.kv_cache_memory_bytes is not None:
        config["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if architecture.startswith("Gemma4"):
        config["max_num_batched_tokens"] = max(config["max_num_batched_tokens"], 3072)
        config["max_model_len"] = max(
            config["max_model_len"],
            math.ceil(config["max_num_batched_tokens"] / args.max_num_seqs),
        )
    if architecture == "Qwen3_5MoeForConditionalGeneration":
        config["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        # The same misclassification corrupts every one-token replay lead after the
        # first batch (stock included), so all-token replay needs a longer lead.
        if (
            args.all_token
            and not args.dense_prefix_fraction
            and args.dense_prefix_tokens < 2
        ):
            raise ValueError(
                "Qwen3.5 all-token replay needs a lead of at least 2 tokens"
            )
    if sparse_policy:
        config.update(
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            model_class_overrides=model_override(args.model),
            additional_config={"tokenwise_phase_metadata": True, "tokenwise": options},
        )
    reports: list[dict[str, Any]] = []
    for input_path, output, panel, _, counts in panels:
        reports.append(
            {
                "complete": False,
                "scope": (
                    "Teacher-forced native likelihoods, not free-running generation. "
                    "Each policy owns its KV. Hybrid scores first target with stock "
                    "prefill; all-token replays the prompt after its stock-prefilled "
                    "lead (first token, or the dense-prefix options). Sparse-prefill "
                    "(evaluation only) runs the policy on every token, prefill "
                    "included, and scores all targets from one prefill (its routed "
                    "read statistics cover prefill and decode tokens). Prefix reuse "
                    "computes a shared context once per engine; read statistics cover "
                    "computed tokens. Reducers sum all target logprobs."
                ),
                "command": sys.argv,
                "revision": args.revision,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
                "vllm_tuned_config_folder": os.environ.get("VLLM_TUNED_CONFIG_FOLDER"),
                "config": config,
                "batch_size": args.batch_size,
                "panel": str(input_path),
                "engine_panels": [str(path) for path in inputs_paths],
                "prepared_requests": bool(args.requests),
                "all_token": args.all_token,
                "sparse_prefill": args.sparse_prefill,
                "prefix_reuse": args.prefix_reuse,
                "dense_prefix_fraction": args.dense_prefix_fraction,
                "dense_prefix_tokens": args.dense_prefix_tokens,
                "fidelity_record": str(args.fidelity_record)
                if args.fidelity_record
                else None,
                "fidelity_reference": str(args.fidelity_reference)
                if args.fidelity_reference
                else None,
                "allocation_record": str(args.allocation_record)
                if args.allocation_record
                else None,
                "panel_role": panel["role"],
                "panel_data": panel["data"],
                "source_counts": dict(counts),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("vllm", "torch", "triton")
                },
                "documents": [],
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports[-1], indent=2) + "\n")
    start = time.perf_counter()
    current = 0
    try:
        llm = LLM(**config)
        memory = llm.collective_rpc("tokenwise_model_memory")
        if args.allocation is not None:
            # Before the fidelity sampler, which must stay outermost.
            recording = llm.collective_rpc(
                "tokenwise_allocation_start",
                args=(
                    str(args.allocation_record) if args.allocation_record else None,
                    max(len(d["target"]) for d in every),
                ),
            )
            if args.allocation_record and sum(recording) != 1:
                raise RuntimeError("Allocation records must come from exactly one rank")
        if fidelity_mode:
            reference = args.fidelity_record or args.fidelity_reference
            installed = llm.collective_rpc(
                "tokenwise_fidelity_start",
                args=(
                    fidelity_mode,
                    str(reference),
                    max(len(d["target"]) for d in every),
                ),
            )
            if sum(installed) != 1:
                raise RuntimeError("Fidelity sampler must run on exactly one rank")
        startup = time.perf_counter() - start
        for current, (_, output, _, documents, _) in enumerate(panels):
            report = reports[current]
            # Parallel runs keep every rank's residency; TP1 keeps its single record.
            report["loaded_model_memory"] = memory[0] if len(memory) == 1 else memory
            report["engine_startup_seconds"] = startup
            if current:
                # Panels share no prompts; each report owns its computed tokens.
                llm.reset_prefix_cache()
            evaluate_panel(llm, args, documents, report, output, scoring, sparse_policy)
            report["engine_preemptions_so_far"] = int(
                sum(
                    getattr(metric, "value", 0)
                    for metric in llm.get_metrics()
                    if metric.name == "vllm:num_preemptions"
                )
            )
            if (
                args.prefix_reuse
                and not scoring
                and report["engine_preemptions_so_far"]
            ):
                raise RuntimeError("Preemption under --prefix-reuse decode")
            report["elapsed_seconds"] = time.perf_counter() - start
            report["complete"] = True
            output.write_text(json.dumps(report, indent=2) + "\n")
        if args.allocation is not None:
            llm.collective_rpc("tokenwise_allocation_finish")
    except BaseException as error:
        for report, (_, output, *_) in zip(
            reports[current:], panels[current:], strict=True
        ):
            report["error"] = repr(error)
            report["elapsed_seconds"] = time.perf_counter() - start
            output.write_text(json.dumps(report, indent=2) + "\n")
        raise


def evaluate_panel(
    llm: LLM,
    args: argparse.Namespace,
    documents: list[dict[str, Any]],
    report: dict[str, Any],
    output_path: Path,
    scoring: bool,
    sparse_policy: bool,
) -> None:
    """Score one panel's documents; fill its report with NLLs and read statistics."""
    if args.policy != "progressive":
        llm.collective_rpc("tokenwise_baseline_reset_work")
    elif args.retention is not None or args.layer_retention is not None:
        llm.collective_rpc("tokenwise_reader_reset_work")
    evaluation_start = time.perf_counter()
    # Whether stock prefill (not the policy) predicted target 0; stock scoring
    # predicts every target in prefill.
    if args.prefix_reuse and scoring:
        scored, reuse = score_shared_contexts(llm, documents, args.batch_size)
        prefill_target = not sparse_policy
        for document in documents:
            append_document(
                report, document, scored[document["id"]], prefill_target, sparse_policy
            )
        prefill_tokens = reuse["computed_tokens"]
        decode_inputs = 0
        report["prefix_reuse_tokens"] = reuse
        report["replayed_tokens"] = 0
        report["replayed_requests"] = reuse["requests"]
    else:
        prefill_tokens, decode_inputs, units = score_units(
            llm, args, documents, report, output_path, scoring, sparse_policy
        )
        report["replayed_tokens"] = 0 if scoring else decode_inputs + units
        report["replayed_requests"] = units
    grouped: dict[str, list[float]] = defaultdict(list)
    for document in report["documents"]:
        if document["decode_nll"] is not None:
            grouped[document["source"]].append(document["decode_nll"])
    report["source_decode_nll"] = {
        source: mean(values) for source, values in grouped.items()
    }
    report["macro_decode_nll"] = (
        mean(report["source_decode_nll"].values()) if grouped else None
    )
    if args.fidelity_reference:
        report["fidelity"] = fidelity_summary(report["documents"])
    if args.policy != "progressive":
        report["baseline_work"] = aggregate_work_reports(
            llm.collective_rpc("tokenwise_baseline_work_report")
        )
        if report["baseline_work"]["read_fraction"] is None:
            raise RuntimeError("No measured baseline decode reads")
    elif args.retention is not None or args.layer_retention is not None:
        reader_work = aggregate_work_reports(
            llm.collective_rpc("tokenwise_reader_work_report")
        )
        if reader_work["read_fraction"] is not None:
            report["reader_work"] = reader_work
    report["evaluation_seconds"] = time.perf_counter() - evaluation_start
    # Computed forward tokens (prefix-cache hits excluded) against scored targets.
    report["prefill_tokens"] = prefill_tokens
    report["decode_input_tokens"] = decode_inputs
    report["scored_tokens"] = sum(len(d["target"]) for d in documents)
    report["forward_tokens_per_second"] = (prefill_tokens + decode_inputs) / report[
        "evaluation_seconds"
    ]
    dimensions = model_dimensions(Path(args.model))
    if dimensions["architecture"] == "Qwen3_5MoeForConditionalGeneration":
        report["shared_expert_work"] = shared_expert_work_report(
            dimensions, prefill_tokens, decode_inputs, args.tensor_parallel_size
        )
    elif dimensions["architecture"].startswith("Gemma4"):
        # Gemma executes its native dense MLP at every layer in addition to MoE.
        dense_work = dense_mlp_work_report(dimensions, prefill_tokens, decode_inputs)
        report["dense_mlp_work"] = dense_work
        routed = report.get("reader_work", report.get("baseline_work"))
        if routed is not None:
            # The routed read statistics cover every token the policy ran on.
            dense_bytes = dense_work["decode_logical_weight_bytes"] + (
                dense_work["prefill_logical_weight_bytes"] if args.sparse_prefill else 0
            )
            report["overall_logical_read_fraction"] = (
                routed["charged_bytes"] + dense_bytes
            ) / (routed["reference_bytes"] + dense_bytes)
    report["replayed_tokens_per_second"] = (
        report["replayed_tokens"] / report["evaluation_seconds"]
    )
    report["requests_per_second"] = len(documents) / report["evaluation_seconds"]


def append_document(
    report: dict[str, Any],
    document: dict[str, Any],
    logprobs: list[float],
    prefill_target: bool,
    sparse_policy: bool,
) -> None:
    """Append one document's target log probabilities and NLLs to the report.

    `prefill_target` says whether stock prefill predicted target 0; that target
    then counts as prefill NLL and the rest as decode NLL.
    """
    report["documents"].append(
        {
            **document,
            "logprobs": logprobs,
            "prefill_target_nll": -logprobs[0] if prefill_target else None,
            "decode_nll": -mean(logprobs[int(prefill_target) :])
            if len(logprobs) > int(prefill_target)
            else None,
            "sparse_scored_tokens": len(logprobs) - int(prefill_target)
            if sparse_policy
            else 0,
        }
    )


def score_units(
    llm: LLM,
    args: argparse.Namespace,
    documents: list[dict[str, Any]],
    report: dict[str, Any],
    output_path: Path,
    scoring: bool,
    sparse_policy: bool,
) -> tuple[int, int, int]:
    """Score every request from its own prefill or trace replay."""
    # All-token requests that share a prompt and have one-token targets (MC
    # letters, yes/no) replay the prompt once: every candidate's log probability
    # comes from the same final-position distribution.
    units: list[list[dict[str, Any]]] = []
    shared: dict[tuple[int, ...], list[dict[str, Any]]] = {}

    def dense(d: dict[str, Any]) -> int:
        # Stock-prefilled prompt tokens: at least the first, never the position
        # that predicts the first target, so every scored prediction is replayed.
        total = len(d["prompt"]) + len(d["target"])
        lead = max(args.dense_prefix_tokens, int(args.dense_prefix_fraction * total))
        return max(1, min(lead, len(d["prompt"]) - 1))

    def prefilled(d: dict[str, Any]) -> int:
        # Scoring prefills every input; the final target is never consumed.
        if scoring:
            return len(d["prompt"]) + len(d["target"]) - 1
        return dense(d) if args.all_token else len(d["prompt"])

    for d in documents:
        if (
            (args.all_token or scoring)
            and not args.parity_check
            and len(d["target"]) == 1
        ):
            unit = shared.get(key := tuple(d["prompt"]))
            # vLLM returns at most 128 requested token IDs per request.
            if unit is None or len({x["target"][0] for x in [*unit, d]}) > 128:
                unit = shared[key] = []
                units.append(unit)
            unit.append(d)
        else:
            units.append([d])
    written = time.perf_counter()
    cached = control_prefill = 0
    for offset in range(0, len(units), args.batch_size):
        unit_batch = units[offset : offset + args.batch_size]
        batch = [unit[0] for unit in unit_batch]
        replay = [(d["prompt"] + d["target"])[prefilled(d) :] for d in batch]
        # The first batch is checked against unforced controls, each a full
        # recompute; it bypasses the prefix cache too, so both run one path.
        checked = offset < 8 and not args.all_token and not scoring
        # The forced token's log probability is always returned; the requested
        # IDs add the other candidates. Scoring reads earlier targets from
        # prompt log probabilities and the final one from the next-token step.
        params = [
            SamplingParams(
                temperature=0,
                max_tokens=1,
                logprob_token_ids=sorted({d["target"][-1] for d in unit}),
                prompt_logprobs=0 if len(unit[0]["target"]) > 1 else None,
            )
            if scoring
            else SamplingParams(
                temperature=0,
                max_tokens=len(tokens),
                trace_decode_token_ids=tokens,
                logprob_token_ids=sorted({d["target"][0] for d in unit}),
                ignore_eos=True,
                skip_reading_prefix_cache=True if checked else None,
                extra_args=(
                    {FIDELITY_KEY: unit[0]["fidelity_key"]}
                    if "fidelity_key" in unit[0]
                    else {}
                )
                | (
                    {
                        allocation.KEY: unit[0]["allocation_key"],
                        allocation.OFFSET: unit[0]["answer_offset"],
                    }
                    if "allocation_key" in unit[0]
                    else {}
                )
                or None,
            )
            for tokens, unit in zip(replay, unit_batch, strict=True)
        ]
        outputs = run_requests(
            llm, [(d["prompt"] + d["target"])[: prefilled(d)] for d in batch], params
        )
        fidelity = (
            next(
                r
                for r in llm.collective_rpc("tokenwise_fidelity_collect")
                if r is not None
            )
            if "fidelity_key" in batch[0]
            else None
        )
        if args.allocation_record:
            recorded = next(
                r
                for r in llm.collective_rpc("tokenwise_allocation_collect")
                if r is not None
            )
            # Every decode step of each window: steps 1..T-1 (0 is the stock prefill).
            if recorded != {d["allocation_key"]: len(d["target"]) - 1 for d in batch}:
                raise RuntimeError("An allocation record missed a decode step")
        for unit, tokens, output in zip(unit_batch, replay, outputs, strict=True):
            cached += output.num_cached_tokens or 0
            result = output.outputs[0]
            head = unit[0]
            if scoring:
                prompt_scores = output.prompt_logprobs
                if result.logprobs is None or (
                    len(head["target"]) > 1 and prompt_scores is None
                ):
                    raise RuntimeError("Missing prefill target log probabilities")
                steps = [
                    *(prompt_scores or [])[len(head["prompt"]) :],
                    result.logprobs[0],
                ]
                boundary = None
            else:
                if list(result.token_ids) != tokens or result.logprobs is None:
                    raise RuntimeError(
                        "Native trace replay did not produce the requested continuation"
                    )
                boundary = len(head["prompt"]) - prefilled(head)
                steps = result.logprobs[boundary:]
            # Whether stock prefill (not the policy) predicted target 0; stock
            # scoring predicts every target in prefill.
            prefill_target = (scoring or boundary == 0) and not (
                args.sparse_prefill and sparse_policy
            )
            for document in unit:
                logprobs = [
                    target_logprob(scores, token)
                    for token, scores in zip(document["target"], steps, strict=True)
                ]
                append_document(
                    report, document, logprobs, prefill_target, sparse_policy
                )
                if fidelity is not None and args.fidelity_reference:
                    # Step 0 is the shared stock prefill; decode steps follow.
                    compared = fidelity[document["fidelity_key"]]
                    report["documents"][-1].update(
                        prefill_step_kl=compared["kl"][0],
                        kl=compared["kl"][1:],
                        top1=compared["top1"][1:],
                        mean_kl=mean(compared["kl"][1:]),
                        top1_agreement=mean(map(float, compared["top1"][1:])),
                    )
        if checked:
            # Independently query the same first target without forcing it.
            # This protects raw-distribution scoring, including rare targets.
            controls = [
                SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    logprob_token_ids=[d["target"][0]],
                    skip_reading_prefix_cache=True,
                )
                for d in batch
            ]
            control_outputs = run_requests(llm, [d["prompt"] for d in batch], controls)
            control_prefill += sum(len(d["prompt"]) for d in batch)
            for document, output in zip(
                report["documents"][-len(batch) :], control_outputs, strict=True
            ):
                scores = output.outputs[0].logprobs
                if scores is None or scores[0] is None:
                    raise RuntimeError("Missing unforced raw-distribution control")
                control = scores[0][document["target"][0]].logprob
                document["prefill_raw_control_logprob"] = control
                if (
                    not math.isfinite(control)
                    or abs(control - document["logprobs"][0]) > 1e-4
                ):
                    raise RuntimeError(
                        "Trace replay changed the first target's raw log probability"
                    )
        if args.parity_check and offset == 0:
            controls = [
                SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    prompt_logprobs=1,
                    logprob_token_ids=[d["target"][-1]],
                )
                for d in batch
            ]
            llm.enqueue(
                [{"prompt_token_ids": d["prompt"] + d["target"][:-1]} for d in batch],
                controls,
                use_tqdm=False,
            )
            controls_out = llm.wait_for_completion(
                output_type=RequestOutput, use_tqdm=False
            )
            parity = []
            choice_deltas = []
            for d, scored, control in zip(
                batch, report["documents"][-len(batch) :], controls_out, strict=True
            ):
                scores = control.prompt_logprobs
                if scores is None:
                    raise RuntimeError("Missing full-prefill prompt logprobs")
                full = []
                for i, token in enumerate(d["target"][:-1], len(d["prompt"])):
                    position = scores[i]
                    if position is None or token not in position:
                        raise RuntimeError("Missing full-prefill target logprob")
                    full.append(position[token].logprob)
                generated = control.outputs[0].logprobs
                if (
                    generated is None
                    or generated[0] is None
                    or d["target"][-1] not in generated[0]
                ):
                    raise RuntimeError("Missing final full-prefill target logprob")
                last = generated[0][d["target"][-1]].logprob
                full.append(last)
                parity.extend(
                    abs(x - y) for x, y in zip(scored["logprobs"], full, strict=True)
                )
                choice_deltas.append(math.fsum(scored["logprobs"]) - math.fsum(full))
            report["parity_max_abs_logprob"] = max(parity)
            report["parity_mean_abs_logprob"] = mean(parity)
            report["parity_choice_sum_deltas"] = choice_deltas
        # Throttled progress snapshot: rewriting the growing report after every
        # batch is quadratic I/O and dominated long all-token runs.
        if time.perf_counter() - written > 300:
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            written = time.perf_counter()
    # Each unit prefills its lead (less prefix-cache hits) and decodes every later
    # input; the final target is emitted but never consumed (scoring decodes nothing).
    report["prefix_cached_tokens"] = cached
    # Unforced first-batch controls bypass the cache; their prefills are extra work.
    report["control_prefill_tokens"] = control_prefill
    prefill_tokens = sum(prefilled(u[0]) for u in units) - cached
    decode_inputs = sum(
        len(u[0]["prompt"]) + len(u[0]["target"]) - prefilled(u[0]) - 1 for u in units
    )
    return prefill_tokens, decode_inputs, len(units)


if __name__ == "__main__":
    main()
