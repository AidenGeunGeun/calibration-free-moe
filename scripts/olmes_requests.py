"""Prepare benchmark requests and reduce scored runs with pinned OLMES or lm-eval.

`prepare` writes a task's token requests (`--protocol olmes`, the default, or
`--protocol prox` for lm-eval 0.4.3 as in the Prox paper); `reduce` replays the
task's own scoring on a score report and writes per-task metrics, standard errors
and per-question results. Both run on CPU from cached datasets and tokenizers.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

OLMES_REVISION = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
# Default tokenizer: the cached OLMoE-1B-7B-0924 snapshot the OLMoE panels used.
MODEL = "allenai/OLMoE-1B-7B-0924"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"


def pinned_revision() -> str:
    """Return the installed OLMES git revision, which must be the pinned one."""
    import oe_eval

    root = Path(next(iter(oe_eval.__path__))).resolve().parent
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OLMES_REVISION:
        raise ValueError(f"Expected OLMES {OLMES_REVISION}, found {revision}")
    return revision


def encode_pair(tokenizer: Any, context: str, continuation: str):
    """Return (prompt, target) token IDs split as lm-eval's causal `_encode_pair`.

    Trailing context whitespace moves into the continuation, and the prompt keeps
    the target tokenizer's own special-token prefix.
    """
    if context == "":
        prefix = tokenizer.encode("", add_special_tokens=True)
        return (prefix or [tokenizer.eos_token_id]), tokenizer.encode(
            continuation, add_special_tokens=False
        )
    trailing = len(context) - len(context.rstrip())
    if trailing:
        continuation = context[-trailing:] + continuation
        context = context[:-trailing]
    whole = tokenizer.encode(context + continuation, add_special_tokens=True)
    prompt = tokenizer.encode(context, add_special_tokens=True)
    return prompt, whole[len(prompt) :]


PROX_TASKS = ("mmlu", "hellaswag", "arc_easy", "arc_challenge", "piqa", "gsm8k_cot")
GENERATION_TASKS = ("gsm8k_cot",)
# lm-eval 0.4.3's unqualified Hub dataset IDs no longer resolve in the
# installed datasets/HF Hub stack. The GSM8K ID redirects to its repository;
# allenai/hellaswag is a parquet copy of Rowan/hellaswag with the same schema and
# split sizes. Only the locator changes, not the task's prompts or metrics.
PROX_DATASET_LOCATORS = {
    "hellaswag": "allenai/hellaswag",
    "mmlu": "cais/mmlu",
    "piqa": "ybisk/piqa",
    "gsm8k_cot": "openai/gsm8k",
}
# lm-eval 0.4.3 evaluator defaults: bootstrap iterations for non-mean stderrs and the
# vLLM backend's generation cap when a task's generation_kwargs set no max_gen_toks.
LM_EVAL_BOOTSTRAP_ITERS = 100_000
LM_EVAL_VLLM_MAX_GEN_TOKS = 256
MMLU_CATEGORIES = ("mmlu_stem", "mmlu_other", "mmlu_social_sciences", "mmlu_humanities")


def harness_generation(gen_kwargs: dict[str, Any], eos: str):
    """Return (max tokens, stop strings, sampling) as lm-eval 0.4.3's vLLM backend.

    Follows `VLLM.generate_until` and `modify_gen_kwargs` for one request; only
    greedy generation is supported.
    """
    kwargs = copy.deepcopy(gen_kwargs)
    until = kwargs.pop("until", None)
    stop = [until] if isinstance(until, str) else list(until or [])
    stop.append(eos)
    max_tokens = kwargs.pop("max_gen_toks", LM_EVAL_VLLM_MAX_GEN_TOKS)
    if kwargs.pop("do_sample", None) is False and "temperature" not in kwargs:
        kwargs["temperature"] = 0.0
    kwargs.setdefault("skip_special_tokens", False)
    kwargs.setdefault("spaces_between_special_tokens", False)
    if kwargs.get("temperature") != 0.0 or set(kwargs) != {
        "temperature",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    }:
        raise ValueError(f"Only greedy lm-eval generation is supported: {gen_kwargs}")
    return max_tokens, stop, kwargs


def park_tasks(
    names: str,
    fewshot: int | None,
    eval_split: str | None = None,
):
    """Instantiate the exact lm-eval 0.4.3 task definitions, without a model backend.

    `names` is a comma-separated subset of the Prox tasks. `eval_split` replaces
    only the evaluated split (the task's `test_split`); the CLI allows it for MMLU,
    whose shots come from dev, so shots, prompts and metrics stay.
    """
    from lm_eval.tasks import TaskManager, get_task_dict

    if importlib.metadata.version("lm_eval") != "0.4.3":
        raise ValueError("The benchmark protocols pin lm-eval 0.4.3 task definitions")
    selected = names.split(",")
    locators = PROX_DATASET_LOCATORS
    if len(selected) != len(set(selected)) or any(
        name not in PROX_TASKS for name in selected
    ):
        raise ValueError(f"Unsupported prox task selection: {names}")
    manager = TaskManager()
    split = {} if eval_split is None else {"test_split": eval_split}
    specs: list[str | dict[str, str]] = []
    for name in selected:
        if name == "mmlu":
            # lm-eval 0.4.3's MMLU group points to an uncached dataset copy;
            # use the identical canonical cais/mmlu splits for its leaf tasks.
            leaves = sorted(
                leaf
                for leaf in manager.all_tasks
                if leaf.startswith("mmlu_")
                and "task" in manager.task_index[leaf]["type"]
                and "/mmlu/default/" in manager.task_index[leaf]["yaml_path"]
            )
            if len(leaves) != 57:
                raise ValueError(f"Expected 57 MMLU subjects, found {len(leaves)}")
            specs.extend(
                {"task": leaf, "dataset_path": locators[name], **split}
                for leaf in leaves
            )
        else:
            locator = {"dataset_path": locators[name]} if name in locators else {}
            overrides = {**locator, **split}
            specs.append({"task": name, **overrides} if overrides else name)
    tasks = cast(dict[str, Any], get_task_dict(cast(Any, specs), task_manager=manager))
    for task in tasks.values():
        # lm-eval 0.4.3 simple_evaluate: an explicit shot count overrides the task's
        # default unless that default is 0; an undefined default becomes 0.
        default = task.get_config("num_fewshot")
        if fewshot is not None and default != 0:
            task.set_config(key="num_fewshot", value=fewshot)
        elif default is None:
            task.set_config(key="num_fewshot", value=0)
        task.set_fewshot_seed(seed=1234)
    return tasks


def prepare_park(args: argparse.Namespace) -> dict[str, Any]:
    """Return the token requests of lm-eval 0.4.3 tasks (`prepare --protocol prox`).

    Likelihood tasks give one prompt/target request per answer choice; generation
    tasks give one prompt with its stop strings and token cap.
    """
    from transformers import AutoTokenizer

    generation = args.kind == "generation"
    if any((name in GENERATION_TASKS) != generation for name in args.task.split(",")):
        raise ValueError(f"--kind {args.kind} does not match tasks {args.task}")
    tasks = park_tasks(args.task, args.num_fewshot, args.eval_split)
    locators = PROX_DATASET_LOCATORS
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bos = tokenizer.bos_token_id
    if args.bos and bos is None:
        raise ValueError("--bos needs a tokenizer BOS token")

    def with_bos(prompt: list[int]) -> list[int]:
        # lm-eval intends BOS for Gemma (add_bos_token=True), but Gemma 4's
        # tokenizer adds none by default; --bos prepends it explicitly.
        return [bos, *prompt] if args.bos and prompt[:1] != [bos] else prompt

    chat_kwargs = args.chat_template_kwargs

    def chat_template(chat_history: list[dict[str, str]]) -> str:
        # lm-eval 0.4.3 VLLM/HFLM.apply_chat_template, plus the template kwargs that
        # later lm-eval passes as enable_thinking/chat_template_args.
        return tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )

    requests: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    sampling: dict[str, Any] | None = None
    for name, task in tasks.items():
        # lm-eval 0.4.3 evaluate() with --apply_chat_template --fewshot_as_multiturn
        # and no --system_instruction.
        task.build_all_requests(
            limit=args.limit,
            apply_chat_template=args.chat_template,
            fewshot_as_multiturn=args.chat_template,
            chat_template=chat_template if args.chat_template else None,
        )
        counts[name] = len({inst.doc_id for inst in task.instances})
        for instance in task.instances:
            if instance.repeats != 1 or instance.request_type != (
                "generate_until" if generation else "loglikelihood"
            ):
                raise ValueError(f"Unexpected {instance.request_type}: {name}")
            if generation:
                # Raw few-shot contexts encode with the tokenizer's add_special_tokens
                # default (see data.encoding for its relation to lm-eval's vLLM
                # add_bos_token). A rendered chat context carries the template's own
                # special tokens and uses lm-eval's add_bos_token default, False.
                context, gen_kwargs = cast(tuple[str, dict[str, Any]], instance.args)
                prompt = with_bos(
                    tokenizer.encode(context, add_special_tokens=not args.chat_template)
                )
                max_tokens, stop, request_sampling = harness_generation(
                    gen_kwargs, tokenizer.decode(tokenizer.eos_token_id)
                )
                if sampling not in (None, request_sampling):
                    raise ValueError(f"Mixed generation sampling in {name}")
                sampling = request_sampling
                # lm-eval would silently left-truncate; require the full context.
                if not prompt or len(prompt) + max_tokens > args.max_model_len:
                    raise ValueError(
                        f"{name}/{instance.doc_id} needs --max-model-len >= "
                        f"{len(prompt) + max_tokens}"
                    )
                requests.append(
                    {
                        "id": f"{name}/{instance.doc_id}/{instance.idx}",
                        "source": name,
                        "doc_id": instance.doc_id,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "stop": stop,
                    }
                )
                continue
            context, continuation = cast(tuple[str, str], instance.args)
            if not context:
                # lm-eval's vLLM backend prefixes BOS here when available, not EOS.
                raise ValueError(f"Empty lm-eval context: {name}/{instance.doc_id}")
            prompt, target = encode_pair(tokenizer, context, continuation)
            prompt = with_bos(prompt)
            # lm_eval vLLM._loglikelihood_tokens retains the rightmost max_length
            # tokens, including all choice tokens. Preserve that context window.
            lead = bos if bos is not None and prompt[:1] == [bos] else None
            prompt = prompt[-(args.max_model_len - len(target)) :]
            if lead is not None and prompt[0] != lead:
                prompt[0] = lead
            if not prompt or not target or len(target) >= args.max_model_len:
                raise ValueError(
                    f"Invalid {name} request {instance.doc_id}/{instance.idx}"
                )
            requests.append(
                {
                    "id": f"{name}/{instance.doc_id}/{instance.idx}",
                    "source": name,
                    "doc_id": instance.doc_id,
                    "choice": instance.idx,
                    "prompt": prompt,
                    "target": target,
                }
            )
    return {
        "role": f"{args.protocol}-lm-eval",
        "data": {
            "protocol": f"{args.protocol}-lm-eval",
            "tasks": list(tasks),
            "task_spec": args.task,
            "kind": args.kind,
            "num_fewshot": args.num_fewshot,
            "eval_split": args.eval_split,
            "task_num_fewshot": {
                name: task.get_config("num_fewshot") for name, task in tasks.items()
            },
            "limit": args.limit,
            "num_documents": counts,
            "max_model_len": args.max_model_len,
            "model": args.model,
            "lm_eval_version": importlib.metadata.version("lm_eval"),
            "dataset_locators": {
                name: locators[name]
                for name in args.task.split(",")
                if name in locators
            },
            # lm-eval's vLLM backend passes add_special_tokens=add_bos_token (False,
            # forced True for "gemma" paths). The OLMoE, Qwen3.5 and Gemma 4 tokenizer
            # defaults add no BOS, so both choices coincide; "bos" records --bos.
            "bos": bos if args.bos else None,
            "encoding": (
                "lm-eval chat-templated generate_until context, "
                "add_special_tokens=False"
                if args.chat_template
                else "lm-eval generate_until context, "
                "tokenizer add_special_tokens default"
                if generation
                else "lm_eval CausalLM._encode_pair "
                "with target tokenizer special tokens"
            ),
            **(
                {
                    "sampling": sampling,
                    "chat_template": args.chat_template,
                    **(
                        {
                            "fewshot_as_multiturn": True,
                            "chat_template_kwargs": chat_kwargs,
                            "system_instruction": None,
                        }
                        if args.chat_template
                        else {}
                    ),
                }
                if generation
                else {}
            ),
            "command": sys.argv,
        },
        "requests": requests,
    }


def lm_eval_statistics(sample_metrics: dict[str, list[float]], aggregators):
    """Return one task's aggregate metrics and stderrs as lm-eval 0.4.3 computes them.

    Follows `TaskOutput.calculate_aggregate_metric`.
    """
    from lm_eval.api.metrics import stderr_for_metric

    metrics, stderr = {}, {}
    for key, items in sample_metrics.items():
        aggregate = aggregators[key.split(",")[0]]
        metrics[key] = aggregate(items)
        stderr_fn = stderr_for_metric(
            metric=aggregate, bootstrap_iters=LM_EVAL_BOOTSTRAP_ITERS
        )
        stderr[key] = stderr_fn(items) if stderr_fn and len(items) > 1 else None
    return metrics, stderr


def pooled_group(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an lm-eval 0.4.3 group score: weighted mean and pooled stderr.

    Members are weighted by their sample counts.
    """
    from lm_eval.api.metrics import aggregate_subtask_metrics, pooled_sample_stderr

    sizes = [member["count"] for member in members]
    keys = set(members[0]["metrics"])
    if any(set(member["metrics"]) != keys for member in members):
        raise ValueError("Group members report different metrics")
    stderrs = {key: [member["stderr"][key] for member in members] for key in keys}
    return {
        "count": sum(sizes),
        "metrics": {
            key: aggregate_subtask_metrics(
                [member["metrics"][key] for member in members], sizes
            )
            for key in keys
        },
        "stderr": {
            key: None if None in values else float(pooled_sample_stderr(values, sizes))
            for key, values in stderrs.items()
        },
    }


def reduce_park(args: argparse.Namespace) -> dict[str, Any]:
    """Reduce scores of lm-eval requests by replaying lm-eval 0.4.3 `evaluate()`.

    Responses, filters, `process_results` and stderrs follow the harness; MMLU
    subjects pool into its four categories and then into `mmlu`.
    """
    panel = json.loads(args.requests.read_text())
    scores = json.loads(args.scores.read_text())
    if panel["role"] != "prox-lm-eval" or not scores.get("complete"):
        raise ValueError("Expected a complete lm-eval score report")
    data = panel["data"]
    if data["lm_eval_version"] != importlib.metadata.version("lm_eval"):
        raise ValueError("lm-eval version differs from prepared requests")
    generation = data.get("kind", "likelihood") == "generation"
    requests = panel["requests"]
    by_id = {row["id"]: row for row in scores["documents"]}
    if len(by_id) != len(requests) or len(scores["documents"]) != len(requests):
        raise ValueError("Incomplete or duplicate scores")
    responses: dict[str, Any] = {}
    for request in requests:
        row = by_id.pop(request["id"])
        fields = ("prompt", "source") if generation else ("prompt", "target", "source")
        if any(row[k] != request[k] for k in fields):
            raise ValueError(f"Request mismatch: {request['id']}")
        if generation:
            if not isinstance(row["continuation"], str):
                raise ValueError(f"Invalid generation: {request['id']}")
            responses[request["id"]] = row["continuation"]
        else:
            if len(row["logprobs"]) != len(request["target"]) or any(
                not math.isfinite(x) for x in row["logprobs"]
            ):
                raise ValueError(f"Invalid choice logprobs: {request['id']}")
            # is_greedy is unused by multiple_choice process_results.
            responses[request["id"]] = (math.fsum(row["logprobs"]), False)
    if by_id:
        raise ValueError("Unexpected scores")
    tasks = park_tasks(
        data.get("task_spec", ",".join(data["tasks"])),
        data["num_fewshot"],
        data.get("eval_split"),
    )
    result = {}
    for name, task in tasks.items():
        # Panels without a stored --limit use their document count as the first-N
        # limit.
        limit = data.get("limit") or data["num_documents"][name]
        task.build_all_requests(limit=limit)
        by_doc: dict[int, list[Any]] = {}
        for instance in task.instances:
            instance.resps.append(
                responses.pop(f"{name}/{instance.doc_id}/{instance.idx}")
            )
            by_doc.setdefault(instance.doc_id, []).append(instance)
        task.apply_filters()
        documents: dict[int, dict[str, Any]] = {}
        sample_metrics: dict[str, list[float]] = {}
        for filter_key in task.instances[0].filtered_resps:
            for doc_id, doc in task.doc_iterator(limit=limit):
                group = sorted(by_doc[doc_id], key=lambda instance: instance.idx)
                values = task.process_results(
                    doc, [instance.filtered_resps[filter_key] for instance in group]
                )
                record = documents.setdefault(doc_id, {"doc_id": doc_id})
                if generation:
                    record.setdefault("filtered", {})[filter_key] = group[
                        0
                    ].filtered_resps[filter_key]
                for metric, value in values.items():
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite {metric} for {name}/{doc_id}")
                    key = metric if filter_key == "none" else f"{metric},{filter_key}"
                    record[key] = float(value)
                    sample_metrics.setdefault(key, []).append(float(value))
        if len(documents) != data["num_documents"][name]:
            raise ValueError(f"Missing {name} documents")
        metrics, stderr = lm_eval_statistics(sample_metrics, task.aggregation())
        result[name] = {
            "count": len(documents),
            "num_fewshot": task.get_config("num_fewshot"),
            "metrics": metrics,
            "stderr": stderr,
            "documents": list(documents.values()),
        }
    if responses:
        raise ValueError("Scores for requests the task did not build")
    groups = {}
    subjects = [name for name in result if name.startswith("mmlu_")]
    if subjects:
        # lm-eval 0.4.3 aggregates subjects into the four categories, then those into
        # mmlu, each weighted by sample count (overall mean over all MMLU questions).
        categories: dict[str, list[str]] = {}
        for name in subjects:
            categories.setdefault(tasks[name].config.group, []).append(name)
        if sorted(categories) != sorted(MMLU_CATEGORIES):
            raise ValueError(f"Unexpected MMLU categories: {sorted(categories)}")
        for category, members in categories.items():
            groups[category] = pooled_group([result[name] for name in members])
        groups["mmlu"] = pooled_group([groups[name] for name in MMLU_CATEGORIES])
        groups["mmlu"]["subjects"] = len(subjects)
    return {
        "complete": True,
        "role": panel["role"],
        "tasks": result,
        "groups": groups,
        "data": data,
        "score_metadata": {k: v for k, v in scores.items() if k != "documents"},
        "command": sys.argv,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Return the token requests of OLMES tasks (`prepare --protocol olmes`).

    Each request keeps its serialized OLMES instance so `reduce` can rebuild the
    task's own scoring.
    """
    from oe_eval.configs.tasks import TASK_CONFIGS
    from oe_eval.launch import resolve_task_suite
    from oe_eval.metrics.metric import MCAccuracy, Metric
    from oe_eval.run_eval import load_task
    from transformers import AutoTokenizer

    from calibration_free_moe.eval.olmes import (
        model_free_task_processing,
        task_for_alias,
    )

    revision = pinned_revision()
    aliases = (
        resolve_task_suite(args.task, {})
        if args.task not in TASK_CONFIGS
        else [args.task]
    )
    if args.form != "both":
        aliases = [a for a in aliases if f":{args.form}::" in a]
    if not aliases:
        raise ValueError(f"No OLMES {args.form} tasks in {args.task}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    requests = []
    ids = set()
    task_configs = {}
    document_counts = {}
    for alias in aliases:
        config = copy.deepcopy(TASK_CONFIGS[alias])
        if args.limit is not None:
            config["limit"] = args.limit
        config["random_subsample_seed"] = args.seed
        config.setdefault("metadata", {})["alias"] = alias
        task = load_task(config)
        # Some pinned metric factories pop task_config metric kwargs (BBH in
        # particular).
        canonical_config = copy.deepcopy(task.task_config)
        metrics = cast(list[Metric] | None, task.make_metrics())
        task.task_config = canonical_config
        if not metrics or (
            args.kind == "likelihood"
            and any(not isinstance(m, MCAccuracy) for m in metrics)
        ):
            raise ValueError(f"Unsupported canonical metrics for {alias}")
        if args.kind == "generation" and any(
            isinstance(m, MCAccuracy) or m.deferred_metric for m in metrics
        ):
            raise ValueError(f"Unsupported generation metric for {alias}")
        with model_free_task_processing([task_for_alias(alias)]):
            task.download()
            task.build_all_requests(limit=config.get("limit"))
        if not task._instances:
            raise ValueError(f"Task produced no requests: {alias}")
        task_configs[alias] = task.task_config
        offset = task.task_config.get("metric_kwargs", {}).get("uncond_docid_offset")
        document_counts[alias] = len(
            {i.doc_id for i in task._instances if offset is None or i.doc_id < offset}
        )
        for instance in task._instances:
            if instance.request_type != (
                "generate_until" if args.kind == "generation" else "loglikelihood"
            ):
                raise ValueError(f"Unsupported request type: {instance.request_type}")
            serialized = instance.to_dict()
            request = cast(dict[str, Any], serialized["request"])
            if args.kind == "generation":
                prompt = tokenizer.encode(request["context"], add_special_tokens=True)
                kwargs = request["generation_kwargs"] or {}
                if (
                    kwargs.get("do_sample", False)
                    or kwargs.get("num_return_sequences", 1) != 1
                    or kwargs.get("repeats", 1) != 1
                ):
                    raise ValueError(f"Non-greedy generation config for {alias}")
                max_tokens = kwargs.get("max_gen_toks", 256)
                context_budget = args.max_model_len - max_tokens
                if context_budget < 1:
                    raise ValueError(
                        f"Generation maximum exceeds model length: {alias}"
                    )
                original_prompt_tokens = len(prompt)
                # Match pinned OLMES vLLM's default left-context truncation.
                starts_with_bos = bool(
                    prompt
                    and tokenizer.bos_token_id is not None
                    and prompt[0] == tokenizer.bos_token_id
                )
                prompt = prompt[-context_budget:]
                if starts_with_bos and prompt[0] != tokenizer.bos_token_id:
                    prompt[0] = tokenizer.bos_token_id
                fields = {
                    "max_tokens": max_tokens,
                    "stop": [
                        *request["stop_sequences"],
                        tokenizer.decode(tokenizer.eos_token_id),
                    ],
                    "original_prompt_tokens": original_prompt_tokens,
                }
            else:
                prompt, target = encode_pair(
                    tokenizer, request["context"], request["continuation"]
                )
                if not target or len(prompt) + len(target) > args.max_model_len:
                    raise ValueError(
                        f"Invalid likelihood request: {alias}/{instance.doc_id}"
                    )
                fields = {"target": target}
            if not prompt:
                raise ValueError(f"Empty prompt: {alias}/{instance.doc_id}")
            request_id = f"{alias}/{instance.doc_id}/{instance.idx}"
            if request_id in ids:
                raise ValueError(f"Duplicate request ID: {request_id}")
            ids.add(request_id)
            row = {
                "id": request_id,
                "source": alias,
                "prompt": prompt,
                "instance": serialized,
                **fields,
            }
            requests.append(row)
    return {
        "role": "development-task",
        "data": {
            "task_configs": task_configs,
            "task_config": task_configs[aliases[0]] if len(aliases) == 1 else None,
            "kind": args.kind,
            "form": args.form,
            "seed": args.seed,
            "olmes_revision": revision,
            "tokenizer": args.model,
            "add_special_tokens": True,
            "pair_encoding": (
                "lm_eval 0.4.3 causal LM._encode_pair (empty context: EOS)"
            ),
            "max_model_len": args.max_model_len,
            "limit": args.limit,
            "num_documents": document_counts,
            "num_requests": len(requests),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("datasets", "pyarrow", "transformers", "lm_eval")
            },
            "command": sys.argv,
        },
        "requests": requests,
    }


def reduce(args: argparse.Namespace) -> dict[str, Any]:
    """Reduce scores of OLMES requests with the pinned OLMES metrics.

    Also reports how many scored tokens stock prefill predicted and how many were
    sparse decode steps.
    """
    from oe_eval.components.instances import RequestInstance
    from oe_eval.metrics.metric import MCAccuracy, Metric
    from oe_eval.run_eval import load_task, merge_lists_of_dicts, merge_metric_outputs
    from oe_eval.utilities.model_results_collation import collate_results

    revision = pinned_revision()
    panel = json.loads(args.requests.read_text())
    report = json.loads(args.scores.read_text())
    if panel["role"] != "development-task" or not report.get("complete"):
        raise ValueError("Expected a development-task panel and complete score report")
    if panel["data"]["olmes_revision"] != revision:
        raise ValueError("Prepared requests and reducer use different OLMES revisions")
    requests = panel["requests"]
    kind = panel["data"].get("kind", "likelihood")
    task_configs = panel["data"].get("task_configs")
    if task_configs is None:
        task_configs = {requests[0]["source"]: panel["data"]["task_config"]}
    document_counts = panel["data"]["num_documents"]
    if isinstance(document_counts, int):
        document_counts = {requests[0]["source"]: document_counts}
    scored = {row["id"]: row for row in report["documents"]}
    if (
        not requests
        or len(scored) != len(report["documents"])
        or len({row["id"] for row in requests}) != len(requests)
        or set(scored) != {row["id"] for row in requests}
    ):
        raise ValueError("Request/score IDs must correspond completely and uniquely")
    by_source: dict[str, tuple[list[Any], list[Any]]] = {}
    target_tokens = 0
    for request in requests:
        row = scored[request["id"]]
        if any(
            row[key] != request[key]
            for key in (
                ("source", "prompt", "target")
                if kind == "likelihood"
                else ("source", "prompt")
            )
        ):
            raise ValueError(f"Scored request differs from prepared input: {row['id']}")
        if kind == "likelihood":
            target = request["target"]
            logprobs = row["logprobs"]
            if not target or len(logprobs) != len(target):
                raise ValueError(f"Incomplete target scores: {row['id']}")
            if any(not math.isfinite(value) for value in logprobs):
                raise ValueError(f"Non-finite logprobs: {row['id']}")
            response = {
                "sum_logits": math.fsum(logprobs),
                "num_tokens": len(target),
                "num_tokens_all": len(request["prompt"]) + len(target),
            }
            target_tokens += len(target)
        else:
            if not isinstance(row["continuation"], str) or not isinstance(
                row["num_tokens"], int
            ):
                raise ValueError(f"Invalid generated response: {row['id']}")
            response = {
                "continuation": row["continuation"],
                "num_tokens": row["num_tokens"],
                "context_tokens": request.get(
                    "original_prompt_tokens", len(request["prompt"])
                ),
            }
        grouped_instances, grouped_outputs = by_source.setdefault(
            request["source"], ([], [])
        )
        grouped_instances.append(RequestInstance.from_dict(request["instance"]))
        grouped_outputs.append(response)
    per_task = {}
    for alias, config in task_configs.items():
        task = load_task(config)
        instances, outputs = by_source[alias]
        results = collate_results(instances, outputs)
        metrics, predictions = [], []
        task_metrics = cast(list[Metric] | None, task.make_metrics())
        if not task_metrics:
            raise ValueError("Task has no canonical metrics")
        for metric in task_metrics:
            if kind == "likelihood" and not isinstance(metric, MCAccuracy):
                raise TypeError("Likelihood requires MCAccuracy")
            metric.compute_for_docs(copy.deepcopy(results))
            metrics.append(
                metric.aggregate_to_task(
                    primary_metric=task.task_config["primary_metric"]
                )
            )
            predictions.append(copy.deepcopy(metric._scores_for_docs))
        documents = merge_metric_outputs(predictions)
        if len(documents) != document_counts[alias]:
            raise ValueError(
                f"Canonical reduction returned incomplete {alias} documents"
            )
        per_task[alias] = {
            "metrics": merge_lists_of_dicts(metrics),
            "documents": documents,
            "num_instances": len(documents),
            "task_config": task.task_config,
        }
    sparse = bool(report["config"].get("additional_config", {}).get("tokenwise"))
    # Sparse-prefill scoring runs the policy on prefill too: no stock predictions.
    prefill_targets = (
        0
        if kind == "likelihood" and sparse and report.get("sparse_prefill")
        else sum(len(row["prompt"]) == 1 for row in requests)
        if kind == "likelihood" and report.get("all_token")
        else len(requests)
        if kind == "likelihood"
        else 0
    )
    generated = (
        sum(row["num_tokens"] for row in report["documents"])
        if kind == "generation"
        else 0
    )
    exposure = (
        {
            "requests": len(requests),
            "target_tokens": target_tokens,
            "first_stock_prefill_target_tokens": prefill_targets,
            "decode_target_tokens": target_tokens - prefill_targets,
            "sparse_decode_target_tokens": target_tokens - prefill_targets
            if sparse
            else 0,
        }
        if kind == "likelihood"
        else {
            "requests": len(requests),
            "generated_tokens": generated,
            "first_stock_prefill_generated_tokens": len(requests),
            "decode_generated_tokens": generated - len(requests),
            "sparse_decode_generated_tokens": generated - len(requests)
            if sparse
            else 0,
        }
    )
    return {
        "complete": True,
        "role": panel["role"],
        "data": panel["data"],
        "tasks": per_task,
        "exposure": exposure,
        "requests_path": str(args.requests.resolve()),
        "raw_per_request_scores_path": str(args.scores.resolve()),
        "scorer_metadata": {k: v for k, v in report.items() if k != "documents"},
        "command": sys.argv,
    }


def main() -> None:
    """Parse the command line, run `prepare` or `reduce` and write the JSON once."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--task", default="hellaswag:rc::olmes")
    prep.add_argument("--protocol", choices=("olmes", "prox"), default="olmes")
    prep.add_argument(
        "--num-fewshot",
        type=int,
        help="lm-eval shot count; default keeps each task's configured count (else 0)",
    )
    prep.add_argument(
        "--eval-split",
        help="Prox MMLU only: evaluate this split (e.g. validation for selection) "
        "instead of test; dev-split shots, prompts and metrics are unchanged",
    )
    prep.add_argument(
        "--kind", choices=("likelihood", "generation"), default="likelihood"
    )
    prep.add_argument("--form", choices=("mc", "rc", "both"), default="both")
    prep.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="OLMES evaluation document subsample seed",
    )
    prep.add_argument(
        "--limit", type=int, help="Documents per task; default keeps the OLMES config"
    )
    prep.add_argument(
        "--model",
        help=f"Tokenizer path; default is the cached {MODEL} snapshot {MODEL_REVISION}",
    )
    prep.add_argument("--max-model-len", type=int, default=None)
    prep.add_argument(
        "--bos",
        action="store_true",
        help="lm-eval protocols: prepend the tokenizer BOS id to every prompt",
    )
    prep.add_argument(
        "--chat-template",
        action="store_true",
        help="lm-eval generation: render through the tokenizer chat template with "
        "few-shot examples as multiturn (lm-eval --apply_chat_template "
        "--fewshot_as_multiturn)",
    )
    prep.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default={},
        help="JSON kwargs for the chat template, e.g. '{\"enable_thinking\": false}'",
    )
    prep.add_argument("--output", type=Path, required=True)
    reduction = commands.add_parser("reduce")
    reduction.add_argument("--requests", type=Path, required=True)
    reduction.add_argument("--scores", type=Path, required=True)
    reduction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if args.command == "prepare":
        args.max_model_len = args.max_model_len or (
            4096 if args.kind == "generation" else 2048
        )
    if args.command == "prepare" and args.bos and args.protocol == "olmes":
        parser.error("--bos applies to the lm-eval (prox) protocol")
    if args.command == "prepare" and (
        not isinstance(args.chat_template_kwargs, dict)
        or (args.chat_template_kwargs and not args.chat_template)
    ):
        parser.error(
            "--chat-template-kwargs is a JSON object used with --chat-template"
        )
    if (
        args.command == "prepare"
        and args.chat_template
        and (args.protocol == "olmes" or args.kind != "generation" or args.bos)
    ):
        parser.error(
            "--chat-template applies to lm-eval generation; the template owns any BOS"
        )
    if args.command == "prepare" and min(args.limit or 1, args.max_model_len) < 1:
        parser.error("--limit and --max-model-len must be positive")
    # Other tasks sample shots from the evaluated split when fewshot_split is unset,
    # so a split override would change their demonstrations; MMLU draws its shots
    # from dev.
    if (
        args.command == "prepare"
        and args.eval_split
        and (args.protocol == "olmes" or args.task != "mmlu")
    ):
        parser.error("--eval-split applies only to lm-eval MMLU (prox protocol)")
    # Set before any task/tokenizer dependency imports; neither stage loads a model.
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1"
    )
    if args.command == "prepare":
        if args.model is None:
            from huggingface_hub import snapshot_download

            args.model = snapshot_download(
                MODEL, revision=MODEL_REVISION, local_files_only=True
            )
        result = prepare_park(args) if args.protocol == "prox" else prepare(args)
    else:
        role = json.loads(args.requests.read_text())["role"]
        result = reduce_park(args) if role == "prox-lm-eval" else reduce(args)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(payload)


if __name__ == "__main__":
    main()
