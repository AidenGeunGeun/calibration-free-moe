"""Prepare chat-template thinking requests and score their generated answers.

`prepare` writes requests from pinned dataset revisions under each model provider's
chat template and sampling settings; `ultrachat_prompts` instead turns
`prepare_chat_panel.py --trace-prompts` output into unscored stock-trace requests.
`reduce` scores an `eval_vllm_generation.py` report. Both run on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

PRESETS: dict[str, dict[str, Any]] = {
    "qwen35": {
        "repository": "Qwen/Qwen3.5-35B-A3B",
        "template_kwargs": {"enable_thinking": True},
        "sampling": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        },
        "provider_max_tokens": 81920,
    },
    "qwen3-thinking": {
        "repository": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "template_kwargs": {"enable_thinking": True},
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "provider_max_tokens": 32768,
    },
    "gemma4": {
        "repository": "google/gemma-4-26b-a4b-it",
        "template_kwargs": {"enable_thinking": True},
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "provider_max_tokens": None,
    },
}
DATASETS = {
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "gpqa_diamond": "Idavidrein/gpqa",
    "ifeval": "google/IFEval",
    "gsm8k": "openai/gsm8k",
}
DATASET_REVISIONS = {
    "mmlu_pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "gpqa_diamond": "83022cefff930aea54f654c0b282e74b9eeda5c6",
    "ifeval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
}
# Qwen's standard math instruction (Qwen2.5-Math / Qwen3 evaluation prompts).
QWEN_MATH_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def fixed_seed(seed: int, item_id: str, sample_index: int) -> int:
    """Return a 32-bit seed derived from the base seed, item and sample index."""
    digest = hashlib.sha256(f"{seed}/{item_id}/{sample_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def select_mmlu(rows: list[dict[str, Any]], limit: int | None, seed: int):
    """Return a subject-stratified sample of `limit` MMLU-Pro rows.

    Quotas follow subject frequency by largest remainder, so every subject is
    represented. Without a limit, or with one covering all rows, return all rows.
    """
    if limit is None or limit >= len(rows):
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    if limit < len(groups):
        raise ValueError("MMLU-Pro subset must include every subject")
    counts = {key: len(group) * limit / len(rows) for key, group in groups.items()}
    quotas = {key: int(value) for key, value in counts.items()}
    for key in sorted(groups, key=lambda name: (-(counts[name] - quotas[name]), name))[
        : limit - sum(quotas.values())
    ]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        rng = random.Random(fixed_seed(seed, key, 0))
        selected.extend(rng.sample(groups[key], quotas[key]))
    return selected


def load_items(
    task: str, limit: int | None, seed: int, dataset_path: Path | None = None
) -> list[dict[str, Any]]:
    """Return the pinned evaluation split of a task, optionally subsampled.

    GPQA is gated: it needs a Hugging Face token or a local copy (`dataset_path`)
    that records the pinned revision.
    """
    from datasets import load_dataset

    if task == "gpqa_diamond" and dataset_path is None:
        from huggingface_hub import get_token

        if not get_token():
            raise RuntimeError(
                "Idavidrein/gpqa is gated and no Hugging Face token is available; "
                "request authenticated access, not an unverified mirror"
            )
    config, split = {
        "mmlu_pro": (None, "test"),
        "gpqa_diamond": ("gpqa_diamond", "train"),
        "ifeval": (None, "train"),
        "gsm8k": ("main", "test"),
    }[task]
    if dataset_path is not None:
        # The local copy of the gated GPQA repository records its source commit in
        # REVISION.txt; accept only the pinned revision.
        recorded = re.findall(
            r"\b[0-9a-f]{40}\b", (dataset_path / "REVISION.txt").read_text()
        )
        expected = DATASET_REVISIONS[task]
        if recorded != [expected]:
            raise ValueError(f"{dataset_path} records {recorded}, not {expected}")
    dataset = (
        load_dataset(str(dataset_path), config, split=split)
        if dataset_path is not None
        else load_dataset(
            DATASETS[task], config, split=split, revision=DATASET_REVISIONS[task]
        )
    )
    # GSM8K has no item IDs; the pinned test-split position is stable.
    rows = [
        {**cast(dict[str, Any], row), **({"index": i} if task == "gsm8k" else {})}
        for i, row in enumerate(dataset)
    ]
    if task == "mmlu_pro":
        return select_mmlu(rows, limit, seed)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return rows


def item_prompt(
    task: str,
    item: dict[str, Any],
    seed: int,
    examples: list[dict[str, Any]] | None = None,
) -> tuple[str, str, Any]:
    """Return an item's stable ID, user prompt and reference answer.

    MMLU-Pro prompts may carry same-subject few-shot `examples`; GPQA options are
    shuffled per item with a fixed seed.
    """
    if task == "mmlu_pro":
        item_id = str(item["question_id"])
        options = item["options"]
        choices = "\n".join(
            f"{chr(65 + i)}. {option}" for i, option in enumerate(options)
        )
        context = ""
        for example in examples or []:
            example_choices = "\n".join(
                f"{chr(65 + i)}. {option}"
                for i, option in enumerate(example["options"])
            )
            context += (
                f"Question: {example['question']}\n{example_choices}\n"
                f"{example['cot_content']}\n"
                f'Final choice: {{"answer": "{example["answer"]}"}}\n\n'
            )
        prompt = (
            "Answer the following multiple-choice question. Give your final "
            'choice as JSON {"answer": "LETTER"}.\n\n'
            f"{context}Question: {item['question']}\n{choices}"
        )
        answer = item["answer"]
    elif task == "gpqa_diamond":
        item_id = hashlib.sha256(item["Question"].encode()).hexdigest()[:16]
        options = [item["Correct Answer"]] + [
            item[f"Incorrect Answer {index}"] for index in (1, 2, 3)
        ]
        order = list(range(4))
        random.Random(fixed_seed(seed, item_id, 0)).shuffle(order)
        choices = "\n".join(
            f"{chr(65 + i)}. {options[index]}" for i, index in enumerate(order)
        )
        prompt = (
            "Answer the following multiple-choice question. Give your final "
            'choice as JSON {"answer": "LETTER"}.\n\n'
            f"{item['Question']}\n{choices}"
        )
        answer = chr(65 + order.index(0))
    elif task == "ifeval":
        item_id = str(item["key"])
        prompt = item["prompt"]
        answer = None
    elif task == "ultrachat_prompts":
        item_id = str(item["row_index"])
        prompt = item["prompt"]
        answer = None
    else:
        item_id = str(item["index"])
        prompt = item["question"] + QWEN_MATH_INSTRUCTION
        answer = numeric_answer(item["answer"].split("####")[-1])
        if answer is None:
            raise ValueError(f"Unexpected GSM8K answer: {item['answer']}")
    return item_id, prompt, answer


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Return the request file: chat-templated token IDs and seeds per sample."""
    from transformers import AutoTokenizer

    preset = PRESETS[args.preset]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    selection: dict[str, Any] | None = None
    if args.task == "ultrachat_prompts":
        prompts = json.loads(args.prompts.read_text())
        source = prompts["data"]["dataset"]
        revision = prompts["data"]["dataset_revision"]
        selection = prompts["data"] | {"prompts_path": str(args.prompts)}
        rows = prompts["prompts"]
    else:
        source, revision = DATASETS[args.task], DATASET_REVISIONS[args.task]
        rows = load_items(args.task, args.limit, args.seed, args.dataset_path)
    examples_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if args.task == "mmlu_pro" and args.fewshot:
        from datasets import load_dataset

        for example in load_dataset(
            source, split="validation", revision=DATASET_REVISIONS[args.task]
        ):
            row = cast(dict[str, Any], example)
            examples_by_subject[row["category"]].append(row)
        if any(len(group) < args.fewshot for group in examples_by_subject.values()):
            raise ValueError(
                "Not enough MMLU-Pro validation examples for few-shot prompts"
            )
    requests = []
    for item in rows:
        examples = (
            examples_by_subject[item["category"]][: args.fewshot]
            if args.task == "mmlu_pro"
            else []
        )
        item_id, prompt, answer = item_prompt(args.task, item, args.seed, examples)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            **preset["template_kwargs"],
        )
        if isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        if not isinstance(rendered, list) or not rendered:
            raise ValueError("Chat template did not produce token IDs")
        if len(rendered) + args.max_tokens > args.max_model_len:
            raise ValueError(
                f"Prompt plus cap exceeds model length: {args.task}/{item_id}"
            )
        for sample_index in range(args.samples):
            requests.append(
                {
                    "id": f"{args.task}/{item_id}/{sample_index}",
                    "source": args.task,
                    "item_id": item_id,
                    "sample_index": sample_index,
                    "seed": fixed_seed(
                        args.seed, f"{args.task}/{item_id}", sample_index
                    ),
                    "prompt": rendered,
                    "max_tokens": args.max_tokens,
                    "answer": answer,
                    "item": item if args.task == "ifeval" else None,
                    **(
                        {
                            "split_role": item["split_role"],
                            "conversation_id": item["conversation_id"],
                        }
                        if selection is not None
                        else {}
                    ),
                }
            )
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("Repeated dataset item IDs")
    return {
        "role": "provider-thinking-evaluation",
        "data": {
            "kind": "generation",
            "protocol": "provider-thinking",
            "task": args.task,
            "dataset": source,
            "dataset_revision": revision,
            **({"selection": selection} if selection is not None else {}),
            "dataset_path": str(args.dataset_path) if args.dataset_path else None,
            "preset": args.preset,
            "provider_repository": preset["repository"],
            "model": args.model,
            "template_kwargs": preset["template_kwargs"],
            "sampling": preset["sampling"],
            "provider_max_tokens": preset["provider_max_tokens"],
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "samples": args.samples,
            "fewshot": args.fewshot if args.task == "mmlu_pro" else 0,
            "fewshot_source": "MMLU-Pro validation, same subject"
            if args.task == "mmlu_pro" and args.fewshot
            else None,
            "dataset_items": len(rows),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("datasets", "transformers")
            },
            "command": sys.argv,
        },
        "requests": requests,
    }


def strip_thinking(text: str, preset: str = "qwen35") -> str:
    """Return the final answer after removing the thinking channel.

    A response whose thinking never closes has no final answer and returns "".
    """
    # Some Qwen templates supply the opening tag in the assistant prefix.
    if preset.startswith("qwen") and "</think>" not in text:
        return ""
    if (
        preset == "gemma4"
        and "<channel|>" not in text
        and "<|channel>final\n" not in text
    ):
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    if preset == "gemma4":
        # Gemma 4's chat template drops thought segments and keeps every other segment.
        # A closing marker before any opening one ends a thought that began in the
        # prompt; after that, a closing marker outside a thought is a redundant
        # delimiter, not the end of another thought, so the final text around it stays.
        opening = text.find("<|channel>thought\n")
        closing = text.find("<channel|>")
        if closing >= 0 and (opening < 0 or closing < opening):
            text = text[closing + len("<channel|>") :]
        text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
        text = text.replace("<channel|>", "")
    else:
        text = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"^.*?<channel\|>", "", text, flags=re.DOTALL)
    text = text.replace("<|channel>final\n", "")
    text = text.split("<|channel>thought\n", 1)[0]
    text = text.split("<think>", 1)[0]
    return text.strip()


def boxed_expressions(text: str, complete_only: bool = False) -> list[str]:
    """Return the contents of brace-balanced boxes, including invalid expressions.

    An unfinished box reads as "" (not a valid answer) unless `complete_only`
    skips it.
    """
    expressions = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    expressions.append(text[start:index])
                    break
        else:
            if not complete_only:
                expressions.append("")
    return expressions


def numeric_answer(expression: str) -> str | None:
    """Return the canonical decimal string, or None if the expression is not one.

    Dollar signs, commas and spaces (including LaTeX spacing) are removed first.
    """
    cleaned = re.sub(r"[\s$,]|\\[ ,;!]", "", expression.replace("\\$", ""))
    if not re.fullmatch(r"-?\d+(?:\.\d*)?|-?\.\d+", cleaned):
        return None
    return f"{Decimal(cleaned).normalize():f}"  # 18.0, 18. and 18 all read 18


def gsm8k_answers(text: str) -> tuple[str | None, str | None]:
    """Return the strict and the flexible GSM8K answer.

    The strict answer is the last closed box. The secondary flexible answer is
    the last number inside that box (units, percent), or in the whole text without
    a box. Currency signs, digit-group commas and LaTeX spacing are removed before
    numbers are read, so signs and leading-dot decimals survive.
    """
    boxed = boxed_expressions(text, complete_only=True)
    strict = numeric_answer(boxed[-1]) if boxed else None
    if strict is not None:
        return strict, strict
    source = re.sub(r"\\?\$", "", boxed[-1] if boxed else text)
    source = re.sub(r"(?<=\d)(?:,|\\[ ,;!])(?=\d)", "", source)
    numbers = re.findall(r"-?(?:\d+(?:\.\d+)?|\.\d+)", source)
    return strict, numeric_answer(numbers[-1]) if numbers else None


def extract_answer(
    task: str, text: str, choices: str = "ABCDEFGHIJ"
) -> str | int | None:
    """Return the scored answer of a final response; IFEval has none.

    Multiple-choice answers prefer JSON, then the MMLU-Pro order: "answer is (X)"
    or "answer is X", "Answer: X", and the final standalone uppercase choice. The
    caller removes the thinking channel first.
    """
    if task == "ifeval":
        return None
    if task == "gsm8k":
        return gsm8k_answers(text)[0]
    matches = re.findall(
        r'"answer"\s*:\s*"?([' + choices + r'])"?(?=\s*[,}]|\s*$)',
        text,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()
    # Normalize explicit answer formatting, never the case of ordinary prose.
    final = re.sub(r"\\boxed\s*\{\s*([A-J])\s*\}", r"\1", text)
    final = re.sub(r"\*\*([A-J])\*\*", r"\1", final)
    choice = "[" + re.escape(choices) + "]"
    for pattern in (
        rf"(?i:answer is)\s+\(?({choice})\)?(?=[^A-Za-z]|$)",
        rf"(?i:answer):\s*({choice})\b",
    ):
        match = re.search(pattern, final)
        if match:
            return match.group(1)
    standalone = re.findall(rf"\b({choice})\b", final)
    return standalone[-1] if standalone else None


def reduce(args: argparse.Namespace) -> dict[str, Any]:
    """Score generated answers against their requests and summarize the task."""
    panel = json.loads(args.requests.read_text())
    raw = json.loads(args.scores.read_text())
    if panel["data"].get("protocol") != "provider-thinking" or not raw["complete"]:
        raise ValueError("Expected a complete provider-thinking evaluation")
    scored = {row["id"]: row for row in raw["documents"]}
    requests = panel["requests"]
    if len(scored) != len(requests) or set(scored) != {r["id"] for r in requests}:
        raise ValueError("Incomplete or duplicate score IDs")
    task = panel["data"]["task"]
    documents = []
    for request in requests:
        row = scored[request["id"]]
        if (
            row["prompt"] != request["prompt"]
            or row["seed"] != request["seed"]
            or row.get("max_tokens") != request["max_tokens"]
            or row["max_tokens"] != panel["data"]["max_tokens"]
            or row.get("sampling") != panel["data"]["sampling"]
            or row["truncated"] != (row["finish_reason"] == "length")
        ):
            raise ValueError(
                f"Prepared request differs from generation: {request['id']}"
            )
        final = strip_thinking(row["continuation"], panel["data"]["preset"])
        if task == "ifeval":
            # Only IFEval needs lm-eval's checker and its langdetect dependency.
            from langdetect import DetectorFactory
            from lm_eval.tasks.ifeval.utils import (
                InputExample,
                test_instruction_following_loose,
                test_instruction_following_strict,
            )

            DetectorFactory.seed = 0
            item = request["item"]
            example = InputExample(
                key=item["key"],
                instruction_id_list=item["instruction_id_list"],
                prompt=item["prompt"],
                kwargs=item["kwargs"],
            )
            strict = test_instruction_following_strict(example, final)
            loose = test_instruction_following_loose(example, final)
            correct: bool | None = (
                strict.follow_all_instructions and not row["truncated"]
            )
            metrics = {
                "prompt_strict": correct,
                "instruction_strict": [
                    followed and not row["truncated"]
                    for followed in strict.follow_instruction_list
                ],
                "prompt_loose": loose.follow_all_instructions and not row["truncated"],
                "instruction_loose": [
                    followed and not row["truncated"]
                    for followed in loose.follow_instruction_list
                ],
            }
            answer = None
        elif task == "gsm8k":
            answer, flexible = gsm8k_answers(final)
            correct = answer == request["answer"] and not row["truncated"]
            metrics = {
                "flexible_answer": flexible,
                "flexible_correct": flexible == request["answer"]
                and not row["truncated"],
            }
        else:
            answer = extract_answer(
                task,
                final,
                choices="ABCD" if task == "gpqa_diamond" else "ABCDEFGHIJ",
            )
            correct = (
                answer == request["answer"]
                and answer is not None
                and not row["truncated"]
            )
            metrics = None
        documents.append(
            {
                "id": request["id"],
                "item_id": request["item_id"],
                "sample_index": request["sample_index"],
                "seed": request["seed"],
                "answer": answer,
                "target": request["answer"],
                "correct": correct,
                "metrics": metrics,
                "final_response": final,
                "continuation": row["continuation"],
                "generated_tokens": row["num_tokens"],
                "prompt_tokens": row["prompt_tokens"],
                "finish_reason": row["finish_reason"],
                "truncated": row["truncated"],
            }
        )
    summary = {
        "accuracy": sum(bool(d["correct"]) for d in documents) / len(documents),
        "mean_generated_tokens": sum(d["generated_tokens"] for d in documents)
        / len(documents),
        "truncation_rate": sum(d["truncated"] for d in documents) / len(documents),
        "generated_tokens": sum(d["generated_tokens"] for d in documents),
        "first_stock_prefill_tokens": len(documents),
        "prompt_tokens": sum(d["prompt_tokens"] for d in documents),
        "tokens_per_second": raw["tokens_per_second"],
        "evaluation_seconds": raw["evaluation_seconds"],
    }
    summary["total_tokens"] = summary["generated_tokens"] + summary["prompt_tokens"]
    summary["decode_generated_tokens"] = summary["generated_tokens"] - len(documents)
    lengths = [d["generated_tokens"] for d in documents]
    summary["truncated"] = sum(d["truncated"] for d in documents)
    summary["median_generated_tokens"] = statistics.median(lengths)
    summary["p95_generated_tokens"] = (
        statistics.quantiles(lengths, n=20, method="inclusive")[-1]
        if len(lengths) > 1
        else lengths[0]
    )
    if task == "gsm8k":
        summary["flexible_accuracy"] = sum(
            d["metrics"]["flexible_correct"] for d in documents
        ) / len(documents)
        summary["unboxed_or_unparsed"] = sum(
            d["answer"] is None and not d["truncated"] for d in documents
        )
    if task != "ifeval":
        # Sample standard error of the mean over items, as lm-eval reports it.
        for name in ("accuracy", "flexible_accuracy"):
            if name in summary and len(documents) > 1:
                p = summary[name]
                summary[f"{name}_stderr"] = (p * (1 - p) / (len(documents) - 1)) ** 0.5
    if task == "ifeval":
        for name in ("prompt_strict", "prompt_loose"):
            summary[name] = sum(bool(d["metrics"][name]) for d in documents) / len(
                documents
            )
        for name in ("instruction_strict", "instruction_loose"):
            all_results = [v for d in documents for v in d["metrics"][name]]
            summary[name] = sum(all_results) / len(all_results)
    return {
        "complete": True,
        "data": panel["data"],
        "summary": summary,
        "documents": documents,
        "requests_path": str(args.requests.resolve()),
        "raw_scores_path": str(args.scores.resolve()),
        "scorer": "lm_eval.tasks.ifeval.utils"
        if task == "ifeval"
        else "fixed answer extraction",
        "scorer_package_version": (
            importlib.metadata.version("lm_eval") if task == "ifeval" else None
        ),
        "command": sys.argv,
    }


def main() -> None:
    """Run `prepare` or `reduce` and write its JSON without overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--task", choices=(*DATASETS, "ultrachat_prompts"), required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--preset", choices=PRESETS, required=True)
    prep.add_argument("--max-model-len", type=int, required=True)
    prep.add_argument("--max-tokens", type=int, required=True)
    prep.add_argument("--samples", type=int, default=1)
    prep.add_argument("--fewshot", type=int, default=5)
    prep.add_argument("--limit", type=int)
    prep.add_argument(
        "--dataset-path",
        type=Path,
        help="GPQA only: local copy of the pinned repository (REVISION.txt must match)",
    )
    prep.add_argument(
        "--prompts",
        type=Path,
        help="ultrachat_prompts only: prepare_chat_panel.py --trace-prompts output",
    )
    # The fixed base seed of item sampling, option order and per-request seeds.
    prep.add_argument("--seed", type=int, default=20260923)
    prep.add_argument("--output", type=Path, required=True)
    reduction = commands.add_parser("reduce")
    reduction.add_argument("--requests", type=Path, required=True)
    reduction.add_argument("--scores", type=Path, required=True)
    reduction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if args.command == "prepare" and (
        args.max_tokens < 1
        or args.max_model_len <= args.max_tokens
        or args.samples < 1
        or args.fewshot < 0
    ):
        parser.error("Require positive samples/tokens and max_model_len > max_tokens")
    if args.command == "prepare" and (
        (args.task == "ultrachat_prompts")
        != (args.prompts is not None and args.limit is None)
    ):
        parser.error("ultrachat_prompts takes --prompts (no --limit), and only it")
    if (
        args.command == "prepare"
        and args.dataset_path is not None
        and args.task != "gpqa_diamond"
    ):
        parser.error("--dataset-path applies only to gpqa_diamond")
    result = prepare(args) if args.command == "prepare" else reduce(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
