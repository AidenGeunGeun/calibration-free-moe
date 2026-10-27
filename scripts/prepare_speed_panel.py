"""Write the speed panel: distinct real-text prompts from the held-out chat DEV pool.

By default each request is a 128-token prompt (P128) taken from the DEV requests
of `prepare_chat_panel.py`, then from further DEV rows in the same seeded order;
`--context-tokens L` instead joins whole unused DEV conversations into one chat of
exactly L tokens. Calibration conversations are never used.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

PROMPT_TOKENS = 128
PANEL_SIZE = 128


def development_continuation(
    development: dict[str, Any], calibration: dict[str, Any], row_count: int, seed: int
) -> tuple[list[int], int, set[str]]:
    """Return the shuffled rows, the position after the DEV split and used groups.

    Used groups are the conversation IDs of the calibration and development
    panels.
    """
    if development["role"] != "development" or calibration["role"] != "calibration":
        raise ValueError("Expected the original development/calibration panels")
    if (
        development["data"]["selected_row_indices"]
        != calibration["data"]["selected_row_indices"]
    ):
        raise ValueError("Development and calibration panels must share a split")
    original = development["requests"]
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    try:
        start = indices.index(original[-1]["row_index"]) + 1
    except ValueError as error:
        raise ValueError(
            "Original development split is outside this dataset"
        ) from error
    if start > development["data"]["last_shuffled_position"] + 1:
        raise ValueError("Original development order is inconsistent")
    seen_groups = {
        row["conversation_id"] for row in (*calibration["documents"], *original)
    }
    return indices, start, seen_groups


def speed_panel(
    development: dict[str, Any],
    calibration: dict[str, Any],
    row_count: int,
    render: Callable[[int], dict[str, Any] | None],
    *,
    seed: int,
    role_pools: Sequence[str],
    count: int = PANEL_SIZE,
) -> dict[str, Any]:
    """Return `count` P128 prompts: the DEV requests, then continued DEV rows.

    DEV requests shorter than 128 prompt tokens are skipped; calibration rows are
    never used.
    """
    indices, start, seen_groups = development_continuation(
        development, calibration, row_count, seed
    )
    original = development["requests"]
    selected = [
        {"id": row["id"], "prompt": row["prompt"][:PROMPT_TOKENS]}
        for row in original
        if len(row["prompt"]) >= PROMPT_TOKENS
    ]
    omitted = [row["id"] for row in original if len(row["prompt"]) < PROMPT_TOKENS]
    extra_ids: list[str] = []
    for position in range(start, row_count):
        if len(selected) == count:
            break
        if role_pools[position % len(role_pools)] != "DEV":
            continue
        candidate = render(indices[position])
        if candidate is None or candidate["conversation_id"] in seen_groups:
            continue
        if len(candidate["prompt"]) < PROMPT_TOKENS:
            continue
        selected.append(
            {"id": candidate["id"], "prompt": candidate["prompt"][:PROMPT_TOKENS]}
        )
        extra_ids.append(candidate["id"])
        seen_groups.add(candidate["conversation_id"])
    if len(selected) != count or len({row["id"] for row in selected}) != count:
        raise ValueError("Not enough distinct real P128 held-out DEV prompts")
    if any(len(row["prompt"]) != PROMPT_TOKENS for row in selected):
        raise ValueError("A speed prompt is not exactly 128 tokens")
    return {
        "role": "development",
        "data": {
            "dataset": development["data"]["dataset"],
            "dataset_revision": development["data"]["dataset_revision"],
            "seed": seed,
            "selection": (
                "original DEV order, omit short contexts, "
                "continue held-out DEV role pool"
            ),
            "prompt_tokens": PROMPT_TOKENS,
            "omitted_short_ids": omitted,
            "additional_heldout_ids": extra_ids,
        },
        "requests": selected,
    }


def long_context_panel(
    development: dict[str, Any],
    calibration: dict[str, Any],
    row_count: int,
    conversation: Callable[[int], tuple[str, list[dict[str, str]]] | None],
    render: Callable[[list[dict[str, str]]], list[int]],
    *,
    seed: int,
    role_pools: Sequence[str],
    context_tokens: int,
    count: int,
    stop: int,
) -> dict[str, Any]:
    """Return requests that each join whole unused DEV conversations into one chat.

    Conversations follow the same continued DEV order as `speed_panel`, each used
    at most once and never from the calibration or development panels. A request's
    chat grows until its rendering reaches `context_tokens`; positions stay below
    `stop` so the reserved held-out range is never read.
    """
    indices, position, seen_groups = development_continuation(
        development, calibration, row_count, seed
    )
    requests: list[dict[str, Any]] = []
    sources: list[dict[str, list[Any]]] = []
    while len(requests) < count:
        messages: list[dict[str, str]] = []
        source: dict[str, list[Any]] = {"row_indices": [], "conversation_ids": []}
        tokens: list[int] = []
        while len(tokens) < context_tokens:
            if position >= stop:
                raise ValueError(
                    f"DEV conversations before position {stop} fill only "
                    f"{len(requests)} of {count} requests"
                )
            index = indices[position]
            is_dev = role_pools[position % len(role_pools)] == "DEV"
            position += 1
            drawn = conversation(index) if is_dev else None
            if drawn is None or drawn[0] in seen_groups:
                continue
            group, turns = drawn
            seen_groups.add(group)
            messages += turns
            source["row_indices"].append(index)
            source["conversation_ids"].append(group)
            tokens = render(messages)
        requests.append(
            {
                "id": f"ultrachat:train_sft:ctx{context_tokens}:{len(requests)}",
                "prompt": tokens[:context_tokens],
            }
        )
        sources.append(source)
    return {
        "role": "development",
        "data": {
            "dataset": development["data"]["dataset"],
            "dataset_revision": development["data"]["dataset_revision"],
            "seed": seed,
            "selection": (
                "continue held-out DEV role pool; whole unused conversations "
                "as one non-thinking chat, truncated"
            ),
            "context_tokens": context_tokens,
            "last_shuffled_position": position - 1,
            "request_sources": sources,
        },
        "requests": requests,
    }


def main() -> None:
    """Check the local dataset against the panels, then write the speed panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--preset", choices=("qwen35", "qwen3-thinking", "gemma4"), required=True
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--dataset-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-tokens",
        type=int,
        help="Join whole conversations into prompts of exactly this many tokens",
    )
    parser.add_argument("--requests", type=int, default=PANEL_SIZE)
    args = parser.parse_args()
    context_tokens: int | None = args.context_tokens
    if args.requests < 1 or (context_tokens is not None and context_tokens < 1):
        parser.error("--requests and --context-tokens must be positive")

    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    from scripts.prepare_chat_panel import (
        CAP,
        HELDOUT_START,
        REVISION,
        ROLE_POOLS,
        SEED,
        WINDOW,
        assistant_turns,
    )
    from scripts.thinking_requests import PRESETS

    development = json.loads(args.development.read_text())
    calibration = json.loads(args.calibration.read_text())
    if development["data"]["dataset_revision"] != REVISION:
        raise ValueError("The pinned UltraChat revision has changed")
    if args.dataset_snapshot.name != REVISION:
        raise ValueError("The local UltraChat snapshot is not the pinned revision")
    shards = sorted((args.dataset_snapshot / "data").glob("train_sft-*.parquet"))
    if len(shards) != 3:
        raise ValueError("The pinned UltraChat train_sft split needs three shards")
    rows = load_dataset(
        "parquet", data_files=[str(path) for path in shards], split="train"
    )
    # A non-streaming split load is a map-style Dataset, which the row indexing
    # below needs.
    if not isinstance(rows, Dataset):
        raise TypeError("Expected a map-style UltraChat split")
    for item in development["requests"][::32]:
        if str(rows[item["row_index"]]["prompt_id"]) != item["conversation_id"]:
            raise ValueError("Local UltraChat row order differs from development panel")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    kwargs = PRESETS[args.preset]["template_kwargs"] | {"enable_thinking": False}

    def render(index: int) -> dict[str, Any] | None:
        row = rows[index]
        turns, _ = assistant_turns(row["messages"], tokenizer, kwargs)
        if not turns:
            return None
        rng = random.Random(f"{SEED}/{index}")
        turn, tokens, first_content, capped_end = rng.choice(turns)
        start = rng.randint(first_content, capped_end - WINDOW)
        if start >= CAP:
            return None
        return {
            "id": f"ultrachat:train_sft:{index}",
            "conversation_id": str(row["prompt_id"]),
            "assistant_turn_index": turn,
            "prompt": tokens[:start],
        }

    def conversation(index: int) -> tuple[str, list[dict[str, str]]] | None:
        row = rows[index]
        messages: list[dict[str, str]] = row["messages"]
        roles = [message["role"] for message in messages]
        # Whole user/assistant exchanges keep the joined chat alternating.
        if not roles or roles != ["user", "assistant"] * (len(roles) // 2):
            return None
        return str(row["prompt_id"]), messages

    def render_chat(messages: list[dict[str, str]]) -> list[int]:
        text: str = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        tokens: list[int] = tokenizer(text, add_special_tokens=False)["input_ids"]
        return tokens

    if context_tokens is None:
        panel = speed_panel(
            development,
            calibration,
            len(rows),
            render,
            seed=SEED,
            role_pools=ROLE_POOLS,
            count=args.requests,
        )
    else:
        panel = long_context_panel(
            development,
            calibration,
            len(rows),
            conversation,
            render_chat,
            seed=SEED,
            role_pools=ROLE_POOLS,
            context_tokens=context_tokens,
            count=args.requests,
            stop=HELDOUT_START,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "prompt_count": len(panel["requests"])}
            | panel["data"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
