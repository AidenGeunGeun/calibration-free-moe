"""Write disjoint UltraChat FIT/TUNE calibration and DEV decode panels.

By default writes `panel.json` (FIT/TUNE documents) and `dev-requests.json`
(128-token windows inside an assistant turn, rendered with the model's chat
template in non-thinking mode). `--held-out N` instead writes
`heldout-requests.json` from a reserved range, and `--trace-prompts N` a
tokenizer-free `trace-prompts.json` of first user turns for stock thinking traces.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from scripts.thinking_requests import PRESETS

DATASET = "HuggingFaceH4/ultrachat_200k"
REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
SEED = 353
CAP = 2048
WINDOW = 128
FIT = 448
TUNE = 64
DEV = 128
MARKER = "UNIQUE_ASSISTANT_CONTENT_BOUNDARY_813719"
# Tokenizer-independent role of each shuffled row (70% FIT, 10% TUNE, 20% DEV). Each
# model fills a role only from its pool, so a conversation never changes role across
# models even though eligibility depends on the tokenizer.
ROLE_POOLS = ("FIT",) * 7 + ("TUNE",) + ("DEV",) * 2
# Tokenizer-independent reserved ranges of the same shuffled order. Each preset's
# FIT/TUNE/DEV selection (checked per preset) and its speed-panel continuation of the
# DEV pool end far below HELDOUT_START; conversations seen before a range never
# enter it.
HELDOUT_START = 4096
PROMPT_START = 8192
# First user turns for stock reasoning traces: 7:1:2 FIT/TUNE/HELDOUT by pool order.
TRACE_ROLES = ("FIT",) * 7 + ("TUNE",) + ("HELDOUT",) * 2
MAX_PROMPT_CHARS = 4000


def shuffled_indices(count: int) -> list[int]:
    """Return every row index in the one seeded order all panels share."""
    indices = list(range(count))
    random.Random(SEED).shuffle(indices)
    return indices


def seen_conversations(rows: Any, indices: list[int], stop: int) -> set[str]:
    """Return the conversation IDs at shuffled positions before `stop`."""
    return {str(rows[index]["prompt_id"]) for index in indices[:stop]}


def render_turn(
    truncated: list[dict[str, str]], tokenizer: Any, kwargs: dict[str, Any]
) -> tuple[list[int], int] | None:
    """Return token IDs ending with the last assistant turn, and its content start.

    Returns None when the rendered context does not tokenize as a prefix of the
    whole turn.
    """
    marked = [*truncated[:-1], {"role": "assistant", "content": MARKER}]
    marked_text = tokenizer.apply_chat_template(marked, tokenize=False, **kwargs)
    if marked_text.count(MARKER) != 1:
        raise ValueError("Template did not preserve assistant content marker")
    prefix, suffix = marked_text.split(MARKER, 1)
    full_text = tokenizer.apply_chat_template(truncated, tokenize=False, **kwargs)
    if not full_text.startswith(prefix) or not full_text.endswith(suffix):
        raise ValueError("Assistant template changed the rendered turn boundaries")
    # Score the answer as the model's own non-thinking turn. Gemma renders history
    # turns without the empty thought channel its generation prompt opens with.
    opening = tokenizer.apply_chat_template(
        truncated[:-1], tokenize=False, add_generation_prompt=True, **kwargs
    )
    if opening != prefix and opening.startswith(prefix):
        full_text = opening + full_text[len(prefix) :]
        prefix = opening
    # Template separators after the generated end-of-turn token are not
    # assistant content (Qwen/Gemma append a newline after that token).
    separator_length = len(suffix) - len(suffix.rstrip())
    if separator_length:
        full_text = full_text[:-separator_length]
    tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    context = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if tokens[: len(context)] != context:
        return None
    return tokens, len(context)


def assistant_turns(
    messages: list[dict[str, str]], tokenizer: Any, kwargs: dict[str, Any]
) -> tuple[list[tuple[int, list[int], int, int]], Counter[str]]:
    """Return eligible (turn index, rendered IDs, content start, capped end)."""
    choices = []
    skipped: Counter[str] = Counter()
    for turn, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        rendered = render_turn(messages[: turn + 1], tokenizer, kwargs)
        if rendered is None:
            skipped["token_boundary_mismatch"] += 1
            continue
        tokens, content_start = rendered
        end = min(len(tokens), CAP)
        if end - content_start < WINDOW:
            skipped["short_or_outside_cap"] += 1
            continue
        choices.append((turn, tokens, content_start, end))
    return choices, skipped


def window_record(
    row: Any,
    row_index: int,
    tokenizer: Any,
    kwargs: dict[str, Any],
    skips: Counter[str],
) -> tuple[dict[str, Any], list[int]] | None:
    """Return a seeded assistant turn's record and tokens, or None if ineligible.

    The record holds a window start drawn uniformly inside the turn's content.
    """
    turns, reasons = assistant_turns(row["messages"], tokenizer, kwargs)
    skips.update(reasons)
    if not turns:
        skips["ineligible_conversations"] += 1
        return None
    row_rng = random.Random(f"{SEED}/{row_index}")
    turn, tokens, first_content, capped_end = row_rng.choice(turns)
    start = row_rng.randint(first_content, capped_end - WINDOW)
    conversation_id = str(row["prompt_id"])
    record = {
        "id": f"ultrachat:train_sft:{row_index}",
        "source": DATASET,
        "group_id": conversation_id,
        "row_index": row_index,
        "conversation_id": conversation_id,
        "assistant_turn_index": turn,
        "assistant_content_start": first_content,
        "window_start": start,
    }
    return record, tokens


def development_request(record: dict[str, Any], tokens: list[int]) -> dict[str, Any]:
    """Return the record as a request: context before the window, window as target."""
    start = record["window_start"]
    return record | {
        "prompt": tokens[:start],
        "target": tokens[start : start + WINDOW],
    }


def prepare(
    rows: Any, tokenizer: Any, preset: str, model: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the FIT/TUNE calibration panel and the DEV request panel.

    Rows are taken in shuffled order; each position's role is fixed by
    `ROLE_POOLS`, and the first eligible rows fill each role's quota.
    """
    # Dataset answers carry no reasoning, so render them in non-thinking mode.
    kwargs = PRESETS[preset]["template_kwargs"] | {"enable_thinking": False}
    indices = shuffled_indices(len(rows))
    quotas = {"FIT": FIT, "TUNE": TUNE, "DEV": DEV}
    chosen: dict[str, list[dict[str, Any]]] = {role: [] for role in quotas}
    skips: Counter[str] = Counter()
    seen_ids: set[str] = set()
    position = -1
    for position, row_index in enumerate(indices):
        role = ROLE_POOLS[position % len(ROLE_POOLS)]
        if len(chosen[role]) == quotas[role]:
            continue
        row = rows[row_index]
        conversation_id = str(row["prompt_id"])
        if conversation_id in seen_ids:
            skips["repeated_conversation_id"] += 1
            continue
        drawn = window_record(row, row_index, tokenizer, kwargs, skips)
        if drawn is None:
            continue
        record, tokens = drawn
        seen_ids.add(conversation_id)
        start = record["window_start"]
        if role == "DEV":
            record = development_request(record, tokens)
        else:
            record |= {
                "split_role": role,
                "tokens": tokens[: start + WINDOW],
                "context_length": start,
            }
        chosen[role].append(record)
        if all(len(chosen[r]) == quotas[r] for r in quotas):
            break
    if any(len(chosen[r]) != quotas[r] for r in quotas):
        raise ValueError("Not enough eligible conversations")
    documents = chosen["FIT"] + chosen["TUNE"]
    requests = chosen["DEV"]
    data = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "split": "train_sft",
        "seed": SEED,
        "selection": (
            "shuffle all row indices; tokenizer-independent 7:1:2 FIT/TUNE/DEV "
            "pools by shuffled position; first eligible rows per pool, seeded turn "
            "and uniform window start"
        ),
        "selected_row_indices": [d["row_index"] for d in documents + requests],
        "conversation_ids": [d["conversation_id"] for d in documents + requests],
        "skips": dict(skips),
        "last_shuffled_position": position,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "datasets")
        },
        "tokenizer": str(model.resolve()),
        "preset": preset,
        "template_kwargs": kwargs,
        "max_model_len": CAP,
        "window_tokens": WINDOW,
        "add_special_tokens": False,
    }
    return (
        {"role": "calibration", "data": data, "documents": documents},
        {"role": "development", "data": data, "requests": requests},
    )


def prepare_held_out(
    rows: Any,
    tokenizer: Any,
    preset: str,
    model: Path,
    count: int,
    existing: dict[str, Any] | None = None,
    start: int = HELDOUT_START,
    stop: int = PROMPT_START,
) -> dict[str, Any]:
    """Return requests of the first `count` fresh conversations in [start, stop).

    Positions are in the shared shuffled order. The preset's own FIT/TUNE/DEV
    selection is recomputed (and compared with the existing panel when given) to
    confirm it ends before `start`; every conversation at an earlier position is
    excluded, so the held-out rows are disjoint from every preset's panels while
    their role stays tokenizer-independent.
    """
    panel, _ = prepare(rows, tokenizer, preset, model)
    selection = panel["data"]
    if existing is not None and (
        existing["data"]["selected_row_indices"] != selection["selected_row_indices"]
    ):
        raise ValueError("Recomputed FIT/TUNE/DEV selection differs from the panel")
    if selection["last_shuffled_position"] >= start:
        raise ValueError("The FIT/TUNE/DEV selection reaches the held-out range")
    kwargs = selection["template_kwargs"]
    indices = shuffled_indices(len(rows))
    seen_ids = seen_conversations(rows, indices, start)
    requests: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    position = start - 1
    for position in range(start, min(stop, len(rows))):
        row_index = indices[position]
        row = rows[row_index]
        conversation_id = str(row["prompt_id"])
        if conversation_id in seen_ids:
            skips["seen_conversation_id"] += 1
            continue
        seen_ids.add(conversation_id)
        drawn = window_record(row, row_index, tokenizer, kwargs, skips)
        if drawn is not None:
            requests.append(development_request(*drawn))
            if len(requests) == count:
                break
    if len(requests) != count:
        raise ValueError("Not enough eligible held-out conversations")
    data = selection | {
        "selection": (
            f"same shuffled order; first {count} eligible rows at positions "
            f"[{start}, {stop}) whose conversation IDs do not occur at an earlier "
            "position; seeded turn and uniform window start"
        ),
        "heldout_positions": [start, stop],
        "calibration_last_shuffled_position": selection["last_shuffled_position"],
        "last_shuffled_position": position,
        "selected_row_indices": [r["row_index"] for r in requests],
        "conversation_ids": [r["conversation_id"] for r in requests],
        "skips": dict(skips),
    }
    return {"role": "held-out", "data": data, "requests": requests}


def unused_first_turns(
    rows: Any, count: int, start: int = PROMPT_START
) -> dict[str, Any]:
    """Return `count` first user turns at shuffled positions from `start`.

    The selection is tokenizer-free. Conversations at earlier positions (every
    chat panel and the held-out range) are excluded; roles follow pool order, so
    each prompt keeps its role across models.
    """
    indices = shuffled_indices(len(rows))
    seen_ids = seen_conversations(rows, indices, start)
    prompts: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    position = start - 1
    for position in range(start, len(rows)):
        row_index = indices[position]
        row = rows[row_index]
        conversation_id = str(row["prompt_id"])
        if conversation_id in seen_ids:
            skips["seen_conversation_id"] += 1
            continue
        seen_ids.add(conversation_id)
        first = row["messages"][0]
        if first["role"] != "user" or not first["content"].strip():
            skips["no_first_user_turn"] += 1
            continue
        if len(first["content"]) > MAX_PROMPT_CHARS:
            skips["long_first_user_turn"] += 1
            continue
        prompts.append(
            {
                "row_index": row_index,
                "conversation_id": conversation_id,
                "prompt": first["content"],
                "split_role": TRACE_ROLES[len(prompts) % len(TRACE_ROLES)],
            }
        )
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise ValueError("Not enough unused UltraChat first user turns")
    return {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "split": "train_sft",
        "seed": SEED,
        "selection": (
            f"same shuffled order from position {start}; first user turns of up to "
            f"{MAX_PROMPT_CHARS} characters whose conversation IDs do not occur at an "
            "earlier position; 7:1:2 FIT/TUNE/HELDOUT by pool order"
        ),
        "first_position": start,
        "last_shuffled_position": position,
        "skips": dict(skips),
        "prompts": prompts,
    }


def main() -> None:
    """Write the panel files of the chosen mode into a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--held-out",
        type=int,
        help="UltraChat: write only heldout-requests.json with this many fresh windows",
    )
    parser.add_argument(
        "--existing-panel",
        type=Path,
        help="Held-out: the preset's panel.json, whose selection must be reproduced",
    )
    parser.add_argument(
        "--trace-prompts",
        type=int,
        help="Write only the tokenizer-free trace-prompts.json with this many unused "
        "first user turns (no --model or --preset; one file for every model)",
    )
    args = parser.parse_args()
    held_out: int | None = args.held_out
    trace_prompts: int | None = args.trace_prompts
    if trace_prompts is not None:
        if (
            trace_prompts < 1
            or args.model is not None
            or args.preset is not None
            or held_out is not None
            or args.existing_panel is not None
        ):
            parser.error("--trace-prompts (positive) takes only --output")
        selection = unused_first_turns(
            load_dataset(DATASET, split="train_sft", revision=REVISION), trace_prompts
        )
        prompts = selection.pop("prompts")
        selection["packages"] = {"datasets": importlib.metadata.version("datasets")}
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "trace-prompts.json").write_text(
            json.dumps({"role": "trace-prompts", "data": selection, "prompts": prompts})
            + "\n"
        )
        print(json.dumps(selection, indent=2))
        return
    if args.model is None or args.preset is None:
        parser.error("--model and --preset are required")
    if (held_out is not None and held_out < 1) or (
        args.existing_panel is not None and held_out is None
    ):
        parser.error("--held-out must be positive; --existing-panel needs it")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if held_out is not None:
        rows = load_dataset(DATASET, split="train_sft", revision=REVISION)
        existing = (
            json.loads(args.existing_panel.read_text()) if args.existing_panel else None
        )
        outputs = {
            "heldout-requests.json": prepare_held_out(
                rows, tokenizer, args.preset, args.model, held_out, existing
            )
        }
    else:
        rows = load_dataset(DATASET, split="train_sft", revision=REVISION)
        panel, dev = prepare(rows, tokenizer, args.preset, args.model)
        outputs = {"panel.json": panel, "dev-requests.json": dev}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        (args.output / name).write_text(json.dumps(value) + "\n")
    print(json.dumps(next(iter(outputs.values()))["data"], indent=2))


if __name__ == "__main__":
    main()
