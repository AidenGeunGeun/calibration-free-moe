import argparse
import json
import random
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("datasets")

from scripts.prepare_chat_panel import (
    FIT,
    TUNE,
    prepare,
    prepare_held_out,
    unused_first_turns,
)
from scripts.trace_fidelity import calibration_panel
from scripts.trace_fidelity import prepare as fidelity_panels


class CharacterTokenizer:
    """One token per character of a plain role-tagged chat rendering."""

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</s>\n" for m in messages)
        return text + ("<assistant>" if kwargs.get("add_generation_prompt") else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def conversation(index: int, prompt_id: int) -> dict[str, Any]:
    rng = random.Random(index)
    answer = "".join(rng.choice("abcdef ") for _ in range(rng.randint(150, 400)))
    return {
        "prompt_id": str(prompt_id),
        "messages": [
            {"role": "user", "content": f"question {prompt_id}"},
            {"role": "assistant", "content": answer},
        ],
    }


def test_held_out_chat_and_trace_prompts_avoid_every_earlier_conversation():
    # Repeated prompt IDs across the ranges must never cross into a later pool.
    rows = [conversation(i, i % 950) for i in range(1100)]
    tokenizer = CharacterTokenizer()
    panel, dev = prepare(rows, tokenizer, "qwen35", Path("model"))
    held_out = prepare_held_out(
        rows, tokenizer, "qwen35", Path("model"), 40, panel, start=760, stop=860
    )
    prompts = unused_first_turns(rows, 60, start=860)["prompts"]
    assert len(panel["documents"]) == FIT + TUNE
    calibration_ids = {d["conversation_id"] for d in panel["documents"]}
    development_ids = {r["conversation_id"] for r in dev["requests"]}
    held_out_ids = {r["conversation_id"] for r in held_out["requests"]}
    prompt_ids = {p["conversation_id"] for p in prompts}
    assert len(held_out_ids) == 40 and len(prompt_ids) == 60
    assert not held_out_ids & (calibration_ids | development_ids)
    assert not prompt_ids & (calibration_ids | development_ids | held_out_ids)
    for request in held_out["requests"]:
        assert len(request["target"]) == 128
        assert request["window_start"] >= request["assistant_content_start"]
    assert [p["split_role"] for p in prompts[:10]] == ["FIT"] * 7 + ["TUNE"] + [
        "HELDOUT"
    ] * 2


def test_trace_panels_force_recorded_ids_and_keep_roles_disjoint(tmp_path):
    tokenizer = CharacterTokenizer()
    roles = ["FIT"] * 7 + ["TUNE"] + ["HELDOUT"] * 2
    requests, documents, recorded = [], [], {}
    for question in range(40):
        request_id = f"ultrachat_prompts/{question}/0"
        prompt = [ord(c) for c in f"<user>q{question}</s>\n<assistant>"]
        trace = "x" * (300 + 20 * question)
        # Odd questions generated other IDs that decode to the same text, as a
        # same-length tokenization substitution does; 0 is the stop token.
        generated = [7 if question % 2 else ord("x")] * len(trace)
        recorded[request_id] = generated
        requests.append(
            {
                "id": request_id,
                "prompt": prompt,
                "split_role": roles[question % 10],
                "conversation_id": f"c{question}",
            }
        )
        documents.append(
            {
                "id": request_id,
                "source": "ultrachat_prompts",
                "prompt": prompt,
                "continuation": trace,
                "num_tokens": len(trace) + 1,
                "finish_reason": "stop",
            }
        )
    (tmp_path / "requests.json").write_text(json.dumps({"requests": requests}))
    (tmp_path / "scores.json").write_text(
        json.dumps(
            {"complete": True, "documents": documents, "token_ids_file": "ids.jsonl"}
        )
    )
    (tmp_path / "ids.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": d["id"], "prompt_token_ids": d["prompt"], "token_ids": [*ids, 0]}
            )
            + "\n"
            for d, ids in zip(documents, recorded.values(), strict=True)
        )
    )
    common = {
        "scores": tmp_path / "scores.json",
        "requests": tmp_path / "requests.json",
        "model": "model",
        "seed": 353,
    }
    calibration_panel(
        argparse.Namespace(
            **common,
            output=tmp_path / "calibration.json",
            max_length=256,
            window=64,
            fit_tokens=None,
        ),
        tokenizer,
    )
    fidelity_panels(
        argparse.Namespace(
            **common,
            output_dir=tmp_path / "fidelity",
            split_role="HELDOUT",
            traces=6,
            depths=[0, 100],
            window=64,
        ),
        tokenizer,
    )
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    calibrated = {d["id"] for d in calibration["documents"]}
    forced = {
        r["id"].rsplit("/", 1)[0]
        for path in (tmp_path / "fidelity").iterdir()
        for r in json.loads(path.read_text())["requests"]
    }
    assert len(calibrated) == 32 and len(forced) == 6
    assert not calibrated & forced
    for document in calibration["documents"]:
        # Stock context is the prompt plus a trace prefix; one window ends the document.
        assert len(document["tokens"]) == document["context_length"] + 64 <= 256
        assert document["context_length"] >= document["prompt_tokens"]
        trace = document["tokens"][document["prompt_tokens"] :]
        assert trace == recorded[document["id"]][: len(trace)]
    odd = sum(int(d["id"].split("/")[1]) % 2 for d in calibration["documents"])
    assert calibration["data"]["reencoded_trace_changes"] == {
        "traces": 32,
        "changed": odd,
    }
    for path in (tmp_path / "fidelity").iterdir():
        for request in json.loads(path.read_text())["requests"]:
            trace = recorded[request["id"].rsplit("/", 1)[0]]
            assert request["target"] == trace[request["depth"] :][:64]
