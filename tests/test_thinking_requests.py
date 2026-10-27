import argparse
import json

import pytest

from scripts.thinking_requests import (
    extract_answer,
    fixed_seed,
    gsm8k_answers,
    item_prompt,
    reduce,
    select_mmlu,
    strip_thinking,
)


def test_thinking_is_never_scored_as_a_final_answer():
    qwen = '<think>Final answer: A</think>\n{"answer": "B"}'
    assert extract_answer("gpqa_diamond", strip_thinking(qwen), "ABCD") == "B"
    assert strip_thinking("Final answer: A</think>\\boxed{379}") == "\\boxed{379}"
    assert (
        extract_answer(
            "gsm8k",
            strip_thinking(
                "<|channel>thought\n\\boxed{1}<channel|>\\boxed{379}", "gemma4"
            ),
        )
        == "379"
    )
    assert extract_answer("gsm8k", strip_thinking("<think>\\boxed{1}")) is None
    assert extract_answer("gpqa_diamond", '{"answer": "AB"}', "ABCD") is None
    assert strip_thinking("Final answer: A") == ""
    assert strip_thinking("\\boxed{1}", "gemma4") == ""
    # A redundant closing marker after Gemma's thought must not delete the final answer.
    gemma = '<|channel>thought\nA?<channel|>{"answer": "A"} so A.<channel|>```'
    # A thought that began in the prompt still ends at the first closing marker.
    late = 'A?<channel|>{"answer": "B"}<|channel>thought\nA?<channel|>'
    assert (
        extract_answer("gpqa_diamond", strip_thinking(gemma, "gemma4"), "ABCD") == "A"
    )
    assert extract_answer("gpqa_diamond", strip_thinking(late, "gemma4"), "ABCD") == "B"


def test_choice_shuffle_and_stratified_selection_are_deterministic():
    gpqa = {
        "Question": "What is the correct answer?",
        "Correct Answer": "correct",
        **{f"Incorrect Answer {i}": f"wrong{i}" for i in (1, 2, 3)},
    }
    _, prompt, letter = item_prompt("gpqa_diamond", gpqa, seed=53)
    assert letter == "C"
    assert prompt.splitlines().index(f"{letter}. correct") > 0
    assert item_prompt("gpqa_diamond", gpqa, seed=54)[2] == "B"
    rows = [{"category": c, "question_id": i} for c in ("a", "b") for i in range(10)]
    subset = select_mmlu(rows, 6, 31)
    assert len(subset) == 6
    assert sum(row["category"] == "a" for row in subset) == 3
    assert subset == select_mmlu(rows, 6, 31)
    assert fixed_seed(31, "gpqa/1", 0) != fixed_seed(31, "gpqa/1", 1)


def test_mmlu_fewshot_uses_validation_answer_without_leaking_target():
    example = {
        "question": "Example?",
        "options": ["wrong", "correct"],
        "answer": "B",
        "cot_content": "Reason through the example.",
    }
    target = {**example, "question_id": 17, "question": "Target?", "answer": "A"}
    item_id, prompt, answer = item_prompt("mmlu_pro", target, 1, [example])
    assert item_id == "17" and answer == "A"
    assert 'Final choice: {"answer": "B"}' in prompt
    assert prompt.endswith("Question: Target?\nA. wrong\nB. correct")


def test_mc_follows_canonical_answer_priority_without_casefolding_articles():
    for task, choices in (("mmlu_pro", "ABCDEFGHIJ"), ("gpqa_diamond", "ABCD")):
        assert (
            extract_answer(
                task,
                "The answer is a number larger than 5. The correct option is B.",
                choices,
            )
            == "B"
        )
        assert extract_answer(task, "The answer is (C). D", choices) == "C"
        assert extract_answer(task, "The answer is C. D", choices) == "C"
        assert extract_answer(task, "Answer: C. D", choices) == "C"
        for final in ("C", "\\boxed{C}", "Final answer: **C**"):
            assert extract_answer(task, final, choices) == "C"
        assert extract_answer(task, "A1 is an identifier", choices) is None
        assert extract_answer(task, '{"answer": "B"} Answer: C', choices) == "B"


def test_gsm8k_scores_last_final_box_numerically():
    _, prompt, target = item_prompt(
        "gsm8k", {"index": 7, "question": "Q?", "answer": "work\n#### 1,000"}, 1
    )
    assert target == "1000" and prompt.endswith("within \\boxed{}.")
    final = strip_thinking(r"<think>\boxed{7}</think>Total \boxed{\$1,000.00}.")
    assert extract_answer("gsm8k", final) == target
    assert gsm8k_answers(r"\boxed{7} then \boxed{8.}") == ("8", "8")
    assert gsm8k_answers("The total is $1,000.") == (None, "1000")
    assert gsm8k_answers(r"999 \boxed{1000 \text{ dollars}}") == (None, "1000")
    assert gsm8k_answers(r"\boxed{7} then \boxed{8") == ("7", "7")
    assert gsm8k_answers("The answer is .5.") == (None, "0.5")
    assert gsm8k_answers(r"\boxed{-.5\text{ kg}}") == (None, "-0.5")
    assert gsm8k_answers(r"\boxed{-\$5\text{ dollars}}") == (None, "-5")
    assert gsm8k_answers(r"\boxed{1\,000\text{ dollars}}") == (None, "1000")
    assert gsm8k_answers("I have 3, 4 and 5 apples") == (None, "5")
    assert extract_answer("gsm8k", strip_thinking(r"<think>\boxed{1000}")) is None


@pytest.mark.parametrize("task", ["gsm8k", "ifeval"])
def test_reducer_counts_correct_looking_length_truncation_as_wrong(tmp_path, task):
    if task == "ifeval":
        pytest.importorskip("langdetect")
        pytest.importorskip("lm_eval")
    item = {
        "key": 2832,
        "prompt": "Include exactly one markdown bullet point.",
        "instruction_id_list": ["detectable_format:number_bullet_lists"],
        "kwargs": [{"num_bullets": 1}],
    }
    continuation = (
        "<think>done</think>\n- Buy this game!"
        if task == "ifeval"
        else r"<think>done</think>\boxed{336}"
    )
    sampling = {"temperature": 1.0, "top_p": 0.95}
    request = {
        "id": "one",
        "item_id": "one",
        "sample_index": 0,
        "seed": 123,
        "prompt": [1, 2],
        "max_tokens": 8192,
        "answer": {"gsm8k": "336", "ifeval": None}[task],
        "item": item if task == "ifeval" else None,
    }
    panel = {
        "data": {
            "protocol": "provider-thinking",
            "task": task,
            "preset": "qwen35",
            "sampling": sampling,
            "max_tokens": 8192,
        },
        "requests": [request],
    }
    row = {
        "id": "one",
        "prompt": [1, 2],
        "seed": 123,
        "max_tokens": 8192,
        "sampling": sampling,
        "continuation": continuation,
        "num_tokens": 8192,
        "prompt_tokens": 2,
        "finish_reason": "length",
        "truncated": True,
    }
    raw = {
        "complete": True,
        "documents": [row],
        "tokens_per_second": 1.0,
        "evaluation_seconds": 1.0,
    }
    requests = tmp_path / "requests.json"
    scores = tmp_path / "scores.json"
    requests.write_text(json.dumps(panel))
    scores.write_text(json.dumps(raw))
    args = argparse.Namespace(requests=requests, scores=scores)
    complete_row = {**row, "truncated": False, "finish_reason": "stop"}
    scores.write_text(json.dumps({**raw, "documents": [complete_row]}))
    complete_result = reduce(args)
    assert complete_result["documents"][0]["correct"] is True
    if task == "ifeval":
        assert all(
            complete_result["summary"][name] == 1
            for name in (
                "prompt_strict",
                "prompt_loose",
                "instruction_strict",
                "instruction_loose",
            )
        )
    scores.write_text(json.dumps(raw))
    result = reduce(args)
    assert result["documents"][0]["truncated"] is True
    assert result["documents"][0]["final_response"]
    assert result["documents"][0]["correct"] is False
    assert result["summary"]["accuracy"] == 0
    if task == "ifeval":
        assert all(
            result["summary"][name] == 0
            for name in (
                "prompt_strict",
                "prompt_loose",
                "instruction_strict",
                "instruction_loose",
            )
        )
    for field, bad_value in (("max_tokens", 32768), ("sampling", {})):
        bad_row = {**row, field: bad_value}
        scores.write_text(json.dumps({**raw, "documents": [bad_row]}))
        with pytest.raises(ValueError, match="Prepared request differs"):
            reduce(args)
