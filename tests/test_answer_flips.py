import json

from scripts.answer_flips import flips


def write(path, report):
    path.write_text(json.dumps(report))
    return path


def lm_eval(docs):
    return {
        "complete": True,
        "role": "prox-lm-eval",
        "tasks": {
            "arc_challenge": {"documents": docs},
            "mmlu_anatomy": {"documents": docs},
            "mmlu_virology": {"documents": docs[:2]},
        },
    }


def test_lm_eval_flips_use_the_table_metric_and_pool_mmlu(tmp_path):
    stock = [
        {"doc_id": 0, "acc": 1.0, "acc_norm": 1.0},
        {"doc_id": 1, "acc": 0.0, "acc_norm": 1.0},
        {"doc_id": 2, "acc": 1.0, "acc_norm": 0.0},
    ]
    policy = [
        {"doc_id": 0, "acc": 1.0, "acc_norm": 0.0},
        {"doc_id": 1, "acc": 1.0, "acc_norm": 1.0},
        {"doc_id": 2, "acc": 0.0, "acc_norm": 1.0},
    ]
    result = flips(
        write(tmp_path / "s.json", lm_eval(stock)),
        write(tmp_path / "p.json", lm_eval(policy)),
    )
    arc = result["tasks"]["arc_challenge"]
    assert arc["headline"] == "acc_norm"
    assert arc["metrics"]["acc_norm"]["correct_to_incorrect"] == 1
    assert arc["metrics"]["acc_norm"]["incorrect_to_correct"] == 1
    mmlu = result["groups"]["mmlu"]["metrics"]["acc"]
    # Anatomy: 1 gained, 1 lost; virology (first two questions): 1 gained.
    assert (
        mmlu["count"],
        mmlu["correct_to_incorrect"],
        mmlu["incorrect_to_correct"],
    ) == (
        5,
        1,
        2,
    )


def test_sampled_answers_pair_by_question_and_sample(tmp_path):
    def thinking(correct):
        return {
            "complete": True,
            "data": {"protocol": "provider-thinking", "task": "gpqa_diamond"},
            "documents": [
                {
                    "item_id": item,
                    "sample_index": sample,
                    "correct": value,
                    "metrics": None,
                    "generated_tokens": 10,
                    "truncated": False,
                }
                for (item, sample), value in correct.items()
            ],
        }

    stock = {("a", 0): True, ("a", 1): False, ("b", 0): True, ("b", 1): True}
    policy = {("a", 1): True, ("a", 0): True, ("b", 0): False, ("b", 1): True}
    result = flips(
        write(tmp_path / "s.json", thinking(stock)),
        write(tmp_path / "p.json", thinking(policy)),
    )
    task = result["tasks"]["gpqa_diamond"]
    correct = task["metrics"]["correct"]
    assert (correct["correct_to_incorrect"], correct["incorrect_to_correct"]) == (1, 1)
    questions = task["questions"]
    assert questions["mean_correctness_increased"] == 1
    assert questions["mean_correctness_decreased"] == 1
    assert questions["mean_abs_change"] == 0.5


def test_panels_must_match(tmp_path):
    import pytest

    def report(split, value):
        return {
            "complete": True,
            "role": "prox-lm-eval",
            "data": {"eval_split": split},
            "tasks": {"arc_easy": {"documents": [{"doc_id": 0, "acc": value}]}},
        }

    stock = write(tmp_path / "s.json", report("validation", 0.0))
    other = write(tmp_path / "o.json", report("test", 1.0))
    with pytest.raises(ValueError, match="panels differ"):
        flips(stock, other)
    same = write(tmp_path / "p.json", report("validation", 1.0))
    acc = flips(stock, same)["tasks"]["arc_easy"]["metrics"]["acc"]
    assert acc["incorrect_to_correct"] == 1
