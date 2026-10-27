import pytest

from calibration_free_moe.eval.olmes import (
    EXPECTED_INSTANCES,
    TASKS,
    LeafScores,
    SelectionStats,
    merge_selection_statistics,
    table_from_leaves,
)


def test_table_uses_mmlu_subject_macro_mean_and_mcf_ties():
    leaves: dict[str, LeafScores] = {}
    for task in TASKS:
        subjects = 57 if task == "mmlu" else 1
        quotient, remainder = divmod(EXPECTED_INSTANCES[task], subjects)
        for subject in range(subjects):
            name = f"mmlu_subject_{subject}" if task == "mmlu" else task
            for form in ("mc", "rc"):
                score = 0.5
                if task == "mmlu":
                    score = 0.25 if form == "mc" else float(subject == 0)
                leaves[f"{name}:{form}::olmes"] = {
                    "num_instances": quotient + int(subject < remainder),
                    "primary_score": score,
                    "primary_metric": "acc_raw",
                    "processing_time": 1.0,
                }

    per_task, table = table_from_leaves(leaves)

    assert per_task["mmlu"]["cf"] == pytest.approx(1 / 57)
    assert per_task["mmlu"]["instances"] == 14042
    assert table["mmlu"]["score"] == 25.0
    assert table["arc_easy"].get("winner") == "MCF"
    assert table["__average__"]["score"] == 47.5
    leaves["mmlu_subject_0:rc::olmes"]["num_instances"] -= 1
    with pytest.raises(RuntimeError, match="scored 14041 instances"):
        table_from_leaves(leaves)


def test_selection_merge_sums_nested_histograms_and_weights_rates_by_counts():
    first: SelectionStats = {
        "token_layers": 1,
        "candidate_expert_occurrences": 2,
        "selected_blocks": 1,
        "mean_blocks_per_token_layer": 1.0,
        "empty_candidate_experts": 1,
        "empty_candidate_expert_frequency": 0.5,
        "total_block_count_histogram": [0, 1, 0],
        "per_layer_block_count_histogram": [[0, 1, 0]],
        "candidate_expert_block_count_histogram": [1, 1],
        "per_layer_candidate_expert_block_count_histogram": [[1, 1]],
        "per_expert_candidate_block_count_histogram": [[0, 1], [1, 0]],
        "per_layer_expert_block_id_hits": [[[1], [0]]],
    }
    second: SelectionStats = {
        "token_layers": 2,
        "candidate_expert_occurrences": 4,
        "selected_blocks": 3,
        "mean_blocks_per_token_layer": 1.5,
        "empty_candidate_experts": 1,
        "empty_candidate_expert_frequency": 0.25,
        "total_block_count_histogram": [0, 1, 1],
        "per_layer_block_count_histogram": [[0, 1, 1]],
        "candidate_expert_block_count_histogram": [1, 3],
        "per_layer_candidate_expert_block_count_histogram": [[1, 3]],
        "per_expert_candidate_block_count_histogram": [[0, 2], [1, 1]],
        "per_layer_expert_block_id_hits": [[[2], [1]]],
    }
    assert merge_selection_statistics([first, second]) == {
        "token_layers": 3,
        "candidate_expert_occurrences": 6,
        "selected_blocks": 4,
        "mean_blocks_per_token_layer": 4 / 3,
        "empty_candidate_experts": 2,
        "empty_candidate_expert_frequency": 1 / 3,
        "total_block_count_histogram": [0, 2, 1],
        "per_layer_block_count_histogram": [[0, 2, 1]],
        "candidate_expert_block_count_histogram": [2, 4],
        "per_layer_candidate_expert_block_count_histogram": [[2, 4]],
        "per_expert_candidate_block_count_histogram": [[0, 3], [2, 1]],
        "per_layer_expert_block_id_hits": [[[3], [1]]],
    }


def test_likelihood_pair_uses_tokenizer_default_special_tokens_only_in_prompt():
    from scripts.olmes_requests import encode_pair

    class Tokenizer:
        eos_token_id = 9

        def __init__(self, has_bos: bool, inserts_bos: bool):
            self.bos_token_id = 2 if has_bos else None
            self.inserts_bos = inserts_bos

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            prefix = [2] if add_special_tokens and self.inserts_bos else []
            return prefix + [ord(character) for character in text]

    for has_bos, inserts_bos in ((False, False), (True, False), (True, True)):
        tokenizer = Tokenizer(has_bos, inserts_bos)
        prompt, target = encode_pair(tokenizer, "ab", "c")
        assert prompt == ([2] if inserts_bos else []) + [97, 98]
        assert target == [99]
        empty_prompt, empty_target = encode_pair(tokenizer, "", "c")
        assert empty_prompt == ([2] if inserts_bos else [9])
        assert empty_target == [99]
