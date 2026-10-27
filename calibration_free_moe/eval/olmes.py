"""Run and reduce the OLMES likelihood suite for OLMoE.

OLMES (``oe_eval``) scores ten multiple-choice tasks in two formulations: cloze
(CF, ``rc`` aliases) and multiple-choice (MCF, ``mc`` aliases). A task's score is
the better of the two, and MMLU's is the macro mean over its 57 subjects. The
helpers here map task aliases, let ``oe_eval`` build requests without loading a
model, merge per-shard selection statistics, and reduce leaf metrics into the
table compared with OLMoE-1B-7B's published OLMES scores.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import NotRequired, TypedDict, overload


class SelectionStats(TypedDict):
    """Block selection counts and histograms of one evaluation shard."""

    token_layers: int
    candidate_expert_occurrences: int
    selected_blocks: int
    mean_blocks_per_token_layer: float | None
    empty_candidate_experts: int
    empty_candidate_expert_frequency: float | None
    total_block_count_histogram: list[int]
    per_layer_block_count_histogram: list[list[int]]
    candidate_expert_block_count_histogram: list[int]
    per_layer_candidate_expert_block_count_histogram: list[list[int]]
    per_expert_candidate_block_count_histogram: list[list[int]]
    per_layer_expert_block_id_hits: list[list[list[int]]]


class Shard(TypedDict):
    """One worker's OLMES leaves and their predicted cost in instances."""

    shard_id: str
    aliases: list[str]
    predicted_weight: int


class LeafScores(TypedDict):
    """Primary score and instance count of one OLMES leaf (task and formulation)."""

    num_instances: int
    primary_score: float
    primary_metric: str
    processing_time: float


class TaskScores(TypedDict):
    """A task's CF and MCF accuracies (fractions) and its instance count."""

    cf: float
    mcf: float
    instances: int


class TableRow(TypedDict):
    """A table row in percent: the task score, published score and both forms."""

    score: float
    published: float
    cf: NotRequired[float]
    mcf: NotRequired[float]
    winner: NotRequired[str]


# OLMoE-1B-7B's OLMES scores in percent, as published in the OLMoE paper.
PUBLISHED = {
    "arc_challenge": 62.1,
    "arc_easy": 84.2,
    "boolq": 79.2,
    "csqa": 72.9,
    "hellaswag": 80.0,
    "mmlu": 54.1,
    "openbookqa": 65.4,
    "piqa": 79.8,
    "socialiqa": 63.0,
    "winogrande": 70.2,
}
# Scored instances per task and formulation under OLMES's splits and subsampling.
EXPECTED_INSTANCES = {
    "arc_challenge": 1172,
    "arc_easy": 1000,
    "boolq": 1000,
    "csqa": 1221,
    "hellaswag": 1000,
    "mmlu": 14042,
    "openbookqa": 500,
    "piqa": 1000,
    "socialiqa": 1000,
    "winogrande": 1267,
}
TASKS = tuple(sorted(PUBLISHED))


def task_for_alias(alias: str) -> str:
    """Map an OLMES leaf alias to its task; every MMLU subject maps to ``mmlu``."""
    return "mmlu" if alias.startswith("mmlu_") else alias.split(":", 1)[0]


def form_for_alias(alias: str) -> str:
    """Return a leaf alias's formulation: ``mc`` (MCF) or ``rc`` (CF)."""
    return alias.split(":")[1]


@contextmanager
def model_free_task_processing(task_names: Collection[str]):
    """Make the pinned document processors independent of the task object.

    WinoGrande, MMLU, PIQA and SocialIQA processors are called with ``self`` set to
    None (MMLU MC receives only its one context setting), so dataset processing does
    not fingerprint the attached model. The processed documents are unchanged.
    """
    from types import SimpleNamespace

    classes = []
    configured = None
    if "winogrande" in task_names:
        from oe_eval.tasks.oe_eval_tasks.winogrande import Winogrande, WinograndeMC

        classes.extend((Winogrande, WinograndeMC))
    if "mmlu" in task_names:
        from oe_eval.tasks.oe_eval_tasks.mmlu import GenericMMLU, GenericMMLU_MC

        classes.extend((GenericMMLU, GenericMMLU_MC))
        configured = GenericMMLU_MC
    if "piqa" in task_names:
        from oe_eval.tasks.oe_eval_tasks.piqa import PiQA, PiQAMC

        classes.extend((PiQA, PiQAMC))
    if "socialiqa" in task_names:
        from oe_eval.tasks.oe_eval_tasks.siqa import SocialIQA, SocialIQAMC

        classes.extend((SocialIQA, SocialIQAMC))
    originals = {cls: cls.__dict__["_process_doc"] for cls in classes}
    try:
        for cls, method in originals.items():
            if cls is configured:
                # MMLU MC reads this one setting; other pinned processors ignore self.
                def processor(task, method=method):
                    state = SimpleNamespace(
                        task_config={
                            "context_kwargs": {
                                "no_label_prefix_space": task.task_config[
                                    "context_kwargs"
                                ]["no_label_prefix_space"],
                            }
                        }
                    )
                    return partial(method, state)

                cls._process_doc = property(processor)
            else:
                cls._process_doc = staticmethod(partial(method, None))
        yield
    finally:
        for cls, method in originals.items():
            cls._process_doc = method


def table_from_leaves(
    leaves: Mapping[str, LeafScores],
) -> tuple[dict[str, TaskScores], dict[str, TableRow]]:
    """Reduce leaf scores to per-task CF/MCF scores and the published-score table.

    A task scores the better formulation (MCF on ties); MMLU averages its 57
    subjects. Instance counts must match ``EXPECTED_INSTANCES``.
    """
    per_task: dict[str, TaskScores] = {}
    for task in TASKS:
        task_leaves = {
            alias: leaf
            for alias, leaf in leaves.items()
            if task_for_alias(alias) == task
        }
        scores = {}
        for form in ("mc", "rc"):
            form_leaves = {
                alias: leaf
                for alias, leaf in task_leaves.items()
                if form_for_alias(alias) == form
            }
            if task == "mmlu":
                if len(form_leaves) != 57:
                    raise RuntimeError(f"MMLU {form} has {len(form_leaves)} subjects")
                instances = sum(leaf["num_instances"] for leaf in form_leaves.values())
                scores[form] = (
                    sum(leaf["primary_score"] for leaf in form_leaves.values()) / 57
                )
            else:
                if len(form_leaves) != 1:
                    raise RuntimeError(f"{task} {form} has {len(form_leaves)} leaves")
                instances = next(iter(form_leaves.values()))["num_instances"]
                scores[form] = next(iter(form_leaves.values()))["primary_score"]
            if instances != EXPECTED_INSTANCES[task]:
                raise RuntimeError(
                    f"{task} {form} scored {instances} instances, "
                    f"expected {EXPECTED_INSTANCES[task]}"
                )
        per_task[task] = {
            "cf": scores["rc"],
            "mcf": scores["mc"],
            "instances": EXPECTED_INSTANCES[task],
        }
    table: dict[str, TableRow] = {
        task: {
            "cf": 100 * row["cf"],
            "mcf": 100 * row["mcf"],
            "score": 100 * max(row["cf"], row["mcf"]),
            "winner": "CF" if row["cf"] > row["mcf"] else "MCF",
            "published": PUBLISHED[task],
        }
        for task, row in per_task.items()
    }
    table["__average__"] = {
        "score": sum(row["score"] for row in table.values()) / len(TASKS),
        "published": 71.1,
    }
    return per_task, table


Histogram = int | Sequence["Histogram"]


@overload
def add_histograms(values: list[int]) -> int: ...


@overload
def add_histograms(values: list[list[int]]) -> list[int]: ...


@overload
def add_histograms(values: list[list[list[int]]]) -> list[list[int]]: ...


@overload
def add_histograms(values: list[list[list[list[int]]]]) -> list[list[list[int]]]: ...


@overload
def add_histograms(values: Sequence[Histogram]) -> Histogram: ...


def add_histograms(values: Sequence[Histogram]) -> Histogram:
    """Add equally shaped nested integer histograms elementwise."""
    if not values:
        raise ValueError("no histograms to add")
    rows: list[Sequence[Histogram]] = []
    numbers: list[int] = []
    for value in values:
        if isinstance(value, int):
            numbers.append(value)
        else:
            rows.append(value)
    if rows:
        if numbers:
            raise TypeError("histograms have inconsistent nesting depths")
        return [
            add_histograms([row[index] for row in rows])
            for index in range(len(rows[0]))
        ]
    return sum(numbers)


def merge_selection_statistics(records: list[SelectionStats]) -> SelectionStats:
    """Merge shards' selection statistics: add counts, recompute the means."""
    token_layers = sum(record["token_layers"] for record in records)
    candidates = sum(record["candidate_expert_occurrences"] for record in records)
    selected = sum(record["selected_blocks"] for record in records)
    empty = sum(record["empty_candidate_experts"] for record in records)
    return {
        "token_layers": token_layers,
        "candidate_expert_occurrences": candidates,
        "selected_blocks": selected,
        "mean_blocks_per_token_layer": selected / token_layers,
        "empty_candidate_experts": empty,
        "empty_candidate_expert_frequency": empty / candidates,
        "total_block_count_histogram": add_histograms(
            [record["total_block_count_histogram"] for record in records]
        ),
        "per_layer_block_count_histogram": add_histograms(
            [record["per_layer_block_count_histogram"] for record in records]
        ),
        "candidate_expert_block_count_histogram": add_histograms(
            [record["candidate_expert_block_count_histogram"] for record in records]
        ),
        "per_layer_candidate_expert_block_count_histogram": add_histograms(
            [
                record["per_layer_candidate_expert_block_count_histogram"]
                for record in records
            ]
        ),
        "per_expert_candidate_block_count_histogram": add_histograms(
            [record["per_expert_candidate_block_count_histogram"] for record in records]
        ),
        "per_layer_expert_block_id_hits": add_histograms(
            [record["per_layer_expert_block_id_hits"] for record in records]
        ),
    }
