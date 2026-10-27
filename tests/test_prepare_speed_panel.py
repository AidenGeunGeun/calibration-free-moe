import random

from scripts.prepare_speed_panel import speed_panel


def test_speed_panel_replaces_short_context_from_next_unused_dev_rows():
    seed = 353
    indices = list(range(300))
    random.Random(seed).shuffle(indices)
    original = [
        {
            "id": f"dev:{i}",
            "conversation_id": f"dev:{i}",
            "prompt": [i] * (129 if i < 127 else 80),
            "row_index": indices[1],
        }
        for i in range(128)
    ]
    data = {
        "dataset": "test/chat",
        "dataset_revision": "pinned",
        "last_shuffled_position": 3,
        "selected_row_indices": [indices[0], indices[1]],
    }
    development = {"role": "development", "data": data, "requests": original}
    calibration = {
        "role": "calibration",
        "data": data,
        "documents": [{"conversation_id": "calibrated"}],
    }
    called = []

    def render(index):
        called.append(index)
        if index == indices[3]:
            return {"id": "dup", "conversation_id": "calibrated", "prompt": [1] * 128}
        if index == indices[5]:
            return {"id": "short", "conversation_id": "short", "prompt": [1] * 90}
        return {
            "id": f"heldout:{index}",
            "conversation_id": f"heldout:{index}",
            "prompt": [2] * 140,
        }

    panel = speed_panel(
        development,
        calibration,
        len(indices),
        render,
        seed=seed,
        role_pools=("FIT", "DEV"),
    )
    assert called == [indices[3], indices[5], indices[7]]
    assert [row["id"] for row in panel["requests"][:127]] == [
        f"dev:{i}" for i in range(127)
    ]
    assert panel["requests"][-1]["id"] == f"heldout:{indices[7]}"
    assert panel["data"]["omitted_short_ids"] == ["dev:127"]
    assert len(panel["requests"]) == 128
    assert {len(row["prompt"]) for row in panel["requests"]} == {128}
