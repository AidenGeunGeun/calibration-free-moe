from typing import TYPE_CHECKING, cast

import pytest
import torch

from calibration_free_moe.activation import GatedActivation, activate

if TYPE_CHECKING or torch.cuda.is_available():
    from calibration_free_moe.baselines.activation_sparsity import ActivationBaseline
    from calibration_free_moe.reader.progressive import channel_major_down

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _case():
    torch.manual_seed(12)
    h, i, experts = 64, 32, 3
    x = torch.randn(2, h, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2], [1, 0]], device="cuda")
    weights = torch.tensor([[0.6, 0.2], [0.4, 0.3]], device="cuda", dtype=torch.float32)
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 8).bfloat16()
    down = channel_major_down(
        (torch.randn(experts, h, i, device="cuda") / 8).bfloat16()
    )
    return x, ids, weights, gu, down


def _dense(x, ids, weights, gu, down, activation: GatedActivation = "silu"):
    outputs = []
    for b in range(len(x)):
        row = torch.zeros(x.shape[-1], device="cuda", dtype=torch.float32)
        for slot in range(ids.shape[1]):
            expert = int(ids[b, slot])
            gate, up = torch.nn.functional.linear(x[b], gu[expert]).chunk(2)
            feature = activate(gate, activation).bfloat16() * up
            row += (
                torch.nn.functional.linear(feature, down[expert]).float()
                * weights[b, slot]
            )
        outputs.append(row.bfloat16())
    return torch.stack(outputs)


@pytest.mark.parametrize("policy", ["cats", "teal"])
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_zero_sparsity_matches_dense(policy, activation: GatedActivation):
    x, ids, weights, gu, down = _case()
    baseline = ActivationBaseline(
        gu,
        down,
        policy,
        -1.0,
        input_threshold=-1.0,
        down_threshold=-1.0,
        sparsity=0,
        activation=activation,
    )
    result = baseline(x, ids, weights)
    torch.testing.assert_close(
        result.float(),
        _dense(x, ids, weights, gu, down, activation).float(),
        atol=0.08,
        rtol=0.04,
    )
    assert baseline.report()["read_fraction"] == pytest.approx(1)


def test_mask_and_read_counts():
    """Infinite thresholds output zero and count only what each rule must read."""
    x, ids, weights, gu, down = _case()
    h, i, k, count = 64, 32, 2, 2
    # Gate-based CATS: all gate, no up/down at threshold infinity.
    baseline = ActivationBaseline(gu, down, "cats", float("inf"))
    torch.testing.assert_close(baseline(x, ids, weights), torch.zeros_like(x))
    report = baseline.report()
    assert report["gate_coefficients"] == count * k * i * h
    assert report["up_coefficients"] == report["down_coefficients"] == 0
    assert report["auxiliary_bf16_equivalent_reads"] == 0
    assert report["read_fraction"] == pytest.approx(1 / 3)
    # TEAL masked input coordinates contribute nothing; down threshold masks all.
    teal = ActivationBaseline(
        gu, down, "teal", -1, input_threshold=float("inf"), down_threshold=float("inf")
    )
    torch.testing.assert_close(teal(x, ids, weights), torch.zeros_like(x))
    report = teal.report()
    assert (
        report["gate_coefficients"]
        == report["up_coefficients"]
        == report["down_coefficients"]
        == 0
    )


def test_masked_coordinate_cannot_affect_output():
    x, ids, weights, gu, down = _case()
    x = torch.zeros_like(x)
    x[:, 0] = 0.01
    x[:, 1] = 2.0
    teal = ActivationBaseline(
        gu, down, "teal", -1, input_threshold=0.1, down_threshold=-1
    )
    before = teal(x, ids, weights)
    modified = x.clone()
    modified[:, 0] = -0.01
    torch.testing.assert_close(teal(modified, ids, weights), before, rtol=0, atol=0)
    assert cast(float, teal.report()["read_fraction"]) < 1.0


@pytest.mark.parametrize("policy", ["wina", "teal-greedy"])
def test_recent_policies_match_independent_routed_reference_with_padding(policy):
    x, ids, weights, gu, down = _case()
    padding = torch.tensor([False, True], device="cuda")
    h, i, k = 64, 32, 2
    gate_norm = gu[:, :i].float().norm(dim=1).bfloat16()
    up_norm = gu[:, i:].float().norm(dim=1).bfloat16()
    down_norm = down.float().norm(dim=1).bfloat16()
    baseline = ActivationBaseline(
        gu,
        down,
        policy,
        0.1,
        gate_threshold=0.2,
        input_threshold=0.15,
        down_threshold=0.1,
        gate_norm=gate_norm if policy == "wina" else None,
        up_norm=up_norm if policy == "wina" else None,
        down_norm=down_norm if policy == "wina" else None,
    )
    output = baseline(x, ids, weights, padding)
    expected = torch.zeros_like(x)
    reads = [0, 0, 0]
    for slot in range(k):
        expert = int(ids[0, slot])
        gscore = x[0].abs() * (gate_norm[expert] if policy == "wina" else 1)
        uscore = x[0].abs() * (up_norm[expert] if policy == "wina" else 1)
        gm, um = gscore > 0.2, uscore > 0.15
        gate = torch.mv(gu[expert, :i].float(), (x[0] * gm).float()).bfloat16()
        up = torch.mv(gu[expert, i:].float(), (x[0] * um).float()).bfloat16()
        feature = torch.nn.functional.silu(gate).bfloat16() * up
        score = feature.abs() * (down_norm[expert] if policy == "wina" else 1)
        down_mask = score > 0.1
        reads[0] += int(gm.sum()) * i
        reads[1] += int(um.sum()) * i
        reads[2] += int(down_mask.sum()) * h
        expected[0] += (
            torch.mv(down[expert].float(), (feature * down_mask).float())
            * weights[0, slot]
        ).bfloat16()
    torch.testing.assert_close(output.float(), expected.float(), atol=0.08, rtol=0.05)
    torch.testing.assert_close(output[1], torch.zeros_like(x[1]))
    report = baseline.report()
    assert [report[f"{name}_coefficients"] for name in ("gate", "up", "down")] == reads
    # Norm gathers are counted for real tokens' slots, never for padding rows.
    auxiliary = int((~padding).sum()) * k * (2 * h + i) if policy == "wina" else 0
    assert report["auxiliary_bf16_equivalent_reads"] == auxiliary
    assert report["read_fraction"] == pytest.approx(
        (sum(reads) + auxiliary) / (k * 3 * i * h)
    )


def test_remote_candidate_norm_gathers_are_charged():
    x, _, _, gu, down = _case()
    h, i = x.shape[1], down.shape[2]
    # One candidate is absent on this rank. safe_ids still gathers its
    # calibration values from the local zero slot on the eager PyTorch path.
    expert_map = torch.tensor([0, -1, 1], device="cuda", dtype=torch.int32)
    local_gu, local_down = gu[[0, 2]], channel_major_down(down[[0, 2]])
    baseline = ActivationBaseline(
        local_gu,
        local_down,
        "wina",
        0.1,
        expert_map=expert_map,
        gate_norm=torch.ones(2, h, device="cuda"),
        up_norm=torch.ones(2, h, device="cuda"),
        down_norm=torch.ones(2, i, device="cuda"),
    )
    baseline(
        x[:1],
        torch.tensor([[0, 1]], device="cuda"),
        torch.tensor([[0.6, 0.4]], device="cuda"),
    )
    # FP32 test metadata: two BF16 equivalents per value, counted only for the
    # rank-local slot; the remote slot's gather is an execution artifact.
    expected = 2 * (2 * h + i)
    assert baseline.report()["auxiliary_bf16_equivalent_reads"] == expected


@pytest.mark.parametrize(
    "policy",
    [
        "cats",
        "cats-topk",
        "teal",
        "teal-greedy",
        "wina",
        "wina-greedy-topk",
    ],
)
def test_expert_parallel_local_and_remote_outputs_with_graph_replay(policy: str):
    x, _, _, gu, down = _case()
    expert_map = torch.tensor([0, -1, 1], device="cuda", dtype=torch.int32)
    ids = torch.tensor([[0, 1], [2, 1]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], device="cuda")
    local_gu, local_down = gu[[0, 2]], channel_major_down(down[[0, 2]])
    local = ActivationBaseline(
        local_gu,
        local_down,
        policy if policy == "wina-greedy-topk" else policy.removesuffix("-topk"),
        0.1,
        top_fraction=0.3 if policy == "cats-topk" else None,
        projection_sparsities=(0.2, 0.6, 0.8) if policy == "wina-greedy-topk" else None,
        gate_threshold=0.2,
        input_threshold=0.15,
        down_threshold=0.1,
        expert_map=expert_map,
        gate_norm=torch.ones(2, 64, device="cuda"),
        up_norm=None
        if policy == "wina-greedy-topk"
        else torch.ones(2, 64, device="cuda"),
        down_norm=None
        if policy == "wina-greedy-topk"
        else torch.ones(2, 32, device="cuda"),
    )
    full = ActivationBaseline(
        gu,
        down,
        policy if policy == "wina-greedy-topk" else policy.removesuffix("-topk"),
        0.1,
        top_fraction=0.3 if policy == "cats-topk" else None,
        projection_sparsities=(0.2, 0.6, 0.8) if policy == "wina-greedy-topk" else None,
        gate_threshold=0.2,
        input_threshold=0.15,
        down_threshold=0.1,
        gate_norm=torch.ones(3, 64, device="cuda"),
        up_norm=None
        if policy == "wina-greedy-topk"
        else torch.ones(3, 64, device="cuda"),
        down_norm=None
        if policy == "wina-greedy-topk"
        else torch.ones(3, 32, device="cuda"),
    )

    def reference():
        return full(x, ids, weights * (expert_map[ids] >= 0))

    torch.testing.assert_close(
        local(x, ids, weights), reference(), atol=0.04, rtol=0.04
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = local(x, ids, weights)
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=0.04, rtol=0.04)
    ids.copy_(torch.tensor([[1, 2], [1, 0]], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=0.04, rtol=0.04)
    assert torch.isfinite(captured).all()


@pytest.mark.parametrize("policy", ["cats", "cats-topk", "teal-greedy", "wina"])
def test_unrouted_slots_contribute_and_charge_nothing_with_graph_replay(policy):
    # The last expert is local, so a wrapped expert_map[-1] would look local.
    # Unwritten grouped rows are poisoned with NaN through the caching allocator.
    x, _, _, gu, down = _case()
    x = torch.cat((x, -x.flip(0)))
    h, i = 64, 32
    ids = torch.tensor(
        [[0, -1], [2, 1], [-1, 2], [1, 0]], device="cuda", dtype=torch.int32
    )
    weights = torch.full((4, 2), 0.5, device="cuda")
    expert_map = torch.tensor([0, -1, 1], device="cuda", dtype=torch.int32)
    local_gu, local_down = gu[[0, 2]], channel_major_down(down[[0, 2]])

    def build(gate_up, down_weight, map_):
        e = gate_up.shape[0]
        return ActivationBaseline(
            gate_up,
            down_weight,
            policy.removesuffix("-topk"),
            0.1,
            top_fraction=0.3 if policy == "cats-topk" else None,
            gate_threshold=0.2,
            input_threshold=0.15,
            down_threshold=0.1,
            expert_map=map_,
            gate_norm=torch.ones(e, h, device="cuda"),
            up_norm=torch.ones(e, h, device="cuda"),
            down_norm=torch.ones(e, i, device="cuda"),
        )

    op = build(local_gu, local_down, expert_map)
    full = build(gu, down, None)

    def reference() -> tuple[torch.Tensor, int]:
        routed = ids >= 0
        owned = routed & (expert_map[ids.long().clamp_min(0)] >= 0)
        # The unrouted slot as a zero-weight expert 0; remote slots weigh 0.
        return full(x, ids.where(routed, 0), weights * owned), int(owned.sum())

    poison = torch.full((1 << 19,), float("nan"), device="cuda", dtype=torch.bfloat16)
    del poison
    expected, owned = reference()
    torch.testing.assert_close(op(x, ids, weights), expected, atol=0.04, rtol=0.04)
    # Reference coefficients: exactly the rank-local routed slots.
    assert op.work[3] == owned * 3 * i * h
    if policy == "cats-topk":
        assert op.work[0] == owned * i * h
        assert op.work[1] == op.work[2] == owned * round(0.3 * i) * h
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, ids, weights)
    for changed in (
        [[-1, 2], [0, -1], [2, 0], [-1, -1]],
        [[0, 2], [2, 1], [1, 0], [0, 2]],
    ):
        ids.copy_(torch.tensor(changed, device="cuda", dtype=torch.int32))
        graph.replay()
        expected, _ = reference()
        torch.testing.assert_close(captured, expected, atol=0.04, rtol=0.04)


@pytest.mark.parametrize("policy", ["cats", "teal", "teal-greedy", "wina"])
@pytest.mark.parametrize("shape", [(5, 3, 6, 256, 128), (2, 8, 64, 2048, 1024)])
def test_routed_masked_reference_at_batched_and_model_dimensions(policy, shape):
    count, k, experts, h, i = shape
    torch.manual_seed(419)
    x = (torch.randn(count, h, device="cuda") / 3).bfloat16()
    ids = torch.randint(experts, (count, k), device="cuda", dtype=torch.int32)
    weights = torch.rand(count, k, device="cuda") / k
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / h**0.5).bfloat16()
    down = (torch.randn(experts, h, i, device="cuda") / i**0.5).bfloat16()
    gate_norm = gu[:, :i].float().norm(dim=1).bfloat16()
    up_norm = gu[:, i:].float().norm(dim=1).bfloat16()
    down_norm = down.float().norm(dim=1).bfloat16()
    padding = torch.zeros(count, device="cuda", dtype=torch.bool)
    padding[-1] = True
    op = ActivationBaseline(
        gu,
        down,
        policy,
        0.2,
        gate_threshold=0.12,
        input_threshold=0.16,
        down_threshold=0.1,
        gate_norm=gate_norm,
        up_norm=up_norm,
        down_norm=down_norm,
    )
    actual = op(x, ids, weights, padding)
    reference = torch.zeros_like(x)
    for token in range(count - 1):
        mixed = torch.zeros(h, dtype=torch.float32, device="cuda")
        for slot in range(k):
            expert = int(ids[token, slot])
            gate_input = x[token]
            up_input = x[token]
            if policy in {"teal", "teal-greedy", "wina"}:
                gm = (
                    x[token].float().abs()
                    * (gate_norm[expert].float() if policy == "wina" else 1)
                ) > 0.12
                um = (
                    x[token].float().abs()
                    * (up_norm[expert].float() if policy == "wina" else 1)
                ) > 0.16
                gate_input = gate_input * gm
                up_input = up_input * um
            gate = (gu[expert, :i].float() @ gate_input.float()).bfloat16()
            activated = torch.nn.functional.silu(gate).bfloat16()
            channel = torch.zeros(i, dtype=torch.bool, device="cuda")
            if policy == "cats":
                channel = activated.float().abs() >= 0.2
                up = (gu[expert, i:].float() @ x[token].float()).bfloat16() * channel
            else:
                up = (gu[expert, i:].float() @ up_input.float()).bfloat16()
            feature = (activated * up).bfloat16()
            if policy in {"teal", "teal-greedy"}:
                channel = feature.abs() > 0.1
            elif policy == "wina":
                channel = feature.float().abs() * down_norm[expert].float() > 0.1
            output = (down[expert].float() @ (feature * channel).float()).bfloat16()
            mixed += output.float() * weights[token, slot]
        reference[token] = mixed.bfloat16()
    torch.testing.assert_close(actual.float(), reference.float(), atol=0.08, rtol=0.06)
    assert op.report()["read_fraction"] is not None


def test_activation_work_counter_replays_with_padding():
    x, ids, weights, gu, down = _case()
    padding = torch.tensor([False, True], device="cuda")
    op = ActivationBaseline(
        gu, down, "teal-greedy", 0.1, input_threshold=0.1, down_threshold=0.1
    )
    op.work.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op(x, ids, weights, padding)
    graph.replay()
    graph.replay()
    reference = ActivationBaseline(
        gu, down, "teal-greedy", 0.1, input_threshold=0.1, down_threshold=0.1
    )
    expected = reference(x, ids, weights, padding)
    torch.testing.assert_close(result, expected, atol=0.03, rtol=0.02)
    torch.testing.assert_close(op.work, reference.work * 2)


@pytest.mark.parametrize(
    "policy,activation,projection_sparsities,expected_counts",
    [
        ("teal", "silu", None, None),
        ("wina", "silu", None, None),
        ("cats", "silu", None, None),
        ("cats", "gelu_tanh", None, None),
        ("wina-greedy-topk", "silu", (0.2, 0.6, 0.8), (51, 25, 6)),
        ("wina-greedy-topk", "silu", (0.0, 1.0, 1.0), (64, 0, 0)),
    ],
)
def test_top_k_configurations_keep_fixed_counts_and_match_reference(
    policy: str,
    activation: GatedActivation,
    projection_sparsities: tuple[float, float, float] | None,
    expected_counts: tuple[int, int, int] | None,
):
    x, ids, weights, gu, down = _case()
    padding = torch.tensor([False, True], device="cuda")
    h, i, k, fraction = 64, 32, 2, 0.3
    keep_gate, keep_up, keep_i = (
        expected_counts
        if expected_counts is not None
        else (round(fraction * h), round(fraction * h), round(fraction * i))
    )
    wina = policy in {"wina", "wina-greedy-topk"}
    norms: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
        gu[:, :i].float().norm(dim=1).bfloat16(),
        gu[:, i:].float().norm(dim=1).bfloat16(),
        down.float().norm(dim=1).bfloat16(),
    )
    baseline = ActivationBaseline(
        gu,
        down,
        policy,
        -1.0,
        top_fraction=fraction if projection_sparsities is None else None,
        projection_sparsities=projection_sparsities,
        activation=activation,
        gate_norm=norms[0] if wina else None,
        up_norm=norms[1] if policy == "wina" else None,
        down_norm=norms[2] if policy == "wina" else None,
    )
    output = baseline(x, ids, weights, padding)

    def top(score, count):
        # Exactly ``count`` entries, like production (no tie expansion).
        order = score.topk(count, sorted=False).indices
        return torch.zeros_like(score, dtype=torch.bool).scatter(0, order, True)

    expected = torch.zeros(h, device="cuda")
    for slot in range(k):
        expert = int(ids[0, slot])
        scale = (
            [norms[0][expert].float(), 1, 1]
            if policy == "wina-greedy-topk"
            else [n[expert].float() if wina else 1 for n in norms]
        )
        gm = True if policy == "cats" else top(x[0].float().abs() * scale[0], keep_gate)
        um = True if policy == "cats" else top(x[0].float().abs() * scale[1], keep_up)
        gate = torch.mv(gu[expert, :i].float(), (x[0] * gm).float()).bfloat16()
        up = torch.mv(gu[expert, i:].float(), (x[0] * um).float()).bfloat16()
        activated = activate(gate, activation).bfloat16()
        feature = activated * up
        dm = top(
            activated.abs() if policy == "cats" else feature.float().abs() * scale[2],
            keep_i,
        )
        expected += (
            torch.mv(down[expert].float(), (feature * dm).float()) * weights[0, slot]
        )
    torch.testing.assert_close(output[0].float(), expected, atol=0.03, rtol=0.05)
    torch.testing.assert_close(output[1], torch.zeros_like(x[1]))
    report = baseline.report()
    # One real token: every routed slot keeps exactly the fixed counts.
    reads = (
        [k * i * h, k * keep_i * h, k * keep_i * h]
        if policy == "cats"
        else [k * keep_gate * i, k * keep_up * i, k * keep_i * h]
    )
    assert [report[f"{name}_coefficients"] for name in ("gate", "up", "down")] == reads
    auxiliary = (
        k * h if policy == "wina-greedy-topk" else k * (2 * h + i) if wina else 0
    )
    assert report["auxiliary_bf16_equivalent_reads"] == auxiliary
    assert report["read_fraction"] == pytest.approx(
        (sum(reads) + auxiliary) / (k * 3 * i * h)
    )
    assert report["resident_calibration_bytes_per_layer"] == (
        (
            norms[0].numel() * 2
            if policy == "wina-greedy-topk"
            else sum(n.numel() * 2 for n in norms)
            if wina
            else 0
        )
        + (12 if projection_sparsities is not None else 0)
    )


@pytest.mark.parametrize("policy", ["teal", "cats"])
def test_top_k_keeps_exact_counts_at_ties_and_allows_zero(policy):
    x, ids, weights, gu, down = _case()
    h, i, k = 64, 32, 2
    # Twelve equal magnitudes straddle the round(0.1 * 64) = 6 boundary.
    x = torch.zeros_like(x[:1])
    x[0, :4] = 3.0
    x[0, 4:16] = torch.tensor([2.0, -2.0] * 6, device="cuda").bfloat16()
    # A zero gate gives CATS tied scores for every channel.
    if policy == "cats":
        gu[:, :i].zero_()
    baseline = ActivationBaseline(gu, down, policy, -1.0, top_fraction=0.1)
    baseline(x, ids[:1], weights[:1])
    report = baseline.report()
    assert report["gate_coefficients"] == (k * i * h if policy == "cats" else k * 6 * i)
    assert report["up_coefficients"] == (
        k * round(0.1 * i) * h if policy == "cats" else k * 6 * i
    )
    assert report["down_coefficients"] == k * round(0.1 * i) * h
    # The rounded keep counts can be 0; CATS still pays for the dense gate.
    empty = ActivationBaseline(gu, down, policy, -1.0, top_fraction=0.005)
    assert not empty(x, ids[:1], weights[:1]).any()
    assert empty.report()["read_fraction"] == pytest.approx(
        1 / 3 if policy == "cats" else 0
    )
