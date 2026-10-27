import socket
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from calibration_free_moe.activation import activate
from calibration_free_moe.analysis.allocation import (
    LayerAllocation,
    TokenContext,
    integer_counts,
)
from calibration_free_moe.baselines import prox as prox_baseline
from calibration_free_moe.baselines.activation_sparsity import (
    grouped_assignment,
    grouped_config,
)
from calibration_free_moe.baselines.prox import (
    ProxBaseline,
    ProxRecipe,
    ProxyScore,
    proxy_scores,
    quantize_proxy,
)
from calibration_free_moe.reader.progressive import channel_major_down
from scripts.calibrate_prox import fit_keep_threshold

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def assert_prox_membership(reference, before, after):
    """Only identities within the measured FP32 boundary uncertainty may differ."""
    torch.testing.assert_close(
        before["input_mask"], after["input_mask"], rtol=0, atol=0
    )
    torch.testing.assert_close(before["local"], after["local"], rtol=0, atol=0)
    local_rows = before["local"].reshape(-1)
    # Grouped assignment can omit unrouted (-1) slots, leaving their torch.empty
    # score rows unwritten. Decode zeros them. Only local rows feed selection;
    # inactive rows must select nothing, not have equal scratch contents.
    torch.testing.assert_close(
        after["proxy"][local_rows], before["proxy"][local_rows], rtol=0.003, atol=1e-6
    )
    for diagnostic in (before, after):
        assert not diagnostic["channel_mask"][~local_rows].any()
    local = before["local"].reshape(-1, 1)
    old = torch.where(local, before["proxy"].abs(), -1.0)
    new = torch.where(local, after["proxy"].abs(), -1.0)
    old_selected, new_selected = before["channel_mask"], after["channel_mask"]
    eligible = local.expand_as(old)
    if reference.channel_scope == "token":
        tokens = before["input_mask"].shape[0]
        old, new = old.reshape(tokens, -1), new.reshape(tokens, -1)
        old_selected = old_selected.reshape(tokens, -1)
        new_selected = new_selected.reshape(tokens, -1)
        eligible = eligible.reshape(tokens, -1)
    changed = old_selected != new_selected
    error = (old - new).abs().amax(-1, keepdim=True)
    scale = old.abs().amax(-1, keepdim=True).clamp_min(1.0)
    # Do not turn an arbitrarily inaccurate proxy into a large allowed tie band.
    assert (error <= 2e-5 * scale).all()
    if reference.channel_keep is None:
        old_cut = new_cut = torch.full_like(error, reference.channel_threshold)
        for score, selected in ((old, old_selected), (new, new_selected)):
            torch.testing.assert_close(selected, eligible & (score >= old_cut))
    else:
        old_cut = old.topk(reference.channel_keep, dim=-1).values[:, -1:]
        new_cut = new.topk(reference.channel_keep, dim=-1).values[:, -1:]
        for score, selected in ((old, old_selected), (new, new_selected)):
            # A near-boundary location alone is not permission to select an
            # arbitrary identity: each observed set must still be a valid top-k
            # of its own scores, including the exact retained count.
            torch.testing.assert_close(
                selected.sum(-1), eligible.sum(-1).clamp_max(reference.channel_keep)
            )
            worst_selected = torch.where(selected, score, torch.inf).amin(-1)
            best_omitted = torch.where(eligible & ~selected, score, -torch.inf).amax(-1)
            assert (worst_selected >= best_omitted).all()
    # The kth order statistic moves by at most the largest score perturbation.
    ulp = torch.finfo(torch.float32).eps * scale
    assert ((old_cut - new_cut).abs() <= error + ulp).all()
    band = error + (old_cut - new_cut).abs() + ulp
    uncertain = (
        eligible & ((old - old_cut).abs() <= band) & ((new - new_cut).abs() <= band)
    )
    differences, bound = int(changed.sum()), int(uncertain.sum())
    assert not (changed & ~uncertain).any()
    assert differences <= bound
    print(
        f"Prox membership H={before['input_mask'].shape[1]} "
        f"I={before['channel_mask'].shape[1]} scope={reference.channel_scope} "
        f"local_proxy_rows={int(local_rows.sum())}/{local_rows.numel()} "
        f"input_differences=0 channel_differences={differences} "
        f"boundary_candidate_bound={bound} max_boundary_band={band.max().item():.3g} "
        "max_changed_reference_margin="
        f"{torch.where(changed, (old - old_cut).abs(), 0).max().item():.3g}"
    )
    return differences


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_prox_proxy_selection_output_padding_and_bytes(activation):
    torch.manual_seed(9)
    h, i, experts = 128, 32, 3
    x = torch.zeros(3, h, device="cuda", dtype=torch.bfloat16)
    x[0, 2], x[0, 80], x[2, 4] = 0.5, 0.6, 0.7
    ids = torch.tensor([[0, 2], [1, 0], [2, 1]], device="cuda")
    mix = torch.tensor([[0.7, 0.2], [0.3, 0.8], [0.4, 0.6]], device="cuda")
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 5).bfloat16()
    down = (torch.randn(experts, h, i, device="cuda") / 5).bfloat16()
    packed, scales = quantize_proxy(gu)
    tx, ts = 0.2, 0.001
    padding = torch.tensor([False, True, False], device="cuda")
    op = ProxBaseline(gu, channel_major_down(down), packed, scales, tx, ts, activation)
    result = op(x, ids, mix, padding)
    expected = torch.zeros_like(x, dtype=torch.float32)
    input_count, channels = 0, 0
    for token in (0, 2):
        masked = torch.where(x[token].abs() >= tx, x[token], 0)
        input_count += int((masked != 0).sum())
        for slot in range(2):
            expert = int(ids[token, slot])
            # Dequantize independently of the Triton nibble arithmetic.
            lo = (packed[expert] & 15).int()
            hi = (packed[expert] >> 4).int()
            lo = torch.where(lo >= 8, lo - 16, lo)
            hi = torch.where(hi >= 8, hi - 16, hi)
            q = torch.stack((lo, hi), dim=-1).flatten(-2).float()
            proxy_gu = q * scales[expert, :, None]
            proxy_gate, proxy_up = torch.nn.functional.linear(
                masked.float(), proxy_gu
            ).chunk(2)
            mask = (activate(proxy_gate, activation) * proxy_up).abs() >= ts
            channels += int(mask.sum())
            gate, up = torch.nn.functional.linear(
                x[token].float(), gu[expert].float()
            ).chunk(2)
            feature = (activate(gate, activation).bfloat16() * up.bfloat16()).bfloat16()
            feature = feature * mask
            expected[token] += (
                torch.nn.functional.linear(feature.float(), down[expert].float())
                * mix[token, slot]
            )
    torch.testing.assert_close(
        result.float(), expected.bfloat16().float(), atol=0.08, rtol=0.04
    )
    torch.testing.assert_close(result[1], torch.zeros_like(result[1]))
    report = op.report()
    # Coordinate-major proxy: one I-byte gate/up run per retained input and routed slot.
    assert report["proxy_bytes"] == input_count * 2 * i
    # Two sectors in token 0 and one in token 2, across both routed experts.
    assert (
        report["row_major_sector_bytes"]
        == 3 * 2 * 2 * i * 32 + cast(float, report["scale_bytes"]) + channels * h * 6
    )
    assert report["accounting_basis"] == "analytical_sector_estimate"
    assert report["priced_proxy_layout"] == "coordinate_major"
    assert report["executed_proxy_layout"] == "row_major"
    assert report["native_execution"] == "dense_grouped_masked"
    assert report["actual_traffic_measured"] is False
    assert report["logical_charged_bytes"] == report["charged_bytes"]
    assert report["coordinate_major_sector_bytes"] == report["charged_bytes"]
    assert report["scale_bytes"] == 2 * 2 * 2 * i * 4
    assert (
        report["gate_coefficients"]
        == report["up_coefficients"]
        == report["down_coefficients"]
        == channels * h
    )
    assert report["reference_bytes"] == 2 * 2 * 3 * i * h * 2
    assert report["read_fraction"] == pytest.approx(
        cast(float, report["charged_bytes"]) / cast(float, report["reference_bytes"])
    )


def test_prox_counter_replays_with_padding():
    torch.manual_seed(73)
    h, i, experts = 64, 32, 3
    x = torch.randn(2, h, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 0]], device="cuda")
    mix = torch.full((2, 2), 0.5, device="cuda")
    padding = torch.tensor([False, True], device="cuda")
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 5).bfloat16()
    down = (torch.randn(experts, h, i, device="cuda") / 5).bfloat16()
    packed, scales = quantize_proxy(gu)
    op = ProxBaseline(gu, channel_major_down(down), packed, scales, 0.2, 0.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = op(x, ids, mix, padding)
    graph.replay()
    graph.replay()
    reference = ProxBaseline(gu, channel_major_down(down), packed, scales, 0.2, 0.03)
    expected = reference(x, ids, mix, padding)
    torch.testing.assert_close(result, expected, atol=0.03, rtol=0.02)
    torch.testing.assert_close(op.work[:112], reference.work[:112] * 2)
    torch.testing.assert_close(op.work[112:], reference.work[112:])


@pytest.mark.parametrize("h,i,experts", [(2048, 1024, 64), (2048, 768, 128)])
def test_grouped_proxy_scores_and_selection(h, i, experts):
    torch.manual_seed(93)
    count, k = 16, 8
    x = (torch.randn(count, h, device="cuda") / 5).bfloat16()
    ids = torch.randint(experts, (count, k), device="cuda")
    shape = (experts, 2 * i, h // 2)
    lo = torch.randint(-7, 8, shape, dtype=torch.int8, device="cuda")
    hi = torch.randint(-7, 8, shape, dtype=torch.int8, device="cuda")
    packed = ((lo & 15) | (hi << 4)).to(torch.uint8)
    scales = torch.rand(experts, 2 * i, device="cuda") / 100
    valid = torch.ones(count, dtype=torch.bool, device="cuda")
    valid[-1] = False
    assignment = grouped_assignment(ids.flatten(), experts)
    assert assignment[0] is not None
    per_slot, per_slot_mask = proxy_scores(x, ids, packed, scales, 0.1, valid)
    grouped, grouped_mask = proxy_scores(x, ids, packed, scales, 0.1, valid, assignment)
    assert torch.equal(per_slot_mask, grouped_mask)
    # Different FP32 reduction order is allowed, not different proxy arithmetic.
    relative = (per_slot - grouped).abs() / per_slot.abs().clamp_min(0.01)
    assert relative.max() < 0.003
    threshold = 0.03
    flips = (per_slot.abs() >= threshold) != (grouped.abs() >= threshold)
    assert not flips.any() or (per_slot.abs() - threshold).abs()[flips].max() < 0.0001
    print(
        f"H={h} I={i} E={experts}: max relative={relative.max().item():.6g}, "
        f"flips={flips.sum().item()}/{flips.numel()}"
    )


def test_grouped_proxy_runs_in_operator_and_matches_per_slot_output(monkeypatch):
    torch.manual_seed(96)
    h, i, experts, count, k = 128, 64, 8, 16, 2
    x = (torch.randn(count, h, device="cuda") / 5).bfloat16()
    ids = torch.randint(experts, (count, k), device="cuda")
    mix = torch.softmax(torch.randn(count, k, device="cuda"), -1)
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 5).bfloat16()
    down = (torch.randn(experts, h, i, device="cuda") / 5).bfloat16()
    packed, scales = quantize_proxy(gu)
    library_scores = prox_baseline.proxy_scores

    def per_slot_path(*args, **kwargs):
        # Drop the grouped assignment: score each (token, slot) pair separately.
        return library_scores(*args[:6])

    monkeypatch.setattr(prox_baseline, "proxy_scores", per_slot_path)
    per_slot_op = ProxBaseline(gu, channel_major_down(down), packed, scales, 0.1, 0.03)
    per_slot = per_slot_op(x, ids, mix)
    monkeypatch.setattr(prox_baseline, "proxy_scores", library_scores)

    def prohibit_per_slot(*args, **kwargs):
        raise AssertionError("grouped path was not used")

    monkeypatch.setattr(prox_baseline, "_proxy", prohibit_per_slot)
    grouped_op = ProxBaseline(gu, channel_major_down(down), packed, scales, 0.1, 0.03)
    grouped = grouped_op(x, ids, mix)
    torch.testing.assert_close(grouped, per_slot, rtol=0.02, atol=0.03)
    torch.testing.assert_close(grouped_op.work, per_slot_op.work)


@pytest.mark.parametrize("count", [3, 16])
def test_prox_expert_parallel_partials_sum_to_full(count):
    # A slot routed to no expert (id -1, as in vLLM's profiling pass) is neither read
    # nor counted, including on the rank that owns the final expert.
    torch.manual_seed(41)
    h, i, experts, k = 128, 64, 8, 2
    x = (torch.randn(count, h, device="cuda") / 2).bfloat16()
    ids = torch.randint(experts, (count, k), device="cuda", dtype=torch.int32)
    ids[0, 1] = -1
    routed = ids >= 0
    mix = torch.softmax(torch.randn(count, k, device="cuda"), -1)
    padding = torch.zeros(count, dtype=torch.bool, device="cuda")
    padding[-1] = True
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 5).bfloat16()
    down = (torch.randn(experts, h, i, device="cuda") / 5).bfloat16()
    packed, scales = quantize_proxy(gu)
    # Reference: the unrouted slot as a zero-weight final expert.
    unrouted = ProxBaseline(gu, channel_major_down(down), packed, scales, 0.3, 0.02)
    expected = unrouted(x, ids.where(routed, experts - 1), mix * routed, padding)
    full = ProxBaseline(gu, channel_major_down(down), packed, scales, 0.3, 0.02)
    torch.testing.assert_close(full(x, ids, mix, padding), expected)
    expected = expected.float()
    total = torch.zeros_like(expected)
    work = torch.zeros_like(full.work)
    half = experts // 2
    for rank in range(2):
        owned = slice(rank * half, (rank + 1) * half)
        expert_map = torch.full((experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[owned] = torch.arange(half, dtype=torch.int32, device="cuda")
        local_packed, local_scales = quantize_proxy(gu[owned].contiguous())
        assert torch.equal(local_packed, packed[owned])
        local = ProxBaseline(
            gu[owned].contiguous(),
            channel_major_down(down[owned]),
            local_packed,
            local_scales,
            0.3,
            0.02,
            expert_map=expert_map,
        )
        total += local(x, ids, mix, padding).float()
        work += local.work
    # Each rank rounds its partial output to BF16 before the EP sum.
    torch.testing.assert_close(total, expected, atol=0.02, rtol=0.02)
    # Rank-local counters sum to the full read counters; only the per-token input
    # threshold comparisons are repeated on every rank.
    torch.testing.assert_close(work[:8], full.work[:8])
    assert work[7] == (routed & ~padding[:, None]).sum()
    torch.testing.assert_close(work[9], full.work[9])


def _joint_prox_ep_worker(
    rank: int,
    port: int,
    recipe: ProxRecipe,
    input_topk: bool,
    score: ProxyScore,
    floor: bool,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        import vllm.distributed as vd

        # Replace only vLLM's group lookup, not the operator or NCCL collective.
        vd.get_ep_group = lambda: SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
            device_group=dist.group.WORLD
        )
        torch.manual_seed(149)
        h, i, experts, count, k, keep = 128, 32, 4, 6, 2, 19
        gu = (torch.randn(experts, 2 * i, h) / 10).bfloat16().cuda()
        down = torch.zeros(experts, h, i, device="cuda", dtype=torch.bfloat16)
        channel = torch.arange(i, device="cuda")
        for expert in range(experts):
            # Disjoint output coordinates prevent cancellation between slots.
            down[expert, expert * i + channel, channel] = 2.0**expert
        down = channel_major_down(down)
        x = (torch.randn(count, h) / 2).bfloat16().cuda()
        x[0] = 0  # Exact joint ties, spanning both owners and reversed expert IDs.
        ids = torch.tensor(
            [[2, 0], [1, 3], [0, 1], [3, -1], [1, 2], [3, 0]],
            device="cuda",
            dtype=torch.int32,
        )
        weights = torch.softmax(torch.randn(count, k), -1).cuda()
        padding = torch.tensor([False] * (count - 1) + [True], device="cuda")

        def operator(
            gate_up: torch.Tensor,
            native_down: torch.Tensor,
            expert_map: torch.Tensor | None = None,
        ) -> ProxBaseline:
            packed, scales = quantize_proxy(gate_up)
            return ProxBaseline(
                gate_up,
                native_down,
                packed,
                scales,
                None if input_topk else 0.2,
                0.02 if floor else None,
                expert_map=expert_map,
                input_keep=41 if input_topk else None,
                channel_keep=None if floor else keep,
                channel_floor=keep if floor else 0,
                channel_scope="token",
                proxy_score=score,
                recipe=recipe,
            )

        full = operator(gu, down)
        owned = slice(rank * 2, (rank + 1) * 2)
        expert_map = torch.full((experts,), -1, device="cuda", dtype=torch.int32)
        expert_map[owned] = torch.arange(2, device="cuda", dtype=torch.int32)
        local = operator(
            gu[owned].contiguous(), channel_major_down(down[owned]), expert_map
        )

        def check(actual: torch.Tensor, decisions: dict[str, torch.Tensor]) -> None:
            expected_decisions: dict[str, torch.Tensor] = {}
            full.work.zero_()
            expected = full(x, ids, weights, padding, diagnostics=expected_decisions)
            torch.testing.assert_close(
                decisions["channel_mask"],
                expected_decisions["channel_mask"] & decisions["local"].reshape(-1, 1),
                rtol=0,
                atol=0,
            )
            selected = decisions["channel_mask"].int()
            dist.all_reduce(selected)
            torch.testing.assert_close(
                selected, expected_decisions["channel_mask"].int(), rtol=0, atol=0
            )
            if recipe == "study":
                # The first token's zero scores resolve by flattened slot index,
                # not rank or expert ID, even though its native output is zero.
                torch.testing.assert_close(
                    selected.view(count, k * i)[0],
                    (torch.arange(k * i, device="cuda") < keep).int(),
                )
            total = actual.float()
            dist.all_reduce(total)
            envelope = 2 * torch.finfo(torch.bfloat16).eps * expected.abs().max().item()
            torch.testing.assert_close(total, expected.float(), rtol=0, atol=envelope)
            local_work = local.work[:11].clone()
            dist.all_reduce(local_work)
            # Token counters repeat, but all weight/proxy coefficients are local.
            torch.testing.assert_close(local_work[:8], full.work[:8], rtol=0, atol=0)
            torch.testing.assert_close(local_work[9:], full.work[9:11], rtol=0, atol=0)
            assert local.report()["ranked_channel_scores"] == (
                int(decisions["local"].sum()) * i
            )

        decisions: dict[str, torch.Tensor] = {}
        check(local(x, ids, weights, padding, diagnostics=decisions), decisions)
        # Warm NCCL and the real kernels before capture. Captured Torch score
        # buffers stay in the graph pool; changed routes must overwrite them.
        for _ in range(3):
            local(x, ids, weights, padding)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])
        graph = torch.cuda.CUDAGraph()
        graph_decisions: dict[str, torch.Tensor] = {}
        with torch.cuda.graph(graph):
            replayed = local(x, ids, weights, padding, diagnostics=graph_decisions)
        for routes, pad in (
            ([[0, 2], [3, 1], [2, 3], [-1, 0], [1, 2], [0, 3]], False),
            ([[2, 0], [1, 3], [0, 1], [3, -1], [2, 3], [3, 0]], True),
        ):
            ids.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
            padding[-1] = pad
            x[1:].mul_(0.5)
            local.work.zero_()
            graph.replay()
            check(replayed, graph_decisions)
    finally:
        # Captured NCCL holds the group until worker exit; explicit destruction
        # can block while the CUDA graph is alive (same as the reader EP test).
        dist.barrier(device_ids=[rank])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize(
    "recipe,input_topk,score,floor",
    [
        (recipe, inputs, score, False)
        for recipe in ("legacy", "study")
        for inputs in (False, True)
        for score in ("prox", "output-unit")
    ]
    + [("study", True, "prox", True)],
)
def test_joint_prox_expert_parallel_matches_tp1(
    recipe: ProxRecipe, input_topk: bool, score: ProxyScore, floor: bool
) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    spawn(
        _joint_prox_ep_worker,
        args=(port, recipe, input_topk, score, floor),
        nprocs=2,
        join=True,
    )


def test_prox_free_output_unit_selection():
    # One-hot down columns give every (expert, channel) its own output coordinate, so
    # the selected set is observable; power-of-two column norms keep them exact.
    torch.manual_seed(29)
    h, i, experts, k, count, keep, inputs = 128, 32, 4, 2, 6, 20, 48
    x = (torch.randn(count, h, device="cuda") / 2).bfloat16()
    ids = torch.stack([torch.randperm(experts)[:k] for _ in range(count)]).cuda()
    mix = torch.softmax(torch.randn(count, k, device="cuda"), -1)
    padding = torch.zeros(count, dtype=torch.bool, device="cuda")
    padding[-1] = True
    valid = ~padding
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 5).bfloat16()
    norms = 2.0 ** torch.randint(-2, 3, (experts, i), device="cuda")
    down = torch.zeros(experts, h, i, device="cuda")
    channel = torch.arange(i, device="cuda")
    for expert in range(experts):
        down[expert, expert * i + channel, channel] = norms[expert]
    down = channel_major_down(down.bfloat16())
    packed, scales = quantize_proxy(gu)

    def op(**options):
        return ProxBaseline(
            gu,
            down,
            packed,
            scales,
            None,
            None,
            channel_keep=keep,
            input_keep=inputs,
            channel_scope="token",
            **options,
        )

    # Independent top-F of |w| n |p| over each token's K slots, from the same proxy.
    assignment = grouped_assignment(
        ids.flatten(), experts, None, grouped_config(gu, down, k, count)
    )
    proxy, _ = proxy_scores(
        x, ids, packed, scales, None, valid, assignment, input_keep=inputs
    )
    proxy = proxy.view(count, k, i)
    selected = {}
    for name, weight in (("prox", 1.0), ("output-unit", mix[..., None] * norms[ids])):
        salience = (weight * proxy.abs()).view(count, k * i)
        index = salience.topk(keep, -1).indices
        mask = torch.zeros_like(salience, dtype=torch.bool).scatter_(-1, index, True)
        selected[name] = mask.view(count, k, i) & valid[:, None, None]
    assert not torch.equal(selected["prox"], selected["output-unit"])
    for name, mask in selected.items():
        operator = op(proxy_score=name)
        result = operator(x, ids, mix, padding)
        observed = torch.zeros_like(mask)
        for slot in range(k):
            coordinate = ids[:, slot, None] * i + channel
            observed[:, slot] = result.gather(1, coordinate) != 0
        assert torch.equal(observed, mask)
        report = operator.report()
        assert report["selection_parameter_bytes"] == (
            4 * i * k * (count - 1) if name == "output-unit" else 0
        )


@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("input_rule", ["threshold", "topk"])
@pytest.mark.parametrize("channel_rule", ["threshold", "topk", "floor"])
def test_keep_rule_matrix_and_joint_floor(activation, input_rule, channel_rule):
    # Membership is observed through disjoint one-hot down columns. The small and
    # large tokens exercise both the floor and a threshold's larger allocation. Input
    # masks and INT4 scores come from independent Torch references.
    torch.manual_seed(57)
    h, i, experts, k, count = 128, 32, 4, 2, 4
    x = torch.randn(count, h, device="cuda").bfloat16()
    x *= torch.tensor([0.02, 0.3, 1.0, 2.0], device="cuda")[:, None]
    ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], device="cuda")
    mix = torch.full((count, k), 0.5, device="cuda")
    padding = torch.tensor([False, False, False, True], device="cuda")
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 10).bfloat16()
    down = torch.zeros(experts, h, i, device="cuda", dtype=torch.bfloat16)
    coordinate = torch.arange(i, device="cuda")
    for expert in range(experts):
        down[expert, expert * i + coordinate, coordinate] = 1
    down = channel_major_down(down)
    packed, scales = quantize_proxy(gu)
    tx = 0.1 if input_rule == "threshold" else None
    inputs = 40 if input_rule == "topk" else None
    ts = None if channel_rule == "topk" else 0.02
    keep = 19 if channel_rule == "topk" else None
    floor = 13 if channel_rule == "floor" else 0

    def operator():
        return ProxBaseline(
            gu,
            down,
            packed,
            scales,
            tx,
            ts,
            activation,
            input_keep=inputs,
            channel_keep=keep,
            channel_scope="token",
            channel_floor=floor,
            recipe="study",
        )

    if inputs is None:
        input_mask = x.abs() >= cast(float, tx)
    else:
        input_mask = torch.zeros_like(x, dtype=torch.bool).scatter_(
            -1, x.abs().argsort(dim=-1, descending=True, stable=True)[:, :inputs], True
        )
    input_mask &= ~padding[:, None]
    lo, hi = (packed & 15).int(), (packed >> 4).int()
    q = torch.stack((lo.where(lo < 8, lo - 16), hi.where(hi < 8, hi - 16)), -1)
    dequantized = q.flatten(-2).float() * scales[..., None]
    scores = torch.empty(count, k, i, device="cuda")
    for token in range(count):
        for slot in range(k):
            g, u = torch.nn.functional.linear(
                (x[token] * input_mask[token]).float(), dequantized[ids[token, slot]]
            ).chunk(2)
            scores[token, slot] = (activate(g, activation) * u).abs()
    scores = scores.view(count, k * i)
    threshold_mask = scores >= (ts if ts is not None else 0.02)
    expected = threshold_mask.clone()
    if keep is not None or floor:
        top = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1,
            scores.argsort(dim=-1, descending=True, stable=True)[:, : keep or floor],
            True,
        )
        expected = top if keep is not None else expected | top
    expected &= ~padding[:, None]
    expected = expected.view(count, k, i)
    op = operator()
    result = op(x, ids, mix, padding)
    observed = torch.stack(
        [
            result.gather(1, ids[:, slot, None] * i + coordinate) != 0
            for slot in range(k)
        ],
        1,
    )
    assert torch.equal(observed, expected)
    counts = expected.flatten(1).sum(-1)[:3]
    if floor:
        assert threshold_mask[0].sum() < floor
        assert threshold_mask[2].sum() > floor
        assert counts[0] == floor
        assert counts[2] == threshold_mask[2].sum()
    report = op.report()
    assert report["ranked_input_scores"] == (3 * h if inputs else 0)
    assert report["ranked_channel_scores"] == (3 * k * i if keep or floor else 0)
    assert report["threshold_comparisons"] == (
        (3 * h if inputs is None else 0) + (3 * k * i if keep is None else 0)
    )
    assert report["proxy_bytes"] == int(input_mask.sum()) * k * i
    assert report["gate_coefficients"] == int(counts.sum()) * h
    distribution = cast(dict[str, Any], report["selected_channels_per_token"])
    assert distribution["tokens"] == 3
    assert distribution["min"] == int(counts.min())
    assert distribution["max"] == int(counts.max())
    assert distribution["mean"] == pytest.approx(float(counts.float().mean()))
    for key, quantile in (("p1_interval", 0), ("p5_interval", 0), ("p50_interval", 1)):
        low, high = distribution[key]
        assert low <= int(counts.sort().values[quantile]) <= high
    assert sum(bin["tokens"] for bin in distribution["histogram"]) == 3
    if floor:
        reference = op.work.clone()
        op.work.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = op(x, ids, mix, padding)
        graph.replay()
        graph.replay()
        torch.testing.assert_close(replayed, result)
        torch.testing.assert_close(op.work[:112], reference[:112] * 2)
        torch.testing.assert_close(op.work[112:], reference[112:])


@pytest.mark.parametrize("input_topk", [False, True])
def test_fitted_threshold_with_floor_realizes_target_reads(input_topk):
    """Fit residual scores after reserving the floor, then measure the real reads."""
    torch.manual_seed(71)
    h, i, experts, k, count, floor = 128, 32, 4, 2, 8, 9
    x = (torch.randn(count, h, device="cuda") / 3).bfloat16()
    x[0] *= 0.01
    ids = torch.tensor([[0, 1], [2, 3]], device="cuda").repeat(count // 2, 1)
    mix = torch.full((count, k), 0.5, device="cuda")
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 8).bfloat16()
    down = channel_major_down(
        (torch.randn(experts, h, i, device="cuda") / 8).bfloat16()
    )
    packed, scales = quantize_proxy(gu)
    inputs = 41 if input_topk else None
    tx = None if input_topk else fit_keep_threshold(x.abs().cpu(), 0.3 * x.numel())
    assignment = grouped_assignment(
        ids.flatten(), experts, None, grouped_config(gu, down, k, count)
    )
    proxy, mask = proxy_scores(
        x,
        ids,
        packed,
        scales,
        tx,
        assignment=assignment,
        input_keep=inputs,
        stable_ties=True,
    )
    tail = (
        proxy.abs()
        .view(count, k * i)
        .topk(k * i - floor, dim=-1, largest=False, sorted=False)
        .values.cpu()
    )
    target_channels = 20
    ts = fit_keep_threshold(tail, count * (target_channels - floor))
    operator = ProxBaseline(
        gu,
        down,
        packed,
        scales,
        tx,
        ts,
        input_keep=inputs,
        channel_scope="token",
        channel_floor=floor,
        recipe="study",
    )
    output = operator(x, ids, mix)
    assert torch.isfinite(output).all()
    report = operator.report()
    assert report["gate_coefficients"] == count * target_channels * h
    proxy_read = int(mask.sum()) / count / (6 * h) + 4 / (3 * h)
    assert report["read_fraction"] == pytest.approx(
        proxy_read + target_channels / (k * i)
    )
    # A zero proxy is a tied boundary: fit must choose the floor alone, not read
    # every channel because >= 0 happens to include the entire tied tail.
    tied = torch.zeros(100, dtype=torch.bfloat16)
    boundary = fit_keep_threshold(tied, 1)
    assert not (tied >= boundary).any()


@pytest.mark.parametrize("floor", [False, True])
def test_study_ties_use_lowest_input_and_joint_channel_indices(floor):
    # Identical channel rows have equal proxies; one-hot down columns expose exact
    # membership; a reversed route order distinguishes slot order from expert ID.
    # The reference is an explicit lowest-index set, not torch.topk.
    h, i, k, keep, inputs = 128, 32, 2, 13, 41
    x = torch.ones(2, h, device="cuda", dtype=torch.bfloat16)
    x[:, 1::2] = -1
    ids = torch.tensor([[1, 0], [0, 1]], device="cuda")
    mix = torch.full((2, k), 0.5, device="cuda")
    gu = (x[:1] / 128).expand(2, 2 * i, h).contiguous()
    down = torch.zeros(2, h, i, device="cuda", dtype=torch.bfloat16)
    coordinate = torch.arange(i, device="cuda")
    for expert in range(2):
        down[expert, expert * i + coordinate, coordinate] = 1
    down = channel_major_down(down)
    packed, scales = quantize_proxy(gu)
    # Even though all BF16 magnitudes tie, precisely coordinates [0, inputs) win.
    _, input_mask = proxy_scores(
        x, ids, packed, scales, None, input_keep=inputs, stable_ties=True
    )
    assert torch.equal(
        input_mask, (torch.arange(h, device="cuda") < inputs).expand_as(x)
    )
    operator = ProxBaseline(
        gu,
        down,
        packed,
        scales,
        2.0 if floor else None,
        1.0 if floor else None,
        input_keep=None if floor else inputs,
        channel_keep=None if floor else keep,
        channel_floor=keep if floor else 0,
        channel_scope="token",
        recipe="study",
    )
    diagnostics = {}
    result = operator(x, ids, mix, diagnostics=diagnostics)
    torch.testing.assert_close(
        diagnostics["input_mask"], torch.zeros_like(input_mask) if floor else input_mask
    )
    expected_channels = torch.arange(k * i, device="cuda") < keep
    torch.testing.assert_close(
        diagnostics["channel_mask"].view(2, k * i), expected_channels.expand(2, -1)
    )
    expected = torch.zeros_like(x, dtype=torch.bool)
    expected.scatter_(1, ids[:, :1] * i + torch.arange(keep, device="cuda"), True)
    assert torch.equal(result != 0, expected)
    # Exercise ties under graph replay as well: no host-side decisions or sync.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = operator(x, ids, mix)
    graph.replay()
    torch.testing.assert_close(replayed, result)


@pytest.mark.parametrize("rule", ["uniform", "threshold", "fractional", "exact"])
def test_allocation_keeps_prox_rules_and_charges_its_counts(rule):
    # Uniform at the top-k count is the joint study top-k; threshold is its mask. A
    # fractional mean follows the depth buffer across graph replays; the exact score
    # ranks exact activations and reads full gate/up rows with no proxy.
    torch.manual_seed(57)
    h, i, experts, k, count, keep = 128, 32, 4, 2, 4, 19
    x = torch.randn(count, h, device="cuda").bfloat16()
    x *= torch.tensor([0.02, 0.3, 1.0, 2.0], device="cuda")[:, None]
    ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], device="cuda")
    mix = torch.full((count, k), 0.5, device="cuda")
    padding = torch.tensor([False, False, False, True], device="cuda")
    gu = (torch.randn(experts, 2 * i, h, device="cuda") / 10).bfloat16()
    down = torch.zeros(experts, h, i, device="cuda", dtype=torch.bfloat16)
    coordinate = torch.arange(i, device="cuda")
    for expert in range(experts):
        down[expert, expert * i + coordinate, coordinate] = 1
    down = channel_major_down(down)
    packed, scales = quantize_proxy(gu)
    exact = rule == "exact"
    context = TokenContext(
        "threshold" if rule == "threshold" else "uniform",
        1,
        count,
        1,
        torch.device("cuda"),
        recording=True,
    )
    target = 19.5 if rule == "fractional" else float(keep)
    allocation = LayerAllocation(context, 0, k * i, keep, target, 0.02)
    op = ProxBaseline(
        gu,
        down,
        packed,
        scales,
        None if exact else 0.1,
        0.02,
        channel_scope="token",
        recipe="study",
        proxy_score="exact" if exact else "prox",
        allocation=allocation,
    )
    tokens = slice(0, count)

    def membership(result):
        return torch.stack(
            [
                result.gather(1, ids[:, slot, None] * i + coordinate) != 0
                for slot in range(k)
            ],
            1,
        ).flatten(1)

    result = op(x, ids, mix, padding, tokens=tokens)
    observed = membership(result)
    counts = observed.sum(-1)
    assert torch.equal(context.counts[:, 0].long(), counts)
    report = op.report()
    assert report["gate_coefficients" if not exact else "down_coefficients"] == (
        int(counts.sum()) * h
    )
    # Recorded per-token bytes sum to the operator's read cost.
    assert context.charged[:, 0].sum().item() == report["charged_bytes"]
    realized = cast(dict[str, Any], report["allocation"])["realized_count"]
    assert realized == counts[:3].double().mean().item()
    if rule in {"uniform", "threshold"}:
        reference = ProxBaseline(
            gu,
            down,
            packed,
            scales,
            0.1,
            None if rule == "uniform" else 0.02,
            channel_keep=keep if rule == "uniform" else None,
            channel_scope="token",
            recipe="study",
        )
        assert torch.equal(result, reference(x, ids, mix, padding))
    elif exact:
        g, u = (
            torch.einsum("th,tsih->tsi", x.float(), gu[ids].float())
            .bfloat16()
            .float()
            .chunk(2, dim=-1)
        )
        score = (torch.nn.functional.silu(g) * u).abs().flatten(1)
        assert torch.equal(counts, torch.tensor([keep] * 3 + [0], device="cuda"))
        worst = torch.where(observed, score, torch.inf).amin(-1)[:3]
        best = torch.where(observed, -torch.inf, score).amax(-1)[:3]
        assert (worst >= best - 1e-2 * score.amax(-1)[:3]).all()
        assert report["proxy_bytes"] == 0 and report["scale_bytes"] == 0
        assert report["up_coefficients"] == 3 * k * i * h
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = op(x, ids, mix, padding, tokens=tokens)
        for start in (0, 5, 11):
            depth = torch.arange(start, start + count, device="cuda")
            context.depth.copy_(depth)
            graph.replay()
            expected = integer_counts(
                torch.full((count,), target, device="cuda"),
                context.seeds[context.slot[:count]],
                depth,
                0,
                k * i,
            ).masked_fill(padding, 0)
            assert torch.equal(context.counts[:, 0].long(), expected)
            assert torch.equal(membership(replayed).sum(-1), expected)
