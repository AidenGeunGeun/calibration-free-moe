import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from calibration_free_moe.activation import GatedActivation, activate
from calibration_free_moe.reader.budget import compile_budget
from calibration_free_moe.reader.progressive import (
    ProgressiveReader,
    SelectionScore,
    channel_major_down,
)
from calibration_free_moe.reader.sketch import build_sketch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused MoE")
@pytest.mark.parametrize("tokens", [1, 32, 128])
def test_fused_moe_prefill_accepts_channel_major_down(tokens: int) -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    torch.manual_seed(79)
    e, h, i, k = 4, 256, 256, 2
    x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
    gu = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16)
    native = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(e, (tokens, k), device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, k, device="cuda")
    expected = fused_experts(x, gu, native, weights, ids)
    actual = fused_experts(x, gu, channel_major_down(native), weights, ids)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.015)


def unpack(codes: Tensor, rows: int, hidden: int) -> Tensor:
    """Decode sketch codes to trits in {-1, 0, 1}, shaped [experts, rows, hidden].

    Reconstructs coordinate runs from full tiles and block-paired tail sectors
    independently of the library's unpacking.
    """
    stride = (2 * ((rows // 2 + 4) // 5) + 15) // 16 * 16
    body, blocks = stride // 32 * 32, hidden // 128
    runs = codes[..., : blocks * 128 * body].reshape(codes.shape[0], blocks, 128, body)
    if stride % 32:
        tails = codes[..., blocks * 128 * body :].reshape(
            -1, (blocks + 1) // 2, 128, 2, 16
        )
        tails = tails.transpose(2, 3).reshape(-1, (blocks + 1) // 2 * 2, 128, 16)[
            :, :blocks
        ]
        runs = torch.cat((runs, tails), dim=-1)
    row = torch.arange(rows, device=codes.device)
    channel = row % (rows // 2)
    trits = (
        runs[..., 2 * (channel // 5) + row // (rows // 2)].long()
        // (3 ** (channel % 5))
    ) % 3 - 1
    return trits.movedim(-1, -3).flatten(-2)


def _hadamard128() -> Tensor:
    """Build the unnormalized 128 x 128 Sylvester Hadamard matrix (entries +-1)."""
    block = torch.ones(1, 1)
    while len(block) < 128:
        block = torch.cat(
            (torch.cat((block, block), dim=1), torch.cat((block, -block), dim=1)),
            dim=0,
        )
    return block


def _dense_block_rotation(signs: Tensor) -> Tensor:
    """Independent Sylvester-matrix reference for the real rotated screen."""
    matrix = torch.block_diag(*([_hadamard128()] * (signs.numel() // 128))) / 128**0.5
    return matrix * signs.float()[None, :]


def _reader_rotation(reader: ProgressiveReader) -> Tensor:
    """Return the reader's sketch rotation as a dense [hidden, hidden] matrix."""
    return _dense_block_rotation(reader.sketch_signs.cpu())


def test_ternary_packing_and_residual_variance():
    torch.manual_seed(33)
    weights = torch.randn(2, 8, 256).bfloat16()
    codes, metadata = build_sketch(weights)
    original = weights.float().reshape(2, 8, 2, 128)
    keep = original.abs() > 0.7 * original.abs().mean(-1, keepdim=True)
    expected_scale = (original.abs() * keep).sum(-1) / keep.sum(-1)
    expected_codes = torch.where(keep, original.sign(), 0).reshape_as(weights)
    torch.testing.assert_close(
        unpack(codes, 8, 256), expected_codes.long(), rtol=0, atol=0
    )
    torch.testing.assert_close(metadata[..., 0], expected_scale.half(), rtol=0, atol=0)
    residual = (
        original - expected_codes.reshape_as(original) * expected_scale[..., None]
    )
    torch.testing.assert_close(
        metadata[..., 1], residual.square().mean(-1).half(), rtol=0, atol=0
    )
    assert codes.numel() + metadata.numel() * 2 == 2 * 2 * (128 * 16 + 8 * 4)


def _reference_screen(
    reader: object, x: Tensor, ids: Tensor, weights: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Recompute the screen in Torch from decoded trits.

    Returns the sketch projections, their residual variances and the selection scores
    (the five-point Gauss-Hermite risk, or the point score).
    """
    from calibration_free_moe.reader.progressive import ProgressiveReader

    assert isinstance(reader, ProgressiveReader)
    p = reader.plan
    h, i = p.hidden_size, p.expert_width
    codes = unpack(reader.sketch_codes.cpu(), 2 * i, h).float()
    scale, variance = reader.sketch_statistics.cpu().float().unbind(-1)
    sketch = codes * scale.repeat_interleave(128, -1)
    x = x.float() @ _reader_rotation(reader).T
    kept = p.screen_coordinates
    if kept < 128:
        # Partial-sketch screen: each block's top-|Rx| coordinates, the rest zero.
        blocks = x.reshape(len(x), -1, 128)
        top = blocks.abs().argsort(dim=-1, descending=True, stable=True)[..., :kept]
        x = torch.zeros_like(blocks).scatter(-1, top, blocks.gather(-1, top))
        x = x.reshape(len(x), h)
    ids = ids.long()
    weights = weights.float()
    ordered = torch.argsort(ids, dim=1, stable=True)
    ids, weights = ids.gather(1, ordered), weights.gather(1, ordered)
    # These are parameterized Gaussian residual variances, not exact errors for x.
    energy = x.reshape(-1, h // 128, 128).square().sum(-1)
    projected = torch.stack(
        [
            (sketch[ids[:, j]] @ x[:, :, None]).reshape(len(x), 2, i)
            for j in range(p.top_k)
        ],
        dim=2,
    ).reshape(len(x), 2, p.candidate_count)
    sigma2 = torch.stack(
        [
            (variance[ids[:, j]] @ energy[:, :, None]).reshape(len(x), 2, i)
            for j in range(p.top_k)
        ],
        dim=2,
    ).reshape(len(x), 2, p.candidate_count)
    nodes = (
        -2.8569700138728056,
        -1.355626179974266,
        0.0,
        1.355626179974266,
        2.8569700138728056,
    )
    quadrature = (
        0.011257411327720693,
        0.2220759220056126,
        0.5333333333333333,
        0.2220759220056126,
        0.011257411327720693,
    )
    results = []
    for t in range(len(x)):
        g, u = projected[t].bfloat16().float()
        sg, su = sigma2[t].sqrt()
        mu = torch.zeros_like(g)
        second = torch.zeros_like(g)
        for node, mass in zip(nodes, quadrature, strict=True):
            a = activate(g + node * sg, reader.activation)
            mu += mass * a * u
            second += mass * a.square() * (u.square() + su.square())
        p_feature = (activate(g, reader.activation) * u).bfloat16().float()
        norm = reader.norms.cpu()[ids[t]].reshape(-1)
        geometry = reader.recovery_geometry
        alpha, beta = (
            geometry.cpu()[ids[t]].reshape(-1, 2).unbind(-1)
            if geometry is not None
            else (torch.zeros_like(norm), torch.zeros_like(norm))
        )
        risk = norm.square() * (second - mu.square()).clamp_min(0)
        risk += (norm * mu - alpha * p_feature).square() + (beta * p_feature).square()
        if reader.selection_score == "point":
            risk = (norm * activate(g, reader.activation) * u).square()
        results.append(risk * weights[t].repeat_interleave(i).square())
    return projected, sigma2, torch.stack(results)


def _reference_output(
    reader: object,
    gu: Tensor,
    down: Tensor,
    encoder: Tensor,
    decoder: Tensor,
    hidden: Tensor,
    ids: Tensor,
    weights: Tensor,
    padding: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recompute selection, native reads and recovery in Torch.

    Rounds to BF16 where the reader does; returns the output, the selected channels and
    the channels computed exactly (the reader's `completed` mask).
    """
    from calibration_free_moe.reader.progressive import ProgressiveReader

    assert isinstance(reader, ProgressiveReader)
    plan = reader.plan
    t, h = hidden.shape
    i, n = plan.expert_width, plan.candidate_count
    projected, _, screen_scores = _reference_screen(reader, hidden, ids, weights)
    outputs = torch.zeros(t, h)
    chosen = torch.zeros(t, n, dtype=torch.bool)
    completed = torch.zeros_like(chosen)
    for row in range(t):
        if padding is not None and bool(padding[row]):
            continue
        order = ids[row].argsort(stable=True)
        experts = ids[row, order].long()
        route = weights[row, order].float()
        actual = []
        for expert in experts:
            g, u = (gu[expert].float() @ hidden[row].float()).chunk(2)
            g, u = g.bfloat16().float(), u.bfloat16().float()
            actual.append(
                (activate(g, reader.activation).bfloat16().float() * u)
                .bfloat16()
                .float()
            )
        activation = torch.cat(actual)
        q = plan.final_count
        if q < n and reader.selection_threshold is not None:
            # Every score at or above the layer threshold, at most Q (best first).
            passing = int(
                (screen_scores[row] >= reader.selection_threshold.cpu()).sum()
            )
            kept_ids = (
                screen_scores[row]
                .argsort(descending=True, stable=True)[: min(passing, q)]
                .sort()
                .values
            )
        elif q < n:
            kept_ids = (
                screen_scores[row]
                .argsort(descending=True, stable=True)[:q]
                .sort()
                .values
            )
        else:
            kept_ids = torch.arange(n)
        completed[row, kept_ids] = True
        chosen[row, kept_ids] = True
        sketch_g, sketch_u = projected[row].bfloat16().float()
        partial = (activate(sketch_g, reader.activation) * sketch_u).bfloat16().float()
        features = torch.where(completed[row], activation, partial)
        selected = torch.zeros(h)
        latent = torch.zeros(decoder.shape[-1])
        for slot, expert in enumerate(experts):
            span = slice(slot * i, (slot + 1) * i)
            read_feature = activation[span] * chosen[row, span]
            selected += (
                down[expert].float() @ read_feature
            ).bfloat16().float() * route[slot]
            latent += (
                encoder[expert].float() @ (features[span] * ~chosen[row, span])
            ).bfloat16().float() * route[slot]
        recovery = (decoder.float() @ latent.bfloat16().float()).bfloat16().float()
        outputs[row] = (selected + recovery).bfloat16().float()
    return outputs.bfloat16(), chosen, completed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
@pytest.mark.parametrize(
    "h,width,activation,grouped,tokens,screen_keep,score,retention",
    [
        (256, 704, "gelu_tanh", False, 4, 0.625, "point", 0.125),
        (256, 512, "silu", True, 4, 0.625, "point", 0.125),
        (2816, 512, "silu", True, 16, 1.0, "recovery-aware", 0.1),
        (256, 512, "silu", False, 2, 0.625, "point", 0.125),
        (2048, 1024, "silu", False, 1, 0.625, "point", 0.1246337890625),
        (2048, 1024, "silu", False, 1, 1.0, "recovery-aware", 0.2),
        (2816, 704, "gelu_tanh", False, 1, 0.625, "point", 0.125),
        (2816, 704, "gelu_tanh", False, 1, 1.0, "recovery-aware", 0.2),
    ],
    ids=[
        "gemma-gelu",
        "grouped-partial-union",
        "grouped-full-gauss-hermite-unequal-splits",
        "two-token-native",
        "olmoe-batch1-keep80",
        "olmoe-batch1-full-gauss-hermite",
        "gemma-batch1-keep80-unequal-splits",
        "gemma-batch1-full-gauss-hermite-unequal-splits",
    ],
)
def test_screen_recovery_shapes_and_graph(
    h: int,
    width: int,
    activation: GatedActivation,
    grouped: bool,
    tokens: int,
    screen_keep: float,
    score: SelectionScore,
    retention: float,
) -> None:
    # H=2048 uses eight equal screen splits; Gemma's H=2816 gives seven three-block
    # splits and one one-block split.
    torch.manual_seed(419)
    e, k, rank = 4, 2, 32
    gu = (torch.randn(e, 2 * width, h, device="cuda") / h**0.5).bfloat16()
    down = (torch.randn(e, h, width, device="cuda") / width**0.5).bfloat16()
    b = (torch.randn(e, rank, width, device="cuda") / width**0.5).bfloat16()
    u = (torch.randn(h, rank, device="cuda") / rank**0.5).bfloat16()
    native_gu = gu.float().cpu()
    native_down = down.float().cpu()
    down = channel_major_down(down)
    plan = compile_budget(
        hidden_size=h,
        expert_width=width,
        top_k=k,
        retention=retention,
        screen_keep=screen_keep,
    )
    reader = ProgressiveReader(
        gu,
        down,
        b,
        u,
        plan,
        tokens,
        selection_score=score,
        activation=activation,
        grouped_tokens=2 if grouped else tokens + 1,
        grouped_screen_tokens=2 if grouped else tokens + 1,
        grouped_screen_splits=4,
    )
    unfused = ProgressiveReader(
        gu,
        down,
        b,
        u,
        plan,
        tokens,
        selection_score=score,
        activation=activation,
        grouped_tokens=2 if grouped else tokens + 1,
        grouped_screen_tokens=2 if grouped else tokens + 1,
        sketch=(reader.sketch_codes, reader.sketch_statistics),
        workspace=reader.workspace,
        fused_accounting=False,
        radix_select=False,
        screen_block_splits=1,
        grouped_screen_splits=1,
    )
    per_token = (
        ProgressiveReader(
            gu,
            down,
            b,
            u,
            plan,
            tokens,
            selection_score=score,
            activation=activation,
            grouped_tokens=tokens + 1,
            sketch=(reader.sketch_codes, reader.sketch_statistics),
        )
        if grouped
        else None
    )
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = (
        torch.tensor([[0, 2], [0, 2], [1, 2], [1, 2]], device="cuda", dtype=torch.int32)
        .repeat(((tokens + 3) // 4, 1))[:tokens]
        .contiguous()
    )
    weights = torch.rand(tokens, k, device="cuda")
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    # Exact block transform H diag(signs) (the unnormalized screen rotation).
    signs = reader.sketch_signs.cpu().double().reshape(h // 128, 128)
    hadamard = _hadamard128().double()

    def rotated_support(rows: Tensor) -> Tensor:
        return ((rows.double().reshape(-1, h // 128, 128) * signs) @ hadamard).ne(0)

    reader(x, ids, weights, padding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader(x, ids, weights, padding)
    for replay in range(2):
        x.normal_()
        if grouped and (screen_keep < 1.0 or replay == 0):
            # Every token has 80 nonzero rotated coordinates, but shared experts read
            # a proper partial union. Full-screen replay 1 instead uses a dense union.
            # x = diag(signs) H y has small dyadic integers, exact in BF16, and its
            # exact rotation 128 y / sqrt(128) is zero off y's support.
            y = torch.zeros(tokens, h // 128, 128, dtype=torch.float64)
            for token in range(tokens):
                for block in range(h // 128):
                    start = (token * 8 + block * 7) % 48
                    y[token, block, start : start + 80] = 1 / 16
            x.copy_(((y @ hadamard) * signs).reshape(tokens, h))
            assert torch.equal(rotated_support(x.cpu()), y.ne(0))
        weights.uniform_(0.2, 0.8)
        padding[min(1, tokens - 1)] = bool(replay)
        ids.copy_(ids.flip(1))
        reader.read_work.zero_()
        graph.replay()
        expected, selected, _ = _reference_output(
            reader,
            native_gu,
            native_down,
            b.cpu(),
            u.cpu(),
            x.cpu(),
            ids.cpu(),
            weights.cpu(),
            padding.cpu(),
        )
        active = ~padding.cpu()
        assert torch.equal(reader.selected[:tokens].cpu()[active], selected[active])
        torch.testing.assert_close(
            output.cpu().float(), expected.float(), rtol=0.04, atol=0.02
        )
        projected, _, scores = _reference_screen(
            reader, x.cpu(), ids.cpu(), weights.cpu()
        )
        omitted = (active[:, None] & ~selected)[:, None, :].expand_as(projected)
        torch.testing.assert_close(
            reader.partial[:tokens].cpu()[omitted],
            projected[omitted],
            rtol=2e-4,
            atol=2e-5,
        )
        torch.testing.assert_close(
            reader.scores[0][:tokens].cpu()[active],
            scores[active],
            rtol=0.025,
            atol=2e-5,
        )
        if per_token is not None:
            reference = per_token(x, ids, weights, padding)
            torch.testing.assert_close(
                output.float(), reference.float(), rtol=0.02, atol=0.01
            )
            assert torch.equal(
                reader.selected[:tokens][~padding],
                per_token.selected[:tokens][~padding],
            )
            requested = 0
            partial_union = False
            for expert in range(e):
                members = ((ids.cpu() == expert).any(-1) & active).nonzero().flatten()
                for members_block in members.split(16):
                    if not members_block.numel():
                        continue
                    union = rotated_support(x.cpu()[members_block]).any(0)
                    partial_union |= bool(
                        ((union.sum(-1) > 80) & (union.sum(-1) < 128)).any()
                    )
                    # I=512: six full sectors and one block-paired tail sector
                    # per coordinate, plus the FP16 scale plane per block.
                    requested += int(union.sum()) * 7 * 32 + (h // 128) * 4 * width * (
                        1 + (score != "point")
                    )
            assert partial_union == (screen_keep < 1.0 or replay == 0)
            assert reader.report()["issued_sketch_bytes"] == requested
        else:
            assert reader.report()["issued_sketch_bytes"] == int(
                active.sum()
            ) * plan.sketch_read_bytes(residual=score != "point")
        # Shared scratch must not retain another call's membership or bucket keys.
        membership = reader.selected[:tokens].clone()
        unfused.read_work.zero_()
        reference = unfused(x, ids, weights, padding)
        assert torch.equal(membership, unfused.selected[:tokens])
        torch.testing.assert_close(
            output.float(), reference.float(), rtol=0.04, atol=0.02
        )
        torch.testing.assert_close(reader.read_work, unfused.read_work, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
@pytest.mark.parametrize("reduction", ["screen", "recovery", "pdl"])
def test_qwen3_p10_last_block_screen_reduction_and_graph(reduction: str) -> None:
    # One deployed Qwen3 layer: the 768-channel expert has a 48-channel final screen
    # tile. Replays vary the input, padding, all-zero ties and equal-score experts;
    # each `reduction` variant is compared with the reader without it.
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )

    if reduction == "pdl" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL requires SM90+")
    torch.manual_seed(421)
    e, h, i, k, rank, capacity = 128, 2048, 768, 8, 32, 8
    gu = torch.empty(e, 2 * i, h, device="cuda", dtype=torch.bfloat16).normal_(
        std=h**-0.5
    )
    down = (
        torch.empty(e, i, h, device="cuda", dtype=torch.bfloat16)
        .normal_(std=i**-0.5)
        .transpose(1, 2)
    )
    # Equal native experts give exact nonzero score ties across routed slots.
    gu[1:k].copy_(gu[:1].expand(k - 1, -1, -1))
    down[1:k].copy_(down[:1].expand(k - 1, -1, -1))
    encoder = torch.empty(e, rank, i, device="cuda", dtype=torch.bfloat16).normal_(
        std=i**-0.5
    )
    decoder = torch.empty(h, rank, device="cuda", dtype=torch.bfloat16).normal_(
        std=rank**-0.5
    )
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=0.123046875,
        screen_keep=0.625,
    )
    fast = ProgressiveReader(
        gu,
        down,
        encoder,
        decoder,
        plan,
        capacity,
        selection_score="point",
        pdl=True,
    )
    reference = ProgressiveReader(
        gu,
        down,
        encoder,
        decoder,
        plan,
        capacity,
        selection_score="point",
        sketch=(fast.sketch_codes, fast.sketch_statistics),
        workspace=fast.workspace,
        screen_last_block_reduction=reduction != "screen",
        recovery_last_block_reduction=reduction != "recovery",
        radix_select=reduction != "screen",
        pdl=False,
    )
    gate = torch.empty(e, h, device="cuda", dtype=torch.bfloat16).normal_(std=h**-0.5)

    def routed(
        reader: ProgressiveReader,
        x: Tensor,
        padding: Tensor,
        early_prepare: bool = False,
    ) -> Tensor:
        input_prepared = early_prepare and reader.prepare_input(x, padding)
        weights, ids, _ = fused_topk(x, F.linear(x, gate), k, renormalize=True)
        return reader(x, ids, weights, padding, input_prepared=input_prepared)

    for tokens in (1, 2, 4, 8):
        x = torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16)
        padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")

        # Compile both paths and initialize the shared recovery events before
        # capturing. Each graph uses caller-owned output, not shared scratch.
        actual = routed(fast, x, padding, early_prepare=True)
        latent = fast.latent[:tokens].clone()
        expected = routed(reference, x, padding, early_prepare=reduction == "pdl")
        if reduction == "recovery":
            torch.testing.assert_close(
                latent, reference.latent[:tokens], rtol=0.015, atol=0.01
            )
            torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.02)
        torch.cuda.synchronize()
        fast_graph, reference_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(fast_graph):
            actual = routed(fast, x, padding, early_prepare=True)
        with torch.cuda.graph(reference_graph):
            expected = routed(reference, x, padding, early_prepare=reduction == "pdl")
        for replay in range(4):
            padding.zero_()
            if replay == 2:
                x.zero_()
            else:
                x.normal_()
            if replay == 1:
                padding[-1] = True
            if replay == 3:
                # Route the equal native experts with equal nonzero weights.
                x[:, 0] = 1
                gate.zero_()
                gate[:k, 0] = 1
            else:
                gate.normal_(std=h**-0.5)
            fast.read_work.zero_()
            reference.read_work.zero_()
            # POINT must not depend on residual-variance scratch from another call.
            fast.variances.fill_(torch.nan)
            fast_graph.replay()
            scores = fast.scores[0][:tokens].clone()
            selected = fast.selected[:tokens].clone()
            latent = fast.latent[:tokens].clone()
            assert not bool(torch.count_nonzero(fast.screen_counters))
            # Independent trit decoding and coordinate reduction, using exactly
            # the rotated coordinates and FP16 scales the real reader kept.
            projected = []
            for token in range(tokens):
                experts = fast.ids[token].long()
                trits = unpack(fast.sketch_codes[experts], 2 * i, h).float()
                trits = trits.reshape(k, 2, i, h // 128, 128)
                coordinates = fast.sketch_coordinates[token].long()
                kept = trits.gather(-1, coordinates.expand(k, 2, i, -1, -1))
                dot = (kept * fast.sketch_planes[token]).sum(-1)
                scale = fast.sketch_statistics[experts, :, :, 0].float()
                estimate = (dot * scale.reshape(k, 2, i, h // 128)).sum(-1)
                projected.append(estimate.permute(1, 0, 2).reshape(2, -1))
            projected = torch.stack(projected)
            splits = fast.screen_splits if tokens == 1 else fast.batch_screen_splits
            parts = fast.screen_parts[: tokens * splits].reshape(
                tokens, splits, 2, plan.candidate_count
            )
            torch.testing.assert_close(parts.sum(1), projected, rtol=1e-5, atol=2e-5)
            projected_gate, projected_up = projected.bfloat16().float().unbind(1)
            norms = fast.norms[fast.ids[:tokens].long()].reshape(tokens, -1)
            point = norms * (activate(projected_gate, "silu") * projected_up)
            route_weights = fast.weights[:tokens].repeat_interleave(i, -1)
            decoded_scores = route_weights.square() * point.square()
            torch.testing.assert_close(scores, decoded_scores, rtol=0.025, atol=2e-5)
            ordered = decoded_scores.argsort(descending=True, stable=True)
            decoded_selection = torch.zeros_like(selected).scatter_(
                1, ordered[:, : plan.final_count], True
            )
            # Torch's different FP32 reduction tree can cross a BF16 boundary.
            cutoff = decoded_scores.gather(
                1, ordered[:, plan.final_count - 1 : plan.final_count]
            )
            error = (scores - decoded_scores).abs().amax(1, keepdim=True)
            flipped = selected ^ decoded_selection
            assert torch.equal(selected.sum(1), decoded_selection.sum(1))
            assert bool(
                (((decoded_scores - cutoff).abs() <= 2 * error) | ~flipped).all()
            )
            if replay == 3:
                assert torch.equal(
                    scores.reshape(tokens, k, i),
                    scores[:, None, :i].expand(tokens, k, i),
                )
            top_ids = (
                scores.cpu()
                .argsort(dim=1, descending=True, stable=True)[:, : plan.final_count]
                .sort(dim=1)
                .values
            )
            membership = torch.zeros_like(selected, device="cpu").scatter_(
                1, top_ids, True
            )
            assert torch.equal(selected.cpu(), membership)
            assert torch.equal(fast.candidate_ids[-1][:tokens].cpu(), top_ids)
            counts = membership.reshape(tokens, k, i).sum(-1)
            starts = counts.cumsum(-1) - counts
            tiles = (counts + 127) // 128
            tile_starts = tiles.cumsum(-1) - tiles
            assert torch.equal(fast.counts[:tokens].cpu(), counts)
            assert torch.equal(fast.starts[:tokens].cpu(), starts)
            assert torch.equal(fast.tile_starts[:tokens].cpu(), tile_starts)
            reference_graph.replay()
            torch.testing.assert_close(
                scores,
                reference.scores[0][:tokens],
                rtol=0,
                atol=0,
            )
            reference_selected = reference.selected[:tokens]
            assert torch.equal(selected, reference_selected)
            if reduction == "recovery":
                torch.testing.assert_close(
                    latent, reference.latent[:tokens], rtol=0.015, atol=0.01
                )
            if replay == 2:
                logical = torch.arange(plan.candidate_count, device="cuda")
                assert torch.equal(
                    selected, (logical < plan.final_count).expand_as(selected)
                )
            # Screen-reduction and PDL variants only reorder execution; recovery
            # reduction changes arithmetic.
            exact = reduction in ("screen", "pdl")
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0 if exact else 0.04,
                atol=0 if exact else 0.02,
            )
            torch.testing.assert_close(
                fast.read_work, reference.read_work, rtol=0, atol=0
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_radix_selection_ties_and_bucket_overflow(tokens: int) -> None:
    # The short final sketch tile must not enter the histogram. All-zero inputs
    # put every channel in one overflowing bucket and select the lowest IDs.
    torch.manual_seed(417)
    h, i, k = 128, 528, 8
    gu = (torch.randn(k, 2 * i, h, device="cuda") / h**0.5).bfloat16()
    down = channel_major_down((torch.randn(k, h, i, device="cuda") / i**0.5).bfloat16())
    plan = compile_budget(
        hidden_size=h, expert_width=i, top_k=k, retention=0.125, screen_keep=0.625
    )
    fast = ProgressiveReader(
        gu,
        down,
        None,
        None,
        plan,
        tokens,
        selection_score="point",
    )
    slow = ProgressiveReader(
        gu,
        down,
        None,
        None,
        plan,
        tokens,
        selection_score="point",
        radix_select=False,
        fused_accounting=False,
        sketch=(fast.sketch_codes, fast.sketch_statistics),
        workspace=fast.workspace,
    )
    ids = (
        torch.arange(k - 1, -1, -1, device="cuda", dtype=torch.int32)
        .expand(tokens, -1)
        .contiguous()
    )
    weights = torch.full((tokens, k), 1 / k, device="cuda")
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    for tied in (False, True):
        if tied:
            x.zero_()
        fast.read_work.zero_()
        slow.read_work.zero_()
        actual = fast(x, ids, weights)
        chosen = fast.selected[:tokens].clone()
        top_ids = (
            fast.scores[0][:tokens]
            .cpu()
            .argsort(dim=1, descending=True, stable=True)[:, : plan.final_count]
            .sort(dim=1)
            .values
        )
        assert torch.equal(fast.candidate_ids[-1][:tokens].cpu(), top_ids)
        reference = slow(x, ids, weights)
        assert torch.equal(chosen, slow.selected[:tokens])
        if tied:
            expected = (
                torch.arange(plan.candidate_count, device="cuda") < plan.final_count
            )
            assert torch.equal(chosen, expected.expand_as(chosen))
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
        torch.testing.assert_close(fast.read_work, slow.read_work, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
@pytest.mark.parametrize(
    "retention,screen_keep", [(0.125, 1.0), (1.0, 1.0), (0.125, 0.5)]
)
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
def test_sketch_reader_output_with_nonpower_width(
    retention: float,
    screen_keep: float,
    activation: GatedActivation,
) -> None:
    """A partial-sketch screen's recovery uses its masked-input sketch features."""
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(77)
    h, i, e, k, rank = 384, 160, 4, 2, 8
    gu = (torch.randn(e, 2 * i, h) / h**0.5).bfloat16()
    down = (torch.randn(e, h, i) / i**0.5).bfloat16()
    encoder = (torch.randn(e, rank, i) / i**0.5).bfloat16()
    decoder = (torch.randn(h, rank) / rank**0.5).bfloat16()
    hidden = torch.randn(2, h).bfloat16()
    ids = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=retention,
        screen_keep=screen_keep,
    )
    reader = ProgressiveReader(
        gu.cuda(),
        channel_major_down(down.cuda()),
        encoder.cuda(),
        decoder.cuda(),
        plan,
        len(hidden),
        activation=activation,
    )
    rotation = _reader_rotation(reader)
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(h), atol=1e-6, rtol=0)
    # The deployed setup rotates FP32 BF16 weights; no intermediate BF16 rounding.
    reference_codes, reference_stats = build_sketch(gu.float() @ rotation.T)
    torch.testing.assert_close(
        reader.sketch_codes.cpu(), reference_codes, atol=0, rtol=0
    )
    torch.testing.assert_close(
        reader.sketch_statistics.cpu(), reference_stats, atol=1e-4, rtol=1e-3
    )
    expected, selected, completed = _reference_output(
        reader,
        gu,
        down,
        encoder,
        decoder,
        hidden,
        ids,
        weights,
    )
    actual = reader(hidden.cuda(), ids.cuda(), weights.cuda()).cpu()
    torch.testing.assert_close(reader.selected.cpu(), selected, rtol=0, atol=0)
    torch.testing.assert_close(reader.completed.cpu(), completed, rtol=0, atol=0)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.025, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reader")
def test_sketch_reader_graph_replay_updates_routes_and_padding() -> None:
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(181)
    h, i, e, k, rank = 128, 64, 4, 2, 8
    gu = (torch.randn(e, 2 * i, h) * 0.08).bfloat16()
    down = (torch.randn(e, h, i) * 0.08).bfloat16()
    b = (torch.randn(e, rank, i) * 0.08).bfloat16()
    u = (torch.randn(h, rank) * 0.08).bfloat16()
    plan = compile_budget(hidden_size=h, expert_width=i, top_k=k, retention=0.25)
    reader = ProgressiveReader(
        gu.cuda(), channel_major_down(down.cuda()), b.cuda(), u.cuda(), plan, 2
    )
    x = torch.randn(2, h, device="cuda").bfloat16()
    ids = torch.tensor([[0, 2], [1, 3]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], device="cuda")
    padding = torch.tensor([False, True], device="cuda")
    reader(x, ids, weights, padding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader(x, ids, weights, padding)
    for seed, routes, pad in (
        (11, [[2, 0], [1, 3]], [False, True]),
        (29, [[3, 1], [2, 0]], [False, False]),
    ):
        x.copy_(
            torch.randn(2, h, generator=torch.Generator().manual_seed(seed))
            .bfloat16()
            .cuda()
        )
        ids.copy_(torch.tensor(routes, dtype=torch.int32, device="cuda"))
        padding.copy_(torch.tensor(pad, device="cuda"))
        expected, selected, _ = _reference_output(
            reader, gu, down, b, u, x.cpu(), ids.cpu(), weights.cpu(), padding.cpu()
        )
        graph.replay()
        torch.testing.assert_close(
            reader.selected.cpu()[~padding.cpu()],
            selected[~padding.cpu()],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output.cpu().float(), expected.float(), rtol=0.025, atol=0.02
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reader")
@pytest.mark.parametrize("retention", [0.125, 1.0])
def test_grouped_reader_matches_per_token_reads(retention: float) -> None:
    # Expert-major execution reassociates FP32 sums. On this fixture no selection sits
    # within that error of its threshold, so decisions match and outputs agree within
    # BF16 tolerance.
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(211)
    h, i, e, k, rank, tokens = 384, 160, 6, 2, 8, 24
    gu = (torch.randn(e, 2 * i, h, device="cuda") / h**0.5).bfloat16()
    down = channel_major_down((torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16())
    b = (torch.randn(e, rank, i, device="cuda") / i**0.5).bfloat16()
    u = (torch.randn(h, rank, device="cuda") / rank**0.5).bfloat16()
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=retention,
    )
    per_token = ProgressiveReader(
        gu, down, b, u, plan, tokens, grouped_tokens=tokens + 1
    )
    sketch = (per_token.sketch_codes, per_token.sketch_statistics)
    grouped = ProgressiveReader(
        gu,
        down,
        b,
        u,
        plan,
        tokens,
        sketch=sketch,
        grouped_tokens=2,
        grouped_screen_tokens=2,
    )
    mixed = ProgressiveReader(
        gu,
        down,
        b,
        u,
        plan,
        tokens,
        sketch=sketch,
        grouped_tokens=2,
        grouped_screen_tokens=tokens + 1,
    )
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(tokens)])
    ids = ids.int().contiguous()
    weights = torch.rand(tokens, k, device="cuda")
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    grouped(x, ids, weights, padding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = grouped(x, ids, weights, padding)
    grouped.read_work.zero_()
    mixed.read_work.zero_()
    screened_blocks = 0
    for seed, pad in ((5, [False] * tokens), (6, [t % 7 == 3 for t in range(tokens)])):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(torch.randn(tokens, h, device="cuda", generator=generator))
        ids.copy_(torch.argsort(torch.rand(tokens, e, device="cuda"), -1)[:, :k])
        padding.copy_(torch.tensor(pad, device="cuda"))
        expected = per_token(x, ids, weights, padding).clone()
        graph.replay()
        after_token_screen = mixed(x, ids, weights, padding)
        active = ~padding
        for reader, result in ((grouped, output), (mixed, after_token_screen)):
            assert torch.equal(
                reader.selected[:tokens][active], per_token.selected[:tokens][active]
            )
            assert torch.equal(
                reader.completed[:tokens][active],
                per_token.completed[:tokens][active],
            )
            assert torch.equal(result[padding], expected[padding])
            # FP32 accumulation order differs; values keep their BF16 boundaries.
            torch.testing.assert_close(
                result.float(), expected.float(), rtol=0.016, atol=2e-3
            )
        # The expert-major screen reads each 16-row expert block's sketch once.
        pairs = torch.bincount(ids[active].flatten(), minlength=e)
        screened_blocks += int(((pairs + 15) // 16).sum())
    token_report, grouped_report = per_token.report(), grouped.report()
    for key in ("charged_bytes", "logical_charged_bytes", "reference_bytes"):
        assert token_report[key] == grouped_report[key] == mixed.report()[key]
    for key in ("issued_gate_up_bytes", "issued_down_bytes"):
        assert float(grouped_report[key] or 0) <= float(token_report[key] or 0)
    padded = sum(t % 7 == 3 for t in range(tokens))
    assert grouped_report["grouped_rows"] == 2 * tokens - padded
    assert not token_report["grouped_rows"]
    # Per-token screens (alone or before grouped stages) request exactly the sketch
    # rows in the read cost; the expert-major screen requests each block's sketches.
    screen = plan.completion_count < plan.candidate_count
    assert token_report["issued_sketch_bytes"] == token_report["physical_sketch_bytes"]
    assert (
        mixed.report()["issued_sketch_bytes"] == token_report["physical_sketch_bytes"]
    )
    assert (
        grouped_report["issued_sketch_bytes"]
        == screened_blocks * plan.sketch_read_bytes() // k
    )
    assert bool(token_report["sketch_bytes"]) == screen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reader")
def test_grouped_reader_repeats_exactly_at_many_pairs() -> None:
    # Expert blocks, hence union order and FP32 sums, are fixed by the routes. With
    # 1024 (token, slot) pairs, vLLM's moe_align_block_size places pairs by atomics,
    # so using it would make replays of one input differ in the last BF16 bit.
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(223)
    h, i, e, k, rank, tokens = 256, 1024, 8, 8, 8, 160
    gu = (torch.randn(e, 2 * i, h, device="cuda") / h**0.5).bfloat16()
    down = channel_major_down((torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16())
    b = (torch.randn(e, rank, i, device="cuda") / i**0.5).bfloat16()
    u = (torch.randn(h, rank, device="cuda") / rank**0.5).bfloat16()
    plan = compile_budget(hidden_size=h, expert_width=i, top_k=k, retention=0.1)
    reader = ProgressiveReader(gu, down, b, u, plan, tokens)
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.stack([torch.randperm(e, device="cuda") for _ in range(tokens)]).int()
    weights = torch.rand(tokens, k, device="cuda")
    reader(x, ids, weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader(x, ids, weights)
    graph.replay()
    first = output.clone()
    for _ in range(8):
        graph.replay()
        assert torch.equal(output, first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reader")
@pytest.mark.parametrize("tokens", [1, 12, 64])
@pytest.mark.parametrize("different_counts", [False, True])
def test_layers_sharing_a_workspace_match_private_workspaces(
    tokens: int, different_counts: bool
) -> None:
    from calibration_free_moe.reader.progressive import (
        ProgressiveReader,
        ReaderWorkspace,
    )

    torch.manual_seed(223)
    h, i, e, k, rank = 256, 512 if different_counts else 128, 6, 2, 8
    capacity = max(16, tokens)
    plans = [
        compile_budget(
            hidden_size=h,
            expert_width=i,
            top_k=k,
            retention=(0.125 if layer == 0 else 0.25) if different_counts else 0.125,
            screen_keep=0.625 if different_counts else 1.0,
        )
        for layer in range(2)
    ]
    layers = [
        (
            (torch.randn(e, 2 * i, h, device="cuda") / h**0.5).bfloat16(),
            channel_major_down(
                (torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16()
            ),
            (torch.randn(e, rank, i, device="cuda") / i**0.5).bfloat16(),
            (torch.randn(h, rank, device="cuda") / rank**0.5).bfloat16(),
        )
        for _ in range(2)
    ]
    private = [
        ProgressiveReader(*layer, plan, capacity)
        for layer, plan in zip(layers, plans, strict=True)
    ]
    shared_scratch = ReaderWorkspace(
        max(plans, key=lambda p: p.final_count),
        e,
        rank,
        capacity,
        torch.device("cuda", 0),
    )
    shared = [
        ProgressiveReader(
            *layer,
            plan,
            capacity,
            sketch=(reader.sketch_codes, reader.sketch_statistics),
            workspace=shared_scratch,
        )
        for layer, plan, reader in zip(layers, plans, private, strict=True)
    ]
    assert shared[0].workspace_bytes == shared_scratch.bytes
    assert shared[1].workspace_bytes == 0
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(tokens)])
    ids = ids.int().contiguous()
    weights = torch.rand(tokens, k, device="cuda")

    def forward(
        readers: list[ProgressiveReader], early_prepare: bool = False
    ) -> tuple[Tensor, Tensor]:
        first = readers[0](
            x,
            ids,
            weights,
            input_prepared=early_prepare and readers[0].prepare_input(x),
        )
        return first, readers[1](
            first,
            ids.flip(-1).contiguous(),
            weights,
            input_prepared=early_prepare and readers[1].prepare_input(first),
        )

    forward(shared, early_prepare=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward(shared, early_prepare=True)
    for reader in shared:
        reader.read_work.zero_()
    for seed in (3, 4):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(torch.randn(tokens, h, device="cuda", generator=generator))
        expected = forward(private)
        graph.replay()
        for actual, reference in zip(captured, expected, strict=True):
            torch.testing.assert_close(
                actual.float(), reference.float(), rtol=0.016, atol=2e-3
            )
        assert torch.equal(
            shared_scratch.selected[:tokens], private[1].selected[:tokens]
        )
    for reader, reference in zip(shared, private, strict=True):
        assert torch.equal(reader.read_work[:11], reference.read_work[:11])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
@pytest.mark.parametrize("tokens", [1, 4])
def test_selection_breaks_score_ties_in_channel_order(tokens: int) -> None:
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(227)
    h, i, e, k, rank = 256, 128, 4, 2, 8
    plan = compile_budget(hidden_size=h, expert_width=i, top_k=k, retention=0.75)
    reader = ProgressiveReader(
        (torch.randn(e, 2 * i, h, device="cuda") / h**0.5).bfloat16(),
        channel_major_down((torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16()),
        (torch.randn(e, rank, i, device="cuda") / i**0.5).bfloat16(),
        (torch.randn(h, rank, device="cuda") / rank**0.5).bfloat16(),
        plan,
        tokens,
    )
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    # Expert 1 sorts into slot 0 with zero routing weight: its scores all tie at zero,
    # and the earliest of them win.
    ids = torch.tensor([[3, 1]] * tokens, device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.8, 0.0]] * tokens, device="cuda")
    reader(x, ids, weights)
    scores = reader.scores[0][:tokens].cpu()
    assert bool((scores[:, :i] == 0).all()) and bool((scores[:, i:] > 0).all())
    expected = torch.zeros(tokens, 2 * i, dtype=torch.bool)
    expected[:, : plan.final_count - i] = True
    expected[:, i:] = True
    assert torch.equal(reader.selected[:tokens].cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
@pytest.mark.parametrize("tokens", [1, 4])
def test_point_score_with_bank_reads_no_geometry(tokens: int) -> None:
    """A recovery map changes no selection and adds only its own reads."""
    torch.manual_seed(241)
    h, i, e, k, rank = 256, 64, 4, 2, 8
    gu = (torch.randn(e, 2 * i, h, device="cuda") / h**0.5).bfloat16()
    down = channel_major_down((torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16())
    encoder = (torch.randn(e, rank, i, device="cuda") / i**0.5).bfloat16()
    decoder = (torch.randn(h, rank, device="cuda") / rank**0.5).bfloat16()
    plan = compile_budget(hidden_size=h, expert_width=i, top_k=k, retention=0.25)
    bank = ProgressiveReader(
        gu, down, encoder, decoder, plan, tokens, selection_score="point"
    )
    sketch = (bank.sketch_codes, bank.sketch_statistics)
    none = ProgressiveReader(
        gu, down, None, None, plan, tokens, sketch=sketch, selection_score="point"
    )
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.argsort(torch.rand(tokens, e, device="cuda"), -1)[:, :k]
    ids = ids.int().contiguous()
    weights = torch.rand(tokens, k, device="cuda")
    bank(x, ids, weights)
    none(x, ids, weights)
    assert bank.recovery_geometry is None
    assert torch.equal(bank.selected[:tokens], none.selected[:tokens])
    with_bank, without = bank.report(), none.report()
    assert with_bank["selection_parameter_bytes"] == tokens * 4 * plan.candidate_count
    assert (
        without["selection_parameter_bytes"] == with_bank["selection_parameter_bytes"]
    )
    assert with_bank["charged_bytes"] == float(without["charged_bytes"] or 0) + float(
        with_bank["physical_recovery_bytes"] or 0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reader")
@pytest.mark.parametrize(
    "score,screen_keep", [("recovery-aware", 1.0), ("point", 0.625)]
)
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 48])
def test_selection_threshold_keeps_token_counts_and_charges_them(
    tokens: int, score: SelectionScore, screen_keep: float
) -> None:
    # The threshold keeps every score at or above it, at most F per token; reads count
    # each token's kept channels plus one threshold sector.
    torch.manual_seed(257)
    h, i, e, k, rank = 384, 160, 6, 3, 8
    n = k * i
    gu = (torch.randn(e, 2 * i, h) / h**0.5).bfloat16()
    down = (torch.randn(e, h, i) / i**0.5).bfloat16()
    encoder = (torch.randn(e, rank, i) / i**0.5).bfloat16()
    decoder = (torch.randn(h, rank) / rank**0.5).bfloat16()
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=(n - 1) / n,
        screen_keep=screen_keep,
    )
    assert plan.final_count == plan.completion_count == n - 1
    # vLLM prepares readers under its BF16 default dtype; the threshold stays FP32.
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        reader = ProgressiveReader(
            gu.cuda(),
            channel_major_down(down.cuda()),
            encoder.cuda(),
            decoder.cuda(),
            plan,
            tokens,
            selection_score=score,
            selection_threshold=0.0,
            grouped_tokens=4,
            grouped_screen_tokens=16,
        )
    finally:
        torch.set_default_dtype(default)
    assert reader.selection_threshold is not None
    assert reader.selection_threshold.dtype == torch.float32
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.argsort(torch.rand(tokens, e, device="cuda"), -1)[:, :k]
    ids = ids.int().contiguous()
    weights = torch.rand(tokens, k, device="cuda")
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    reader(x, ids, weights, padding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader(x, ids, weights, padding)
    reader.read_work.zero_()
    calls = kept_total = 0
    # A mid-range threshold keeps a token-dependent count; zero keeps the N - 1 cap
    # and the largest finite float keeps nothing. Input seed 5 puts two rotated
    # coordinates 6e-7 apart (relative) at the partial screen's keep boundary,
    # where FP32 rotation reassociation decides; these inputs keep clear gaps.
    cases = (
        (7, [False] * tokens, None),
        (8, [t % 3 == 1 for t in range(tokens)], None),
        (11, [False] * tokens, 0.0),
        (12, [False] * tokens, float(torch.finfo(torch.float32).max)),
    )
    counts: list[int] = []
    for seed, pad, fixed in cases:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(torch.randn(tokens, h, device="cuda", generator=generator))
        ids.copy_(torch.argsort(torch.rand(tokens, e, device="cuda"), -1)[:, :k])
        padding.copy_(torch.tensor(pad, device="cuda"))
        threshold = fixed
        if threshold is None:
            # The widest relative gap among the middle fifth of reference scores:
            # FP32 reassociation of the kernel's scores cannot cross it.
            _, _, scores = _reference_screen(reader, x.cpu(), ids.cpu(), weights.cpu())
            values = scores.flatten().sort().values
            middle = values[2 * len(values) // 5 : 3 * len(values) // 5]
            gap = int((middle[1:] / middle[:-1].clamp_min(1e-30)).argmax())
            threshold = float((middle[gap] * middle[gap + 1]).sqrt())
        reader.selection_threshold.fill_(threshold)
        expected, selected, completed = _reference_output(
            reader,
            gu,
            down,
            encoder,
            decoder,
            x.cpu(),
            ids.cpu(),
            weights.cpu(),
            padding.cpu(),
        )
        graph.replay()
        active = ~padding.cpu()
        chosen = reader.selected[:tokens].cpu()
        assert torch.equal(chosen[active], selected[active])
        assert torch.equal(reader.completed[:tokens].cpu()[active], completed[active])
        torch.testing.assert_close(
            output.cpu().float(), expected.float(), rtol=0.025, atol=0.02
        )
        counts += chosen[active].sum(-1).tolist()
        calls += int(active.sum())
        kept_total += int(chosen[active].sum())
    middle = counts[: 2 * tokens - sum(t % 3 == 1 for t in range(tokens))]
    assert 0 < sum(middle) < len(middle) * (n - 1)
    if tokens > 1:
        assert len(set(middle)) > 1
    assert counts[-2 * tokens : -tokens] == [n - 1] * tokens
    assert counts[-tokens:] == [0] * tokens
    report = reader.report()
    residual = score != "point"
    # Kept channels' native rows; skipped channels' encoder rows and the decoder;
    # screen metadata per candidate plus one FP32 threshold (one sector) per token.
    assert report["native_expert_coefficient_bytes"] == 2 * 3 * h * kept_total
    assert report["recovery_bytes"] == 2 * rank * (calls * (n + h) - kept_total)
    geometry = 12 if score == "recovery-aware" else 4
    assert report["selection_parameter_bytes"] == calls * (geometry * n + 4)
    assert report["physical_selection_parameter_bytes"] == calls * (geometry * n + 32)
    assert report["sketch_bytes"] == calls * plan.sketch_read_bytes(residual, False)
    assert report["selection_threshold_resident_bytes"] == 4
    macs = report["matrix_macs_per_token"]
    expected_macs = (
        k * 2 * h * i + 3 * h * kept_total / calls + rank * (n - kept_total / calls + h)
    )
    assert macs == pytest.approx(expected_macs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reader")
@pytest.mark.parametrize("retention", [0.1, 0.125])
@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_no_bank_reader_matches_zero_bank_without_recovery_reads(
    tokens: int, retention: float
) -> None:
    # Without a recovery map the reader also reads no score geometry, only the FP32
    # norm per candidate.
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(233)
    h, i, e, k, rank = 256, 128, 6, 2, 8
    gu = (torch.randn(e, 2 * i, h, device="cuda") / h**0.5).bfloat16()
    down = channel_major_down((torch.randn(e, h, i, device="cuda") / i**0.5).bfloat16())
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=retention,
    )
    zero = ProgressiveReader(
        gu,
        down,
        torch.zeros(e, rank, i, device="cuda", dtype=torch.bfloat16),
        torch.zeros(h, rank, device="cuda", dtype=torch.bfloat16),
        plan,
        tokens,
    )
    sketch = (zero.sketch_codes, zero.sketch_statistics)
    none = ProgressiveReader(gu, down, None, None, plan, tokens, sketch=sketch)
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(tokens)])
    ids = ids.int().contiguous()
    weights = torch.rand(tokens, k, device="cuda")
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    none(x, ids, weights, padding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = none(x, ids, weights, padding)
    none.read_work.zero_()
    calls = 0
    for seed, pad in ((5, [False] * tokens), (6, [t % 3 == 1 for t in range(tokens)])):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(torch.randn(tokens, h, device="cuda", generator=generator))
        ids.copy_(torch.argsort(torch.rand(tokens, e, device="cuda"), -1)[:, :k])
        padding.copy_(torch.tensor(pad, device="cuda"))
        expected = zero(x, ids, weights, padding).clone()
        graph.replay()
        active = ~padding
        calls += int(active.sum())
        for name in ("selected", "completed"):
            assert torch.equal(
                getattr(none, name)[:tokens][active],
                getattr(zero, name)[:tokens][active],
            )
        assert torch.equal(output, expected)
    bank_free, reference = none.report(), zero.report()
    assert bank_free["recovery_bytes"] == bank_free["physical_recovery_bytes"] == 0
    assert bank_free["issued_recovery_encoder_bytes"] == 0
    assert reference["physical_recovery_bytes"]
    n = plan.candidate_count
    # The zero bank's recovery reads and its two geometry values per candidate.
    assert bank_free["charged_bytes"] == (
        float(reference["charged_bytes"] or 0)
        - float(reference["physical_recovery_bytes"] or 0)
        - 8 * n * calls
    )
    assert bank_free["logical_charged_bytes"] == calls * (
        plan.sketch_read_bytes(sectors=False)
        + h  # rotation signs
        + plan.selected_weight_bytes()
        + plan.selection_read_bytes(geometry=False)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reader")
@pytest.mark.parametrize(
    "score,screen_keep,tied",
    [
        ("recovery-aware", 1.0, False),
        ("point", 1.0, False),
        ("point", 0.75, False),
        ("point", 0.625, False),
        ("point", 0.625, True),
        ("recovery-aware", 0.5, False),
    ],
)
@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_bank_free_ablations_match_reference_and_read_cost(
    tokens: int, score: SelectionScore, screen_keep: float, tied: bool
) -> None:
    # A partial-sketch screen estimates partial dot products over each block's top-|Rx|
    # coordinates and reads coordinate-major code runs of the kept coordinates plus
    # every scale (and residual for the recovery-aware score).
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(239)
    # Three candidates of a non-power-of-two width (F = 50).
    h, i, e, k = 384, 160, 6, 3
    gu = (torch.randn(e, 2 * i, h) / h**0.5).bfloat16()
    down = (torch.randn(e, h, i) / i**0.5).bfloat16()
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=0.105,
        screen_keep=screen_keep,
    )
    reader = ProgressiveReader(
        gu.cuda(),
        channel_major_down(down.cuda()),
        None,
        None,
        plan,
        tokens,
        selection_score=score,
        # Exercise the expert-major stages and screen at 4 and 16 tokens.
        grouped_tokens=4,
        grouped_screen_tokens=16,
    )
    signs = reader.sketch_signs.cpu()
    rotation = _reader_rotation(reader)
    assert torch.equal(
        reader.sketch_codes.cpu(), build_sketch(gu.float() @ rotation.T)[0]
    )
    zero_bank = (torch.zeros(e, 8, i), torch.zeros(h, 8))
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.argsort(torch.rand(tokens, e, device="cuda"), -1)[:, :k]
    ids = ids.int().contiguous()
    weights = torch.rand(tokens, k, device="cuda")
    padding = torch.zeros(tokens, dtype=torch.bool, device="cuda")
    reader(x, ids, weights, padding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader(x, ids, weights, padding)
    reader.read_work.zero_()
    calls = 0
    tied_union_bytes = 0
    for seed, pad in ((5, [False] * tokens), (6, [t % 3 == 1 for t in range(tokens)])):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(torch.randn(tokens, h, device="cuda", generator=generator))
        if tied:
            # x = diag(signs) H (1/16, ..., 1/16) per block is 8 sign_0 at each
            # block's first coordinate; every rotated coordinate equals 8 / sqrt(128)
            # exactly, so the first KEEP must win in both paths.
            x.zero_()
            x[:, ::128] = 8 * signs[::128].to(device="cuda", dtype=x.dtype)
        ids.copy_(torch.argsort(torch.rand(tokens, e, device="cuda"), -1)[:, :k])
        padding.copy_(torch.tensor(pad, device="cuda"))
        expected, selected, _ = _reference_output(
            reader,
            gu,
            down,
            *zero_bank,
            x.cpu(),
            ids.cpu(),
            weights.cpu(),
            padding.cpu(),
        )
        graph.replay()
        active = ~padding.cpu()
        calls += int(active.sum())
        if tied and tokens >= 16:
            pairs = torch.bincount(ids.cpu()[active].flatten(), minlength=e)
            blocks = int(((pairs + 15) // 16).sum())
            tied_union_bytes += blocks * plan.sketch_read_bytes(residual=False) // k
        chosen = reader.selected[:tokens].cpu()
        assert torch.equal(chosen[active], selected[active])
        # Skipped channels keep their screen estimates; exact computation overwrites
        # the F selected ones.
        projected, _, scores = _reference_screen(
            reader, x.cpu(), ids.cpu(), weights.cpu()
        )
        # Scores may straddle a BF16 projection rounding boundary after FP32
        # reassociation; skipped channels' FP32 projections give the tighter check.
        torch.testing.assert_close(
            reader.scores[0][:tokens].cpu()[active],
            scores[active],
            rtol=0.025,
            atol=2e-5,
        )
        omitted = (active[:, None] & ~chosen)[:, None, :].expand_as(projected)
        torch.testing.assert_close(
            reader.partial[:tokens].cpu()[omitted],
            projected[omitted],
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            output.cpu().float(), expected.float(), rtol=0.025, atol=0.02
        )
    report = reader.report()
    if tied_union_bytes:
        assert report["issued_sketch_bytes"] == tied_union_bytes
    assert report["rotation_sign_bytes"] == calls * h
    kept = round(screen_keep * 128)
    residual = score != "point"
    rest = h + plan.selected_weight_bytes() + plan.selection_read_bytes(geometry=False)
    # I=160: 320 row trits occupy exactly 64 bytes per coordinate; each
    # separate scale/residual plane occupies 640 bytes. No sector padding here.
    assert (2 * i + 4) // 5 == 64
    sketch = physical = k * (h // 128) * (64 * kept + 640 * (1 + residual))
    assert report["sketch_bytes"] == calls * sketch
    assert report["physical_sketch_bytes"] == calls * physical
    assert report["logical_charged_bytes"] == calls * (sketch + rest)
    assert report["charged_bytes"] == calls * (physical + rest)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
def test_sparse_prefill_chunks_match_one_call_and_read_cost() -> None:
    """Prefill-sized batches run in capacity chunks of the nonreentrant reader."""
    from calibration_free_moe.reader.progressive import ProgressiveReader
    from calibration_free_moe.runtime.experts import reader_rows

    torch.manual_seed(211)
    # Chunks of 4, 4, 4 and a one-row tail: the tail runs the single-token path.
    h, i, e, k, rank, tokens = 256, 64, 4, 2, 8, 13
    weights_and_bank = (
        (torch.randn(e, 2 * i, h) * 0.05).bfloat16().cuda(),
        channel_major_down((torch.randn(e, h, i) * 0.05).bfloat16().cuda()),
        (torch.randn(e, rank, i) * 0.05).bfloat16().cuda(),
        (torch.randn(h, rank) * 0.05).bfloat16().cuda(),
        compile_budget(hidden_size=h, expert_width=i, top_k=k, retention=0.25),
    )
    whole_reader = ProgressiveReader(*weights_and_bank, 16)
    chunk_reader = ProgressiveReader(*weights_and_bank, 4)
    x = torch.randn(tokens, h, device="cuda").bfloat16()
    ids = torch.randint(e, (tokens, k), device="cuda", dtype=torch.int32)
    weights = torch.rand(tokens, k, device="cuda")
    padding = torch.zeros(tokens, device="cuda", dtype=torch.bool)
    padding[5] = True
    whole = reader_rows(whole_reader, x, ids, weights, padding, 16)
    chunked = reader_rows(chunk_reader, x, ids, weights, padding, 4)
    # Chunking changes expert-block unions and summation order, so outputs
    # agree within BF16 tolerance rather than bitwise.
    active = ~padding
    torch.testing.assert_close(
        chunked[active].float(), whole[active].float(), rtol=0.016, atol=2e-3
    )
    # The read-cost counters (the first eleven) are exact; the trailing requested-byte
    # counters (issued_*) depend on each chunk's unions.
    torch.testing.assert_close(
        chunk_reader.read_work[:11], whole_reader.read_work[:11], atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reader")
@pytest.mark.parametrize("batch", [1, 2, 16])
def test_screen_matches_independent_gauss_hermite_score_and_read_cost(batch: int):
    from calibration_free_moe.reader.progressive import ProgressiveReader

    torch.manual_seed(37)
    h, i, e, k, rank = 256, 64, 4, 2, 8
    gu = (torch.randn(e, 2 * i, h) * 0.02).bfloat16().cuda()
    down = (torch.randn(e, h, i) * 0.02).bfloat16().cuda()
    encoder = (torch.randn(e, rank, i) * 0.02).bfloat16().cuda()
    decoder = (torch.randn(h, rank) * 0.02).bfloat16().cuda()
    plan = compile_budget(hidden_size=h, expert_width=i, top_k=k, retention=0.125)
    reader = ProgressiveReader(
        gu, channel_major_down(down), encoder, decoder, plan, batch
    )
    x = torch.randn(batch, h).bfloat16()
    ids = torch.tensor([[3, 1]] * batch, dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3]] * batch)
    projected, variance, scores = _reference_screen(reader, x, ids, weights)
    result = reader(x.cuda(), ids.cuda(), weights.cuda())
    assert torch.isfinite(result).all()
    # Exact computation overwrites the selected channels' sketch projections; skipped
    # channels keep theirs.
    omitted = ~reader.completed.cpu()
    torch.testing.assert_close(
        reader.partial.cpu()[omitted[:, None].expand_as(reader.partial.cpu())],
        projected[omitted[:, None].expand_as(projected)],
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(reader.variances.cpu(), variance, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(reader.scores[0].cpu(), scores, atol=2e-3, rtol=3e-4)
    selected = (
        scores.argsort(descending=True, stable=True)[:, : plan.final_count]
        .sort()
        .values
    )
    torch.testing.assert_close(
        reader.candidate_ids[-1].cpu(), selected.int(), atol=0, rtol=0
    )
    assert torch.equal(
        reader.selected.cpu().sum(-1), torch.full((batch,), plan.final_count)
    )
    report = reader.report()
    expected = (
        plan.sketch_read_bytes(sectors=False)
        + h  # one model-wide sign vector load per active token-layer
        + plan.selected_weight_bytes()
        + plan.recovery_read_bytes(rank)
        + plan.selection_read_bytes()
    )
    assert report["logical_charged_bytes"] == batch * expected
    assert float(report["charged_bytes"] or 0) >= float(
        report["logical_charged_bytes"] or 0
    )
    assert report["sketch_bytes"] == batch * plan.sketch_read_bytes(sectors=False)
    assert report["reference_bytes"] == batch * plan.native_read_bytes()
    assert (
        report["physical_native_expert_coefficient_bytes"]
        == report["native_expert_coefficient_bytes"]
    )
    omitted = (~reader.selected.view(batch, k, i)).reshape(batch, k, i // 2, 2)
    assert report["physical_recovery_bytes"] == (
        32 * omitted.any(-1).sum().item() + batch * 2 * rank * h
    )
