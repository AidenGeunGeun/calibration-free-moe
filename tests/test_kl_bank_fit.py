import copy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from calibration_free_moe.activation import GatedActivation
from calibration_free_moe.reader.budget import compile_budget
from calibration_free_moe.reader.progressive import (
    ProgressiveReader,
    SelectionScore,
    channel_major_down,
)
from calibration_free_moe.reader.sketch import (
    ACQUISITION,
    build_sketch,
    differentiable_sketch_feature,
    require_sketch_acquisition,
    unpack_sketch,
)
from scripts.fit_kl_bank import Calibration, moe_block, place_model, selected_output
from scripts.fit_plan_recovery import candidate_activations, extracted_model_output


def test_nonzero_recovery_bank_requires_matching_acquisition() -> None:
    bank: dict[str, object] = {
        "format": "pca-joint-recovery-kl-v1",
        "decoder": torch.zeros(2, 3),
        "encoder": torch.zeros(2, 3),
    }
    require_sketch_acquisition(bank)
    bank["encoder"] = torch.ones(2, 3)
    with pytest.raises(ValueError, match=ACQUISITION):
        require_sketch_acquisition(bank)
    bank["plan"] = {"acquisition": "ternary-sketch-v1"}
    with pytest.raises(ValueError, match=ACQUISITION):
        require_sketch_acquisition(bank)
    bank["plan"] = {"acquisition": ACQUISITION}
    require_sketch_acquisition(bank)


def test_cached_sketch_matches_packed_forward_and_input_gradient() -> None:
    torch.manual_seed(19)
    hidden, width = 128, 16
    signs = torch.where(torch.rand(hidden) > 0.5, 1, -1).to(torch.int8)
    weights = torch.randn(2 * width, hidden).bfloat16()[None]
    codes, statistics = build_sketch(weights, signs=signs)
    input_x = torch.randn(4, hidden).bfloat16().requires_grad_()
    packed = differentiable_sketch_feature(input_x, codes[0], statistics[0], signs)
    cached = differentiable_sketch_feature(
        input_x,
        codes[0],
        statistics[0],
        signs,
        unpacked=unpack_sketch(codes, 2 * width, hidden)[0],
    )
    checkpointed = differentiable_sketch_feature(
        input_x,
        codes[0],
        statistics[0],
        signs,
        unpacked=unpack_sketch(codes, 2 * width, hidden)[0],
        checkpoint_cached=True,
    )
    torch.testing.assert_close(cached, packed, rtol=0, atol=0)
    torch.testing.assert_close(checkpointed, packed, rtol=0, atol=0)
    grad_packed = torch.autograd.grad(packed.float().sum(), input_x, retain_graph=True)[
        0
    ]
    grad_cached = torch.autograd.grad(cached.float().sum(), input_x, retain_graph=True)[
        0
    ]
    grad_checkpointed = torch.autograd.grad(checkpointed.float().sum(), input_x)[0]
    torch.testing.assert_close(grad_cached, grad_packed, rtol=0, atol=0)
    torch.testing.assert_close(grad_checkpointed, grad_packed, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("tied_head", [False, True])
def test_remote_decoder_layers_receive_reused_rotary_inputs(tied_head: bool) -> None:
    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 8) * 0.1)

        def forward(
            self,
            hidden: torch.Tensor,
            *,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            assert torch.cuda.current_device() == hidden.device.index
            cos, sin = position_embeddings
            return (hidden @ self.weight) * cos + sin

    class Text(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 8)
            self.rotary_emb = torch.nn.Identity()
            self.layers = torch.nn.ModuleList([Decoder() for _ in range(4)])
            self.norm = torch.nn.LayerNorm(8)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Text()
            self.lm_head = torch.nn.Linear(8, 4)
            if tied_head:
                self.lm_head.weight = self.model.embed_tokens.weight

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            hidden = self.model.embed_tokens(indices)
            position_embeddings = (
                self.model.rotary_emb(hidden).cos(),
                self.model.rotary_emb(hidden).sin(),
            )
            for layer in self.model.layers:
                hidden = layer(hidden, position_embeddings=position_embeddings)
            return self.lm_head(self.model.norm(hidden))

    torch.manual_seed(25)
    single, divided = Model(), Model()
    divided.load_state_dict(single.state_dict())
    place_model(cast(Any, single), [torch.device("cuda:0")])
    place_model(cast(Any, divided), [torch.device("cuda:0"), torch.device("cuda:1")])
    if tied_head:
        assert divided.lm_head.weight is divided.model.embed_tokens.weight
        assert divided.model.embed_tokens.weight.device == torch.device("cuda:1")
    tokens = torch.tensor([[0, 1, 2]], device="cuda:0")
    expected, actual = single(tokens), divided(tokens)
    torch.testing.assert_close(actual, expected)
    expected.float().sum().backward()
    actual.float().sum().backward()
    for (_, left), (_, right) in zip(
        single.named_parameters(), divided.named_parameters(), strict=True
    ):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(right.grad.to("cuda:0"), left.grad)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_real_reader_on_second_device_matches_single_device(tmp_path: Path) -> None:
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        max_position_embeddings=64,
    )
    config.head_dim = 32
    config.architectures = ["Qwen3MoeForCausalLM"]
    config.save_pretrained(tmp_path)
    torch.manual_seed(353)
    single = Qwen3MoeForCausalLM(config).bfloat16().eval()
    divided = copy.deepcopy(single)
    first, second = torch.device("cuda:0"), torch.device("cuda:1")
    place_model(single, [first])
    place_model(divided, [first, second])
    plan = compile_budget(hidden_size=128, expert_width=32, top_k=2, retention=0.25)
    single_fit = Calibration(single, plan, 4, tmp_path, devices=[first])
    divided_fit = Calibration(divided, plan, 4, tmp_path, devices=[first, second])
    single_fit.refresh()
    divided_fit.refresh()
    seen_devices: list[int] = []
    moe_block(divided, 1).gate.register_forward_pre_hook(
        lambda _module, _args: seen_devices.append(torch.cuda.current_device())
    )
    torch.cuda.set_device(first)
    with torch.no_grad():
        expected = single_fit.logits([1, 2, 3, 4], sparse_from=1, keep=2)
        actual = divided_fit.logits([1, 2, 3, 4], sparse_from=1, keep=2)
    assert seen_devices and set(seen_devices) == {1}
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA reader")
def test_single_layer_sensitivity_keeps_other_layers_stock_and_reuses_teacher(
    tmp_path: Path,
) -> None:
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        max_position_embeddings=64,
    )
    config.head_dim = 32
    config.architectures = ["Qwen3MoeForCausalLM"]
    config.save_pretrained(tmp_path)
    model = Qwen3MoeForCausalLM(config).bfloat16().eval()
    place_model(model, [torch.device("cuda:0")])
    originals = [moe_block(model, layer).forward for layer in range(2)]
    plan = compile_budget(
        hidden_size=128, expert_width=32, top_k=2, retention=0.25, screen_keep=0.625
    )
    cal = Calibration(
        model,
        plan,
        4,
        tmp_path,
        sparse_layers={0},
        reader_output=True,
        selection_score="point",
    )
    checked: set[int] = set()

    def check_stock(
        _module: object, inputs: tuple[Tensor, ...], output: Tensor, *, layer: int
    ) -> None:
        if cal.sparse_layers is not None and layer not in cal.sparse_layers:
            torch.testing.assert_close(output, originals[layer](*inputs))
            checked.add(layer)

    for layer in range(2):
        moe_block(model, layer).register_forward_hook(
            lambda module, inputs, output, layer=layer: check_stock(
                module, inputs, output, layer=layer
            )
        )
    document = {"id": "fit-0", "tokens": [1, 2, 3, 4, 5, 6], "context_length": 2}
    with torch.no_grad():
        cal.refresh()
        assert torch.isfinite(cal.kl(document))
        teacher = cal.teacher["fit-0"]
        cal.sparse_layers = {1}
        cal.refresh()
        assert torch.isfinite(cal.kl(document))
        assert cal.teacher["fit-0"] is teacher
    assert checked == {0, 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA reader")
@pytest.mark.parametrize("activation", ["silu", "gelu_tanh"])
@pytest.mark.parametrize(
    ("screen_keep", "selection_score"), [(1.0, "recovery-aware"), (0.625, "point")]
)
def test_sparse_moe_matches_reader_extraction(
    activation: GatedActivation, screen_keep: float, selection_score: SelectionScore
) -> None:
    torch.manual_seed(353)
    h, i, k, rank, experts = 128, 32, 2, 32, 4
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=0.25,
        screen_keep=screen_keep,
    )
    x = (torch.randn(2, h, device="cuda") * 0.1).bfloat16()
    gu = (torch.randn(experts, 2 * i, h, device="cuda") * 0.05).bfloat16()
    down = (torch.randn(experts, h, i, device="cuda") * 0.05).bfloat16()
    u = (torch.randn(h, rank, device="cuda") * 0.01).bfloat16()
    b = (torch.randn(experts, rank, i, device="cuda") * 0.01).bfloat16()
    reader = ProgressiveReader(
        gu,
        channel_major_down(down),
        b,
        u,
        plan,
        len(x),
        activation=activation,
        selection_score=selection_score,
    )
    ids = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    weights = torch.full((2, k), 0.3, device="cuda", dtype=torch.float32)
    reader(x, ids, weights)
    exact = reader.activations[: len(x)].float()
    features = candidate_activations(reader, len(x))
    target = extracted_model_output(reader, down, exact, features, b, u)
    master_u = u.float().requires_grad_()
    master_b = b.float().requires_grad_()
    actual = selected_output(
        x, reader, gu, down, master_u, master_b, plan, activation=activation
    ).float()
    relative = (actual - target).norm() / target.norm()
    assert relative.item() < 0.06
    # Skipped-channel recovery alone must use the deployed sketch activation;
    # larger inputs leave the near-linear region where SiLU and GELU coincide.
    x = x * 16
    reader(x, ids, weights)
    zero = torch.zeros_like(down)
    features = candidate_activations(reader, len(x))
    target = extracted_model_output(reader, zero, exact, features, b, u)
    recovery = selected_output(
        x, reader, gu, zero, u.float(), b.float(), plan, activation=activation
    ).float()
    assert ((recovery - target).norm() / target.norm()).item() < 0.02
    # The reusable Triton workspace may be overwritten before autograd finishes.
    reader(-x, ids, weights)
    actual.square().mean().backward()
    assert master_u.grad is not None and master_b.grad is not None
    assert torch.isfinite(master_u.grad).all() and torch.isfinite(master_b.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA reader")
def test_omitted_recovery_backpropagates_to_prior_bank() -> None:
    torch.manual_seed(499)
    h, i, k, rank, experts = 128, 32, 2, 32, 4
    plan = compile_budget(hidden_size=h, expert_width=i, top_k=k, retention=0.125)
    gu = (torch.randn(experts, 2 * i, h, device="cuda") * 0.2).bfloat16()
    down = torch.zeros(experts, h, i, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(experts, rank, i, device="cuda") * 0.2).bfloat16()
    u = (torch.randn(h, rank, device="cuda") * 0.2).bfloat16()
    prior_bank = torch.nn.Parameter(torch.randn(rank, h, device="cuda") * 0.1)
    latent = torch.randn(2, rank, device="cuda")
    x = (torch.randn(2, h, device="cuda") * 0.25 + latent @ prior_bank).bfloat16()
    x.retain_grad()
    reader = ProgressiveReader(gu, channel_major_down(down), b, u, plan, len(x))
    ids = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.65, 0.35], [0.4, 0.6]], device="cuda")
    reader(x.detach(), ids, weights)
    assert bool((~reader.selected).any())
    output = selected_output(x, reader, gu, down, u.float(), b.float(), plan)
    output.float().square().mean().backward()
    assert x.grad is not None and prior_bank.grad is not None
    assert x.grad.float().abs().sum().item() > 0
    assert prior_bank.grad.float().abs().sum().item() > 0
