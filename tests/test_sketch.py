import pytest
import torch

from calibration_free_moe.reader.budget import compile_budget
from calibration_free_moe.reader.sketch import (
    block_hadamard_signs,
    build_sketch,
    differentiable_sketch_feature,
    rotate_blocks,
    unpack_sketch,
)


@pytest.mark.parametrize("keep", [80, 128])
@pytest.mark.parametrize("input_is_rotated", [False, True])
def test_sketch_feature_matches_kept_coordinate_reference(
    keep: int, input_is_rotated: bool
) -> None:
    torch.manual_seed(71)
    hidden, width = 256, 16
    signs = block_hadamard_signs(hidden, "cpu")
    codes, statistics = build_sketch(
        torch.randn(1, 2 * width, hidden) * 0.1, signs=signs
    )
    # Integer magnitudes force ties, including at the keep boundary. Distinct
    # blocks and tokens must choose coordinates independently.
    x = torch.randint(-3, 4, (3, hidden)).float().requires_grad_()
    rotated = x if input_is_rotated else rotate_blocks(x, signs)
    blocks = rotated.reshape(-1, 128)
    mask = torch.zeros_like(blocks, dtype=torch.bool)
    for row, values in enumerate(blocks.detach().tolist()):
        order = sorted(range(128), key=lambda c: (-abs(values[c]), c))
        mask[row, order[:keep]] = True
    kept = torch.where(mask, blocks, 0.0).reshape(3, 2, 128)
    trits = (
        unpack_sketch(codes, 2 * width, hidden)[0].float().reshape(2 * width, 2, 128)
    )
    projection = (
        torch.einsum("tbc,rbc->trb", kept, trits) * statistics[0, ..., 0].float()
    ).sum(-1)
    g, u = projection.chunk(2, dim=-1)
    expected = (
        torch.nn.functional.silu(g.bfloat16().float()) * u.bfloat16().float()
    ).bfloat16()
    actual = differentiable_sketch_feature(
        x,
        codes[0],
        statistics[0],
        signs,
        input_is_rotated=input_is_rotated,
        screen_coordinates=keep,
    )
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.float().sum(), x)[0]
    actual_grad = torch.autograd.grad(actual.float().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    if input_is_rotated:
        assert torch.count_nonzero(actual_grad.reshape_as(mask)[~mask]) == 0


@pytest.mark.parametrize("width,hidden", [(512, 384), (704, 256)])
def test_sector_tiled_sketch_preserves_rows_and_prices_loads(
    width: int, hidden: int
) -> None:
    """Real packing at a half-sector width, including an odd final rotation block."""
    torch.manual_seed(83)
    weights = torch.randn(1, 2 * width, hidden).bfloat16()
    codes, statistics = build_sketch(weights)
    original = weights.float().reshape(1, 2 * width, hidden // 128, 128)
    keep = original.abs() > 0.7 * original.abs().mean(-1, keepdim=True)
    signed = torch.where(keep, original.sign(), 0)
    torch.testing.assert_close(
        unpack_sketch(codes, 2 * width, hidden),
        signed.flatten(-2).to(torch.int8),
        rtol=0,
        atol=0,
    )
    assert (codes.numel() + statistics.numel() * 2) * 8 / weights.numel() < 2
    useful = 2 * ((width + 4) // 5)
    stride = (useful + 15) // 16 * 16
    body = stride // 32 * 32
    blocks = hidden // 128
    sectors = 0
    # Mirror the public layout's actual byte requests, not a ceil(run/32) shortcut.
    for block in range(blocks):
        for coordinate in range(80):
            for start in range(0, useful, 32):
                byte = torch.arange(start, min(start + 32, useful))
                if start < body:
                    address = (block * 128 + coordinate) * body + byte
                else:
                    address = (
                        blocks * 128 * body
                        + (block // 2 * 128 + coordinate) * 32
                        + block % 2 * 16
                        + byte
                        - body
                    )
                sectors += len(set((address // 32).tolist()))
                assert len(set((address // 32).tolist())) == 1
                packed = codes[0, address].int()
                # Vector loads may include final padding bytes, but must stay
                # inside exactly the sector the byte loader already counted.
                word_address = address[::4]
                word_bytes = word_address[:, None] + torch.arange(4)
                assert set((word_bytes // 32).flatten().tolist()) == set(
                    (address // 32).tolist()
                )
                words = codes[0].view(torch.int32)[word_address // 4]
                expanded = ((words[:, None] >> (torch.arange(4) * 8)) & 255).flatten()
                torch.testing.assert_close(
                    expanded[: len(address)].int(), packed, rtol=0, atol=0
                )
                for digit in range(5):
                    channel = byte // 2 * 5 + digit
                    valid = channel < width
                    row = byte % 2 * width + channel
                    decoded = (packed // 3**digit) % 3 - 1
                    torch.testing.assert_close(
                        decoded[valid].float(),
                        signed[0, row[valid], block, coordinate],
                        rtol=0,
                        atol=0,
                    )
    plan = compile_budget(
        hidden_size=hidden,
        expert_width=width,
        top_k=1,
        retention=0.1,
        screen_keep=0.625,
    )
    assert plan.sketch_read_bytes(residual=False) == sectors * 32 + blocks * 4 * width
