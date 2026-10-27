"""Ternary sketch of the native gate and up expert rows, read by the screen.

The sketch is a resident low-bit copy: each 128-coordinate block of a
block-Hadamard-rotated row keeps one trit per coordinate (five per byte) and an
FP16 scale and residual variance.
"""

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from calibration_free_moe.activation import GatedActivation, activate
from calibration_free_moe.reader.budget import sketch_code_stride, sketch_storage_bytes

# Acquisition rule a nonzero fitted recovery map (`bank`) must declare to be used
# with this sketch.
ACQUISITION = "ternary-hadamard-sketch-v1"


def block_hadamard_signs(hidden: int, device: torch.device | str) -> Tensor:
    """Return the rotation's random signs, one int8 +-1 per hidden coordinate.

    The signs are model-wide and drawn with seed 0, so the reader, prebuilt
    sketches and offline recovery-map fitting all use the same rotation.
    """
    if hidden % 128:
        raise ValueError("Hadamard sketch requires hidden width divisible by 128")
    return (
        torch.randint(0, 2, (hidden,), generator=torch.Generator().manual_seed(0))
        .mul_(2)
        .sub_(1)
        .to(device=device, dtype=torch.int8)
    )


def rotate_blocks(x: Tensor, signs: Tensor | None) -> Tensor:
    """Apply blockdiag(H128)/sqrt(128) times diag(signs) in FP32 (identity if None)."""
    if signs is None:
        return x.float()
    blocks = signs.numel() // 128
    value = x.float().reshape(-1, blocks, 128) * signs.float().reshape(1, blocks, 128)
    for width in (1, 2, 4, 8, 16, 32, 64):
        paired = value.reshape(-1, blocks, 128 // (2 * width), 2, width)
        left, right = paired.unbind(-2)
        value = torch.stack((left + right, left - right), dim=-2).reshape(
            -1, blocks, 128
        )
    return (value * (1 / math.sqrt(128))).reshape(x.shape)


def mask_rotated_blocks(x: Tensor, keep: int = 128) -> Tensor:
    """Keep each block's largest |Rx|, breaking ties by coordinate order.

    Membership is detached; gradients pass only through retained rotated values.
    """
    if keep == 128:
        return x
    blocks = x.reshape(*x.shape[:-1], -1, 128)
    with torch.no_grad():
        magnitude = blocks.abs()
        threshold = magnitude.sort(dim=-1, descending=True).values[..., keep - 1 : keep]
        above = magnitude > threshold
        tied = magnitude == threshold
        ties = keep - above.sum(dim=-1, keepdim=True)
        kept = above | (tied & (tied.cumsum(dim=-1) <= ties))
    return torch.where(kept, blocks, 0.0).reshape_as(x)


def require_sketch_acquisition(bank: dict[str, object]) -> None:
    """Reject a map fitted to another acquisition rule; permit an all-zero map."""
    decoder, encoder = bank["decoder"], bank["encoder"]
    if not isinstance(decoder, Tensor) or not isinstance(encoder, Tensor):
        raise TypeError("Recovery bank parameters must be tensors")
    if bool(torch.count_nonzero(decoder)) or bool(torch.count_nonzero(encoder)):
        plan = bank.get("plan")
        if not isinstance(plan, dict) or plan.get("acquisition") != ACQUISITION:
            raise ValueError(
                f"Nonzero recovery bank requires {ACQUISITION} acquisition"
            )


def build_sketch(weights: Tensor, signs: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Pack five TWN trits per byte, with FP16 scale and residual variance per block.

    weights are the native gate and up rows [experts, 2 I, H]. Each rotated
    128-coordinate block of a row is ternarized by the ternary weight network
    (TWN) rule: coordinates above 0.7 times the mean magnitude keep their sign,
    and the scale is the mean magnitude of those kept. Returns the uint8 codes
    [experts, sketch_storage_bytes(2 I, H)] and the FP16 statistics
    [experts, 2 I, H // 128, 2] (scale, residual variance).

    Flat bytes per expert pair a gate byte and an up byte for five channels.
    Full 32-byte tiles are coordinate-major. A final 16-byte tile shares a
    sector with the same coordinate of the adjacent rotation block. Thus each
    tile load touches one sector without padding every coordinate to 32 bytes.
    Statistics expose [expert, row, block, moment], with contiguous moment planes.
    Construct one expert at a time so the FP32 copy is transient.
    """
    experts, rows, hidden = weights.shape
    if hidden % 128 or rows % 2:
        raise ValueError(
            "Ternary sketch requires 128-coordinate blocks and paired gate/up rows"
        )
    blocks = hidden // 128
    width = rows // 2
    stride = sketch_code_stride(rows)
    body = stride // 32 * 32
    tail = stride % 32
    # Padding byte 121 = 1 + 3 + 9 + 27 + 81 encodes five zero trits.
    codes = torch.full(
        (experts, sketch_storage_bytes(rows, hidden)),
        121,
        device=weights.device,
        dtype=torch.uint8,
    )
    moments = torch.empty(
        (experts, blocks, 2, rows), device=weights.device, dtype=torch.float16
    ).permute(0, 3, 1, 2)
    powers = torch.tensor([1, 3, 9, 27, 81], device=weights.device, dtype=torch.int16)
    for expert in range(experts):
        value = weights[expert].float().reshape(rows, blocks, 128)
        if signs is not None:
            value = rotate_blocks(value, signs)
        keep = value.abs() > 0.7 * value.abs().mean(dim=-1, keepdim=True)
        scale = (value.abs() * keep).sum(-1) / keep.sum(-1).clamp_min(1)
        signed = torch.where(keep, value.sign(), 0)
        residual = (value - signed * scale.unsqueeze(-1)).square().mean(-1)
        trits = (
            signed.to(torch.int16).reshape(2, width, blocks, 128).permute(2, 3, 0, 1)
            + 1
        )
        trits = torch.nn.functional.pad(trits, (0, -width % 5), value=1).reshape(
            blocks, 128, 2, -1, 5
        )
        packed = (trits * powers).sum(-1).transpose(-1, -2).flatten(-2).to(torch.uint8)
        packed = torch.nn.functional.pad(
            packed, (0, stride - packed.shape[-1]), value=121
        )
        codes[expert, : blocks * 128 * body] = packed[..., :body].flatten()
        if tail:
            tails = torch.nn.functional.pad(
                packed[..., body:], (0, 0, 0, 0, 0, blocks % 2), value=121
            )
            codes[expert, blocks * 128 * body :] = (
                tails.reshape(-1, 2, 128, 16).transpose(1, 2).flatten()
            )
        moments[expert] = torch.stack((scale, residual), dim=-1).to(torch.float16)
    return codes, moments


def differentiable_sketch_feature(
    x: Tensor,
    codes: Tensor,
    statistics: Tensor,
    signs: Tensor | None,
    *,
    unpacked: Tensor | None = None,
    checkpoint_cached: bool = False,
    activation: GatedActivation = "silu",
    input_is_rotated: bool = False,
    screen_coordinates: int = 128,
) -> Tensor:
    """Recompute the sketch activation phi(g) u in BF16, with gradients to x.

    This is the feature recovery encodes for a skipped channel, computed from
    the sketch projections as the reader does. Gradients reach x only, not the
    partial screen's coordinate membership. Uncached projections are
    checkpointed; multi-device fitting also checkpoints cached trits so the
    backward graph of the larger models fits in memory.
    """
    blocks = statistics.shape[-2]

    def project(input_x: Tensor) -> Tensor:
        trit = (
            (
                unpacked
                if unpacked is not None
                else unpack_sketch(codes, statistics.shape[-3], blocks * 128)
            )
            .float()
            .reshape(-1, blocks, 128)
        )
        scale = statistics[..., 0].float()
        projection = (
            torch.einsum(
                "tbc,rbc->trb",
                mask_rotated_blocks(
                    input_x if input_is_rotated else rotate_blocks(input_x, signs),
                    screen_coordinates,
                ).reshape(-1, blocks, 128),
                trit,
            )
            * scale[None]
        ).sum(-1)
        g, u = projection.chunk(2, dim=-1)
        return (
            activate(g.bfloat16().float(), activation) * u.bfloat16().float()
        ).bfloat16()

    return (
        checkpoint(project, x, use_reentrant=False)
        if unpacked is None or checkpoint_cached
        else project(x)
    )


def unpack_sketch(codes: Tensor, rows: int, hidden: int) -> Tensor:
    """Decode packed sketch codes into int8 trits [..., rows, hidden].

    Offline recovery-map fitting keeps the decoded trits as signed bytes.
    """
    width, blocks = rows // 2, hidden // 128
    stride = sketch_code_stride(rows)
    body, tail = stride // 32 * 32, stride % 32
    row = torch.arange(rows, device=codes.device, dtype=torch.int64)
    coordinate = torch.arange(hidden, device=codes.device, dtype=torch.int64)
    byte = 2 * ((row % width) // 5) + row // width
    offset = coordinate[:, None] * body + byte[None, :]
    if tail:
        tail_offset = (
            blocks * 128 * body
            + ((coordinate // 256) * 128 + coordinate % 128)[:, None] * 32
            + ((coordinate // 128) % 2)[:, None] * 16
            + byte[None, :]
            - body
        )
        offset = torch.where(byte[None, :] < body, offset, tail_offset)
    packed = codes[..., offset].int()
    signed = (packed // (3 ** ((row % width) % 5))) % 3 - 1
    return signed.transpose(-1, -2).to(torch.int8)
