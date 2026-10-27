import pytest
import torch
import torch.nn.functional as F

from calibration_free_moe.reader.budget import compile_budget
from calibration_free_moe.reader.progressive import (
    ProgressiveReader,
    channel_major_down,
)
from scripts.fit_plan_recovery import (
    candidate_activations,
    extract,
    extracted_model_output,
    route,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_extracted_last_paid_estimator_matches_reader() -> None:
    torch.manual_seed(503)
    e, h, i, k, rank, t = 4, 256, 64, 2, 8, 3
    device = "cuda"
    gate_up = (torch.randn(e, 2 * i, h, device=device) / h**0.5).bfloat16()
    down = channel_major_down((torch.randn(e, h, i, device=device) / i**0.5).bfloat16())
    encoder = (torch.randn(e, rank, i, device=device) / i**0.5).bfloat16()
    decoder = (torch.randn(h, rank, device=device) / rank**0.5).bfloat16()
    gate = (torch.randn(e, h, device=device) / h**0.5).bfloat16()
    hidden = torch.randn(t, h, device=device).bfloat16()
    ids, weights = route(hidden, gate, k)
    plan = compile_budget(
        hidden_size=h,
        expert_width=i,
        top_k=k,
        retention=0.125,
    )
    reader = ProgressiveReader(gate_up, down, encoder, decoder, plan, t)
    actual = reader(hidden, ids, weights).float().clone()
    partial = candidate_activations(reader, t)
    extracted_ids, weighted_p, omitted_output = extract(
        hidden, gate_up, down, gate, encoder, decoder, plan, t
    )
    torch.testing.assert_close(extracted_ids, reader.ids[:t], rtol=0, atol=0)
    expected_weighted_p = (
        partial.view(t, k, i).float()
        * ~reader.selected[:t].view(t, k, i)
        * reader.weights[:t, :, None]
    )
    torch.testing.assert_close(weighted_p, expected_weighted_p, rtol=0, atol=0)
    assert bool((~reader.completed[:t]).any())
    assert bool(reader.completed[:t].any())

    # Dense candidate truth is computed independently from the fitter's features.
    activation = torch.empty(t, k, i, device=device)
    for slot in range(k):
        gu = torch.bmm(
            gate_up[reader.ids[:t, slot].long()].float(), hidden.float().unsqueeze(-1)
        ).squeeze(-1)
        g, u = gu[:, :i].bfloat16().float(), gu[:, i:].bfloat16().float()
        activation[:, slot] = (F.silu(g).bfloat16().float() * u).bfloat16().float()
    expected_omitted = torch.zeros(t, h, device=device)
    for slot in range(k):
        expected_omitted += reader.weights[:t, slot, None] * torch.bmm(
            down[reader.ids[:t, slot].long()].float(),
            (
                activation[:, slot] * ~reader.selected[:t].view(t, k, i)[:, slot]
            ).unsqueeze(-1),
        ).squeeze(-1)
    torch.testing.assert_close(omitted_output, expected_omitted, rtol=0.001, atol=1e-5)
    modeled = extracted_model_output(
        reader, down, activation, partial, encoder, decoder
    )
    torch.testing.assert_close(actual, modeled, rtol=0.03, atol=0.02)
    # A broken feature extraction must materially change the compared output.
    broken = extracted_model_output(
        reader, down, activation, torch.zeros_like(partial), encoder, decoder
    )
    assert (actual - broken).abs().max() > 0.1
