from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from calibration_free_moe.reader.budget import compile_budget
from calibration_free_moe.reader.progressive import (
    ProgressiveReader,
    channel_major_down,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reader_worker(rank: int, port: int, output: str, screen_keep: float) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        import vllm.distributed as vd

        # Only the serving group's lookup is replaced; the reader, kernels,
        # PyTorch NCCL collectives, and vLLM-style global ID map are real.
        vd.get_ep_group = lambda: SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
            device_group=dist.group.WORLD, rank_in_group=rank
        )
        torch.manual_seed(37)
        h, i, e, k, r = 128, 64, 4, 2, 8
        gu = (torch.randn(e, 2 * i, h) * 0.1).bfloat16().cuda()
        down = (torch.randn(e, h, i) * 0.1).bfloat16().cuda()
        encoder = (torch.randn(e, r, i) * 0.01).bfloat16()
        decoder = torch.randn(h, r).bfloat16()
        expert_map = torch.full((e,), -1, dtype=torch.int32)
        expert_map[rank * 2 : (rank + 1) * 2] = torch.arange(2, dtype=torch.int32)
        plan = compile_budget(
            hidden_size=h,
            expert_width=i,
            top_k=k,
            retention=0.1,
            screen_keep=screen_keep,
        )
        # The single-GPU oracle and the EP reader run per token (a grouped
        # threshold above capacity): the oracle below reconstructs per-token
        # encoder/down partials, and the EP reader is the grouped reference.
        per_token = 3
        full = (
            ProgressiveReader(
                gu,
                channel_major_down(down),
                encoder,
                decoder,
                plan,
                2,
                grouped_tokens=per_token,
            )
            if rank == 0
            else None
        )
        local = ProgressiveReader(
            gu[rank * 2 : (rank + 1) * 2].contiguous(),
            channel_major_down(down[rank * 2 : (rank + 1) * 2]),
            encoder[rank * 2 : (rank + 1) * 2],
            decoder,
            plan,
            2,
            expert_map=expert_map,
            grouped_tokens=per_token,
        )
        rows: list[dict[str, float | int | str]] = []
        expected_outputs: dict[str, torch.Tensor] = {}

        def check(
            x: torch.Tensor,
            ids: torch.Tensor,
            weights: torch.Tensor,
            padding: torch.Tensor,
            stage: str,
            graph: tuple[torch.cuda.CUDAGraph, torch.Tensor] | None = None,
        ) -> None:
            n = x.shape[0]
            expected = (
                full(x, ids, weights, padding).clone() if full is not None else None
            )
            if graph is None:
                actual = local(x, ids, weights, padding).clone()
            else:
                graph[0].replay()
                actual = graph[1][:n].clone()
            scores = local.scores[0][:n].clone()
            selected = local.candidate_ids[-1][:n].clone()
            mixed = local.latent[:n].clone()
            # Capture the rank-local FP32 router mixture using the per-expert
            # BF16 encoder boundary. The library has already all-reduced mixed.
            per_expert = local.encoder_parts[:n].sum(dim=2).bfloat16().float()
            local_mixed = (per_expert * local.weights[:n, :, None]).sum(1)
            local_mixed_parts = [torch.empty_like(local_mixed) for _ in range(2)]
            dist.all_gather(local_mixed_parts, local_mixed)
            global_mixed = local_mixed_parts[0] + local_mixed_parts[1]
            routed_local = torch.zeros_like(x, dtype=torch.float32)
            for token in range(n):
                for slot in range(k):
                    tiles = (int(local.counts[token, slot]) + 127) // 128
                    if tiles:
                        per_expert_down = (
                            local.down_parts.view(-1, k, local.down_splits, h)[
                                token, slot, :tiles
                            ]
                            .sum(0)
                            .bfloat16()
                            .float()
                        )
                        routed_local[token] += (
                            per_expert_down * local.weights[token, slot]
                        )
            routed_global = routed_local.clone()
            dist.all_reduce(routed_global)
            dist.all_reduce(actual)
            if rank == 0:
                assert full is not None and expected is not None
                torch.testing.assert_close(scores, full.scores[0][:n], rtol=0, atol=0)
                torch.testing.assert_close(
                    selected, full.candidate_ids[-1][:n], rtol=0, atol=0
                )
                torch.testing.assert_close(mixed, global_mixed, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(
                    mixed.bfloat16().float(),
                    full.latent[:n].bfloat16().float(),
                    rtol=0,
                    atol=2
                    * torch.finfo(torch.bfloat16).eps
                    * full.latent[:n].abs().max().item(),
                )
                assert torch.equal(actual[padding], torch.zeros_like(actual[padding]))
                assert torch.isfinite(actual).all()
                # Per-rank BF16 partial outputs plus the reduced BF16 boundary
                # stay within a few BF16 ulps of the single-GPU output.
                envelope = (
                    2
                    * torch.finfo(torch.bfloat16).eps
                    * expected.float().abs().max().item()
                )
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=0, atol=envelope
                )
                expected_outputs[stage] = expected.float()
                max_abs = (actual.float() - expected.float()).abs().max().item()
                # The final local BF16 boundary remains a distinct observed
                # numerical effect until the runner can reduce FP32 routed output.
                rows.append(
                    {
                        "stage": stage,
                        "envelope": envelope,
                        "max_abs_output": max_abs,
                        "max_abs_latent_fp32": (mixed - full.latent[:n])
                        .abs()
                        .max()
                        .item(),
                        "max_abs_local_bf16_rounding": (
                            actual.float() - routed_global.bfloat16().float()
                        )
                        .abs()
                        .max()
                        .item()
                        if stage == "routed_only"
                        else 0.0,
                        "active": int((~padding).sum()),
                    }
                )

        x = torch.randn(2, h).bfloat16().cuda()
        ids = torch.tensor([[0, 2], [1, 3]], device="cuda", dtype=torch.int32)
        weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], device="cuda")
        padding = torch.tensor([False, False], device="cuda")
        check(x, ids, weights, padding, "eager")

        # Zero U only after preparation: keep the identical Gauss-Hermite score
        # geometry and selected identities, isolate the routed-only output.
        local_decoder = local.decoder
        full_decoder = full.decoder if full is not None else None
        assert local_decoder is not None
        local_decoder.zero_()
        if full_decoder is not None:
            full_decoder.zero_()
        check(x, ids, weights, padding, "routed_only")
        if rank == 0:
            # Recovery must exceed the envelope several times over, so the eager
            # assertion fails if EP drops or duplicates the recovery contribution.
            recovery = expected_outputs["eager"] - expected_outputs["routed_only"]
            assert recovery.abs().max().item() > 4 * float(rows[0]["envelope"])
        local_decoder.copy_(decoder.cuda())
        if full_decoder is not None:
            full_decoder.copy_(decoder.cuda())

        # Keep fixed capacity and replay changed routes, data and padding.
        xbuf, ibuf, wbuf, pbuf = (
            x.clone(),
            ids.clone(),
            weights.clone(),
            padding.clone(),
        )
        local(xbuf, ibuf, wbuf, pbuf)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = local(xbuf, ibuf, wbuf, pbuf)
        local.read_work.zero_()
        for seed, stage, routed, pad in (
            (101, "graph_local", [[0, 1], [2, 3]], [False, True]),
            (102, "graph_cross", [[0, 2], [1, 3]], [False, False]),
        ):
            xbuf.copy_(
                (torch.randn(2, h, generator=torch.Generator().manual_seed(seed)))
                .bfloat16()
                .cuda()
            )
            ibuf.copy_(torch.tensor(routed, device="cuda", dtype=torch.int32))
            pbuf.copy_(torch.tensor(pad, device="cuda"))
            check(xbuf, ibuf, wbuf, pbuf, stage, (graph, graph_output))
        report = local.report()
        reports: list[dict[str, Any]] = [{} for _ in range(2)]
        dist.all_gather_object(reports, report)
        # Expert-major EP execution against the per-token EP reference: remote
        # pairs join no block, so rank-local and reduced outputs, the reduced
        # screen scores and the selections match, eagerly and in graph replay.
        grouped = ProgressiveReader(
            gu[rank * 2 : (rank + 1) * 2].contiguous(),
            channel_major_down(down[rank * 2 : (rank + 1) * 2]),
            encoder[rank * 2 : (rank + 1) * 2],
            decoder,
            plan,
            2,
            expert_map=expert_map,
            sketch=(local.sketch_codes, local.sketch_statistics),
            grouped_tokens=2,
            grouped_screen_tokens=2,
        )
        grouped(xbuf, ibuf, wbuf, pbuf)
        torch.cuda.synchronize()
        grouped_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(grouped_graph):
            grouped_output = grouped(xbuf, ibuf, wbuf, pbuf)
        for replay, routed, pad in (
            (False, [[0, 2], [1, 3]], [False, False]),
            (False, [[0, 2], [1, 3]], [False, True]),
            (True, [[0, 1], [2, 3]], [False, True]),
            (True, [[3, 0], [2, 1]], [False, False]),
        ):
            ibuf.copy_(torch.tensor(routed, device="cuda", dtype=torch.int32))
            pbuf.copy_(torch.tensor(pad, device="cuda"))
            expected = local(xbuf, ibuf, wbuf, pbuf).clone()
            if replay:
                grouped_graph.replay()
                actual = grouped_output.clone()
            else:
                actual = grouped(xbuf, ibuf, wbuf, pbuf).clone()
            active = ~pbuf
            assert torch.equal(grouped.selected[:2][active], local.selected[:2][active])
            # Reduced screen scores differ only by FP32 reassociation, which is
            # bounded by the score scale rather than each (cancelling) score.
            reference_scores = local.scores[0][:2][active]
            torch.testing.assert_close(
                grouped.scores[0][:2][active],
                reference_scores,
                rtol=0,
                atol=1e-3 * reference_scores.abs().max().item(),
            )
            assert torch.equal(actual[pbuf], torch.zeros_like(actual[pbuf]))
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=0.016, atol=2e-3
            )
            dist.all_reduce(actual)
            dist.all_reduce(expected)
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=0.016, atol=2e-3
            )
        if rank == 0:
            expected_rows = sum(int(row["active"]) for row in rows[-2:])
            reference_bytes = sum(float(part["reference_bytes"]) for part in reports)
            charged_bytes = sum(
                float(part["logical_charged_bytes"]) for part in reports
            )
            assert reference_bytes == expected_rows * plan.native_read_bytes()
            assert charged_bytes == expected_rows * (
                plan.selected_weight_bytes()
                + plan.recovery_read_bytes(r)
                + plan.selection_read_bytes()
                + plan.sketch_read_bytes(sectors=False)
                + 2 * h  # both EP ranks rotate their own screening inputs
            )
            Path(output).write_text(
                json.dumps({"rows": rows, "charged_bytes": charged_bytes})
            )
    finally:
        # Graph-captured NCCL operations retain their process group until the
        # worker exits. Destroying it explicitly blocks on the captured graph.
        dist.barrier(device_ids=[rank])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("screen_keep", [1.0, 0.625])
def test_expert_parallel_reader_same_state_and_graph(
    tmp_path: Path, screen_keep: float
) -> None:
    output = tmp_path / "ep-reader.json"
    spawn(
        _reader_worker,
        args=(_free_port(), str(output), screen_keep),
        nprocs=2,
        join=True,
    )
    rows = json.loads(output.read_text())["rows"]
    assert [row["stage"] for row in rows] == [
        "eager",
        "routed_only",
        "graph_local",
        "graph_cross",
    ]
