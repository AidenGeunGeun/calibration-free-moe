"""Evaluation-only fidelity to stock: full-vocabulary KL and top-1 agreement.

`FidelitySampler` wraps vLLM's V2 sampler on the first tensor-parallel rank. At
every sampled step of a request with a fidelity key, it records the raw logits (a
stock reference run) or compares them with the stored stock logits of the same
teacher-forced step; stored logits keep the model's output dtype (BF16), so the
reference is exact. It reads the logits before sampling, outside any CUDA graph,
and never changes logits, sampled tokens or logprobs. Tracking is per worker slot,
so a preempted and resumed request fails instead of being scored.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.sample.sampler import Sampler

# SamplingParams.extra_args key that marks a request for fidelity tracking.
KEY = "tokenwise_fidelity_key"


def reference_path(reference: Path, key: str) -> Path:
    """Return the file holding one request's stored stock logits."""
    return reference / f"{key}.pt"


def kl_and_top1(
    reference_logits: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute row-wise KL(P_reference || P) in FP32 and argmax agreement."""
    reference_logprobs = torch.log_softmax(reference_logits.float(), dim=-1)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = reference_logprobs.exp()
    # Zero-probability reference tokens contribute nothing (0 log 0 = 0).
    terms = torch.where(
        probabilities > 0,
        probabilities * (reference_logprobs - logprobs),
        torch.zeros_like(probabilities),
    )
    agree = reference_logits.argmax(dim=-1) == logits.argmax(dim=-1)
    return terms.sum(dim=-1), agree


class FidelitySampler:
    """Delegate to the native sampler; record or compare tracked requests."""

    def __init__(
        self,
        sampler: Sampler,
        mode: Literal["record", "compare"],
        reference: Path,
        max_steps: int,
    ) -> None:
        self._sampler: Sampler = sampler
        self.mode: Literal["record", "compare"] = mode
        self.reference: Path = reference
        self.max_steps: int = max_steps
        states = sampler.req_states
        self.device: torch.device = states.device
        slots = states.max_num_reqs
        self.keys: dict[int, str] = {}
        self.lengths: torch.Tensor = torch.zeros(
            slots, dtype=torch.int64, device=self.device
        )
        self.seen: torch.Tensor = torch.zeros(
            slots, max_steps, dtype=torch.bool, device=self.device
        )
        self.kl: torch.Tensor = torch.zeros(slots, max_steps, device=self.device)
        self.top1: torch.Tensor = torch.zeros(
            slots, max_steps, dtype=torch.bool, device=self.device
        )
        # [slot, step, vocab] logits: recorded (stock) or loaded (reference).
        self.logits: torch.Tensor | None = None
        self.pending: dict[int, torch.Tensor] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sampler, name)

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        """Register a request with the native sampler and track it if keyed.

        In compare mode the stored stock logits are loaded here and copied to the
        GPU buffer on the next sampler call.
        """
        if req_idx in self.keys:
            raise RuntimeError("A tracked request's slot was reused before collection")
        self._sampler.add_request(req_idx, prompt_len, sampling_params)
        key = (sampling_params.extra_args or {}).get(KEY)
        if not isinstance(key, str):
            return
        steps = sampling_params.max_tokens
        if steps is None or steps > self.max_steps:
            raise ValueError("Fidelity request exceeds the configured step count")
        self.keys[req_idx] = key
        self.lengths[req_idx] = steps
        self.seen[req_idx] = False
        if self.mode == "compare":
            stored: torch.Tensor = torch.load(
                reference_path(self.reference, key),
                map_location="cpu",
                weights_only=True,
            )
            if stored.shape[0] != steps:
                raise ValueError(f"Reference {key} has {stored.shape[0]} steps")
            # The logits buffer is sized on the first sampler call.
            self.pending[req_idx] = stored

    def _buffer(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logits is None:
            self.logits = torch.empty(
                self.lengths.shape[0],
                self.max_steps,
                logits.shape[-1],
                dtype=logits.dtype,
                device=self.device,
            )
        for slot, stored in self.pending.items():
            if stored.shape[-1] != logits.shape[-1] or stored.dtype != logits.dtype:
                raise ValueError("Reference logits do not match the model's logits")
            self.logits[slot, : stored.shape[0]].copy_(stored)
        self.pending.clear()
        return self.logits

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        """Observe tracked rows' logits, then sample natively."""
        rows = [
            row
            for row, slot in enumerate(input_batch.idx_mapping_np.tolist())
            if slot in self.keys
        ]
        if rows:
            if logits.shape[0] != input_batch.num_reqs:
                raise RuntimeError("Fidelity needs one logits row per request")
            self._observe(logits, input_batch, rows)
        return self._sampler(logits, input_batch)

    def _observe(
        self, logits: torch.Tensor, input_batch: InputBatch, rows: list[int]
    ) -> None:
        buffer = self._buffer(logits)
        index = torch.tensor(rows, device=self.device)
        slots = input_batch.idx_mapping[index].long()
        # The row predicts output token (position + 1 - prompt_len): the prompt's
        # last position predicts token 0; partial prefill chunks fall below 0.
        positions = input_batch.positions[input_batch.logits_indices[index]]
        prompt = self._sampler.req_states.prompt_len.gpu[slots]
        steps = (positions + 1 - prompt).long()
        valid = (steps >= 0) & (steps < self.lengths[slots])
        steps = steps.clamp(0, self.max_steps - 1)
        current = logits[index]
        if self.mode == "record":
            buffer[slots, steps] = torch.where(
                valid[:, None], current, buffer[slots, steps]
            )
        else:
            kl, agree = kl_and_top1(buffer[slots, steps], current)
            self.kl[slots, steps] = torch.where(valid, kl, self.kl[slots, steps])
            self.top1[slots, steps] = torch.where(valid, agree, self.top1[slots, steps])
        self.seen[slots, steps] |= valid

    def collect(self) -> dict[str, dict[str, Any]]:
        """Finish every tracked request; save references or return comparisons."""
        results: dict[str, dict[str, Any]] = {}
        for slot, key in sorted(self.keys.items()):
            steps = int(self.lengths[slot])
            if not bool(self.seen[slot, :steps].all()):
                raise RuntimeError(f"Fidelity request {key} missed a sampled step")
            if self.mode == "record":
                assert self.logits is not None
                torch.save(
                    self.logits[slot, :steps].cpu().clone(),
                    reference_path(self.reference, key),
                )
                results[key] = {"steps": steps}
            else:
                results[key] = {
                    "kl": self.kl[slot, :steps].tolist(),
                    "top1": self.top1[slot, :steps].tolist(),
                }
        self.keys.clear()
        return results


class FidelityWorkerExtension:
    """Worker RPCs that install and drain the evaluation-only fidelity sampler."""

    def tokenwise_fidelity_start(
        self: Any, mode: str, reference: str, max_steps: int
    ) -> bool:
        """Install the fidelity sampler on the first TP rank; return whether it did."""
        from vllm.distributed import get_tensor_model_parallel_rank

        runner = self.model_runner
        if get_tensor_model_parallel_rank() != 0 or runner.sampler is None:
            return False
        if mode not in ("record", "compare"):
            raise ValueError(f"Unknown fidelity mode {mode!r}")
        if runner.batch_sharder is not None or runner.rejection_sampler is not None:
            raise RuntimeError("Fidelity needs unsharded, non-speculative sampling")
        runner.sampler = FidelitySampler(
            runner.sampler, mode, Path(reference), max_steps
        )
        return True

    def tokenwise_fidelity_collect(self: Any) -> dict[str, dict[str, Any]] | None:
        """Collect the installed sampler's results; None on other ranks."""
        sampler = self.model_runner.sampler
        return sampler.collect() if isinstance(sampler, FidelitySampler) else None
