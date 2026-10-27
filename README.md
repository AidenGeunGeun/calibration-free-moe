# calibration-free-moe

Code for "Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data".

During decoding, each routed expert of a frozen MoE model reads only some of its intermediate
channels. A small ternary sketch of every expert's gate and up rows stays resident on the GPU and
screens all candidate channels (the intermediate channels of a token's routed experts), scoring them
from the token's hidden state with no calibration data. The top-k channels per token and layer, the selected channels, are
computed exactly from the native BF16 weights, and a weight-only low-rank recovery map estimates the
output of the skipped channels. No step uses calibration data. The repository also holds
the calibrated baselines the paper compares against (Prox, TEAL, WINA, CATS) and the scripts and
reducers behind the paper's numbers.

Everything runs inside vLLM: stock vLLM prefills the prompt and the sparse reader runs decode.

## Layout

| Path | Contents |
|---|---|
| `calibration_free_moe/reader/` | Our reader: read budgets (`budget.py`), the resident sketch (`sketch.py`), and the Triton operator that screens, selects, computes the selected channels, recovers the skipped ones and records the read cost (`progressive.py`) |
| `calibration_free_moe/runtime/` | vLLM bindings for OLMoE, Qwen3, Qwen3.5 and Gemma 4 experts, and read reports |
| `calibration_free_moe/baselines/` | Prox, TEAL, WINA and CATS (`activation_sparsity.py`), and HiRE |
| `calibration_free_moe/eval/` | Full-vocabulary KL against stock, and the OLMES likelihood suite |
| `calibration_free_moe/analysis/` | Per-token budget allocation used by the Section 4 analyses |
| `scripts/` | Command-line entry points: task preparation, evaluation, calibration, recovery maps, speed |
| `reproduce/` | Reducers that turn stored run files into the paper's tables, figures and counts |
| `patches/` | Two patches to vLLM 0.29.0 that the sparse runtime needs |
| `tests/` | Contract tests against independent PyTorch references (most need a CUDA GPU) |

## Installation

The runtime needs Linux, an NVIDIA driver for CUDA 12.9 or newer, and Python 3.12. The paper's
results use the CUDA 12.9 builds of vLLM 0.29.0 and PyTorch 2.13.0:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu129 \
  https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
pip install -e .                    # vLLM 0.29.0, PyTorch 2.13.0, Triton 3.7.1
```

Sparse decoding needs vLLM with the two patches in `patches/` applied. Keep stock vLLM untouched for
the stock baseline and build a patched copy of the `vllm` package next to it:

```bash
SITE=$(python -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
mkdir -p overlay && cp -r "$SITE/vllm" overlay/vllm
git apply --directory=overlay --include='overlay/vllm/*' \
  patches/vllm-0.29.0-phase-metadata.patch patches/vllm-0.29.0-gdn-first-chunk.patch
```

Run this from the repository root: inside a Git checkout, `git apply` resolves paths from the
root, so `--directory=overlay` places them in the copy. The patches also touch vLLM's own tests,
which `--include` leaves out.

`phase-metadata` lets the sparse operator tell decode rows from prefill rows and accepts the
channel-major down-projection layout. `gdn-first-chunk` is upstream vLLM pull request 51565; it is
needed only for Qwen3.5 with chunked prefill. Sparse runs put the overlay and this repository first on
`PYTHONPATH`; stock runs use neither:

```bash
export PYTHONPATH=$PWD/overlay:$PWD VLLM_PLUGINS=""
```

Task preparation and likelihood reduction use a second environment, because they pin library
versions that the vLLM environment does not carry: `datasets==3.6.0`, `lm_eval==0.4.3` with
the IFEval checker's `langdetect`, `immutabledict` and `nltk`, and the OLMES evaluation suite at commit `5a51f502` (for the OLMoE likelihood tasks).
The reducers in `reproduce/` need only `pip install -e ".[reproduce]"`.

## Models

All checkpoints are the official BF16 releases from the Hugging Face Hub.

| Name in the paper | Checkpoint |
|---|---|
| Qwen3-Thinking | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| Gemma 4 | `google/gemma-4-26B-A4B-it` |
| Qwen3.5 | `Qwen/Qwen3.5-35B-A3B` |
| OLMoE | `allenai/OLMoE-1B-7B-0924` |

Each script takes a local checkpoint directory through `--model`.

## Running our reader

1. Build the weight-only recovery map (`bank` in the scripts and flags) once per model (no data, a
   few minutes on one GPU):

   ```bash
   python scripts/build_weight_recovery_bank.py --model MODEL --output BANK_DIR --source-revision REV
   ```

2. Prepare a benchmark (`gpqa_diamond`, `ifeval` or `mmlu_pro`) in the model's chat format and
   sampling settings (`--preset qwen3-thinking`, `gemma4` or `qwen35`), in the preparation
   environment. Dataset revisions are pinned in the script. GPQA-Diamond needs approved access on
   the Hub. Generation lengths and sample counts for each task are in the paper's appendix.

   ```bash
   python scripts/thinking_requests.py prepare --task gpqa_diamond --model MODEL \
       --preset qwen3-thinking --max-model-len MAX_LEN --max-tokens MAX_TOKENS --samples 4 \
       --output REQUESTS
   ```

3. Generate with our reader, then score:

   ```bash
   python scripts/eval_vllm_generation.py --model MODEL --requests REQUESTS --revision REV \
       --bank BANK_DIR/weight-subspace-s1.pt --retention 0.123046875 \
       --selection-score point --screen-keep 0.625 --output SCORES
   python scripts/thinking_requests.py reduce --requests REQUESTS --scores SCORES --output METRICS
   ```

   Without `--bank`, `--retention`, `--selection-score` and `--screen-keep`, and in the
   unpatched environment, the same command runs the full model. The paper's two heavy
   budgets use `--selection-score point --screen-keep 0.625` with these retentions; the 28% and 38%
   budgets use `--retention 0.2` and `0.3` without those two flags. Budgets are read costs (the
   bytes a method must read, priced in 32-byte sectors) as a share of dense routed-expert reads.

   | Model | about 13% reads | about 18% reads |
   |---|---|---|
   | Qwen3-Thinking | 0.07291666666666667 | 0.123046875 |
   | Gemma 4 | 0.07492897727272728 | 0.125 |
   | Qwen3.5 | 0.069824 | 0.120361 |
   | OLMoE | 0.074462890625 | 0.1246337890625 |

   Qwen3-Thinking and Gemma 4 fit one 96 GB GPU, or two 48 GB GPUs with
   `--tensor-parallel-size 2 --enable-expert-parallel`; Qwen3.5 ran on one 96 GB GPU.

## Baselines

Calibrated baselines fit their thresholds on chat data first (`scripts/prepare_chat_panel.py`,
then `scripts/collect_moe_inputs.py` for the expert inputs), then evaluate with the same
`eval_vllm_generation.py` command and a policy flag:

| Selector in the paper | Fit | Evaluate with |
|---|---|---|
| Prox as published, calibrated | `calibrate_prox.py --panel PANEL --target-reads 0.13 0.18` | `--policy prox --thresholds FILE` |
| Prox score, top-k | none | `--policy prox-free --retention F --input-keep 0.3` |
| TEAL as published, calibrated | `calibrate_baselines.py --states STATES --policy teal-greedy --sparsity S` | `--policy teal-greedy --thresholds FILE` |
| WINA as published, top-k | `calibrate_baselines.py --states STATES --policy wina-greedy-topk --sparsity S` | `--policy wina-greedy-topk --thresholds FILE` |
| WINA score, calibrated | `calibrate_baselines.py --states STATES --policy wina --sparsity S` | `--policy wina --thresholds FILE` |
| CATS, calibrated / top-k | `calibrate_baselines.py --states STATES --policy cats --sparsity S` | `--policy cats --thresholds FILE`, or `--policy cats-topk --retention F` |
| Our reader, calibrated | `calibrate_selection_thresholds.py --panel PANEL --bank BANK --target-retention F` | `--selection-thresholds FILE --retention (N-1)/N`, with the same recovery map, score and screen flags |

For our reader with a threshold, the threshold sets each token's count and the retention is only
a cap: N is the number of candidate channels (routed experts times expert width).
`S` is the input sparsity the method is calibrated for (0.8 targets 20% of inputs). Prox's study
recipe (Section 5) fits with `--recipe study --input-keep 0.3` and evaluates with
`--prox-recipe study --input-keep 0.3`.

Each script's `--help` lists the remaining options.

## Likelihood and KL (OLMoE)

`scripts/olmes_requests.py prepare` builds the OLMES or lm-eval tasks (`--protocol prox` for Prox's
own suite), `scripts/eval_vllm_continuation.py` scores them with stock prefill or, with
`--sparse-prefill`, with a sparse prompt, and `scripts/olmes_requests.py reduce` applies the tasks'
own metrics. For KL against the full model, record stock once with `--fidelity-record DIR` and
score every sparse arm with `--fidelity-reference DIR`.

## Speed

`scripts/speed_block.py` times the MoE block on captured decode inputs, and `scripts/speed_harness.py`
times whole-model decoding (time per output token) on real chat prompts from
`scripts/prepare_speed_panel.py`. Both run each arm in a fresh process on one GPU and compare
against the faster of stock vLLM's default and tuned fused-MoE configurations;
`scripts/reduce_speed_results.py` produces the speedups. The router runs as stock in both arms.

## Reproducing the paper's numbers

`reproduce/README.md` maps every number in the main text to the reducer and the stored runs it
reads. The reducers need no GPU.

## Tests

```bash
pytest tests
```

Most tests compare the CUDA operator against an independent PyTorch reference and need a GPU; the
rest run on CPU. Run them with the sparse `PYTHONPATH` above. Tests of task preparation and IFEval
scoring need the second environment's packages and are skipped without them; run those files
there with `PYTHONPATH=$PWD pytest tests/test_heldout_panels.py tests/test_thinking_requests.py`.

## License

Apache License 2.0; see `LICENSE`.
