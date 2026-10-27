# Reproducing the paper's numbers from stored runs

Each module reads the result files the evaluation scripts write (`metrics.json` and
`scores.json`, or zstd-compressed `.json.zst` twins) and prints the paper's numbers or
writes figure data. The inputs of `context_cut`, `layer_budgets` and `drift_reads`, the
allocation records of `allocation_controls` and the layer files of `components` must be plain
JSON. Run from the repository root with Python 3.12, `numpy` and `zstandard`; `context_cut`
and `allocation_controls` also need torch (CPU is enough), and `context_cut` `safetensors`.
No module needs vLLM or a GPU.

`RUNS` holds the run directories named below (for example `RUNS/standard-protocols/arms`).
`ALLOC` is the allocation-study root: Qwen3-Thinking at the top level (`deep/`,
`deep/reference/`, `cut/`, `sweep/`), Gemma 4 and Qwen3.5 under `gem/` and `q35c/` (each with
`deep/`, `reference/` and `cut/`), and the generation runs in `gen/` and `gen2/`. `DEEP` is
the deep-analysis root.

Reads are the routed-expert read cost (bytes a method must read, priced in 32-byte sectors)
as a percentage of dense routed reads; Prox is priced on its best single layout. Paired
comparisons use one value per question (the mean over its samples) and the standard error of
the per-question differences; a difference counts beyond two standard errors. Printed SEs are
one SE.

## Headline, keep rule and deployment reads

| Paper item | Command | Inputs under `RUNS` |
|---|---|---|
| Table 1 (`tab:accuracy`), Figure 1a, `figures/headline_data.json` (Figure 1, headline grid, Figure 4c, drift table) | `python -m reproduce.headline --runs RUNS --output headline_data.json` | `headline-matrix`, `standard-protocols`, `gpqa-x8`, `gemma-repro`, `wina-published` |
| Section 5: our reader with a calibrated threshold (34 ties, 2 top-k leads of 36), CATS (16 of 18), Prox on Qwen3.5 GPQA-Diamond at 13% | same command (printed after Table 1) | same |
| Figure 5, Section 5 counts (40 of 43, 37 of 41, OLMoE Prox 7 of 10, 5 of 11 post-trained, +11.5 +- 1.9), `figures/data/{scal,rec}-{target,matched}.json`, `figures/tab-keep-pooled.tex` values | `python -m reproduce.keep_rule --runs RUNS --output figures/data` | `score-calibration`, `keep-rule-study`, `prox-reconcile`, `standard-protocols`, `benchmark-suite` (OLMoE full model) |
| Section 4.1 and Figure 4c: 72 of 72 calibrated runs below budget, Prox at 18% reads 12.9% (minimum 9.9%), our reader's threshold version 0.4 to 5.8 points below budget; all 216 calibrated runs (drift appendix) | `python -m reproduce.drift_reads --headline headline_data.json` (or `--runs RUNS`) | as for `reproduce.headline` |

## Analysis reducers (Sections 4.1, 4.2 and 5)

| Paper item | Command | Inputs |
|---|---|---|
| Sec. 4.1 Prox context cut, 57% more (App. E table: 9,983 / 11,423 / 15,422 / 17,283; +15.0, +56.6; Gemma 4 +12.5) | `python -m reproduce.context_cut --prox ALLOC ALLOC/gem ALLOC/q35c` | `ALLOC/cut/keep{0,1000}.rec`, `ALLOC/deep/b13-chat/d0*/threshold/rec` (and under `gem/`, `q35c/`) |
| App. E exact-score cut +30.1, +10.4, +19.3 | `python -m reproduce.context_cut --cut-json DEEP/cut-thk/cut.json` (stored aggregate) or `--states DEEP/cut-thk/states --model QWEN3_THINKING --mechanism DEEP/p0-thk/mechanism.json` (recompute, CPU) | `DEEP/cut-thk/` |
| Sec. 4.1 fixed counts 3.3, 12.5, 10.5; constant vs decay at most 1.0; App. E equal reads, schedules | `python -m reproduce.allocation_controls --deep ALLOC/deep ALLOC/gem/deep ALLOC/q35c/deep` | `ALLOC/{,gem/,q35c/}deep/b1{3,8}-{chat,trace}/d0*/*/scores.json` |
| Sec. 4.1 count vs loss Spearman -0.29 to +0.14; App. E sweep correlations | add `--sweep ALLOC/sweep --references ALLOC/deep/reference ALLOC/gem/reference ALLOC/q35c/reference` | `ALLOC/sweep/`, threshold `rec/` and reference scores |
| Sec. 4.2 Qwen3.5 GPQA-Diamond +7.1; App. E generation pairs | add `--pair ALLOC/gen2/q35-gpqa-uniform ALLOC/gen2/q35-gpqa-threshold` (and the `ALLOC/gen/{ifeval,gpqa}-*`, `ALLOC/gen2/q35-ifeval-*` pairs); `--runs ALLOC/gen2/gem-*-uniform` | `ALLOC/gen/`, `ALLOC/gen2/` |
| Sec. 4.2 calibrated layer budgets on reasoning text (App.: -15.7, -18.7, +2.4) | `python -m reproduce.layer_budgets --layer-oracle DEEP/layer-oracle` | `DEEP/layer-oracle/{eval,panel-eval.json}` |
| App. layer-budget accuracy (-1.77, -1.85 Qwen3-Thinking; -0.76, +1.11 Gemma 4) | add `--pair RUNS/layer-alloc/arms/<m>/<task>/cal-p10 RUNS/standard-protocols/arms/<m>/<task>/isp-p10` | `RUNS/layer-alloc/arms/`, `RUNS/standard-protocols/arms/` |
| Sec. 4.2 calibrated recovery map 0 of 18 (App.: weight-only +2.1 on Qwen3-Thinking MMLU-Pro at 13% reads) | `python -m reproduce.calibrated_recovery --arms RUNS/standard-protocols/arms` | `*/calrec-p*`, `*/isp-p*`, `{gem,thk}/ifeval/stack-p05` |
| Sec. 5 keep rule, output scaling 1.3-1.5x, INT4 gate / TEAL 2.8x / 5.1x, energy Spearman -0.96, recovery vs exact channels, partial screen 0.488 -> 0.310 and 0.245 -> 0.187, Gemma 4 +11.6 and +13.7; App. components table | `python -m reproduce.components --fidelity RUNS/fidelity --component-arms RUNS/component-arms-olmoe/arms --energy RUNS/energy-kl --arms RUNS/standard-protocols/arms` | `RUNS/fidelity/{olmoe-confirm,olmoe,gemma4,qwen3-thinking,qwen35}/`, `RUNS/component-arms-olmoe/arms/`, `RUNS/energy-kl/match-g*.json`, `RUNS/standard-protocols/arms/gem/` |
