# Merge notes — UtilityTransfer → MPUP

Branch: `merge-utilitytransfer`. **Nothing has been pushed.** Review before merging to `main`.

A parallel implementation of the same research idea (`UtilityTransfer/`) ran ~35k utility
labels across 7 LLMs and 3 datasets. This branch ports the parts of it that MPUP was missing
and fixes three things its measurements showed were wrong. MPUP's architecture, feature
stack (D1–D5), HyDE design, and test discipline are kept as the foundation.

Test count: **93 → 110**, all passing.

---

## Why the two efforts were compared at all

MPUP's reported results use RAGBench's `all_utilized_sentence_keys` as the utility label;
UtilityTransfer runs an LLM twice per passage. Rather than argue, both labels were computed
on identical (query, passage) pairs — 300 RAGBench HotpotQA questions, 1,199 passages, with
gold answers recovered by joining back to HotpotQA (100% matched).

**The proxy held up well.** It correlates with true LLM utility at per-query ρ = 0.558
(Qwen) / 0.595 (Mistral), against a ceiling of 0.836 measured from Qwen–Mistral agreement on
the same pairs — roughly 67–71% of what another LLM captures. Training on it recovers 96%
(Mistral) and 110% (Qwen) of training on true labels, and a proxy-trained predictor reaches
73–85% of an in-domain reference on NQ and TriviaQA, **datasets with no annotations at all**.

So the cheap label is a legitimate way to scale past a single GPU, and that is a real result.
Two limits to keep in view:

1. `u_proxy` correlates with document length at −0.413 against −0.12 for true utility, a
   direct consequence of dividing by sentence count. This is why D2 dominates the ablation
   at Δρ = −0.312: a predictor trained on it is substantially learning to invert length.
2. It is one label shared by every LLM, so it cannot address whether utility is LLM-specific.

---

## What changed

### `src/evaluation/metrics.py`

- **`token_f1_score` used `set()` overlap**, discarding token multiplicity: `"paris paris
  paris"` against gold `"paris"` scored 1.0 where SQuAD F1 gives 0.5. Now multiset.
- **Added `relaxed_match_score`** (DPR `has_answer`): gold answer as a contiguous token span
  of the prediction. Instruction-tuned models answer in sentences, so strict EM marks them
  wrong; since utility is a *difference* of correctness, that collapses labels toward zero.
  Token spans rather than raw substrings avoid matching `"it"` inside `"withering"`.
- **Added `relaxed_combined_score`**, `_normalize` now applies NFD.
- **Added `spearman_per_query` and `top1_accuracy`.** Global Spearman pools all pairs, so
  part of it is separating easy queries from hard ones — which no reranker can act on. Every
  finding from the other effort is stated in per-query terms. `top1_accuracy` replaces
  "top-k accuracy", which is trivially 1.0 when there are k or fewer passages per query.

### `src/predictor/calibration.py`

- **Added `FeatureSpaceAdapter`.** `LinearCalibrator` is kept and unchanged — it is the right
  tool for calibration error, cutting MSE by an order of magnitude. But `α·f + β` with α > 0
  preserves order exactly, so it **provably cannot move Spearman or NDCG**; the README
  already notes this, and on real data it reproduces the zero-shot score to six decimals. At
  k = 4 it can even *hurt*, since a noisy fit can give α < 0 and invert the ranking. The new
  adapter fits a regularized ridge in feature space and blends with the standardized base
  score, which does change ranking (+0.17 ρ at k=32 on a distant target LLM).

### `src/data/llm_backend.py` (new) and `src/data/utility_labeler.py`

- **`QuantizedLLM`** — `SourceLLMAnswerer` loads fp16 and answers one passage per call. A 7B
  model in fp16 needs ~15GB and **cannot run on an 8GB card at all**. This loads 4-bit NF4
  (~5GB for 7B), batches generation, and exposes token-level logprobs. It scores only the
  continuation positions: upcasting the full logits tensor costs batch × seq_len × vocab
  floats, which at a 152k vocabulary exhausted both VRAM and host RAM before it was fixed
  (7858 MiB → 4326 MiB on the same job).
- **`BatchedUtilityLabeler`** — batches across queries and passages, generating the
  no-context answer once per query. ~0.3s per labelled passage on an RTX 4060. Both original
  classes are untouched.

Smoke-tested end to end on a real GPU with Qwen2.5-0.5B: loads in 5.2s, labels correctly,
logprobs sane, unloads cleanly.

---

## Corrected benchmark numbers

`results/ragbench_hotpotqa_cuda.json` — 100 questions, full D1–D5 on **CUDA** (Qwen2.5-7B,
4-bit), split by query, Tian baseline retrained.

| | published (MLX, row split, zero-masked baseline) | corrected |
|---|---|---|
| MPUP zero-shot ρ | 0.632 | **0.577** |
| NDCG@10 | 0.765 | 0.764 |
| Tian et al. baseline ρ | 0.452 | **0.038** |

Two things to note, and the second is good news for MPUP:

**The baseline was the larger error, and fixing it widens MPUP's margin** — from
0.632 vs 0.452 to 0.577 vs 0.038. The retrained figure also now agrees with the ablation's
`d5_only = 0.0379`, where the two tables previously disagreed.

**The row split was inflating results by about the amount MPUP's headline dropped.** Run on
one cached feature matrix with nothing changed but the split:

| split | ρ | NDCG@10 |
|---|---|---|
| by query (correct) | 0.4354 | 0.5414 |
| by row (leaky) | 0.4873 | 0.6414 |

Leakage is worth **+0.052 ρ and +0.100 NDCG**, which closely matches the −0.055 the real run
lost. So the drop is attributable to the split fix, not to swapping MLX Llama-3.2-3B for
CUDA Qwen2.5-7B.

**The ablation's NDCG contradiction shrinks but does not vanish**: 4 of 5 leave-one-out
configs beat the full stack before, 2 of 5 now (dropping D3 or D5). The "every group
contributes" claim is now true for ρ and still false for NDCG.

**D5 is much weaker on Qwen2.5-7B than on MLX Llama-3.2-3B** — `d5_only` falls from 0.2975 to
0.0379, and dropping D5 now costs only Δ−0.018. Worth investigating before D5 is presented as
a contribution: a smoke test showed ΔH coming out *negative* (context raising entropy), the
opposite of the direction the feature's name implies.

`--cache` now stores the feature matrix so the split, the baselines and the ablation can be
re-examined without paying for GPU extraction again; `--split row` reproduces the old
behaviour for comparison.

## Two things in the current results that should be fixed before publication

Neither inflates MPUP's claims — the first actually works against them — but both would be
caught by a reviewer.

1. **The "Tian et al. 2026" baseline is not retrained.** `_only_d5()` takes the full 14-dim
   matrix, zeroes the 11 non-D5 columns, and feeds that to a predictor trained on all
   features. A tree model given out-of-distribution zeros returns arbitrary values. The
   ablation script *does* retrain and reports D5-only at **0.2975**, while the baseline table
   reports the same quantity as **0.452**. The correct baseline is weaker, so fixing it makes
   MPUP look better.
2. **The ablation's NDCG column contradicts its stated conclusion.** The README says "Every
   group contributes; the full stack beats every 4-group subset." On NDCG, four of five
   leave-one-out configs beat the full model (0.797, 0.832, 0.800, 0.832 vs 0.765). Only
   dropping D2 hurts. The claim holds for ρ only.

---

## Still to do

- **Port the analysis scripts.** The other effort's scientific results — cross-LLM agreement,
  the knowledge-gap law, judge transfer, learning curves, feature diagnostics — live in
  `UtilityTransfer/scripts/` and read parquet files. They need adapting to MPUP's module
  layout, or MPUP needs a thin compatibility layer. Where these should live is a decision for
  both authors.
- **Decide the feature system.** MPUP has D1–D5 (14-dim); the other effort has its own 14
  features. They overlap but are not identical. The most important difference: the other
  effort's answerability feature is an **LLM prompt** ("can this passage answer the
  question?") scored from the Yes/No logits, and it was by a wide margin the single strongest
  feature — on its own it beat the other thirteen combined. MPUP's D3 answerability is
  lexical overlap against the HyDE pseudo-answer, which is cheaper. Worth measuring both.
- **Write a CUDA HyDE generator.** `HyDEGenerator` and `MLXProber` require `mlx`/`mlx-lm`,
  which is Apple Silicon only. `LogitProber` already has a CUDA path; HyDE does not, so the
  full stack cannot currently run on the Windows/RTX 4060 machine. Small job, but it blocks
  one of the two collaborators.
- **Bring the label data across.** ~17MB of parquet in `UtilityTransfer/data/` (35k labels,
  7 models, 3 datasets). Not copied yet to avoid two sources of truth.

---

## The finding that applies to both projects

Three independent measurements say **feature quality is the bottleneck**, not labels, not the
model class, and not the calibration step:

- Fitting on the evaluation data itself (an oracle upper bound) beats the honest Ridge score
  by only +0.068 — better modelling is close to exhausted.
- A single answerability prompt outperforms the other thirteen features combined.
- The raw RAGBench annotation, with no model and no features at all, outranks both trained
  predictors.

MPUP's HyDE and ΔH are the most promising untested ideas either project has for moving that,
which is a good argument for building on this codebase rather than the other one.
