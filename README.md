# MPUP — Meta-Probing Utility Predictor

Predict RAG document utility scores **without fine-tuning the target LLM**.

MPUP scores retrieved passages by combining five feature groups (D1–D5) and training a lightweight predictor, so the target model stays frozen during deployment.

---

## How it works

```
Query + Passages
     │
     ├─ BM25 / Dense Retrieval ──────────────────────── D1 (4-dim)
     ├─ Document Quality (length, readability, cred.) ── D2 (3-dim)
     ├─ Entity Overlap + HyDE Answerability ─────────── D3 (2-dim)
     ├─ NLI Faithfulness (DeBERTa) ───────────────────── D4 (2-dim)
     └─ Logit Probing ΔH = H_base − H_ctx ───────────── D5 (3-dim)
              │
        x_i ∈ ℝ¹⁴
              │
        GBR / MLP Predictor
              │
        û_i (utility score) ──► rank desc ──► top-m RAG prompt
```

Ground-truth utility labels:

```
u_i_true = clip(combined_score(with p_i) − combined_score(no ctx), −1, 1)
combined_score = 0.5 × EM + 0.5 × token-F1
```

---

## Feature groups

| Group | Dims | Signal |
|-------|------|--------|
| D1 | 4 | BM25 score, dense cosine sim, rank, score gap |
| D2 | 3 | Token length, Flesch-Kincaid grade, source credibility |
| D3 | 2 | Query entity overlap (spaCy), HyDE answerability (lexical overlap vs. pseudo-answer) |
| D4 | 2 | Entailment prob, contradiction score — NLI cross-encoder, premise=passage / hypothesis=HyDE pseudo-answer |
| D5 | 3 | Base perplexity, context perplexity, ΔH = H_base − H_ctx |

The HyDE pseudo-answer is generated once per query (MLX Llama-3.2-3B) and shared by D3 and D4. Without a generator, a rule-based question→statement stub is used.

---

## Project structure

```
src/
  data/           — utility labeler, dataset loaders
  retrieval/      — BM25Retriever, DenseRetriever
  features/       — D1–D5 extractors + concatenate.py
  probing/        — LogitProber, MockProber
  predictor/      — MPUPPredictor (GBR/MLP), LinearCalibrator
  evaluation/     — EM, F1, Spearman ρ, NDCG@k, MRR
  pipeline/       — MPUPPipeline (rank + RAG prompt builder)
  features/hyde.py — HyDE pseudo-answer generator (MLX Llama / rule-based mock)
experiments/
  ablation.py     — leave-one-group-out ablation (retrained per config)
  evaluate_baselines.py — vs No Retrieval / Full Unranked / logit-probe-only
  run_ragbench_benchmark.py — end-to-end RAGBench benchmark, full D1–D5
tests/            — 115 tests (≈65% total line coverage; see audit below)
configs/
  default.yaml
```

---

## Datasets

| Dataset | HF path |
|---------|---------|
| RAGBench | `galileo-ai/ragbench` (one config per subset, e.g. `hotpotqa`) |
| Natural Questions | `google-research-datasets/natural_questions` |
| MS MARCO | `microsoft/ms_marco`, config `v2.1` |
| TriviaQA | `mandarjoshi/trivia_qa`, config `unfiltered` |

---

## Results — RAGBench HotpotQA (100 questions, 400 pairs)

Full **D1–D5** stack. D5 ΔH + HyDE pseudo-answers from **Qwen2.5-7B-Instruct** (4-bit, 8GB
CUDA); D4 NLI from `cross-encoder/nli-deberta-v3-small`. Labels: RAGBench
`all_utilized_sentence_keys` proxy. **Split by query, 80/20** — a row split leaks features
shared across one query's passages and inflates ρ by ~0.05.

| Method | Spearman ρ | NDCG@10 |
|--------|-----------|---------|
| No Retrieval | 0.000 | 0.251 |
| Full Unranked | 0.000 | 0.251 |
| Logit-probe only (D5) | 0.038 | 0.280 |
| **MPUP Zero-Shot (D1–D5)** | **0.577** | **0.764** |

Few-shot *linear* calibration with a positive slope leaves ρ and NDCG unchanged — α·f+β is
order-preserving, so it cannot improve a rank metric. It only corrects the scale of û for
threshold-based top-m selection. With a small k the fitted slope can come out negative, which
*inverts* the ranking (see the audit below). Use `FeatureSpaceAdapter` for k-shot adaptation
that actually changes the ranking.

### Ablation — leave-one-group-out (retrained per config)

| Config | ρ | NDCG@10 | Δρ vs. full | group alone (ρ) |
|--------|-----|---------|-------------|-----------------|
| **ALL (D1–D5)** | **0.577** | 0.764 | — | — |
| drop D2 | 0.439 | 0.367 | **−0.138** | 0.362 |
| drop D1 | 0.514 | 0.615 | −0.063 | 0.269 |
| drop D4 | 0.521 | 0.669 | −0.057 | 0.047 |
| drop D3 | 0.525 | 0.776 | −0.052 | 0.221 |
| drop D5 | 0.559 | 0.774 | −0.018 | 0.038 |

Every group contributes; the full stack beats every 4-group subset. No single group exceeds
ρ≈0.37 alone — the accuracy comes from the combination, not from any one signal. Ablations are
**retrained** per configuration; zero-masking columns on a predictor trained with all features
returns arbitrary values (it gave 0.452 for the D5-only quantity that honest retraining puts
at 0.038).

### Audit: re-run on Apple Silicon (2026-10)

The tables above come from a CUDA/Qwen2.5-7B run on a single 80/20 split of 100 questions
(about 20 test questions). An independent re-run with the MLX backend (Llama-3.2-3B-Instruct
4-bit for D5 and HyDE, same 100 questions) gave a different picture. The audit scripts are
not part of this repository; the numbers are reproducible only in outline via
`run_ragbench_benchmark --backend mlx`.

| Quantity | Result |
|----------|--------|
| Benchmark split (seed 42), MPUP zero-shot ρ | 0.515 (README: 0.577) |
| MPUP ρ over 20 random query splits | 0.554 ± 0.089 (min 0.388, max 0.688) |
| "Shortest document first" baseline, same 20 splits | 0.406 ± 0.096 |
| Paired leave-one-group-out Δρ over 30 splits | D2 −0.126, D1 −0.084, D4 −0.022, D5 −0.017, D3 −0.016 |
| Same benchmark with random mock D5 + rule-based HyDE | ρ ≈ 0.48–0.52 |
| Linear calibrator, k=5: slope α<0 | 25 of 200 draws; mean ρ 0.368 vs 0.515 zero-shot (worst −0.515) |
| Linear calibrator, k=10 / k=20: slope α<0 | 4.5% / 0.5% of draws |

Reading this: a single split is too noisy to rank methods (±0.09 across splits). The
RAGBench proxy label correlates −0.41 with document length (390 test questions), so a length-only
baseline already reaches ρ≈0.41 and should be reported next to MPUP. D2 (which contains
length) is the largest contributor. Replacing the LLM-derived D5/HyDE with random and rule-based
stand-ins lowers mean ρ by about 0.07 (0.48 vs 0.55 over the same 20 splits), so the LLM features
contribute something, but roughly half of MPUP's margin over the length baseline (+0.08 of
+0.15) is reachable without them. These are single-seed model runs on one dataset, and the
mock-vs-real gap was not tested for significance.

Test status: 115 tests pass, but total line coverage is about 65%. The "≥95% on core modules"
holds for the predictor, pipeline, metrics and D1/D2/D5 modules; `llm_backend`,
`dense_retriever` (0%), `logit_probe` (37%) and `hyde` (49%) are largely untested.

---

## Baselines

| Method | Description |
|--------|-------------|
| No Retrieval | Lower bound — no context |
| Full Unranked | All passages, uniform score |
| Logit-probe only | D5 features alone, retrained (isolates the ΔH signal) |
| MPUP Zero-Shot | Full D1–D5, no calibration |
| MPUP Few-Shot k={5,10,20} | Feature-space ridge adapter fitted on k examples |

---

## Quickstart

```bash
pip install -e ".[dev]"
pip install mlx mlx-lm            # Apple Silicon — D5 + HyDE
python -m spacy download en_core_web_sm
pytest                           # 115 tests

python -m experiments.run_ragbench_benchmark --subset hotpotqa --n 100
python -m experiments.run_ragbench_benchmark --subset hotpotqa --n 100 --mock  # no MLX
```

### Python API

```python
from src.pipeline.mpup_pipeline import MPUPPipeline
from src.predictor.mpup_predictor import MPUPPredictor
from src.probing.logit_probe import MLXProber       # Apple Silicon — no GPU
from src.features.hyde import HyDEGenerator

predictor = MPUPPredictor.load("checkpoints/mpup", algorithm="xgboost")
prober    = MLXProber("mlx-community/Llama-3.2-3B-Instruct-4bit")
hyde      = HyDEGenerator("mlx-community/Llama-3.2-3B-Instruct-4bit")
pipeline  = MPUPPipeline(predictor=predictor, prober=prober,
                         hyde_generator=hyde, top_m=5)

ranked = pipeline.rank(query, passages)
prompt = pipeline.build_rag_prompt(query, ranked)
```

---

## Requirements

- Python ≥ 3.11
- `sentence-transformers`, `rank-bm25`, `transformers`, `torch`
- `scikit-learn`, `spacy`, `textstat`, `datasets`
- `mlx`, `mlx-lm` (Apple Silicon — D5 logit probing + HyDE)

```bash
pip install -e .
python -m spacy download en_core_web_sm
pip install mlx mlx-lm
```
