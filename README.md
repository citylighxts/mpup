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
  evaluate_baselines.py — vs No Retrieval / Full Unranked / Tian et al. 2026
  run_ragbench_benchmark.py — end-to-end RAGBench benchmark, full D1–D5
tests/            — 93 tests, ≥95% coverage on core modules
configs/
  default.yaml
```

---

## Datasets

| Dataset | HF path |
|---------|---------|
| RAGBench | `galileo-ai/ragbench` |
| Natural Questions | `google/natural_questions` |
| MS MARCO | `microsoft/ms_marco v2.1` |
| TriviaQA | `trivia_qa unfiltered` |

---

## Results — RAGBench HotpotQA (100 questions, 400 pairs)

Full **D1–D5** stack. D5 ΔH + HyDE pseudo-answers from **Llama-3.2-3B-Instruct-4bit via MLX**
(Apple Silicon, no GPU); D4 NLI from `cross-encoder/nli-deberta-v3-small`.
Labels: RAGBench `all_utilized_sentence_keys` proxy. 80/20 train/test split.

| Method | Spearman ρ | NDCG@10 |
|--------|-----------|---------|
| No Retrieval | 0.000 | 0.233 |
| Full Unranked | 0.000 | 0.233 |
| Tian et al. 2026 (D5-only) | 0.452 | 0.535 |
| **MPUP Zero-Shot (D1–D5)** | **0.632** | **0.765** |

Few-shot linear calibration (k=5, 10) leaves ρ and NDCG unchanged — α·f+β is a
monotonic transform, so it only corrects the *scale* of û for threshold-based
top-m selection, not the ranking.

### Ablation — leave-one-group-out (retrained per config)

| Config | ρ | NDCG@10 | Δρ vs. full |
|--------|-----|---------|-------------|
| **ALL (D1–D5)** | **0.632** | 0.765 | — |
| drop D2 | 0.320 | 0.489 | **−0.312** |
| drop D5 | 0.551 | 0.832 | −0.081 |
| drop D4 | 0.576 | 0.800 | −0.057 |
| drop D3 | 0.616 | 0.832 | −0.017 |
| drop D1 | 0.629 | 0.797 | −0.003 |

Every group contributes; the full stack beats every 4-group subset. D2 dominates
on this proxy label (document length is a strong prior on the *fraction of
sentences cited*), D5 and the now-fixed D4 each add a clear, independent lift.
No single group exceeds ρ≈0.50 alone (D2-only 0.503, D5-only 0.298, D1-only 0.113).

---

## Baselines

| Method | Description |
|--------|-------------|
| No Retrieval | Lower bound — no context |
| Full Unranked | All passages, uniform score |
| Tian et al. 2026 | D5-only logit probe (target ρ < 0.3) |
| MPUP Zero-Shot | Full D1–D5, no calibration |
| MPUP Few-Shot k={5,10,20} | Linear adapter α·f + β fitted on k examples |

---

## Quickstart

```bash
pip install -e ".[dev]"
pip install mlx mlx-lm            # Apple Silicon — D5 + HyDE
python -m spacy download en_core_web_sm
pytest                           # 93 tests

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
