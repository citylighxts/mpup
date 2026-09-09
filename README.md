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
| D3 | 2 | Query entity overlap (spaCy), HyDE answerability |
| D4 | 2 | Entailment prob, contradiction score (DeBERTa NLI) |
| D5 | 3 | Base perplexity, context perplexity, ΔH |

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
experiments/
  ablation.py     — per-group ρ drop study
  evaluate_baselines.py — vs No Retrieval / Full Unranked / Tian et al. 2026
tests/            — 86 tests, ≥95% coverage on core modules
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
pytest                     # 86 tests
make features              # extract D1-D5
make train                 # fit predictor
make evaluate              # run baselines
make ablation              # per-group ρ drop
```

### Python API

```python
from src.pipeline.mpup_pipeline import MPUPPipeline
from src.predictor.mpup_predictor import MPUPPredictor
from src.probing.logit_probe import MockProber

predictor = MPUPPredictor.load("checkpoints/mpup", algorithm="xgboost")
pipeline  = MPUPPipeline(predictor=predictor, prober=MockProber(), top_m=5)

ranked = pipeline.rank(query, passages)
prompt = pipeline.build_rag_prompt(query, ranked)
```

---

## Requirements

- Python ≥ 3.11
- `sentence-transformers`, `rank-bm25`, `transformers`, `torch`
- `xgboost`, `scikit-learn`, `spacy`, `textstat`

```bash
pip install -e .
python -m spacy download en_core_web_sm
```
