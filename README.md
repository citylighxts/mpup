# MPUP Pipeline — Meta-Probing Utility Predictor

MPUP memprediksi **utility score** dokumen kandidat terhadap target LLM untuk RAG, tanpa fine-tuning pada model target. Menggabungkan 1-pass logit probing (D5) dengan fitur NLP eksternal (D1–D4).

> Referensi: *Technical Documentation: MPUP Pipeline & Logit Probing Architecture* (Notion, 2026-09-07)

---

## Arsitektur

```
Query q
  └─→ Candidate Retrieval (BM25 / Dense)
         └─→ Candidate Passages {p1..pK}
                ├─→ Logit Probing M_target      → D5: H_base, H_ctx, ΔH
                ├─→ Retriever scores             → D1: BM25, dense sim, rank, gap
                ├─→ Document quality (textstat)  → D2: length, readability, credibility
                ├─→ Semantic utility (NER/HyDE)  → D3: entity overlap, answerability
                └─→ Faithfulness (DeBERTa-NLI)   → D4: entailment, contradiction
                       └─→ x_i = concat(D1..D5)
                              └─→ MPUP Predictor f_source (XGBoost / MLP)
                                     ├─→ Zero-Shot:  û_i = f_source(x_i)
                                     └─→ Few-Shot:   û_i = α × f_source(x_i) + β
                                            └─→ Top-m passages → RAG Prompt → M_target → Answer
```

---

## Feature Matrix

| Kode | Kategori | Variabel | Tools |
|------|----------|----------|-------|
| D1 | Retriever-Centric | BM25 score, dense sim, rank, score gap | rank-bm25, FAISS |
| D2 | Document-Quality | Token length, Flesch-Kincaid, credibility | textstat, NLTK |
| D3 | Semantic Utility | Entity overlap (NER), HyDE answerability | spaCy |
| D4 | Faithfulness | Entailment prob, contradiction score | DeBERTa-v3 NLI |
| D5 | LLM-Aware | Base/ctx perplexity, ΔH = H_base − H_ctx | Target LLM (1-pass) |

### Interpretasi ΔH (D5)

| Nilai ΔH | Interpretasi |
|----------|-------------|
| ΔH ≫ 0 | Kebingungan LLM berkurang drastis → dokumen ber-utility tinggi |
| ΔH ≈ 0 | Tidak ada perubahan → redundan / neutral |
| ΔH < 0 | Kebingungan bertambah → dokumen merusak / meningkatkan noise |

---

## Dataset Benchmark (Notion Section 6.3)

| Dataset | Source | Split |
|---------|--------|-------|
| RAGBench | `galileo-ai/ragbench` (HF) | test |
| Natural Questions | `google/natural_questions` (HF) | validation |
| MS MARCO | `microsoft/ms_marco` v2.1 (HF) | validation |
| TriviaQA | `trivia_qa` unfiltered (HF) | validation |

**Ground-truth utility:**
```
u_i_true = combined_score(answer | context=p_i, query=q)
         - combined_score(answer | no context, query=q)
```
di mana `combined_score = 0.5 × EM + 0.5 × token-F1`.

---

## Baseline Pembanding (Notion Section 5)

| Method | Perlu Fine-tuning? | Target ρ |
|--------|-------------------|---------|
| No Retrieval | — | lower bound |
| Full Unranked | — | naive |
| Tian et al. (2026) | Tidak | ρ < 0.3 |
| LURE-RAG | Ya (per LLM) | upper bound |
| RRPO | Ya (per LLM) | upper bound |
| **MPUP Zero-Shot** | Tidak | >> 0.3 |
| **MPUP Few-Shot** | Tidak (linear adapter) | ≈ fine-tuned |

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Alur Kerja Lengkap

### Menggunakan Makefile

```bash
# Full pipeline (ragbench, 500 samples)
make all DATASET=ragbench SOURCE_MODEL=meta-llama/Llama-3.2-1B-Instruct \
         TARGET_MODEL=meta-llama/Llama-3.1-8B-Instruct MAX_SAMPLES=500

# Atau step-by-step:
make label    # Step 1: label u_i_true dengan source LLM
make score    # Step 2: tambah BM25 scores
make features # Step 3: ekstrak D1-D5 feature vectors
make train    # Step 4: train predictor
make evaluate # Step 5: evaluasi vs semua baseline
make ablation # Step 6: ablation study per feature group
```

### Manual (per script)

```bash
# Step 1 – Label utility scores
python scripts/label_utility.py \
  --dataset ragbench \
  --source-model meta-llama/Llama-3.2-1B-Instruct \
  --output data/raw/ragbench_labeled.jsonl \
  --max-samples 500

# Step 2 – Tambah BM25 retrieval scores
python scripts/retrieve_and_score.py \
  --input data/raw/ragbench_labeled.jsonl \
  --output data/raw/ragbench_scored.jsonl

# Step 3 – Ekstrak feature matrix D1-D5
python scripts/extract_features.py \
  --data data/raw/ragbench_scored.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir data/processed

# Step 4 – Train MPUP predictor
python scripts/train.py \
  --features data/processed/features.npy \
  --labels data/processed/labels.npy \
  --output checkpoints/mpup_ragbench

# Step 5 – Evaluasi vs baselines
python experiments/evaluate_baselines.py \
  --features data/processed/features.npy \
  --labels data/processed/labels.npy \
  --checkpoint checkpoints/mpup_ragbench \
  --dataset ragbench \
  --output-dir results
```

---

## Metrik Evaluasi (Notion Section 6.1)

| Metrik | Deskripsi |
|--------|-----------|
| Spearman ρ | Korelasi ranking ûᵢ vs uᵢ_true — **metrik utama** |
| NDCG@k, MRR | Kualitas ranking top-m dokumen |
| EM / F1 | Akurasi jawaban end-to-end dari M_target |
| Computational overhead | Waktu inferensi vs biaya fine-tuning |

---

## Struktur Project

```
mpup-pipeline/
├── src/
│   ├── data/
│   │   ├── loaders.py          # Dataset loaders (ragbench, NQ, MSMARCO, TriviaQA)
│   │   └── utility_labeler.py  # Compute u_i_true via source LLM inference
│   ├── retrieval/
│   │   └── retriever.py        # BM25Retriever
│   ├── probing/
│   │   └── logit_probe.py      # 1-pass logit probe → ΔH (D5)
│   ├── features/
│   │   ├── d1_retriever_centric.py
│   │   ├── d2_document_quality.py
│   │   ├── d3_semantic_utility.py
│   │   ├── d4_faithfulness.py
│   │   └── d5_llm_aware.py
│   ├── predictor/
│   │   ├── mpup_predictor.py   # XGBoost / MLP regressor
│   │   └── calibration.py      # Linear adapter α, β (few-shot)
│   ├── pipeline/
│   │   └── mpup_pipeline.py    # End-to-end rank → top-m → RAG prompt
│   └── evaluation/
│       └── metrics.py          # Spearman ρ, NDCG@k, MRR, EM, F1
├── experiments/
│   ├── ablation.py             # Feature ablation study D1-D5
│   └── evaluate_baselines.py   # Compare MPUP vs all baselines
├── scripts/
│   ├── label_utility.py        # Step 1: label u_i_true
│   ├── retrieve_and_score.py   # Step 2: add BM25 scores
│   ├── extract_features.py     # Step 3: build feature matrix
│   └── train.py                # Step 4: train predictor
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                    # JSONL output dari label + score
│   └── processed/              # features.npy, labels.npy
├── checkpoints/                # Trained predictor
├── results/                    # Evaluation JSON results
├── Makefile
└── requirements.txt
```
