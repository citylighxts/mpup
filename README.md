# MPUP — Meta-Probing Utility Predictor

Predict RAG document utility scores **without fine-tuning the target LLM**.

MPUP scores retrieved passages by combining five feature groups (D1–D5) and training a lightweight predictor, so the target model stays frozen during deployment.

---

## How it works

```
Query + Passages
     │
     ├─ BM25 (÷ best score of the query) + dense cosine ─ D1 (4-dim)
     ├─ Document Quality (length, readability, cred.) ─── D2 (3-dim)
     ├─ Entity Overlap + HyDE Answerability ───────────── D3 (2-dim)
     ├─ NLI Faithfulness (DeBERTa) ─────────────────────── D4 (2-dim)
     └─ Logit Probing ΔH = H_base − H_ctx ─────────────── D5 (3-dim)
              │
        x_i ∈ ℝ¹⁴
              │
        GBR / Ridge / MLP predictor (optionally query-centred)
              │
        û_i (utility score) ──► rank desc ──► top-m RAG prompt
```

Ground-truth utility labels (Eq. 1 of the paper):

```
u_i_true = clip(Acc(with p_i) − Acc(no ctx), −1, 1)
Acc      = 0.5 × relaxed-EM + 0.5 × token-F1      relaxed-EM: a gold answer appears as a token span
```

Relaxed EM, because instruction-tuned models answer in sentences and strict EM would score almost
every answer 0, collapsing the labels.

---

## Feature groups

| Group | Dims | Signal |
|-------|------|--------|
| D1 | 4 | BM25 score divided by the query's best score, dense cosine similarity (e5), rank, score gap |
| D2 | 3 | Token length, Flesch-Kincaid grade, source credibility (**constant 0.5 in every dataset so far**: it carries no information) |
| D3 | 2 | Query entity overlap (spaCy), HyDE answerability (lexical overlap vs. pseudo-answer) |
| D4 | 2 | Entailment prob, contradiction score — NLI cross-encoder, premise=passage / hypothesis=HyDE pseudo-answer |
| D5 | 3 | Base perplexity (= exp of the next-token entropy), context perplexity, ΔH = H_base − H_ctx |

The HyDE pseudo-answer is generated once per query and shared by D3 and D4. One LLM `M`
(Llama-3.2-3B-Instruct 4-bit on MLX; Qwen2.5-7B-Instruct 4-bit on CUDA) writes it, does the logit
probing and labels the utility, so it is loaded once.

---

## Project structure

```
src/
  data/           — benchmarks.py (the 12 datasets), mlx_backend.py / llm_backend.py (batched LLM),
                    utility_labeler.py (Eq. 1), loaders.py (legacy loaders)
  retrieval/      — stream_bm25.py (exact BM25 over the 21M-passage Wikipedia corpus without an index),
                    BM25Retriever, DenseRetriever, candidate_scores (D1 normalisation)
  features/       — D1–D5 extractors + concatenate.py, hyde.py
  probing/        — LogitProber, QuantizedProber, MLXProber, MockProber
  predictor/      — MPUPPredictor (GBR/Ridge/MLP, optional query-centring), LinearCalibrator
  evaluation/     — EM, F1, Spearman ρ, NDCG@k, MRR; protocol.py (query splits, LODO, label guard)
  pipeline/       — MPUPPipeline (rank + RAG prompt builder)
experiments/
  prepare_data.py          — stage 1: questions, top-5 BM25 passages, corpus statistics
  extract_multidataset.py  — stage 2: Eq. 1 labels + the 14 features, per dataset, resumable
  run_multidataset.py      — stage 3: pooled CV, nested leave-one-dataset-out, proxy transfer
  nested_selection.py      — predictor selection without optimism (nested by dataset)
  cv_report.py             — repeated query-grouped CV and leave-one-dataset-out reports
  run_ragbench_benchmark.py, evaluate_baselines.py, ablation.py — legacy single-dataset scripts
tests/            — 204 tests
configs/
  default.yaml
results/
  multidataset.{md,json}   — the multi-dataset results below
```

---

## Datasets

Twelve datasets, 11 with gold answers (utility of Eq. 1) and RAGBench with sentence annotations
(a different label, never pooled with the others). 100 questions are sampled per dataset with a
fixed seed. Sources were checked against the originals: FlashRAG's HotpotQA context is the
*fullwiki* retrieval, not the distractor set (71.8% of questions lose a gold paragraph), and its
MuSiQue drops the paragraphs, so those two come from the original Hub repos.

| Dataset | Source | Candidates | Filter |
|---|---|---|---|
| NQ-open, TriviaQA, WebQuestions, EntityQuestions, PopQA, SQuAD, AmbigQA, Bamboogle | FlashRAG files (EntityQuestions: authors' zip) | top-5 BM25 over Wikipedia | PopQA: drop duplicated question text |
| HotpotQA (distractor) | `hotpotqa/hotpot_qa` | the 10 provided paragraphs | drop yes/no |
| 2WikiMultiHopQA | FlashRAG `2wikimultihopqa` | the 10 provided paragraphs | drop yes/no |
| MuSiQue | `dgslibisey/MuSiQue` | the ≤20 provided paragraphs | answerable only |
| RAGBench-HotpotQA | `galileo-ai/ragbench`, config `hotpotqa` | the 4 provided documents | proxy label (Eq. 5) |

The Wikipedia corpus is FlashRAG's `wiki18_100w` (21,015,324 passages of 100 words, the DPR Dec-2018
dump), 5.1 GB; sampled-question answers occur in it for 94–99% of questions, but for PopQA's rarest
entities (popularity < 100) the subject's article is in it only 36% of the time.

---

## Results — 12 datasets (Apple Silicon, MLX Llama-3.2-3B-Instruct 4-bit)

Full tables, CIs and per-dataset rows: [`results/multidataset.md`](results/multidataset.md). All
numbers are **within-query Spearman ρ** (the quantity a reranker controls), averaged over queries,
95% bootstrap interval over queries. 719 queries have a rankable passage set.

**Headline: nested leave-one-dataset-out.** For each held-out dataset, the predictor
configuration is chosen using only the other datasets, refit on them, and scored on the held-out one.

| Method | macro ρ | top-1 | NDCG@10 |
|---|---|---|---|
| **MPUP (nested selection)** | **0.286** | 0.54 | 0.79 |
| MPUP, configuration stated in the paper (raw features, 200 trees of depth 5) | 0.211 | 0.49 | 0.76 |
| Dense similarity alone (e5 cosine) | 0.256 | 0.52 | 0.78 |
| BM25 alone | 0.110 | 0.41 | 0.72 |

Paired over all 719 held-out queries: MPUP − BM25 **+0.16 [+0.12, +0.20]**, MPUP − paper
configuration **+0.07 [+0.04, +0.10]**, MPUP − dense **+0.02 [+0.00, +0.05]**.

What this does and does not show:

- MPUP clearly beats the relevance signal (BM25) and the configuration as first specified.
- **Against a single dense-similarity feature it is only marginally ahead** (the interval touches
  zero) and is not significantly ahead on any dataset except NQ (MuSiQue's interval touches zero). The paper-stated
  predictor (raw features, 200 trees of depth 5) is *worse* than dense similarity alone
  (−0.04 [−0.07, −0.01]): it overfits the noisy per-query label offsets.
- The two changes that made the difference were **training on within-query deviations**
  (`center_by_query`) and **lower capacity**; the selection picked a query-centred model for all
  11 held-out datasets (shallow GBR 6×, ridge 5×).
- The selection is part of the method, so it is evaluated nested: choosing the best configuration
  on the held-out dataset itself would have claimed ρ 0.314, **0.028 higher** than the honest 0.286.
- Feature groups (selected configuration, paired Δρ when removed): D1 +0.05 [+0.02, +0.07],
  D3 +0.02 [+0.00, +0.03], D5 +0.01 [−0.00, +0.03], D2 and D4 ≈ 0. Removing the document-length
  group changes nothing here: the large D2 effect in the older RAGBench results below was a
  property of that proxy label (it correlates −0.41 with length), not of utility.
- **Proxy transfer** (train on RAGBench annotation labels, no labels from the target): macro ρ
  0.194 with the paper configuration, above BM25 (0.110) and below dense similarity (0.256).
- Single run, one LLM (3B), 100 questions per dataset, one seed: intervals reflect the questions,
  not model or sampling variance.

The deployable predictor is the configuration selected on all 11 datasets
(`checkpoints/mpup_pooled`, not tracked). Its scores are *within-query* (better or worse than the
average candidate), so `utility_threshold` then drops below-average candidates.

### Reproduce

```bash
pip install -e ".[dev]"
pip install mlx mlx-lm                  # Apple Silicon; on CUDA use QuantizedLLM (bitsandbytes)
python -m spacy download en_core_web_sm

# 1. questions + top-5 BM25 over Wikipedia (needs wiki18_100w.zip, 5.1 GB, from FlashRAG_datasets)
python -m experiments.prepare_data --corpus path/to/wiki18_100w.zip
# 2. Eq. 1 labels + D1–D5 features (≈1.5 h on an M5; resumable per dataset)
python -m experiments.extract_multidataset
# 3. pooled CV, nested leave-one-dataset-out, proxy transfer -> results/multidataset.{md,json}
python -m experiments.run_multidataset --repeats 5
pytest
```

---

## Earlier results — RAGBench HotpotQA, one CUDA run (superseded)

Kept for the record. 100 questions, 400 pairs, Qwen2.5-7B-Instruct 4-bit on an 8 GB CUDA card,
RAGBench `all_utilized_sentence_keys` proxy label, one 80/20 split by query. These numbers use a
label that correlates −0.41 with passage length and a single split, and were not reproducible on
Apple Silicon (ρ 0.515 on the same split); read them with the audit below.

| Method | Spearman ρ | NDCG@10 |
|--------|-----------|---------|
| No Retrieval | 0.000 | 0.251 |
| Full Unranked | 0.000 | 0.251 |
| Logit-probe only (D5) | 0.038 | 0.280 |
| MPUP Zero-Shot (D1–D5) | 0.577 | 0.764 |

Leave-one-group-out (retrained per configuration): drop D2 0.439, D1 0.514, D4 0.521, D3 0.525,
D5 0.559. Ablations are retrained; zero-masking columns on a predictor trained with all features
returns arbitrary values (0.452 for a D5-only quantity that honest retraining puts at 0.038).

### Audit: re-run on Apple Silicon (2026-10)

| Quantity | Result |
|----------|--------|
| Benchmark split (seed 42), MPUP zero-shot ρ | 0.515 (above: 0.577) |
| MPUP ρ over 20 random query splits | 0.554 ± 0.089 (min 0.388, max 0.688) |
| "Shortest document first" baseline, same 20 splits | 0.406 ± 0.096 |
| Paired leave-one-group-out Δρ over 30 splits | D2 −0.126, D1 −0.084, D4 −0.022, D5 −0.017, D3 −0.016 |
| Same benchmark with random mock D5 + rule-based HyDE | ρ ≈ 0.48–0.52 |
| Linear calibrator, k=5: slope α<0 | 25 of 200 draws; mean ρ 0.368 vs 0.515 zero-shot (worst −0.515) |

The linear calibrator now falls back to a pure shift when the fitted slope is not positive, so it
can no longer invert a ranking.

The claim that a row split "leaks and inflates ρ by ~0.05" does **not** hold: the `--split row`
option cuts the file at 80% of its rows, and since a question's passages are contiguous it split
zero questions (on the cached run, ρ 0.696 against 0.570 ± 0.104 for 40 random question splits,
1.2 sd). A *random* row split would leak; this one does not.

---

## Evaluation protocol

`src/evaluation/protocol.py`, `experiments/cv_report.py` and `experiments/nested_selection.py`:

1. **Split by query.** `group_splits` keeps all passages of a query on one side; results are
   repeated K-fold over queries, not one 80/20 split (which moves ρ by about ±0.09).
2. **Within-query metrics with intervals.** ρ, NDCG@10 and top-1 are computed per query and
   averaged, with a bootstrap interval over queries, next to BM25, dense and length baselines and
   the **paired** difference to each. Pooled Spearman is only a reference.
3. **Transfer across datasets** is tested with leave-one-dataset-out.
4. **One label definition per pool.** `pool_tables` refuses to mix the RAGBench sentence-proxy
   with the Eq. (1) utility.
5. **Nested selection.** Anything chosen from data (hyperparameters, query-centring, model class)
   is chosen inside the training datasets only (`nested_lodo`); a test checks that scrambling the
   held-out dataset's labels cannot change the choice made for it. On the earlier single-dataset
   run the first version of this question was "is nested CV needed with fixed hyperparameters?"
   (no); once the audit showed the fixed configuration underperforms and the configuration was
   selected, it became necessary, and the oracle-vs-nested gap above (0.028) is the measured cost
   of skipping it.

---

## Baselines

| Method | Description |
|--------|-------------|
| BM25 | Rank by BM25 score (classic relevance) |
| Dense | Rank by e5 cosine similarity |
| Shortest-first | Rank by passage length (checks for a length artefact) |
| MPUP, paper configuration | Raw features, 200 trees of depth 5 (no selection) |
| MPUP, nested selection | Configuration selected by nested leave-one-dataset-out |
| Proxy transfer | Paper configuration trained on RAGBench proxy labels only |

---

## Python API

```python
from src.data.mlx_backend import MLXLLM
from src.features.hyde import CUDAHyDEGenerator     # works with any generate(prompts) LLM
from src.pipeline.mpup_pipeline import MPUPPipeline
from src.predictor.mpup_predictor import MPUPPredictor
from src.probing.logit_probe import MLXProber       # Apple Silicon — no GPU
from src.retrieval.dense_retriever import DenseRetriever

llm = MLXLLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
pipeline = MPUPPipeline(
    predictor=MPUPPredictor.load("checkpoints/mpup_pooled"),
    prober=MLXProber(llm=llm),
    hyde_generator=CUDAHyDEGenerator(llm=llm),
    dense_retriever=DenseRetriever("intfloat/multilingual-e5-large"),   # D1 dense feature
    top_m=5,
)

ranked = pipeline.rank(query, passages)   # passages: RetrievedPassage with BM25 already ÷ best score
prompt = pipeline.build_rag_prompt(query, ranked)
```

`rank` treats the passages as the candidates of **one** query. Features from `pipeline.rank` equal
the training matrix exactly (checked on five queries from two datasets: max difference 0).

---

## Requirements

- Python ≥ 3.11
- `sentence-transformers`, `rank-bm25`, `transformers`, `accelerate`, `torch`
- `scikit-learn`, `spacy`, `textstat`, `datasets`
- `mlx`, `mlx-lm` (Apple Silicon — labelling, D5 logit probing, HyDE); `bitsandbytes` on CUDA

Known limits: `source_credibility` is constant; one 3B LLM and one seed; 100 questions per
dataset; the Natural Questions *full* loader in `loaders.py` still yields every long-answer
candidate (median 66 per question) and the legacy TriviaQA loader whole pages — the 12-dataset
pipeline above does not use them.
