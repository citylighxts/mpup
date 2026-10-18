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
(Qwen2.5-7B-Instruct on CUDA, in bfloat16 or 4-bit NF4; Llama-3.2-3B-Instruct 4-bit on MLX for Apple
Silicon) writes it, does the logit probing and labels the utility, so it is loaded once. The
numbers in the paper come from the CUDA run in bfloat16.

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
  evaluate_fresh_questions.py — frozen configuration scored on a second sample of questions
  run_ragbench_benchmark.py, evaluate_baselines.py, ablation.py — legacy single-dataset scripts
tests/            — 206 tests
configs/
  default.yaml
results/
  multidataset_rtx6000_qwen7b.{md,json} — the paper's main run (Qwen2.5-7B, bfloat16, RTX 6000 Ada)
  fresh_questions.{md,json}             — frozen configuration on a second question sample
  hardware_rtx6000.json                 — peak GPU memory and extraction time of that run
  multidataset.{md,json}                — the earlier Apple Silicon run (Llama-3.2-3B, MLX)
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

## Results — 12 datasets (Qwen2.5-7B-Instruct, bfloat16, NVIDIA RTX 6000 Ada)

This is the run reported in the paper. Full tables, CIs and per-dataset rows:
[`results/multidataset_rtx6000_qwen7b.md`](results/multidataset_rtx6000_qwen7b.md). All numbers are
**within-query Spearman ρ** (the quantity a reranker controls), averaged over queries, with a 95%
bootstrap interval over queries. 747 queries have a rankable passage set (11 datasets with the
utility of Eq. 1; RAGBench is only used for the proxy-transfer test).

**Headline: nested leave-one-dataset-out.** For each held-out dataset, the predictor
configuration is chosen using only the other datasets, refit on them, and scored on the held-out one.

| Method | macro ρ | top-1 | NDCG@10 |
|---|---|---|---|
| **MPUP (nested selection)** | **0.333** | 0.56 | 0.80 |
| MPUP, configuration stated in the paper (raw features, 200 trees of depth 5) | 0.227 | 0.50 | 0.77 |
| Dense similarity alone (e5 cosine) | 0.289 | 0.54 | 0.79 |
| BM25 alone | 0.117 | 0.43 | 0.71 |

Paired over all 747 held-out queries: MPUP − BM25 **+0.20 [+0.16, +0.24]**, MPUP − paper
configuration **+0.10 [+0.06, +0.13]**, MPUP − dense **+0.04 [+0.02, +0.06]**.

What this does and does not show:

- MPUP clearly beats the relevance signal (BM25) and the configuration as first specified.
- **Against a single dense-similarity feature the gain is small.** Per dataset, the interval of
  MPUP − dense excludes zero only for MuSiQue (+0.05) and SQuAD (+0.14). The paper-stated
  predictor (raw features, 200 trees of depth 5) is *worse* than dense similarity alone: in pooled
  cross-validation the paired difference is −0.06 [−0.09, −0.04].
- The two changes that made the difference were **training on within-query deviations**
  (`center_by_query`) and **lower capacity**; the selection picked a query-centred model for all
  11 held-out datasets (shallow GBR 6×, ridge 5×).
- The selection is part of the method, so it is evaluated nested: choosing the best configuration
  on the held-out dataset itself would have claimed ρ 0.357, **0.024 higher** than the honest 0.333.
  The grid was designed after an earlier run on the same questions, so even the nested estimate is
  slightly optimistic.
- **Fresh questions, configuration frozen** (`experiments/evaluate_fresh_questions.py`): a second
  sample of questions (another seed, already-seen ones removed), 693 rankable queries,
  `gbr(100,d2)-centered` fitted on the first sample and not selected again: ρ 0.312 against dense
  0.277, BM25 0.155 and the paper configuration 0.186. Paired over dense: +0.052 [+0.026, +0.079].
  The datasets are the same, so this does not test transfer to new datasets.
- Feature groups (selected configuration, paired Δρ when removed): D1 +0.06 [+0.03, +0.09],
  D5 +0.02 [+0.00, +0.04], D3 +0.01 [+0.00, +0.03], D2 and D4 ≈ 0. Most of the signal is in the
  retriever features. The large D2 effect in the older RAGBench results below was a property of
  that proxy label (it correlates −0.41 with length), not of utility.
- **Proxy transfer** (train on RAGBench annotation labels, no labels from the target): macro ρ
  0.239 with the paper configuration, above BM25 (0.117) on 9 of 11 datasets and below dense
  similarity (0.289).
- Single run per LLM, 100 questions per dataset (390 for RAGBench): intervals reflect the
  questions, not model or sampling variance. Transfer from one source LLM to a different target
  LLM is not tested, and answer accuracy with the selected top-m passages is not measured.

**Cost** (`results/hardware_rtx6000.json`): peak GPU memory 20.8 GiB in bfloat16 and 7.3 GiB in
4-bit NF4, both measured on 40 questions from MuSiQue and HotpotQA (the longest contexts, batch size
32); 22.3 GiB over the whole extraction run of the second question sample (bfloat16). Labelling and
feature extraction for all 12 datasets (9,554 pairs) took about 27 minutes in bfloat16, and that
includes the labelling with the true utility. Selection-time latency, the 4-bit setting end to end
and an 8 GB card were not measured.

### Earlier run — Apple Silicon, MLX Llama-3.2-3B-Instruct 4-bit

Same questions and candidates, 719 rankable queries ([`results/multidataset.md`](results/multidataset.md)).
The ordering of the methods is the same, the values are lower; the two runs differ in model,
quantization and hardware at once.

| Method | macro ρ | top-1 | NDCG@10 |
|---|---|---|---|
| **MPUP (nested selection)** | **0.286** | 0.54 | 0.79 |
| MPUP, paper configuration | 0.211 | 0.49 | 0.76 |
| Dense similarity alone | 0.256 | 0.52 | 0.78 |
| BM25 alone | 0.110 | 0.41 | 0.72 |

Paired: MPUP − BM25 +0.16 [+0.12, +0.20], MPUP − paper configuration +0.07 [+0.04, +0.10],
MPUP − dense +0.02 [+0.00, +0.05]. Oracle selection would have claimed 0.314 (0.028 above nested).
Proxy transfer: 0.194.

The deployable predictor is the configuration selected on all 11 datasets
(`checkpoints/mpup_pooled`, not tracked). Its scores are *within-query* (better or worse than the
average candidate), so no threshold is applied by default (`utility_threshold=None`, as in the
paper); setting one would drop below-average candidates.

### Reproduce

```bash
pip install -e ".[dev]"
pip install mlx mlx-lm                  # Apple Silicon (MLX backend)
pip install bitsandbytes                # CUDA, only needed for --quantization nf4
python -m spacy download en_core_web_sm  # without it D3 falls back to word overlap (a warning is printed)

# 1. questions + top-5 BM25 over Wikipedia (needs wiki18_100w.zip, 5.1 GB, from FlashRAG_datasets)
python -m experiments.prepare_data --corpus path/to/wiki18_100w.zip
# 2. Eq. 1 labels + D1–D5 features (resumable per dataset)
#    paper run, CUDA, Qwen2.5-7B-Instruct in bfloat16 (≈27 min on an RTX 6000 Ada):
python -m experiments.extract_multidataset --backend cuda
#    4-bit weights: add --quantization nf4.  Apple Silicon (≈1.5 h on an M5): omit --backend
# 3. pooled CV, nested leave-one-dataset-out, proxy transfer -> results/multidataset.{md,json}
python -m experiments.run_multidataset --repeats 5
# 4. second question sample (prepare_data --seed 1 --out data/processed/multidataset_seed1, then
#    extract_multidataset --data on it), frozen configuration:
python -m experiments.evaluate_fresh_questions
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
   selected, it became necessary, and the oracle-vs-nested gap above (0.024 on the paper run, 0.028
   on the MLX run) is the measured cost of skipping it.

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

Known limits: `source_credibility` is constant; one source LLM per run and one seed; 100 questions
per dataset (390 for RAGBench); on the CUDA path the probing prompt is cut at 512 tokens from the
right, which by a word-count estimate affects about 0.3% of the pairs (long 2Wiki/HotpotQA paragraphs); the Natural Questions *full* loader in `loaders.py` still yields every long-answer
candidate (median 66 per question) and the legacy TriviaQA loader whole pages — the 12-dataset
pipeline above does not use them.
