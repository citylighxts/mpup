# Paper outline — working draft

**Status:** structure proposed, not written. Every number below is measured and reproducible
from this repository; nothing is projected or assumed.

**Recommended venue: IPM over SIGIR.** The work that survived contact with data is a
*measurement study* with a predictive law and practical consequences, not a new method.
SIGIR rewards methods; IPM publishes exactly this kind of empirical characterisation, and
gives the room the negative results need. This reverses the README's original preference and
should be a deliberate decision, not a drift.

---

## The one-sentence claim

> Two LLMs agree about which retrieved passages are useful in proportion to how similar
> their prior knowledge of the question is — not their family, and not their size.

Everything else in the paper is either evidence for that, or a consequence of it.

---

## Why the original plan does not survive

The project began as *UtilityTransfer*: predict LLM-specific retrieval utility for an unseen
LLM without fine-tuning, using a feature predictor plus a k-shot affine calibrator. Three of
its premises failed under measurement, and saying so plainly is part of the contribution.

| Original premise | What we measured |
|---|---|
| Utility is LLM-specific and "non-transferable" | 7–8B models agree at per-query ρ 0.810–0.820 regardless of family; passages that help one model and harm another are **under 1.2%** of pairs |
| A k-shot affine calibrator recovers cross-LLM performance (ρ 0.31 → 0.51) | `α·f + β` preserves order, so it **cannot change Spearman or NDCG at all** — reproduced to six decimals in four independent settings |
| Perplexity-gap and answerability are the key novel features | Answerability is confirmed as the strongest single feature; **perplexity gap is worthless** (univariate ρ = 0.005, permutation importance 0.0005) |

---

## Structure

### 1. Introduction

Frame the question the field has been assuming an answer to. Recent work treats
LLM-specific utility as established and builds methods to estimate it per model. We ask how
large the effect actually is, and find it is small among comparable models, systematic in its
variation, and governed by a single measurable variable.

Contributions:
1. A measurement protocol for LLM-specific utility, including a **reliability ceiling** — the
   agreement of one model with a quantised copy of itself — which no prior work establishes.
2. The **knowledge-gap law** (§5).
3. Three practical consequences: a strong judge's features beat a model's own; cheap
   annotation labels train as well as expensive ones; low-capacity predictors transfer better.
4. Methodological corrections the emerging literature needs (§7).
5. A released corpus of ~30k utility labels across 3 datasets and 7 model configurations.

### 2. Related work

Zhang et al. (2026) on LLM-specific utility — **we contradict its central framing** and must
engage carefully: their claim may hold for the model pairs they study, and our knowledge-gap
law predicts *when* it holds. Tian et al. (2026) on utility prediction — we reproduce their
feature families and find the reader-centric ones near-worthless on our labels.
Perez-Beltrachini & Lapata (2025), Dai et al. (2025) on SePer.

### 3. Measurement protocol

- **Utility**: `u(q,p,M) = correct(M | q,p) − correct(M | q)`.
- **Correctness**: relaxed EM (DPR `has_answer`), with strict EM and token-F1 also stored.
  *Justify this carefully* — strict EM marks instruction-tuned sentence answers wrong, and
  since utility is a difference of correctness, that drives labels to zero. This is a
  measurement-validity point, not a detail.
- **Metric**: per-query Spearman ρ as primary; global ρ reported alongside. Note that top-k
  accuracy is degenerate at k passages per query, so top-1 is used.
- **Reliability ceiling**: Qwen2.5-0.5B in NF4 vs bf16 — same weights, only precision differs
  — agree at **0.8455**. Every agreement number is read against this.
- Datasets: NQ, TriviaQA (both via published DPR retrieval results), RAGBench HotpotQA.
  Models: Qwen2.5 {0.5B, 1.5B, 3B, 7B}, Mistral-7B-v0.3, Llama-3-8B. 500 queries × 5 passages
  per dataset; ~30k labels total.

### 4. RQ1 — How LLM-specific is retrieval utility?

| pair | per-query agreement |
|---|---|
| quantisation control (same weights) | 0.8455 |
| mistral-7b ↔ llama3-8b | 0.820 |
| qwen2.5-7b ↔ llama3-8b | 0.819 |
| qwen2.5-7b ↔ mistral-7b | 0.810 |
| qwen2.5-7b ↔ qwen2.5-0.5b | 0.767 |
| mistral-7b ↔ qwen2.5-0.5b | 0.700 |

Three 7–8B models from three families reach **96–97% of the ceiling**. Direct conflict never
exceeds 1.2%. On TriviaQA agreement is lower (0.555–0.729) — which §5 explains rather than
excuses.

**The framing that lands hardest:** our best predictor reaches ρ ≈ 0.48, while the LLMs agree
with *each other* at 0.70–0.88. Another LLM's labels are a better predictor of a model's
utility than any feature-based model we built.

### 5. RQ2 — What governs it? *(the paper's core)*

Pooling all 27 available LLM pairs across both datasets, with prior-knowledge gaps from
0.012 to 0.518:

| | agreement vs **knowledge gap** | vs **mean knowledge** (control) |
|---|---|---|
| NQ (n=21) | **r = −0.646, p = 0.002** | r = −0.172, p = 0.456 |
| TriviaQA (n=6) | **r = −0.917, p = 0.010** | r = +0.716, p = 0.110 |
| pooled (n=27) | **r = −0.820, p < 0.001** | r = −0.657, p < 0.001 |

The dissociation holds *within* each dataset, so it is not an artefact of the two datasets
sitting at different levels. The pooled control correlation is significant only through the
cross-dataset confound — say so explicitly.

Also report the within-family capability sweep (Qwen2.5 0.5B→7B) and be honest that it looks
like a **threshold** rather than a gradient, at n = 6 pairs and p = 0.097 — suggestive, not
established.

### 6. RQ3 — Consequences for utility prediction

**6.1 Use one strong judge, not per-LLM features.** Features extracted with a judge model beat
each served model's own features, every model, both datasets:

| served | own | best judge | Δ |
|---|---|---|---|
| qwen2.5-0.5b | 0.399 | 0.496 | **+0.097** |
| qwen2.5-3b | 0.433 | 0.526 | **+0.094** |
| qwen2.5-1.5b | 0.427 | 0.495 | +0.068 |
| mistral-7b | 0.429 | 0.473 | +0.044 |

This inverts the premise: per-LLM feature extraction is not merely reducible, it is
*counterproductive*.

**6.2 The bottleneck is features, not labels or models.** Three independent measurements:
oracle-fit ceiling only +0.068 above the honest score; one answerability prompt beats the
other thirteen features combined; learning curves saturate at 50–200 queries.

**6.3 What actually predicts utility.** Answerability (LLM prompt, P(Yes) from logits)
dominates: univariate ρ 0.318/0.297, permutation importance 0.191. Six of fourteen features
contribute nothing measurable. Cutting to six *improves* performance (0.484 → 0.526).
`answer_likelihood` is the instructive trap — high univariate ρ (0.278), zero permutation
importance, entirely redundant with answerability.

**6.4 Lower capacity transfers better.** Ridge beats gradient boosting in every setting and
by 2.3× on a distant target (0.369 vs 0.162). Caveat honestly: boosting's learning curve is
still rising and the gap narrows with data, so the claim is scale-qualified.

### 7. RQ4 — Can utility be labelled cheaply?

Annotation-derived labels (RAGBench `all_utilized_sentence_keys`) against true LLM utility on
identical pairs:

- Correlate at per-query ρ 0.558 / 0.595, against the 0.836 ceiling on the same data — **67–71%
  of what another LLM captures**.
- **Training on them recovers 96% (Mistral) and 110% (Qwen)** of training on true labels.
- Generalise to unannotated datasets at **73–85%** of an in-domain reference.
- But correlate with document length at **−0.413** against −0.12 for true utility, a direct
  consequence of dividing by sentence count. Report this as a caution, not a footnote.

Consequence: the GPU ceases to be the ceiling on dataset size, provided evaluation stays on
true utility.

### 8. Methodological corrections

Short, pointed, and useful to the field.

1. **Affine calibration cannot improve rank metrics.** Prove it, then show ρ unchanged to six
   decimals across four settings, and that at small k it can *invert* the ranking and hurt.
   A feature-space adapter does change ranking (+0.17 ρ at k=32 on a distant target).
2. **Split by query, never by row.** Passages of one query share features; a row split leaks.
   Measured cost: **+0.052 ρ and +0.100 NDCG** of inflation on an otherwise identical matrix.
3. **Ablation baselines must be retrained, not feature-masked.** Zero-masking columns and
   feeding them to a model trained on all features returns arbitrary values — it produced
   0.452 where retraining gives 0.038 for the same quantity.
4. **Quantisation is not a free variable.** NF4 vs bf16 on the same weights changes which
   passages help a model *more than doubling its parameter count does* (0.846 vs 0.877).

### 9. Discussion and limitations

- Three datasets, all English, all short-answer QA. No multi-hop-specific analysis.
- The knowledge-gap law rests on 27 pairs; the threshold reading rests on 6.
- Relaxed EM is a proxy for correctness and will over-credit answers containing a gold span
  incidentally.
- Utility is defined per single passage; real RAG conditions on a set, where interactions
  matter.

### 10. Conclusion

---

## What is still missing — prioritised

1. **More LLM pairs for §5.** This is the headline claim and it has 27 pairs, most from one
   family. Adding 3–4 models spanning a wide capability range (a 14B, a base-vs-instruct pair,
   a non-Qwen small model) would roughly double the pairs and settle the threshold-vs-gradient
   question. **Highest value per GPU hour of anything remaining.**
2. **Confidence intervals.** Almost every number is a point estimate. Bootstrap over queries
   is cheap and would materially strengthen the paper.
3. **A third proper dataset.** RAGBench HotpotQA is used only for the proxy comparison, at 300
   queries. Either promote it to a full dataset or add one (MS MARCO was in the original plan).
4. **§6.4 at full scale.** The Ridge-vs-boosting claim is qualified by data scale; settling it
   is the one good reason to run 2,000 queries.
5. **Human spot-check of labels.** A few hundred manually verified utility labels would let us
   state the label noise floor rather than assume it.

## What is *not* needed

- More features of the kind we already have. Three independent measurements say the ceiling
  is near, and the union of both projects' feature sets performs *worse* than one project's.
- More calibration work. The affine result is settled and the feature-space adapter adds
  little once the base model is well chosen.
- Scaling queries for prediction accuracy. The learning curves are flat past ~200.

## Division of labour (proposal, for both authors to agree)

- The measurement sections (§4, §5) rest on the utility-label corpus and the agreement
  analyses.
- §7 rests on the RAGBench proxy work and is the natural home for the MPUP contribution — it
  is what turns the labelling cost problem from a constraint into a solved sub-problem.
- §8 items 2–3 were found in the MPUP benchmark; item 1 in the UtilityTransfer calibrator.
  Both codebases contributed errors and both contributed fixes, which is worth stating plainly
  in the paper rather than smoothing over.
