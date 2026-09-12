# Introduction

> **Provenance note — read before submitting.**
> Claims marked **[V]** are backed by a result file in this repository
> (`results/ragbench_hotpotqa_cuda.json`, `results/ragbench_hotpotqa_benchmark.json`) or are
> provable from the code. Claims marked **[U]** come from the *UtilityTransfer* measurement
> track and are recorded in `paper/OUTLINE.md` but have **no backing data file in this
> repository**. Every **[U]** claim must have its result file attached before this text goes to
> a journal. Do not submit with an unresolved **[U]**.

---

## Hook

Retrieval-augmented generation works better when passages are selected for their *utility* to
the reading model — the change they cause in answer correctness — than when they are selected
for topical relevance. That much is settled. What is not settled is how a practitioner is
supposed to obtain a utility estimate. Every published route carries a cost that scales with
deployment: distilling a selector from a 32-billion-parameter teacher, retraining the
retriever's embedding space, or running the target model twice over every query–passage pair
to produce labels. On a single 8 GB GPU none of these is available.

This paper asks what utility-aware selection looks like when commodity hardware is the binding
constraint, and reports how much accuracy that constraint actually costs.

## Current knowledge

Retrieval-augmented generation (RAG) conditions a large language model (LLM) on retrieved
passages so it can answer questions beyond its parametric knowledge (Lewis et al., 2020).
Retrieval in such pipelines has conventionally been tuned for topical relevance, yet relevance
only approximates what determines downstream quality.

The field has since formalised utility as an objective in its own right (Zhang et al., 2026).
Evaluation frameworks now separate the positive contribution of helpful passages from the
active harm done by related-but-distracting ones — documents that conventional IR metrics treat
as neutral but which measurably degrade generation (Trappolini et al., 2025).

Utility has been operationalised in several ways. SePer measures it as the reduction in
semantic perplexity once evidence is supplied, and correlates strongly with human judgment
(Dai et al., 2025). A complementary line trains a lightweight neural model to predict passage
usefulness for a target QA model, using the estimate to quantify that model's uncertainty
(Perez-Beltrachini & Lapata, 2025). LLM-generated annotations have been shown to capture
usefulness beyond human relevance labels, with LLM annotations plus 20% of human labels
matching full human annotation (Zhang et al., 2025a).

Deployable selectors follow the same trajectory. Compact students distilled from a Qwen3-32B
teacher make utility-based selection affordable at serving time (Zhang et al., 2025b).
Utility-Aligned Embeddings instead inject utility signal into the retriever through a modulated
InfoNCE objective, avoiding test-time LLM inference at the price of retraining the embedding
space (Sandhu et al., 2026).

Beneath this literature sits a structural assumption: that utility is model-dependent and
non-transferable, so each served LLM needs evidence selected for it specifically (Zhang et al.,
2025c).

## Gap

Three costs separate this literature from a selector a practitioner can actually run, and they
compound.

**Teacher cost.** Distillation needs a large teacher to generate supervision (Zhang et al.,
2025b); embedding alignment needs the retriever retrained (Sandhu et al., 2026). Neither fits a
commodity budget.

**Label cost.** Utility is defined as a difference in model correctness, so labelling means
running the target LLM twice for every query–passage pair. In practice this, not model
capacity, bounds dataset size.

**Per-model cost.** If utility is genuinely non-transferable (Zhang et al., 2025c), both costs
above multiply by the number of deployed models.

The third cost is the one that has never been measured against an interpretable scale. No prior
study reports a *reliability ceiling* — the agreement a model reaches with a near-identical
copy of itself. Without that reference point, an observed cross-model agreement cannot be
called high or low, and the necessity of per-model estimation is assumed rather than shown.

A fourth gap is methodological, and we encountered all of it in our own pipeline before
correcting it. **[V]** Affine calibration of the form `α·f(x) + β` is order-preserving and
therefore cannot change Spearman correlation or NDCG at all, yet it is reported as improving
them; in our runs the k=5 and k=10 calibrated scores are identical to the zero-shot score.
**[V]** Splitting train and test by row rather than by query leaks features shared across one
query's passages. **[V]** Ablating a feature group by zero-masking its columns while reusing a
predictor trained on all features returns values that do not reflect that group's contribution:
the same D5-only quantity reads 0.452 under masking and 0.038 under honest retraining.

## Objective and hypotheses

We ask: **can retrieval utility be predicted accurately enough to rank passages using only
commodity hardware, freely available labels, and a single predictor shared across target LLMs?**

We answer with MPUP (Meta-Probing Utility Predictor), a utility-aware passage selector that
never fine-tunes the target LLM. MPUP scores each candidate with fourteen features across five
families — retriever-centric signals, document-quality statistics, semantic overlap against a
generated hypothetical answer, NLI-based faithfulness, and LLM-aware logit probing — then ranks
by predicted utility to assemble the top-*m* context. Extraction costs one short decode per
query plus one forward pass and one NLI pass per passage, which runs in 4-bit precision on an
8 GB GPU or on Apple Silicon via MLX.

**H1 — a heterogeneous feature set outperforms any single LLM-aware signal. [V]** On RAGBench
HotpotQA under a query-level split, the full set reaches ρ = 0.577 and NDCG@10 = 0.764, against
ρ = 0.038 for the logit-probe features alone and NDCG@10 = 0.251 for unranked context.
Leave-one-group-out retraining confirms every family contributes, and no family exceeds
ρ = 0.362 on its own: the accuracy is in the combination, not in any one signal.

**H2 — one shared predictor suffices across target LLMs. [U]** Against a reliability ceiling of
0.846, three 7–8B models from three families agree on per-query utility at ρ = 0.810–0.820,
reaching 96–97% of that ceiling, with direct conflict below 1.2% of pairs. Features extracted
by one strong judge further outperform each served model's own features, on every model tested
(up to +0.097 ρ). Where k-shot adaptation is still wanted, we use a feature-space ridge adapter
rather than affine calibration, precisely because the latter cannot alter a ranking **[V]**.

**H3 — freely available annotation-derived labels substitute for LLM-derived ones. [U]**
Training on RAGBench sentence-utilisation annotations recovers 96–110% of the performance of
training on true LLM-derived utility labels, and transfers to unannotated datasets at 73–85% of
an in-domain reference. One caveat is reported rather than buried: such labels correlate with
document length at −0.413, against −0.12 for true utility, because they divide by sentence
count.

We also state the limit of the approach plainly. **[U]** Our best feature-based predictor
reaches ρ ≈ 0.48 on true utility labels, while the LLMs agree with one another at ρ = 0.70–0.88.
On this evidence another model's labels remain a better predictor of a model's retrieval utility
than any feature set we were able to build — which bounds what feature-based utility prediction
should be expected to deliver, and is reported here rather than left for a reader to infer.
