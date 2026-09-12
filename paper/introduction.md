# 1. Introduction

## Hook

Selecting retrieved passages by their *utility* to the reading model, rather than by topical
relevance, is now established as the better objective for retrieval-augmented generation. What
is not established is how to obtain a utility estimate cheaply. Every published route carries a
cost that scales with deployment: distilling a selector from a 32B teacher, retraining the
retriever's embedding space, or collecting utility labels by running the target model over
every query–passage pair. Under a commodity hardware budget — a single 8GB GPU, or an Apple
Silicon laptop — none of these is available. This paper asks what a utility-aware selector
looks like when that budget is the binding design constraint, and shows that the resulting
method loses little.

## Current knowledge

Retrieval-augmented generation (RAG) conditions a large language model (LLM) on retrieved
passages so that it can answer questions beyond its parametric knowledge (Lewis et al., 2020).
Retrieval in such pipelines has conventionally been optimised for topical relevance, but
relevance only approximates the quantity that determines downstream quality: *utility*, the
change in answer correctness caused by supplying a passage.

The distinction is now a research programme in its own right (Zhang et al., 2026). Evaluation
frameworks have been reformulated to separate the positive contribution of helpful passages
from the active harm caused by related-but-distracting ones — documents that conventional IR
metrics treat as merely neutral, but which measurably degrade generation (Trappolini et al.,
2025).

Several operationalisations of utility have followed. SePer measures it as the reduction in
semantic perplexity when evidence is supplied, correlating strongly with human judgment (Dai
et al., 2025). A complementary line trains a lightweight neural model to predict passage
usefulness for a target QA model, using that estimate to quantify the model's uncertainty
(Perez-Beltrachini & Lapata, 2025). LLM-generated annotations have been shown to capture
usefulness beyond human relevance labels, with the combination of LLM annotations and 20% of
human labels matching full human annotation (Zhang et al., 2025a).

Deployable selectors have followed the same trajectory. Compact student models distilled from a
Qwen3-32B teacher make utility-based selection affordable at serving time, reducing
computational cost while improving answer quality (Zhang et al., 2025b). Utility-Aligned
Embeddings instead inject utility signal directly into the retriever through a modulated
InfoNCE objective, avoiding test-time LLM inference at the price of retraining the embedding
space (Sandhu et al., 2026).

Running beneath this literature is a structural assumption: that utility is *model-dependent
and non-transferable*, so each served LLM requires evidence selected for it specifically
(Zhang et al., 2025c). That work benchmarks four LLMs across three QA datasets, finds each
performs best with its own utilitarian evidence, and concludes that generator-tailored
selection is necessary.

## Gap

Three costs stand between this literature and a deployable selector, and they compound.

**Teacher cost.** Distillation-based selectors require a large teacher to generate supervision
(Zhang et al., 2025b); embedding-alignment approaches require retraining the retriever (Sandhu
et al., 2026). Neither fits a single commodity GPU.

**Label cost.** Utility labels are defined by a difference in model correctness, so obtaining
them means running the target LLM twice over every query–passage pair. This, not model
capacity, is what bounds dataset size in practice.

**Per-model cost.** If utility is genuinely non-transferable (Zhang et al., 2025c), both costs
above multiply by the number of deployed models. This is the assumption that makes the whole
programme expensive — and it has never been measured against a scale on which its magnitude
can be judged. No prior study reports a *reliability ceiling*: the agreement a model reaches
with a near-identical copy of itself. Without that reference point, an observed cross-model
agreement is uninterpretable, and the necessity of per-model estimation is assumed rather than
demonstrated.

A fourth gap is methodological. Three practices inflate reported results in this area, and we
encountered all three in our own pipeline before correcting them: affine calibration of the
form `α·f(x) + β` is order-preserving and therefore provably cannot change Spearman correlation
or NDCG, though it is reported as improving them; splitting train and test by row rather than by
query leaks features shared across a query's passages, worth +0.052 ρ of inflation on an
otherwise identical matrix; and ablating a feature group by zero-masking its columns while
reusing a predictor trained on all features returns arbitrary values — on our data it yielded
ρ = 0.452 where honest retraining gives 0.038 for the same quantity.

## Objective and hypothesis

This paper asks: **can retrieval utility be predicted accurately enough to rank passages using
only commodity hardware, freely available labels, and a single predictor shared across target
LLMs?**

We answer it with MPUP (Meta-Probing Utility Predictor), a utility-aware passage selector that
requires no fine-tuning of the target LLM. MPUP scores each candidate passage with fourteen
features spanning five families — retriever-centric signals, document-quality statistics,
semantic overlap against a generated hypothetical answer, NLI-based faithfulness, and
LLM-aware logit probing — and ranks passages by predicted utility to assemble the top-*m*
context. Feature extraction costs one short decode per query plus one forward pass and one NLI
pass per passage, which runs in 4-bit precision on an 8GB GPU or on Apple Silicon via MLX.

Three hypotheses are tested.

**H1: a compact heterogeneous feature set outperforms any single LLM-aware signal.** On
RAGBench HotpotQA under a query-level split, the full feature set reaches ρ = 0.577 and
NDCG@10 = 0.764, against ρ = 0.038 for the logit-probe features alone and NDCG@10 = 0.251 for
unranked context. Leave-one-group-out retraining confirms each family contributes, and a
six-feature subset *improves* on all fourteen (0.484 → 0.526), locating the accuracy in a small
number of signals rather than in the ensemble's size.

**H2: one predictor suffices across target LLMs.** Against a reliability ceiling of 0.846 —
one model versus a quantised copy of itself — three 7–8B models from three families agree on
per-query utility at ρ = 0.810–0.820, reaching 96–97% of that ceiling, with direct conflict
below 1.2% of pairs. Features extracted by a single strong judge model further outperform each
served model's own features, for every model tested and on both datasets (up to +0.097 ρ).
Per-model feature extraction is therefore not merely avoidable but counterproductive, and MPUP
is designed around one shared predictor. Where adaptation is still wanted, we use a
feature-space ridge adapter rather than affine calibration, which gains +0.17 ρ at k = 32 on a
distant target precisely because it is not order-preserving.

**H3: freely available annotation-derived labels substitute for expensive LLM-derived ones.**
Training on RAGBench sentence-utilisation annotations recovers 96–110% of the performance of
training on true LLM-derived utility labels, and transfers to unannotated datasets at 73–85% of
an in-domain reference. This removes the label-cost bottleneck, provided evaluation remains on
true utility — and we report the one caveat honestly, that such labels correlate with document
length at −0.413 against −0.12 for true utility.
