# Journal title options

Target venue: **Information Processing & Management** (Elsevier). IPM favours descriptive,
problem-first titles over slogans, and publishes empirical characterisation work that a
methods-only venue such as SIGIR would reject for lacking novelty.

---

## Scenario A — the UtilityTransfer measurements are attached

The paper then covers both the method and the transferability evidence that justifies sharing
one predictor. Scope: 3 datasets, 7 model configurations, ~30k utility labels.

**Recommended:**

> **Utility-aware passage selection for retrieval-augmented generation without per-model
> training or LLM-derived labels**

States both cost removals in the title, which is the contribution. No slogan, no colon — reads
as an IPM title.

**Alternatives:**

> One predictor for many readers: commodity-hardware utility prediction for retrieval-augmented
> generation

> How transferable is retrieval utility? Evidence for a single shared predictor in
> retrieval-augmented generation

> Cheap utility estimation for retrieval-augmented generation: heterogeneous features, shared
> predictors, and annotation-derived labels

---

## Scenario B — only this repository's results are used

Scope shrinks to RAGBench HotpotQA, 100 questions / 400 pairs, proxy labels. H2 and H3 drop
out; the paper becomes a focused methods-and-measurement contribution. **This is the honest
scope today**, and it is a short-paper or a methods-note rather than a full IPM article.

**Recommended:**

> **Beyond the logit probe: heterogeneous features for retrieval utility prediction under a
> commodity compute budget**

The strongest verifiable result is ρ = 0.577 for the full feature set against ρ = 0.038 for
logit probing alone, so the title should lead with that contrast rather than with
transferability.

**Alternatives:**

> What actually predicts retrieval utility? A retrained leave-one-group-out analysis of
> fourteen features

> Three ways to overstate a retrieval utility predictor, and how to avoid them

The last one is a genuine option: the three methodological corrections — affine calibration
cannot move a rank metric, row splits leak, masked ablations misreport — are verifiable here,
are mistakes currently present in published work, and would make a well-cited short
contribution on their own.

---

## Titles to avoid

| Title shape | Why not |
|---|---|
| "...for **LLM-specific** retrieval utility" | The measurements say utility largely transfers among comparable models. A title asserting specificity contradicts the results. |
| "**The first** method to..." | Perez-Beltrachini & Lapata (2025) already predict passage utility for a target model without fine-tuning it. Priority claims invite a reviewer to look for a counterexample. |
| Anything foregrounding **few-shot calibration** | The affine calibrator provably cannot change a rank metric. Naming it in the title advertises the paper's weakest claim. |
| Anything foregrounding **perplexity gap** | Univariate ρ = 0.005 on the measurement track. It is not a contribution. |
