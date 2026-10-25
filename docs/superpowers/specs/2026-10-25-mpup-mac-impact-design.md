# MPUP on Apple Silicon: raising the effect size over dense similarity

Date: 2026-10-25. Status: approved by the author on 2026-10-25, including the success criteria and cut order below.

## Intent and why

The current paper (`paper/mpup.tex`) reports that MPUP ranks passages by LLM-specific utility slightly better
than dense similarity: pooled paired gain +0.04 in within-query Spearman rho (0.333 vs 0.289), p = 0.001, but a
standardised effect of only d_z = 0.12, and no single dataset is significant after Holm correction. Most of the
signal comes from retriever features (D1); the LLM features add +0.01 to +0.02; D2 and D4 add nothing. Answer
accuracy of the selected passages was never measured.

The author wants MPUP to be more impactful, meaning a larger effect size and a measurable benefit in answer
accuracy, not merely a smaller p-value. The author also wants the whole pipeline to run on their own Mac
(Apple M5, 24 GiB unified memory, MLX 0.32.2) instead of the RTX 6000 GPU, because that hardware matches the
paper's "lightweight, single device" claim and makes the cost numbers measured instead of estimated.

## Goals

1. Find out, before changing the method, how much room exists: answer accuracy of MPUP vs dense vs BM25 vs random
   vs oracle, using the existing `experiments/evaluate_answer_accuracy.py`.
2. Try the changes most likely to raise the effect size, in this order of expected value:
   a. a hidden-state probe (D5b): last-token hidden states of the target LLM for (query, passage), with a small
      head on top, replacing or adding to the entropy-delta features;
   b. a ranking objective (LightGBM `lambdarank` or pairwise) instead of regression on within-query deviations;
   c. dropping D2 and D4 (no measurable gain, NLI is the costliest extractor);
   d. a smoother utility target (change in gold-answer log-likelihood with vs without the passage) as an
      additional training label, evaluated against the existing relaxed-EM/F1 utility;
   e. harder candidate sets (top-20 from the dense retriever, plus the provided distractors) alongside the
      current 4-20 passage sets;
   f. a cross-encoder reranker (e.g. bge-reranker) as a baseline and as an extra feature.
3. Run everything on the Mac with MLX, and report measured peak memory and selection latency, which the paper
   currently lists as not measured.
4. Update the paper to whatever the results support, including Table I, the Abstract, Limitations and Conclusion.

## Non-goals and anti-goals

- Not a goal: a smaller p-value by adding queries. Query count may grow only to tighten intervals, never as the
  way to make the claim "significant".
- Anti-goal: choosing the final method by looking at the held-out numbers. Every new variant enters the nested
  leave-one-dataset-out selection (`experiments/nested_selection.py`) as a pre-declared grid entry. The second
  question sample (already used once for a frozen check) is not used again for selection. Final confirmation uses
  a newly drawn third sample with a frozen configuration, scored once.
- Anti-goal: hiding a negative result. If no variant beats dense similarity by a clearly larger margin, the paper
  keeps its current honest framing (small gain, mostly retriever signal) and adds the answer-accuracy and Mac
  cost results.
- Not a goal: new datasets, new target LLMs beyond the model below, or fine-tuning any LLM.

## Constraints

- Hardware: one Apple M5 Mac, 24 GiB unified memory. No CUDA.
- No fine-tuning of the target LLM or the retriever (criterion C2 of the paper must stay true).
- The metric, label definition (Eq. 1 with relaxed EM + token F1, greedy, 32 new tokens), splits by query, and the
  nested leave-one-dataset-out protocol stay the same, so numbers remain comparable with the paper.
- Existing RTX 6000 / Qwen2.5-7B bf16 results stay in the paper as a secondary comparison row (the author's
  decision). They are not re-run and not overwritten.

## Decided by the author

- Run on the Mac, not the GPU.
- Keep the RTX bf16 results in the paper as a secondary comparison.

## My calls (check these)

- Primary model: `mlx-community/Qwen2.5-7B-Instruct-4bit` (about 5 GB). Same model family as the RTX run, so the
  comparison with it is meaningful; Llama-3.2-3B 4-bit gave a weaker gain (+0.02) and stays only as the earlier
  MLX run already in the paper.
- Hidden states are cached to disk once per (query, passage) so predictor experiments run on CPU without
  re-running the LLM. Which layer(s) to cache is a decision for the builder; caching the last layer plus one
  middle layer is my default.
- The first deliverable is the answer-accuracy and oracle-gap measurement (step 1), because its outcome decides
  how much effort steps 2a-2f deserve: a small gap to the oracle means the limit is the candidate set and labels,
  a large one means features and objective are worth working on.
- Time: a full extraction run on the M5 is unmeasured; my guess is a few hours. The first run measures it.

- Grid (fixed before any new extraction, so nothing is chosen from held-out numbers): one-at-a-time changes from
  the current selected configuration, namely +D5b probe, lambdarank objective, minus D2/D4, +smooth label, plus
  one combined entry (all changes that were not worse alone). Not a full cross product. Step 1 only decides
  whether to proceed with a step; it never adds or edits grid entries. Cache once, sweep on CPU.
- Samples: the first sample (100 questions per dataset, 390 for RAGBench) is the development and selection set,
  and Step 1 runs on it. The second sample stays a frozen-check set as in the paper. The third sample is
  drawn with a new seed, same size, with questions overlapping earlier samples removed, scored once.
- The Mac run re-labels and re-extracts everything on the same first-sample queries with the 4-bit model. Labels
  therefore differ from the RTX labels; the RTX row is a side-by-side, and the paper says the two runs differ in
  hardware, precision and labels at once.
- Answer accuracy (already fixed by `evaluate_answer_accuracy.py`): top-m with m = 1 and 3 in the labelling
  prompt, scored with the paper's Acc; the answering LLM is the one that produced the labels; "oracle" ranks by
  the true utility; "none" is the no-passage prompt. Kept as is.
- 2e (harder candidates) is an extra evaluation condition with its own labelled extraction, not a change to
  training data. It runs last and only on datasets whose retrieval corpus the repo already covers (see the
  dataset notes: wiki18_100w coverage is 94-99% for NQ, TriviaQA, WebQ, SQuAD, EntityQuestions); if the Mac is
  slow it is the first item cut.
- Proposed (needs author's confirmation, see below): headline endpoint, thresholds and cut order.

## Success criteria and budget (proposed, awaiting the author)

- Primary endpoint: pooled paired gain over dense similarity in within-query rho on the frozen third sample, as
  in the paper. Secondary: answer-accuracy difference vs dense at m = 1 and 3.
- "Worth continuing" after Step 1: oracle accuracy exceeds dense accuracy by at least 3 points at m = 1
  (otherwise the candidate sets leave no room and Step 2 stops at 2e/2d).
- "Clearly larger" means a pooled gain of at least +0.08 rho (about double today's) with an interval that
  excludes zero. Below that, the paper keeps the honest small-gain framing.
- Cut order if time runs out: 2f, then 2e, then the middle-layer cache, then 2d.

## Left to the builder

- Exact layers, head architecture and regularisation for the hidden-state probe.
- LightGBM vs pairwise implementation details, within the declared grid.
- How to batch and cache MLX forward passes to fit 24 GiB.
- How to shape the new paper tables and wording, following the existing style and the conventions already
  applied in `mpup.tex`.

## Done means

- `results/` holds Mac MLX runs of the full pipeline, with measured peak memory and selection latency.
- Answer-accuracy results for MPUP, dense, BM25, random, oracle and no-passage, with paired intervals and tests.
- A nested-selection comparison of the pre-declared variants, and one frozen run on a new question sample.
- `paper/mpup.tex` and `mpup.pdf` updated to match, with the RTX results kept as a secondary row.
