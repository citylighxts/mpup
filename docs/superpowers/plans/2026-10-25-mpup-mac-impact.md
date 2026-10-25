# MPUP on Apple Silicon: Effect-Size Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the whole MPUP pipeline on the Mac (MLX, Qwen2.5-7B 4-bit), measure how much room exists for a better ranker, try the pre-declared method changes under the existing nested protocol, confirm once on a fresh question sample, and update the paper to whatever the numbers show.

**Architecture:** Extraction is done once and cached (14 features, two layers of last-token hidden states, optional gold-answer log-likelihood label); every predictor variant is a `Config` in a new fixed grid evaluated by the existing `nested_lodo`, so selection never sees held-out data. A frozen configuration is scored once on a third question sample.

**Tech Stack:** Python 3.14, mlx 0.32 / mlx_lm 0.31, numpy, scikit-learn, xgboost 3.4 (`XGBRanker`; lightgbm is not installed and not added), pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-mpup-mac-impact-design.md` (approved 2026-10-25, with its success criteria and cut order).

## Global Constraints

- Hardware is the Apple M5 with 24 GiB; no CUDA. Target LLM is `mlx-community/Qwen2.5-7B-Instruct-4bit` (already in the HF cache).
- No fine-tuning of the target LLM or the retriever.
- Metric (within-query Spearman), label definition (Eq. 1, relaxed EM + token F1, greedy, 32 new tokens), query-level splits and nested leave-one-dataset-out stay as they are.
- RTX 6000 results (`results/multidataset_rtx6000_qwen7b.*`, `results/fresh_questions.*`, `results/significance_*`) are never overwritten; new Mac outputs go under `results/mac_qwen7b/`. The paper keeps the RTX numbers as a secondary row.
- The grid is fixed before the extraction that feeds it. Step-1 results may stop a task, never add or edit grid entries.
- The second question sample is not used for selection. The third sample (new seed, 100 per dataset, 390 for RAGBench, questions seen in samples 1 and 2 removed) is scored once with a frozen configuration.
- Success criteria: primary = pooled paired gain over dense similarity in rho on the third sample; "clearly larger" = at least +0.08 with a 95% interval excluding zero. Step-1 gate = oracle accuracy minus dense accuracy at m = 1 of at least 0.03; if it fails, only Task 4, the `base+smooth` entry of Task 5/6 and Task 9a run before Task 10.
- Cut order if time runs out: Task 9b, then 9a, then the middle-layer cache, then Task 4.
- Commit messages carry no `Co-Authored-By` line and PR text no "Generated with" line (the user's global rule overrides any tooling default).
- Paper prose follows the conventions already applied in `paper/mpup.tex`: no sentence opens with This, These, Such, a personal pronoun, A or An; p-values come from the interval-based paired t statistic as already described.

## Review Focus

- Memory: 7B-4bit plus 512-token prompts at batch 8 can exceed 24 GiB; extraction must resume after a crash without redoing finished datasets and must record the batch size used (Task 2).
- Hidden states must come from exactly the tokens the entropy features use (BOS kept, tail kept when a prompt exceeds 512 tokens), otherwise D5 and the probe describe different inputs (Task 1).
- Queries with a single candidate or identical utilities must not crash the ranker or the probe's cross-fitting (Task 5).
- The probe's out-of-fold scores must never see the query they score; a leak here inflates rho (Task 5).
- Hidden rows, loglik rows and feature rows must align by (query_id, passage_id) even after an interrupted run; a mismatch silently scrambles labels (Tasks 2, 4, 6).

---

### Task 0: Branch and record the design

**Files:**
- Create: none (adds the two docs already written)

- [ ] **Step 1:** `git switch -c mac-mpup-impact` from `paper-results-update`; `git add docs/superpowers/specs/2026-10-25-mpup-mac-impact-design.md docs/superpowers/plans/2026-10-25-mpup-mac-impact.md`; `git commit -m "docs: design and plan for the Mac effect-size experiments"`. Leave `paper/mpup_pitch.pptx` untracked.

---

### Task 1: Hidden-state probe on MLX

**Files:**
- Modify: `src/probing/logit_probe.py` (`MLXProber`)
- Test: `tests/probing/test_logit_probe.py`

**Interfaces:**
- Produces: `MLXProber.hidden_batch(self, query: str, passages: list[str], layers: tuple[int, ...] = (14, -1), max_length: int = 512) -> np.ndarray` of shape `(len(passages), len(layers), hidden_size)`, float32. Layer index `i >= 0` is the residual stream after transformer block `i`; `-1` is the final, post-norm state (what `model.model(x)` returns).

- [ ] **Step 1: Write the failing tests** with a tiny random `mlx_lm.models.qwen2.Model` (2 layers, hidden 16) and a fake tokenizer whose `encode` maps characters to ids: `test_hidden_batch_shape` (3 passages, `layers=(0, -1)` gives `(3, 2, 16)`); `test_hidden_last_layer_equals_model_output` (`layers=(-1,)` row equals `model.model(mx.array([tokens]))[0, -1]` for the same prompt `f"Context: {p}\nQuery: {q}"`, `np.allclose`); `test_hidden_uses_same_truncation_as_entropy` (a prompt longer than `max_length` yields tokens `[bos] + tail`, checked by capturing the token list both paths feed the model).
- [ ] **Step 2:** `python3 -m pytest tests/probing/test_logit_probe.py -v` — new tests FAIL (`hidden_batch` missing).
- [ ] **Step 3:** Implement `hidden_batch`; factor the tokenize-and-truncate code in `_entropy` into one private `_tokens(prompt, max_length)` used by both. Run the layers manually (`embed_tokens`, loop over `model.model.layers` with `create_attention_mask`, final `norm`) and take position `-1`.
- [ ] **Step 4:** Same pytest command — all PASS, including the existing probe tests.
- [ ] **Step 5:** Commit `src/probing/logit_probe.py tests/probing/test_logit_probe.py` — "probing: expose last-token hidden states of the MLX model".

---

### Task 2: Extraction caches hidden states and records Mac run info

**Files:**
- Modify: `experiments/extract_multidataset.py`
- Test: `tests/evaluation/test_extract_multidataset.py`

**Interfaces:**
- Consumes: `MLXProber.hidden_batch` (Task 1).
- Produces: `item_hidden(item: dict, prober, layers: tuple[int, ...]) -> np.ndarray` `(n_passages, n_layers, d)`; `process_dataset(..., hidden_layers: tuple[int, ...] | None = None)`; file `hidden/<dataset>.npz` with `H` (float16, `(n_pairs, n_layers, d)`), `query_ids`, `passage_ids` in exactly the row order of `features/<dataset>.npz`; CLI `--hidden-layers 14 -1`; `run_info_mlx.json` with `model`, `peak_gib` (from `mx.get_peak_memory()`, falling back to `mx.metal.get_peak_memory()`), `batch_size`, `datasets`, `limit`, `seconds`.

- [ ] **Step 1: Write the failing tests** reusing the file's fake LLM/prober style, the fake prober gaining `hidden_batch` that returns a deterministic array per (query, passage): `test_hidden_file_aligned_with_features` (ids in `hidden/x.npz` equal the ids in `features/x.npz`, `H.shape == (n_pairs, 2, d)`); `test_no_hidden_file_without_flag`; `test_resume_recomputes_dataset_missing_hidden` (features exist, hidden requested but absent → dataset is processed again).
- [ ] **Step 2:** `python3 -m pytest tests/evaluation/test_extract_multidataset.py -v` — FAIL.
- [ ] **Step 3:** Implement. The skip rule becomes "features exist and, when `--hidden-layers` is given, hidden exists". Write `run_info_mlx.json` in the `mlx` branch of `main`, like the existing CUDA one.
- [ ] **Step 4:** Same command — PASS; then `python3 -m pytest tests -q` — whole suite PASS.
- [ ] **Step 5:** Commit — "extract: cache hidden states and write MLX run info".

---

### Task 3: Mac run on the first sample and the Step-1 gate

**Files:**
- Create: `data/processed/mac_qwen7b/` (git-ignored), `results/mac_qwen7b/multidataset.{json,md}`, `results/mac_qwen7b/answer_accuracy.{json,md}`, `results/mac_qwen7b/step1.md`

- [ ] **Step 1:** `mkdir -p data/processed/mac_qwen7b && cp data/processed/multidataset/items.json data/processed/multidataset/stats.json data/processed/mac_qwen7b/` (1,490 items; the 5 GB wiki corpus is not needed).
- [ ] **Step 2: Smoke run.** `python3 -m experiments.extract_multidataset --data data/processed/mac_qwen7b --backend mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit --hidden-layers 14 -1 --datasets nq --limit 10` — completes, prints peak memory below 22 GiB. If it runs out of memory, repeat with `--batch-size 4` and use that value from here on.
- [ ] **Step 3: Full run** (long; run in the background, note wall-clock time): same command without `--datasets/--limit`. Expected: 12 `features/*.npz` and 12 `hidden/*.npz`, `run_info_mlx.json`.
- [ ] **Step 4:** `python3 -m experiments.run_multidataset --data data/processed/mac_qwen7b --out results/mac_qwen7b/multidataset`, then `python3 -m experiments.evaluate_answer_accuracy --data data/processed/mac_qwen7b --results results/mac_qwen7b/multidataset.json --backend mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit --out results/mac_qwen7b/answer_accuracy`.
- [ ] **Step 5: Gate.** In `answer_accuracy.json` read `all.vs_dense["oracle@1"].mean_diff`. Write `results/mac_qwen7b/step1.md` with the table, extraction wall-clock time, peak memory, and the verdict "gate passed" (>= 0.03) or "gate failed: run only Task 4, the base+smooth entry and Task 9a". Commit results and `step1.md` — "results: Mac MLX run on the first sample and the Step-1 gate".

---

### Task 4: Smooth label (gold-answer log-likelihood)

**Files:**
- Modify: `src/data/mlx_backend.py`
- Create: `experiments/extract_loglik.py`
- Test: `tests/data/test_mlx_backend.py`, `tests/evaluation/test_extract_loglik.py`

**Interfaces:**
- Produces: `MLXLLM.answer_logprobs(self, prompts: list[str], answers: list[str]) -> list[float]` (mean token log-probability of `answer` after the chat-formatted prompt); `loglik/<dataset>.npz` with `u_smooth` (= logprob with passage − logprob without, first gold answer, clipped to [−5, 5]), `query_ids`, `passage_ids`, same row order as features. CLI `python3 -m experiments.extract_loglik --data DIR --model M`.

- [ ] **Step 1: Write the failing tests.** Backend: with the fake `_loaded` pattern of `test_mlx_backend.py`, `test_answer_logprobs_prefers_likely_answer` (a model whose logits favour token A gives a higher value for answer "A" than "B"). Script: `test_loglik_rows_align_with_features_and_are_clipped` using a fake LLM returning fixed values.
- [ ] **Step 2:** `python3 -m pytest tests/data/test_mlx_backend.py tests/evaluation/test_extract_loglik.py -v` — FAIL.
- [ ] **Step 3:** Implement both. Prompts are the Eq. (1) prompts with and without the passage (`NO_CONTEXT_PROMPT` for the latter); queries without gold answers (RAGBench) are skipped.
- [ ] **Step 4:** Same command — PASS.
- [ ] **Step 5:** Run it on `data/processed/mac_qwen7b` (background, note time), then commit code and tests — "labels: gold-answer log-likelihood as a smooth training label".

---

### Task 5: Predictor variants and the fixed grid

**Files:**
- Modify: `src/features/concatenate.py`, `src/predictor/mpup_predictor.py`, `experiments/cv_report.py` (`make_fit_predict`), `experiments/nested_selection.py`
- Create: `src/predictor/hidden_probe.py`
- Test: `tests/features/test_concatenate.py`, `tests/predictor/test_hidden_probe.py`, `tests/predictor/test_mpup_predictor.py`, `tests/evaluation/test_nested_selection.py`

**Interfaces:**
- Produces (layout): extended matrices are `[X14 | H.reshape(n, n_layers*d) | u_smooth]`; `split_extended(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returns `(X14, H, u_smooth)`; width 14 means the base layout with no hidden part.
- Produces (probe): `HiddenProbe(alphas=(1e2, 1e3, 1e4, 1e5))` with `.oof_scores(H, y, groups, n_splits=5) -> np.ndarray` (query-grouped cross-fit on within-query-centred, standardised H, ridge) and `.fit(H, y, groups)` / `.predict(H) -> np.ndarray`.
- Produces (predictor): `MPUPPredictor(algorithm="rank", ...)` using `xgboost.XGBRanker` (`objective="rank:pairwise"`, groups from `groups`); if XGBRanker cannot run under Python 3.14, say so in the commit message and use pairwise logistic regression on within-query score differences instead.
- Produces (config): `make_fit_predict(algorithm, center_by_query, drop_groups: tuple[str, ...] = (), probe: bool = False, smooth_label: bool = False, **params)`; `Config` gains the same three fields; `grid_v2(combined: tuple[str, ...] = ()) -> list[Config]` with names `base` (= `gbr(100,d2)-centered`), `base-noD2D4`, `base+probe`, `base-rank`, `base+smooth`, and `combined` only when `combined` lists variant names. `probe=True` appends the out-of-fold probe score as a 15th feature in training and the full-fit score at prediction; `smooth_label=True` trains on `u_smooth` instead of `y`.

- [ ] **Step 1: Write the failing tests:** `test_split_extended_roundtrip`; `test_probe_oof_never_sees_own_query` (permuting a held-out query's labels leaves that query's OOF scores unchanged); `test_probe_recovers_planted_signal` (H with a column equal to centred y gives OOF rho > 0.8); `test_rank_handles_single_candidate_and_constant_queries` (no exception, finite scores); `test_drop_groups_ignores_columns` (changing D2/D4 columns does not change predictions); `test_smooth_label_trains_on_last_column`; `test_grid_v2_names_and_base_equals_frozen_config`; `test_nested_lodo_runs_with_grid_v2_on_extended_toy_data`.
- [ ] **Step 2:** `python3 -m pytest tests/features tests/predictor tests/evaluation/test_nested_selection.py -v` — new tests FAIL.
- [ ] **Step 3:** Implement the interfaces above. Existing 14-column behaviour and `GRID`/`PAPER` must not change.
- [ ] **Step 4:** Same command — PASS; `python3 -m pytest tests -q` — PASS.
- [ ] **Step 5:** Commit — "predictor: hidden-state probe, ranking objective, group drops, smooth label and the v2 grid".

---

### Task 6: Run the variants on the first sample

**Files:**
- Modify: `experiments/run_multidataset.py` (`load_tables`, CLI), `experiments/evaluate_answer_accuracy.py` (`load_utility_features`, `outer_predictions`, CLI)
- Test: `tests/evaluation/test_run_multidataset.py`, `tests/evaluation/test_answer_accuracy.py`
- Create: `results/mac_qwen7b/variants.{json,md}`

**Interfaces:**
- Produces: `load_tables(data_dir: Path, extended: bool = False)` and `load_utility_features(data_dir: Path, extended: bool = False)` (extended joins `hidden/` and `loglik/` by `(query_id, passage_id)`, raising on any mismatch); CLI flags `--extended` and `--grid {v1,v2}` on both scripts; `outer_predictions(feats, selected, grid=GRID)`.

- [ ] **Step 1: Write the failing tests:** `test_load_tables_extended_joins_by_ids` (shuffled hidden rows still land on the right pairs), `test_load_tables_extended_raises_on_missing_pair`, `test_outer_predictions_accepts_grid_v2`.
- [ ] **Step 2:** `python3 -m pytest tests/evaluation -v` — FAIL.
- [ ] **Step 3:** Implement. `--grid v2` runs the five one-at-a-time entries.
- [ ] **Step 4:** Same command — PASS.
- [ ] **Step 5: One-at-a-time run.** `python3 -m experiments.run_multidataset --data data/processed/mac_qwen7b --extended --grid v2 --out results/mac_qwen7b/variants_single`. For every entry other than `base`, read the pooled paired difference to `base`. Build `combined` from the variants whose point estimate is not below zero, rerun with `grid_v2(combined=...)` to `results/mac_qwen7b/variants`, and write which variants were combined into `variants.md`. The nested final configuration of that run is **the frozen configuration**; record its name in `variants.md`.
- [ ] **Step 6:** Commit code, tests and results — "experiments: v2 grid run on the first sample; freeze the configuration".

---

### Task 7: Third sample and the frozen confirmation

**Files:**
- Create: `experiments/drop_seen_questions.py`, `tests/evaluation/test_drop_seen_questions.py`
- Modify: `experiments/evaluate_fresh_questions.py`, `experiments/evaluate_answer_accuracy.py`
- Create: `results/mac_qwen7b/third_sample.{json,md}`, `results/mac_qwen7b/third_sample_answer_accuracy.{json,md}`

**Interfaces:**
- Produces: `drop_seen_questions(items: list[dict], seen_ids: set[str]) -> list[dict]`; CLI `python3 -m experiments.drop_seen_questions --data DIR --seen-dirs DIR... --seen-seeds 1` (ids of earlier samples come from the `items.json` of the given dirs and from `build_dataset(name, n=100, seed=s)` for each seed; the original file is kept as `items_all.json`); `evaluate_fresh_questions` flags `--grid v2 --extended --min-gain 0.08`, which prints "criterion met" or "criterion not met" using the pooled paired gain over dense and its 95% interval; `evaluate_answer_accuracy --train-data DIR --frozen-config NAME` fits the named config on `DIR` and ranks the data in `--data`.

- [ ] **Step 1: Write the failing test** `test_drop_seen_questions_removes_only_seen_ids_and_keeps_original`.
- [ ] **Step 2:** `python3 -m pytest tests/evaluation/test_drop_seen_questions.py -v` — FAIL. Implement; PASS.
- [ ] **Step 3: Precondition.** `ls` for the FlashRAG `wiki18_100w.zip`; it is not on this machine. If absent, stop and ask the user whether to download it (about 5 GB); do not substitute another corpus.
- [ ] **Step 4:** `python3 -m experiments.prepare_data --corpus <zip> --out data/processed/mac_qwen7b_s3 --seed 2`, then `python3 -m experiments.drop_seen_questions --data data/processed/mac_qwen7b_s3 --seen-dirs data/processed/mac_qwen7b --seen-seeds 1`.
- [ ] **Step 5:** Extract (`extract_multidataset` with `--hidden-layers 14 -1`, plus `extract_loglik` only if the frozen configuration uses `smooth_label`), then run the confirmation once: `python3 -m experiments.evaluate_fresh_questions --train data/processed/mac_qwen7b --fresh data/processed/mac_qwen7b_s3 --config "<frozen name>" --grid v2 --extended --min-gain 0.08 --out results/mac_qwen7b/third_sample` and the answer-accuracy run with `--train-data data/processed/mac_qwen7b --frozen-config "<frozen name>"`. Do not rerun with another configuration.
- [ ] **Step 6:** Commit — "results: frozen confirmation on the third question sample".

---

### Task 8: Measured selection cost on the Mac

**Files:**
- Create: `experiments/measure_selection_cost.py`
- Test: `tests/evaluation/test_measure_selection_cost.py`
- Create: `results/mac_qwen7b/cost.json`

**Interfaces:**
- Produces: `time_stages(stages: dict[str, Callable[[dict], object]], items: list[dict], clock: Callable[[], float] = time.perf_counter) -> dict` returning per-stage median and p90 seconds per query plus the total; CLI measures, on 40 questions of MuSiQue and HotpotQA, the stages the frozen pipeline really runs (HyDE generation, probing incl. hidden states, NLI only if D4 is kept, regressor) and the MLX peak memory.

- [ ] **Step 1: Write the failing test** `test_time_stages_reports_median_p90_and_total` with a fake clock advancing a known amount per call.
- [ ] **Step 2:** `python3 -m pytest tests/evaluation/test_measure_selection_cost.py -v` — FAIL. Implement; PASS.
- [ ] **Step 3:** Run the CLI on the Mac, write `results/mac_qwen7b/cost.json`, commit — "cost: measured selection latency and peak memory on the M5".

---

### Task 9 (only if time allows; cut order 9b first, then 9a): extra conditions

**9a Harder candidates.** Dense retrieval over the 21M-passage corpus is infeasible on 24 GiB, so candidates are BM25 top-100 re-ordered by E5 to top-20 (my call). Add `--hard-k 20` to `experiments/prepare_data.py`, write to `data/processed/mac_qwen7b_hard`, extract and score with the frozen configuration trained on the first sample, no selection. Test: a unit test that the dense re-order keeps exactly the top-k by cosine. Needs the corpus (see Task 7 Step 3).

**9b Cross-encoder.** Add `experiments/score_cross_encoder.py` (`sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")`) writing `ce/<dataset>.npz` and a `ce` entry in `BASELINES`; report it beside dense and MPUP on the first and third samples. Test: scores align by ids with a fake model.

- [ ] **Step 1:** For each executed item, write its failing test, implement, run `python3 -m pytest tests -q` (PASS), run the experiment, commit under "results: hard-candidate condition" / "results: cross-encoder baseline".

---

### Task 10: Update the paper

**Files:**
- Modify: `paper/mpup.tex`, `paper/mpup.pdf`, `paper/make_forest_figure.py` and `paper/fig_forest.tex` only if the figure data changes

- [ ] **Step 1:** Replace the run description with the Mac setup, keep the RTX bf16 results as a clearly labelled secondary row/paragraph, and say that the two runs differ in hardware, precision and labels at once.
- [ ] **Step 2:** Add the answer-accuracy results (Step 1 and third sample), the variants table with paired intervals and Holm-adjusted p-values, the third-sample confirmation with the "criterion met / not met" outcome, and the measured Mac memory and latency (Table V and the Cost subsection).
- [ ] **Step 3:** Remove the statements that no longer hold from Limitations and Conclusion ("answer accuracy ... not measured", "latency ... not measured", "4-bit end to end not measured"), and rewrite Abstract and Conclusion to the actual outcome: a larger effect if the criterion was met, otherwise the current small-gain framing plus the new measurements.
- [ ] **Step 4:** `cd paper && tectonic -X compile mpup.tex` — builds without errors; grep that no new sentence starts with This, These, Such, a personal pronoun, A or An.
- [ ] **Step 5:** Commit `paper/mpup.tex paper/mpup.pdf` (plus figure files if touched) — "paper: report the Mac run, answer accuracy and the variant comparison".
