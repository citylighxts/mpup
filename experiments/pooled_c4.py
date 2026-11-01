"""C4 pooled over the target LLMs: the reader tier against each utility judgment of the target LLM, one method at a
time across Qwen2.5-7B, Mistral-7B and OLMo-2-7B, and against the best judgment of each LLM (an oracle choice).

Statistic: the macro mean over the LLMs of the mean per-query difference in r_s. The same question is rankable for
several LLMs, so the bootstrap resamples questions and the sign-flip test flips all differences of a question together.
Writes results/c4_pooled.{json,md}.

    python -m experiments.pooled_c4
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import run_big
from experiments.run_criteria import (PROBE_METHODS, PROBES, SEEN, TARGETS, V_PROBE, answer_labels, passage_ids,
                                      reader_candidates, setup, write)
from src.evaluation.protocol import per_query_scores

B, PERM = 10_000, 20_000


def rho_by_query(dm: dict, tt: np.ndarray, pred: np.ndarray) -> dict:
    s = per_query_scores(dm["y"][tt], pred, dm["q"][tt])
    return dict(zip(s["query_ids"], s["spearman"]))


def pooled(diffs: dict[str, dict], rng: np.random.Generator) -> dict:
    """diffs: LLM -> {question: difference}. Macro mean over LLMs, question-clustered bootstrap and sign flip."""
    llms = list(diffs)
    qs = sorted(set().union(*[set(v) for v in diffs.values()]))
    qi = {q: i for i, q in enumerate(qs)}
    # per LLM: question index and difference
    cols = {m: (np.array([qi[q] for q in v]), np.array(list(v.values()))) for m, v in diffs.items()}

    def stat(w: np.ndarray, sign: np.ndarray) -> float:
        return float(np.mean([np.sum(w[i] * sign[i] * x) / np.sum(w[i]) for i, x in cols.values()]))

    one = np.ones(len(qs))
    obs = stat(one, one)
    boot = np.array([stat(np.bincount(rng.integers(0, len(qs), len(qs)), minlength=len(qs)).astype(float), one)
                     for _ in range(B)])
    flips = np.array([stat(one, rng.choice([-1.0, 1.0], len(qs))) for _ in range(PERM)])
    return {"macro_diff": obs, "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_signflip": float((np.sum(np.abs(flips) >= abs(obs)) + 1) / (PERM + 1)),
            "per_llm": {m: float(np.mean(list(v.values()))) for m, v in diffs.items()},
            "n_questions": len(qs), "n_pairs_llm_question": int(sum(len(v) for v in diffs.values()))}


def main() -> None:
    rng = np.random.default_rng(0)
    big = Path("data/processed/big")
    sel = json.loads(Path("results/reader_select.json").read_text())["selected"]
    d, dev, _ = setup()
    lab, yesno = answer_labels(big)
    dev_a = dev[~yesno[dev]]
    reader = run_big.fit_predict(reader_candidates(d, big, lab)[sel], dev_a, np.arange(len(d["y"])), V_PROBE)
    row = {k: i for i, k in enumerate(zip(d["q"], passage_ids(big)))}

    r_reader, r_meth, macro = {}, {}, {}
    for tname, stem in PROBES.items():
        path, layer = TARGETS[tname]
        dm = run_big.load(Path(path), layer)
        tt = np.flatnonzero(run_big.split(dm["q"], run_big.seen_ids([Path(s) for s in SEEN])))
        iq = np.array([row[k] for k in zip(dm["q"], passage_ids(Path(path)))])
        sc = np.load(stem + "_scores.npz", allow_pickle=True)
        r_reader[tname] = rho_by_query(dm, tt, reader[iq[tt]])
        r_meth[tname] = {m: rho_by_query(dm, tt, np.asarray(sc[m], dtype=float)) for m in PROBE_METHODS}
        macro[tname] = {"reader": float(np.mean(list(r_reader[tname].values()))),
                        **{m: float(np.mean(list(v.values()))) for m, v in r_meth[tname].items()}}

    def diff(tname: str, m: str) -> dict:
        a, b = r_reader[tname], r_meth[tname][m]
        return {q: a[q] - b[q] for q in a}

    best = {t: max(PROBE_METHODS, key=lambda m: macro[t][m]) for t in macro}
    res = {"selected_reader": sel, "macro_r_s": macro, "best_per_llm": best,
           "vs_method": {m: pooled({t: diff(t, m) for t in macro}, rng) for m in PROBE_METHODS},
           "vs_best_per_llm": pooled({t: diff(t, best[t]) for t in macro}, rng)}

    L = ["# C4 pooled over three target LLMs", "",
         f"Reader tier ({sel}, answer labels, fitted once on Qwen's development questions) against the utility "
         "judgments of each target LLM. Difference = macro mean over Qwen2.5-7B, Mistral-7B and OLMo-2-7B of the mean "
         f"per-query difference in r_s; 95% bootstrap CI over questions ({B} draws) and sign-flip p with all LLMs of a "
         f"question flipped together ({PERM} draws).", "",
         "| comparison | Qwen | Mistral | OLMo | macro difference [95% CI] | p |", "|---|---|---|---|---|---|"]
    for name, r in list(res["vs_method"].items()) + [("best judgment of each LLM (oracle: "
                                                          + ", ".join(best.values()) + ")", res["vs_best_per_llm"])]:
        pl = r["per_llm"]
        L.append(f"| reader − {name} | {pl['Qwen2.5-7B']:+.3f} | {pl['Mistral-7B']:+.3f} | {pl['OLMo-2-7B']:+.3f} | "
                 f"{r['macro_diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p_signflip']:.2g} |")
    L += ["", f"{res['vs_best_per_llm']['n_questions']} distinct test questions, "
          f"{res['vs_best_per_llm']['n_pairs_llm_question']} (LLM, question) pairs.", "",
          "r_s below is the mean over questions (the paper's r_s averages over datasets first, so it is about "
          "0.005 lower); the differences above are means over questions in both.", "",
          "| r_s | " + " | ".join(macro) + " | macro |", "|---|" + "---|" * (len(macro) + 1)]
    for k in ["reader"] + PROBE_METHODS:
        v = [macro[t][k] for t in macro]
        L.append(f"| {k} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.3f} |")
    write("results/c4_pooled", res, L)


if __name__ == "__main__":
    main()
