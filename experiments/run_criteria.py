"""Evidence for criteria C3 to C6 of Table I, on the test questions of the larger study.

  python -m experiments.run_criteria light  --out results/light            # C3: cost tiers of MPUP
  python -m experiments.run_criteria labels --out results/answer_labels    # C6: labels from the gold answers
  python -m experiments.run_criteria transfer --out results/transfer       # C5: one predictor, several LLMs
  python -m experiments.run_criteria matrix --out results/transfer_matrix  # C5: every source LLM x every target LLM
  python -m experiments.run_criteria probing --out results/probing          # C4: tiers vs target-LLM judgments
  python -m experiments.run_criteria reader --out results/reader_select     # one config for C1-C6: choice on dev
  python -m experiments.run_criteria reader-test --out results/reader_test  # ... and its one look at the test part

Every predictor is the variant chosen in the larger study (`gbr(500,d4)`, query-centred, the same probe), fitted on
the development questions only; no setting is chosen on the test questions. Feature columns follow
`src.features.concatenate.FEATURE_GROUPS`:

  LLM-free   D1 (BM25, dense, rank, gap), D2 (length, readability, credibility), D3 entity overlap: columns 0-7
  HyDE       D3 answerability, D4 entailment and contradiction against the HyDE answer: columns 8-10 (one
             generation of M per query)
  D5         base and context perplexity and the entropy change: columns 11-13 (forward passes of M)

C3 tiers (trained on the Eq. (1) utility of M = Qwen2.5-7B):
  full          all 14 + cross-encoder + probe on M's hidden state (MPUP)
  no-generation D1, D2, D3 entity overlap, D5 + cross-encoder + probe on M: no generation at selection time,
                forward passes of M only
  reader-1.5B   LLM-free columns + cross-encoder + probe on the hidden state of Qwen2.5-1.5B-Instruct: M is not run
  reader-0.5B   the same with Qwen2.5-0.5B-Instruct
  no-LM         LLM-free columns + cross-encoder: no language model is run at all

C6 replaces the Eq. (1) labels of the development questions with a label from the dataset's own annotation: 1 when
a gold answer occurs in the passage (token sequence after normalisation, the relaxed match of Eq. (3)), else 0, the
distant supervision used to train dense retrievers. Questions whose gold answers are all yes/no are left out of
training (containment says nothing there). No LLM is run to make these labels. Scores are still against the Eq. (1)
utility of each target LLM.

C5 scores one predictor, fitted once, against the utilities of every target LLM (Qwen2.5-7B, Mistral-7B,
OLMo-2-7B) on the same test questions.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from experiments import run_big
from experiments.significance_tests import describe, holm
from src.evaluation.metrics import _is_subsequence, _normalize
from src.evaluation.protocol import LABEL_UTILITY, leave_one_dataset_out, per_query_scores
from src.predictor.mpup_predictor import center_by_group

LLM_FREE = list(range(8))
NO_GEN = [c for c in range(14) if c not in (8, 9, 10)]
V_PROBE, V_CE = "gbr(500,d4)|u|ce|probe", "gbr(500,d4)|u|ce"
SEEN = ["data/processed/multidataset", "data/processed/multidataset_seed1"]
READERS = {"reader-1.5B": ("extra_q15", 21), "reader-0.5B": ("extra_q05", 18)}
TARGETS = {"Qwen2.5-7B": ("data/processed/big", 21), "Mistral-7B": ("data/processed/big_mistral", 24),
           "OLMo-2-7B": ("data/processed/big_olmo", 24)}


def _files(data_dir: Path) -> list[Path]:
    """The feature files that `run_big.load` uses, in its order."""
    return [p for p in sorted((data_dir / "features").glob("*.npz"))
            if str(np.load(p)["label_source"]) == LABEL_UTILITY and (data_dir / "extra" / p.name).exists()]


def passage_ids(data_dir: Path) -> np.ndarray:
    return np.concatenate([np.load(p)["passage_ids"] for p in _files(data_dir)])


def load_hidden(data_dir: Path, sub: str, layer: int) -> np.ndarray:
    """Hidden states of another reader model (`extract_extra --out-dir sub`), aligned with `run_big.load`."""
    out = []
    for p in _files(data_dir):
        e = np.load(data_dir / sub / p.name)
        assert (e["passage_ids"] == np.load(p)["passage_ids"]).all(), p.name
        out.append(e[f"hidden_{layer}"])
    return np.concatenate(out).astype(np.float32)


def _golds(g) -> list[str]:
    if isinstance(g, str):
        try:
            g = ast.literal_eval(g)
        except (ValueError, SyntaxError):
            g = [g]
    return [str(a) for a in (g if isinstance(g, (list, tuple)) else [g])]


def answer_labels(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """1 if a gold answer occurs in the passage, else 0; and True for rows whose gold answers are all yes/no."""
    items = {str(it["query_id"]): it for it in json.loads((data_dir / "items.json").read_text())}
    lab, yesno = [], []
    for p in _files(data_dir):
        z = np.load(p)
        for qid, pid in zip(z["query_ids"], z["passage_ids"]):
            it = items[str(qid)]
            text = {x["id"]: x["text"] for x in it["passages"]}[str(pid)]
            golds = [g for g in (_normalize(a).split() for a in _golds(it["gold_answers"])) if g]
            toks = _normalize(text).split()
            lab.append(float(any(_is_subsequence(g, toks) for g in golds)))
            yesno.append(all(" ".join(g) in ("yes", "no") for g in golds))
    return np.array(lab), np.array(yesno)


def setup(data: str = "data/processed/big", layer: int = run_big.LAYER):
    d = run_big.load(Path(data), layer)
    test = run_big.split(d["q"], run_big.seen_ids([Path(p) for p in SEEN]))
    return d, np.flatnonzero(~test), np.flatnonzero(test)


def view(d: dict, cols: list[int], H: np.ndarray | None = None, y: np.ndarray | None = None) -> dict:
    out = {**d, "X": d["X"][:, cols]}
    if H is not None:
        out["H"] = H
    if y is not None:
        out["y"] = y
    return out


def summary(c: dict, ref: str, names: list[str]) -> dict:
    out = {"macro": {n: c["macro"][n] for n in names}}
    out["vs"] = {n: {"diff": c["paired_ci"][n], "p_perm": c["tests"][n]["p_permutation"],
                     "p_t": c["tests"][n]["p_ttest"], "dz": c["tests"][n]["cohen_dz"]}
                 for n in names if n != ref}
    return out


def md_table(c: dict, names: list[str]) -> list[str]:
    L = ["| ranking | ρ | NDCG@10 | top-1 |", "|---|---|---|---|"]
    return L + [f"| {n} | {c['macro'][n]['spearman']:.3f} | {c['macro'][n]['ndcg']:.3f} | {c['macro'][n]['top1']:.3f} |"
                for n in names]


def md_vs(c: dict, ref: str, names: list[str]) -> list[str]:
    L = [f"", f"Paired difference of {ref} (95% bootstrap CI; sign-flip permutation p; paired t p; d_z):", ""]
    for n in names:
        if n == ref:
            continue
        m, lo, hi = c["paired_ci"][n]
        t = c["tests"][n]
        L.append(f"- vs {n}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {t['p_permutation']:.2g}, "
                 f"p_t = {t['p_ttest']:.2g}, d_z = {t['cohen_dz']:.2f}")
    return L


def ahead_and_significant(c: dict, a: str, b: str) -> tuple[int, int]:
    """Datasets where `a` is ahead of baseline `b`, and where it is ahead with Holm p < 0.05 (`run_big._compare`)."""
    ahead = {k: c["per_dataset"][k]["spearman"][a] > c["per_dataset"][k]["spearman"][b] for k in c["datasets"]}
    return int(sum(ahead.values())), int(sum(ahead[k] and x["p_holm"] < 0.05 for k, x in c["per_dataset_tests"][b].items()))


# ------------------------------------------------------------------ C3

def run_light(args) -> None:
    d, dev, te = setup()
    preds = {"full (MPUP)": run_big.fit_predict(d, dev, te, V_PROBE),
             "no-generation": run_big.fit_predict(view(d, NO_GEN), dev, te, V_PROBE)}
    info = {}
    for name, (sub, layer) in READERS.items():
        if (Path("data/processed/big") / sub).exists():
            preds[name] = run_big.fit_predict(view(d, LLM_FREE, load_hidden(Path("data/processed/big"), sub, layer)),
                                              dev, te, V_PROBE)
            ri = Path("data/processed/big") / sub / "run_info.json"
            info[name] = json.loads(ri.read_text()) if ri.exists() else None
    preds["no-LM"] = run_big.fit_predict(view(d, LLM_FREE), dev, te, V_CE)
    preds.update({b: f(d, te) for b, f in run_big.BASELINES.items()})
    tiers = [n for n in preds if n not in run_big.BASELINES]
    res = {"n_test_queries": int(len(np.unique(d["q"][te]))), "reader_run_info": info,
           "comparisons": {t: run_big._compare(d, te, preds, t) for t in tiers}}
    names = list(preds)
    c0 = res["comparisons"][tiers[0]]
    L = ["# C3: cost tiers of MPUP", "", f"{c0['n_queries']} rankable test questions, target LLM Qwen2.5-7B. "
         "Each tier fitted on the development questions with the Eq. (1) utility of Qwen2.5-7B.", ""]
    L += md_table(c0, names)
    for t in tiers:
        L += md_vs(res["comparisons"][t], t, names)
        c = res["comparisons"][t]
        for b in ("dense", "cross-encoder"):
            better, up = ahead_and_significant(c, t, b)
            L.append(f"- per dataset vs {b}: ahead in {better}/{len(c['datasets'])}, ahead with Holm p < 0.05 in {up}")
    L += ["", "Reader runs (hidden states + cross-encoder, all 128,467 pairs):", ""]
    L += [f"- {n}: {i}" for n, i in info.items()]
    write(args.out, res, L)


# ------------------------------------------------------------------ C6

def run_labels(args) -> None:
    big = Path("data/processed/big")
    d, dev, te = setup()
    lab, yesno = answer_labels(big)
    dev_a = dev[~yesno[dev]]
    H15 = load_hidden(big, *READERS["reader-1.5B"]) if (big / READERS["reader-1.5B"][0]).exists() else None
    preds = {"MPUP (Eq. 1 labels)": run_big.fit_predict(d, dev, te, V_PROBE),
             "MPUP (answer labels)": run_big.fit_predict(view(d, list(range(14)), y=lab), dev_a, te, V_PROBE)}
    if H15 is not None:
        preds["reader-1.5B (Eq. 1 labels)"] = run_big.fit_predict(view(d, LLM_FREE, H15), dev, te, V_PROBE)
        preds["reader-1.5B (answer labels)"] = run_big.fit_predict(view(d, LLM_FREE, H15, lab), dev_a, te, V_PROBE)
    preds["no-LM (Eq. 1 labels)"] = run_big.fit_predict(view(d, LLM_FREE), dev, te, V_CE)
    preds["no-LM (answer labels)"] = run_big.fit_predict(view(d, LLM_FREE, y=lab), dev_a, te, V_CE)
    preds["answer containment (uses the gold answer)"] = lab[te]
    preds.update({b: f(d, te) for b, f in run_big.BASELINES.items()})
    refs = [n for n in preds if "answer labels" in n]
    res = {"n_test_queries": int(len(np.unique(d["q"][te]))),
           "labels": {"share_positive_dev": float(lab[dev_a].mean()), "n_dev_rows": int(len(dev_a)),
                      "n_dev_rows_yesno_left_out": int(yesno[dev].sum()),
                      "pearson_centred_with_u_dev": float(np.corrcoef(center_by_group(lab[dev_a], d["q"][dev_a]),
                                                                      center_by_group(d["y"][dev_a], d["q"][dev_a]))[0, 1])},
           "comparisons": {r: run_big._compare(d, te, preds, r) for r in refs}}
    names = list(preds)
    c0 = res["comparisons"][refs[0]]
    L = ["# C6: training on labels from the dataset's gold answers", "",
         f"{c0['n_queries']} rankable test questions, scored against the Eq. (1) utility of Qwen2.5-7B. "
         f"Answer labels on {res['labels']['n_dev_rows']} development pairs ({100 * res['labels']['share_positive_dev']:.1f}% "
         f"positive; {res['labels']['n_dev_rows_yesno_left_out']} yes/no pairs left out); centred Pearson with u "
         f"{res['labels']['pearson_centred_with_u_dev']:.3f}.", ""]
    L += md_table(c0, names)
    for r in refs:
        L += md_vs(res["comparisons"][r], r, names)
    write(args.out, res, L)


# ------------------------------------------------------------------ C5

def run_transfer(args) -> None:
    """Predictors fitted once on Qwen2.5-7B data, scored against each target LLM's utility on its test questions."""
    big = Path("data/processed/big")
    d, dev, te = setup()
    lab, yesno = answer_labels(big)
    dev_a = dev[~yesno[dev]]
    H15 = load_hidden(big, *READERS["reader-1.5B"])
    allr = np.arange(len(d["y"]))
    fitted = {  # each fitted once on the Qwen development questions; predictions for every row
        "MPUP (Qwen labels)": run_big.fit_predict(d, dev, allr, V_PROBE),
        "reader-1.5B (Qwen labels)": run_big.fit_predict(view(d, LLM_FREE, H15), dev, allr, V_PROBE),
        "reader-1.5B (answer labels)": run_big.fit_predict(view(d, LLM_FREE, H15, lab), dev_a, allr, V_PROBE),
        "no-LM (answer labels)": run_big.fit_predict(view(d, LLM_FREE, y=lab), dev_a, allr, V_CE),
    }
    row = {k: i for i, k in enumerate(zip(d["q"], passage_ids(big)))}
    res = {"targets": {}}
    L = ["# C5: one predictor for several target LLMs", "",
         "Every predictor is fitted once on the development questions of Qwen2.5-7B (its utility or the answer labels) "
         "and scored unchanged against the Eq. (1) utility of each target LLM on its test questions. 'native' is "
         "MPUP fitted on the target LLM's own labels, features and hidden states (an upper reference).", ""]
    for tname, (path, layer) in TARGETS.items():
        p = Path(path)
        if not (p / "extra").exists() or not any((p / "extra").glob("*.npz")):
            continue
        dm = run_big.load(p, layer)
        iq = np.array([row[k] for k in zip(dm["q"], passage_ids(p))])
        test = run_big.split(dm["q"], run_big.seen_ids([Path(s) for s in SEEN]))
        dv, tt = np.flatnonzero(~test), np.flatnonzero(test)
        preds = {n: v[iq[tt]] for n, v in fitted.items()}
        if tname != "Qwen2.5-7B":
            preds["native MPUP (own labels)"] = run_big.fit_predict(dm, dv, tt, V_PROBE)
            preds["u (Qwen2.5-7B)"] = d["y"][iq[tt]]
        preds.update({b: f(dm, tt) for b, f in run_big.BASELINES.items()})
        refs = list(fitted)
        cmp_ = {r: run_big._compare(dm, tt, preds, r) for r in refs}
        um, uq = dm["y"], d["y"][iq]
        res["targets"][tname] = {"n_test_queries": cmp_[refs[0]]["n_queries"], "comparisons": cmp_,
                                 "pearson_centred_with_qwen": float(np.corrcoef(center_by_group(uq, dm["q"]),
                                                                                center_by_group(um, dm["q"]))[0, 1])}
        names = list(preds)
        L += [f"## {tname} ({cmp_[refs[0]]['n_queries']} rankable test questions; centred Pearson of its utility "
              f"with Qwen's {res['targets'][tname]['pearson_centred_with_qwen']:.3f})", ""]
        L += md_table(cmp_[refs[0]], names)
        for r in refs:
            L += md_vs(cmp_[r], r, [n for n in names if n in ("dense", "cross-encoder", "bm25", "native MPUP (own labels)")
                                    or n == r])
        L.append("")
    write(args.out, res, L)


def run_matrix(args) -> None:
    """Source x target: MPUP fitted on one LLM's development questions (its labels, features and hidden states),
    scored against another LLM's utility on that LLM's test questions (the same pairs, read by the source LLM)."""
    big = Path("data/processed/big")
    d, _, _ = setup()
    key_q = {k: i for i, k in enumerate(zip(d["q"], passage_ids(big)))}
    llms = {}
    for name, (path, layer) in TARGETS.items():
        p = Path(path)
        if name != "Qwen2.5-7B" and not any((p / "extra").glob("*.npz")):
            continue
        dm = run_big.load(p, layer)
        test = run_big.split(dm["q"], run_big.seen_ids([Path(s) for s in SEEN]))
        keys = list(zip(dm["q"], passage_ids(p)))
        llms[name] = {"d": dm, "dev": np.flatnonzero(~test), "te": np.flatnonzero(test),
                      "row": {k: i for i, k in enumerate(keys)}, "keys": keys}
    pred = {s: run_big.fit_predict(v["d"], v["dev"], np.arange(len(v["d"]["y"])), V_PROBE) for s, v in llms.items()}
    res = {"sources": {s: int(len(np.unique(v["d"]["q"][v["dev"]]))) for s, v in llms.items()}, "cells": {}}
    for t, vt in llms.items():
        tt = vt["te"]
        preds = {f"from {s}": pred[s][[vs["row"][vt["keys"][i]] for i in tt]] for s, vs in llms.items()}
        preds.update({b: f(vt["d"], tt) for b, f in run_big.BASELINES.items()})
        for s in llms:
            c = run_big._compare(vt["d"], tt, preds, f"from {s}")
            res["cells"][f"{s} -> {t}"] = {"n_queries": c["n_queries"], "macro": c["macro"][f"from {s}"]["spearman"],
                                           "cross-encoder": c["macro"]["cross-encoder"]["spearman"],
                                           "dense": c["macro"]["dense"]["spearman"],
                                           "vs": {b: {"diff": c["paired_ci"][b], "p_perm": c["tests"][b]["p_permutation"],
                                                      "p_t": c["tests"][b]["p_ttest"], "dz": c["tests"][b]["cohen_dz"]}
                                                  for b in ("cross-encoder", "dense")},
                                           "ahead_holm_vs_ce_below_005": ahead_and_significant(c, f"from {s}", "cross-encoder")[1],
                                           "n_datasets": len(c["datasets"])}
    names = list(llms)
    L = ["# Transfer matrix: MPUP fitted on one LLM, scored against another LLM's utility", "",
         "Rows: the LLM whose development labels, features and hidden states MPUP is fitted on (development questions: "
         + ", ".join(f"{s} {n}" for s, n in res["sources"].items()) + "). Columns: the LLM whose Eq. (1) utility is "
         "predicted, on its test questions. Cell: r_s (difference to the cross-encoder [95% CI], d_z).", "",
         "| fitted on \\ target | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for s in names:
        cells = []
        for t in names:
            c = res["cells"][f"{s} -> {t}"]
            m, lo, hi = c["vs"]["cross-encoder"]["diff"]
            cells.append(f"{c['macro']:.3f} ({m:+.3f} [{lo:+.3f}, {hi:+.3f}], {c['vs']['cross-encoder']['dz']:.2f})")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("| cross-encoder | " + " | ".join(f"{res['cells'][f'{names[0]} -> {t}']['cross-encoder']:.3f}" for t in names) + " |")
    L.append("| dense | " + " | ".join(f"{res['cells'][f'{names[0]} -> {t}']['dense']:.3f}" for t in names) + " |")
    L += ["", "Datasets where the cell is ahead of the cross-encoder with Holm p < 0.05:", ""]
    L += [f"- {k}: {v['ahead_holm_vs_ce_below_005']}/{v['n_datasets']}" for k, v in res["cells"].items()]
    write(args.out, res, L)


# ------------------------------------------------------------------ C4

PROBES = {"Qwen2.5-7B": "results/probe_baselines", "Mistral-7B": "results/probe_baselines_mistral",
          "OLMo-2-7B": "results/probe_baselines_olmo"}
PROBE_METHODS = ["PVS w/o A", "PVS w/ A", "likelihood", "LVR w/ A", "answer confidence"]


def per_dataset_holm(d: dict, te: np.ndarray, a: np.ndarray, b: np.ndarray, datasets: list[str]) -> list[float]:
    """Holm-adjusted sign-flip p of a - b per dataset (as `run_big._compare` does for the baselines)."""
    ra, rb = run_big.rho(d, te, a), run_big.rho(d, te, b)
    ps = []
    for name in datasets:
        m = np.isin(ra["query_ids"], np.unique(d["q"][te][d["ds"][te] == name]))
        ps.append(describe(ra["spearman"][m] - rb["spearman"][m], n_perm=50_000, seed=0)["p_permutation"])
    return holm(ps)


def run_probing(args) -> None:
    """The MPUP tiers against the estimates that query the target LLM (`run_probe_baselines`, cached scores),
    on the same test pairs of each target LLM."""
    big = Path("data/processed/big")
    d, dev, _ = setup()
    lab, yesno = answer_labels(big)
    dev_a = dev[~yesno[dev]]
    H15 = load_hidden(big, *READERS["reader-1.5B"])
    allr = np.arange(len(d["y"]))
    fitted = {"MPUP (Qwen labels)": run_big.fit_predict(d, dev, allr, V_PROBE),
              "reader-1.5B (answer labels)": run_big.fit_predict(view(d, LLM_FREE, H15, lab), dev_a, allr, V_PROBE),
              "no-LM (answer labels)": run_big.fit_predict(view(d, LLM_FREE, y=lab), dev_a, allr, V_CE)}
    title = ["# C4: MPUP tiers against utility estimates that query the target LLM", "",
             "Probing methods: SpecUBench-style judgments of the target LLM (Zhang et al. 2025c), scores cached by "
             "`run_probe_baselines`. MPUP tiers fitted on Qwen2.5-7B development questions (Qwen labels or answer "
             "labels); 'native MPUP' is fitted on the target LLM itself.", ""]
    compare_on_targets(d, fitted, title, args.out)


def compare_on_targets(d: dict, fitted: dict[str, np.ndarray], title: list[str], out: str,
                       only_with_probes: bool = True) -> None:
    """Predictions for every row of the Qwen data (`fitted`), scored against each target LLM's utility on its test
    pairs, next to the cached judgments of that LLM, the native MPUP and the relevance baselines."""
    big = Path("data/processed/big")
    row = {k: i for i, k in enumerate(zip(d["q"], passage_ids(big)))}
    res = {"targets": {}}
    L = list(title)
    for tname, stem in PROBES.items():
        cache = Path(stem + "_scores.npz")
        path, layer = TARGETS[tname]
        p = Path(path)
        if (only_with_probes and not cache.exists()) or not any((p / "extra").glob("*.npz")):
            continue
        dm = run_big.load(p, layer)
        test = run_big.split(dm["q"], run_big.seen_ids([Path(s) for s in SEEN]))
        dv, tt = np.flatnonzero(~test), np.flatnonzero(test)
        sc = np.load(cache, allow_pickle=True) if cache.exists() else None
        iq = np.array([row[k] for k in zip(dm["q"], passage_ids(p))])
        preds = {}
        if tname != "Qwen2.5-7B":
            preds["native MPUP"] = run_big.fit_predict(dm, dv, tt, V_PROBE)
        preds.update({n: v[iq[tt]] for n, v in fitted.items()})
        if sc is not None:
            preds.update({m: np.asarray(sc[m], dtype=float) for m in PROBE_METHODS if m in sc.files})
        preds.update({b: f(dm, tt) for b, f in run_big.BASELINES.items()})
        tiers = [n for n in preds if "MPUP" in n or "answer labels" in n]
        cmp_ = {r: run_big._compare(dm, tt, preds, r) for r in tiers}
        timing = json.loads(Path(stem + "_timing.json").read_text()) if sc is not None else None
        res["targets"][tname] = {"n_test_queries": cmp_[tiers[0]]["n_queries"], "timing_s": timing,
                                 "comparisons": {r: summary(c, r, list(preds)) for r, c in cmp_.items()},
                                 "per_dataset": {}}
        names = list(preds)
        L += [f"## {tname} ({cmp_[tiers[0]]['n_queries']} rankable test questions)", ""]
        L += md_table(cmp_[tiers[0]], names)
        for r in tiers:
            others = [n for n in names if n in PROBE_METHODS or n in ("cross-encoder", "dense")]
            L += md_vs(cmp_[r], r, others + [r])
            c = cmp_[r]
            for m in others:
                ph = per_dataset_holm(dm, tt, preds[r], preds[m], c["datasets"])
                ahead = [c["per_dataset"][k]["spearman"][r] > c["per_dataset"][k]["spearman"][m] for k in c["datasets"]]
                sig_up = sum(a and q < 0.05 for a, q in zip(ahead, ph))
                sig_down = sum(not a and q < 0.05 for a, q in zip(ahead, ph))
                res["targets"][tname]["per_dataset"][f"{r} vs {m}"] = {"ahead": int(sum(ahead)),
                                                                       "ahead_holm_below_005": int(sig_up),
                                                                       "behind_holm_below_005": int(sig_down),
                                                                       "n": len(ph)}
                L.append(f"- per dataset vs {m}: ahead in {sum(ahead)}/{len(ph)}; Holm p < 0.05 ahead in {sig_up}, "
                         f"behind in {sig_down}")
        L += ["", f"Timing (s, all test pairs): {json.dumps(timing)}", ""]
    write(out, res, L)


# ------------------------------------------------------------------ one configuration for all six criteria

READER_SETS = {"1.5B": ("features_q15", "extra_q15", 21), "3B": ("features_q3", "extra_q3", 27)}


def load_reader_X(data_dir: Path, sub: str) -> np.ndarray | None:
    """The 14 features recomputed with a small reader as M (`extract_multidataset --features-only sub`), aligned with
    `run_big.load`; None while any dataset is missing."""
    out = []
    for p in _files(data_dir):
        f = data_dir / sub / p.name
        if not f.exists():
            return None
        z = np.load(f)
        assert (z["passage_ids"] == np.load(p)["passage_ids"]).all(), p.name
        out.append(z["X"])
    return np.concatenate(out)


def reader_candidates(d: dict, data_dir: Path, lab: np.ndarray) -> dict[str, dict]:
    """Every reader configuration fitted to the answer labels: the LLM-free columns or all 14 features recomputed with
    the reader, plus the ridge probe on the reader's hidden state, plus the cross-encoder. No column comes from M."""
    out = {}
    for size, (fsub, hsub, layer) in READER_SETS.items():
        if not (data_dir / hsub).exists() or not all((data_dir / hsub / p.name).exists() for p in _files(data_dir)):
            continue
        H = load_hidden(data_dir, hsub, layer)
        out[f"reader {size}, hidden"] = view(d, LLM_FREE, H, lab)
        X = load_reader_X(data_dir, fsub)
        if X is not None:
            assert np.allclose(X[:, LLM_FREE], d["X"][:, LLM_FREE]), "LLM-free columns must not depend on the reader"
            out[f"reader {size}, all features"] = {**view(d, LLM_FREE, H, lab), "X": X}
    return {k: {kk: v[kk] for kk in ("X", "y", "q", "ds", "ce", "H", "soft")} for k, v in out.items()}


def _dev_fold(v: dict, u: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple[float, float]:
    pred = run_big.fit_predict(v, tr, te, V_PROBE)
    return (float(per_query_scores(v["y"][te], pred, v["q"][te])["spearman"].mean()),
            float(per_query_scores(u[te], pred, v["q"][te])["spearman"].mean()))


def run_reader(args) -> None:
    """Choose the reader configuration on the development questions only.

    Rule, fixed before the test questions are scored: leave-one-dataset-out over the development datasets, every
    candidate fitted to answer containment, and the candidate with the highest macro r_s against answer containment
    on the held-out dataset is selected (no LLM label enters the choice). The r_s against the Qwen2.5-7B utility of
    the same folds is reported for information only."""
    from joblib import Parallel, delayed

    big = Path("data/processed/big")
    d, dev, _ = setup()
    lab, yesno = answer_labels(big)
    dev_a = dev[~yesno[dev]]
    cands = reader_candidates(d, big, lab)
    folds = leave_one_dataset_out(d["ds"][dev_a])
    jobs = [(c, name, dev_a[tr], dev_a[te]) for c in cands for name, tr, te in folds]
    scores = Parallel(n_jobs=min(len(jobs), 44))(delayed(_dev_fold)(cands[c], d["y"], tr, te) for c, _, tr, te in jobs)
    res = {"rule": "max macro dev LODO r_s against answer containment", "n_dev_queries": int(len(np.unique(d["q"][dev_a]))),
           "candidates": {c: {"per_dataset": {}} for c in cands}}
    for (c, name, _, _), (s_lab, s_u) in zip(jobs, scores):
        res["candidates"][c]["per_dataset"][name] = {"answer": s_lab, "utility_qwen": s_u}
    for c, v in res["candidates"].items():
        v["macro_answer"] = float(np.mean([x["answer"] for x in v["per_dataset"].values()]))
        v["macro_utility_qwen"] = float(np.mean([x["utility_qwen"] for x in v["per_dataset"].values()]))
    res["selected"] = max(res["candidates"], key=lambda c: res["candidates"][c]["macro_answer"])
    L = ["# Reader configuration chosen on the development questions", "", res["rule"] + f"; {res['n_dev_queries']} "
         "development questions (yes/no left out). Utility r_s for information only.", "",
         "| candidate | dev r_s vs answer containment | dev r_s vs Qwen utility |", "|---|---|---|"]
    L += [f"| {c}{' **(selected)**' if c == res['selected'] else ''} | {v['macro_answer']:.3f} | {v['macro_utility_qwen']:.3f} |"
          for c, v in res["candidates"].items()]
    write(args.out, res, L)


def run_reader_test(args) -> None:
    """One look at the test questions for the selected reader configuration, against every target LLM."""
    big = Path("data/processed/big")
    sel = json.loads(Path("results/reader_select.json").read_text())["selected"]
    d, dev, _ = setup()
    lab, yesno = answer_labels(big)
    dev_a = dev[~yesno[dev]]
    cands = reader_candidates(d, big, lab)
    allr = np.arange(len(d["y"]))
    fitted = {"MPUP (Qwen labels)": run_big.fit_predict(d, dev, allr, V_PROBE),
              f"MPUP-R ({sel}, answer labels)": run_big.fit_predict(cands[sel], dev_a, allr, V_PROBE),
              # for the cost-tier table only, which is fitted to the Qwen utility like the other tiers
              f"{sel} (Qwen labels)": run_big.fit_predict({**cands[sel], "y": d["y"]}, dev, allr, V_PROBE)}
    if sel != "reader 1.5B, hidden":
        fitted["reader 1.5B, hidden (answer labels)"] = run_big.fit_predict(cands["reader 1.5B, hidden"], dev_a, allr, V_PROBE)
    title = ["# The selected reader configuration on the test questions of every target LLM", "",
             f"Selected on the development questions (`results/reader_select.md`): {sel}. Fitted once on the Qwen "
             "development questions with answer-containment labels; applied unchanged to every target LLM.", ""]
    compare_on_targets(d, fitted, title, args.out, only_with_probes=False)


def write(out: str, res: dict, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    Path(out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    Path(out).with_suffix(".md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["light", "labels", "transfer", "matrix", "probing", "reader", "reader-test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    {"light": run_light, "labels": run_labels, "transfer": run_transfer, "matrix": run_matrix, "probing": run_probing,
     "reader": run_reader, "reader-test": run_reader_test}[args.what](args)


if __name__ == "__main__":
    main()
