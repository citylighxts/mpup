"""A second LLM: does MPUP+ work for another target LLM, and can one LLM read for another?

  python -m experiments.run_cross_llm --out results/cross_llm

`data/processed/big_mistral` holds every test question of the larger study and about 40% of its development
questions, labelled with Eq. (1) by a second LLM (Mistral-7B-Instruct-v0.3), with the 14 features computed with
that LLM (HyDE answer, logit probing) by `extract_multidataset` and its hidden states by `extract_extra`.
The split is the same hash split as `run_big`, so the test questions are the same ones. Every score is the
within-query Spearman correlation with the *second* LLM's utility on the test questions:

  target LLM       MPUP and MPUP+ fitted on the second LLM's development labels, its features and its probe
  Qwen as reader   second LLM's labels, but every LLM-side feature and the probe computed with Qwen (the reader
                   model differs from the target LLM)
  Qwen probe only  second LLM's features, Qwen's probe
  zero-shot        MPUP+ of the larger study (Qwen labels, all its development questions) used unchanged
  u (Qwen)         Qwen's own Eq. (1) utility of the passage: how far the two LLMs agree

The variant is the one chosen in the larger study and is not re-selected here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import run_big
from src.evaluation.protocol import LABEL_UTILITY, per_query_scores
from src.predictor.mpup_predictor import center_by_group

VARIANT = "gbr(500,d4)|u|ce|probe"


def load_ids(data_dir: Path) -> np.ndarray:
    """Passage ids in the row order of `run_big.load`."""
    out = []
    for path in sorted((data_dir / "features").glob("*.npz")):
        z = np.load(path)
        if str(z["label_source"]) == LABEL_UTILITY and (data_dir / "extra" / path.name).exists():
            out.append(z["passage_ids"])
    return np.concatenate(out)


def shuffled_probe(d, tr, te, seed: int = 0) -> np.ndarray:
    """Control: the probe fitted to labels permuted within each training query."""
    rng = np.random.default_rng(seed)
    t = d["y"][tr].copy()
    q = d["q"][tr]
    for qid in np.unique(q):
        m = np.flatnonzero(q == qid)
        t[m] = t[rng.permutation(m)]
    return run_big._probe_fit_predict(d["H"][tr], t, q, d["H"][te], d["q"][te])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/big_mistral")
    ap.add_argument("--layer", type=int, default=24, help="3/4 of the depth (32 layers); Qwen uses 21 of 28")
    ap.add_argument("--qwen-data", default="data/processed/big")
    ap.add_argument("--seen", nargs="*", default=["data/processed/multidataset", "data/processed/multidataset_seed1"])
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dm, dq = run_big.load(Path(args.data), args.layer), run_big.load(Path(args.qwen_data))
    row = {k: i for i, k in enumerate(zip(dq["q"], load_ids(Path(args.qwen_data))))}
    iq = np.array([row[k] for k in zip(dm["q"], load_ids(Path(args.data)))])
    seen = run_big.seen_ids([Path(p) for p in args.seen])
    test = run_big.split(dm["q"], seen)
    dev, te = np.flatnonzero(~test), np.flatnonzero(test)
    q_dev = np.flatnonzero(~run_big.split(dq["q"], seen))
    assert set(dm["q"][te]) <= set(dq["q"][iq[te]]) and not set(dq["q"][q_dev]) & set(dm["q"][te])

    reader = {**dm, "X": dq["X"][iq], "H": dq["H"][iq]}
    mixed = {**dm, "H": dq["H"][iq]}
    v, old = args.variant, run_big.OLD_VARIANT
    preds = {
        "mpup+ (target LLM)": run_big.fit_predict(dm, dev, te, v),
        "mpup (target LLM)": run_big.fit_predict(dm, dev, te, old),
        "mpup+ (Qwen as reader)": run_big.fit_predict(reader, dev, te, v),
        "mpup (Qwen as reader)": run_big.fit_predict(reader, dev, te, old),
        "mpup+ (Qwen probe only)": run_big.fit_predict(mixed, dev, te, v),
        "mpup+ (zero-shot from Qwen)": run_big.fit_predict(dq, q_dev, iq[te], v),
        "probe alone (target LLM)": run_big._probe_fit_predict(dm["H"][dev], dm["y"][dev], dm["q"][dev], dm["H"][te], dm["q"][te]),
        "probe, shuffled labels": shuffled_probe(dm, dev, te),
        "u (Qwen)": dq["y"][iq[te]],
    }
    preds.update({b: f(dm, te) for b, f in run_big.BASELINES.items()})
    refs = ("mpup+ (target LLM)", "mpup+ (Qwen as reader)", "mpup+ (zero-shot from Qwen)")
    res = {"variant": v, "layer": args.layer, "n_dev_queries": int(len(np.unique(dm["q"][dev]))),
           "n_test_queries": int(len(np.unique(dm["q"][te]))),
           "comparisons": {r: run_big._compare(dm, te, preds, r) for r in refs}}

    # how the two LLMs' utilities relate on the same pairs
    uq, um = dq["y"][iq], dm["y"]
    agree = per_query_scores(um[te], uq[te], dm["q"][te])["spearman"]
    res["labels"] = {"mean_u": {"qwen": float(uq.mean()), "target": float(um.mean())},
                     "nonzero": {"qwen": float((uq != 0).mean()), "target": float((um != 0).mean())},
                     "pearson_centred": float(np.corrcoef(center_by_group(uq, dm["q"]), center_by_group(um, dm["q"]))[0, 1]),
                     "within_query_spearman_test": float(agree.mean())}
    if (Path(args.data) / "run_info_none.json").exists():
        res["run_info"] = json.loads((Path(args.data) / "run_info_none.json").read_text())

    s = res["comparisons"][refs[0]]
    L = ["# A second target LLM", "",
         f"Target LLM labels (Eq. 1) on {s['n_queries']} rankable test questions of the larger study; "
         f"{res['n_dev_queries']} development questions for fitting. Variant `{v}`, probe layer {args.layer}.", "",
         f"Labels: mean u target {res['labels']['mean_u']['target']:+.3f} vs Qwen {res['labels']['mean_u']['qwen']:+.3f}; "
         f"nonzero {100 * res['labels']['nonzero']['target']:.0f}% vs {100 * res['labels']['nonzero']['qwen']:.0f}%; "
         f"within-query correlation of the two utilities (centred Pearson) {res['labels']['pearson_centred']:.3f}.", "",
         "| ranking | ρ | NDCG@10 | top-1 |", "|---|---|---|---|"]
    L += [f"| {n} | {m['spearman']:.3f} | {m['ndcg']:.3f} | {m['top1']:.3f} |" for n, m in s["macro"].items()]
    for r in refs:
        c = res["comparisons"][r]
        L += ["", f"Paired difference of {r} (95% bootstrap CI; sign-flip permutation p):", ""]
        for n in ("dense", "cross-encoder", "mpup (target LLM)", "mpup+ (target LLM)", "u (Qwen)"):
            if n in c["paired_ci"]:
                m, lo, hi = c["paired_ci"][n]
                L.append(f"- vs {n}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {c['tests'][n]['p_permutation']:.2g}")
        for b in ("dense", "cross-encoder"):
            ph = [x["p_holm"] for x in c["per_dataset_tests"][b].values()]
            better = sum(c["per_dataset"][k]["spearman"][r] > c["per_dataset"][k]["spearman"][b] for k in c["datasets"])
            L.append(f"- per dataset vs {b}: ahead in {better}/{len(ph)}, Holm p < 0.05 in {sum(p < 0.05 for p in ph)}/{len(ph)}")
    names = list(s["macro"])
    L += ["", "| dataset | queries | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for ds, x in s["per_dataset"].items():
        L.append(f"| {ds} | {x['n_queries']} | " + " | ".join(f"{x['spearman'][n]:.3f}" for n in names) + " |")
    text = "\n".join(L) + "\n"
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    Path(args.out).with_suffix(".md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
