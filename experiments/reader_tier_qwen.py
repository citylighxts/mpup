"""Cost-tier row for the selected reader (3B, all features) fitted to the Qwen2.5-7B utility, against the baselines
(Table tab:tiers). Writes results/reader_tier_qwen.json."""
import json
from pathlib import Path
from experiments import run_big
from experiments.run_criteria import setup, answer_labels, reader_candidates, V_PROBE, ahead_and_significant
d, dev, te = setup()
big = Path("data/processed/big")
lab, _ = answer_labels(big)
c = reader_candidates(d, big, lab)["reader 3B, all features"]
preds = {"tier": run_big.fit_predict({**c, "y": d["y"]}, dev, te, V_PROBE)}
preds.update({b: f(d, te) for b, f in run_big.BASELINES.items()})
cmp_ = run_big._compare(d, te, preds, "tier")
res = {"macro": cmp_["macro"], "ci": cmp_["paired_ci"], "tests": cmp_["tests"]}
res["ahead_holm_vs_ce"] = ahead_and_significant(cmp_, "tier", "cross-encoder")
Path("results/reader_tier_qwen.json").write_text(json.dumps(res, default=float, indent=1))
print(res["macro"], res["ci"]["cross-encoder"], res["ahead_holm_vs_ce"])
