"""Write paper/fig_forest.tex: MPUP minus dense similarity, per held-out dataset, as a TikZ forest plot.

  python paper/make_forest_figure.py

Reads the paired differences of the nested leave-one-dataset-out run (Qwen2.5-7B, bfloat16) from
results/multidataset_rtx6000_qwen7b.json, so the figure always matches Table I. A filled marker means
the 95% interval excludes zero; an open marker means it includes zero. The last row is the paired
difference over all held-out queries together (the interval quoted in the abstract).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "multidataset_rtx6000_qwen7b.json"
OUT = Path(__file__).resolve().parent / "fig_forest.tex"

NAMES = {
    "2wikimultihopqa": "2WikiMultiHopQA", "ambigqa": "AmbigQA", "bamboogle": "Bamboogle",
    "entityquestions": "EntityQuestions", "hotpotqa": "HotpotQA", "musique": "MuSiQue", "nq": "NQ",
    "popqa": "PopQA", "squad": "SQuAD", "triviaqa": "TriviaQA", "webquestions": "WebQuestions",
}
X_MIN, X_MAX, X_PER_UNIT = -0.15, 0.30, 12.0  # data units, and cm per 1.0 of rho difference (1.2 cm per 0.1)
ROW = 0.38  # cm between rows


def build() -> str:
    nested = json.loads(SOURCE.read_text())["nested_lodo"]
    rows = []
    for key, d in nested["per_dataset"].items():
        mean, lo, hi = d["paired_vs"]["dense"]["spearman"]
        rows.append((f"{NAMES[key]} ({d['n_queries']})", mean, lo, hi))
    rows.sort(key=lambda r: -r[1])
    pooled = nested["pooled_paired_vs"]["dense"]["spearman"]

    x = lambda v: (v - X_MIN) * X_PER_UNIT  # noqa: E731
    n = len(rows)
    y_pooled = -(n + 0.6) * ROW
    y_axis = y_pooled - 0.7 * ROW
    L = [
        r"\begin{tikzpicture}[x=1cm, y=1cm, font=\scriptsize, >=Stealth]",
        f"  \\draw[gray!70, thin] ({x(0):.3f},{ROW * 0.6:.3f}) -- ({x(0):.3f},{y_axis:.3f});",
        f"  \\draw[thin] ({x(X_MIN):.3f},{y_axis:.3f}) -- ({x(X_MAX):.3f},{y_axis:.3f});",
    ]
    ticks = [-0.1, 0.0, 0.1, 0.2, 0.3]
    for t in ticks:
        L.append(f"  \\draw[thin] ({x(t):.3f},{y_axis:.3f}) -- ({x(t):.3f},{y_axis - 0.08:.3f}) "
                 f"node[below, inner sep=1.5pt] {{${t:+.1f}$}};".replace("+0.0", "0.0"))
    L.append(f"  \\node[anchor=north, inner sep=0pt] at ({x((X_MIN + X_MAX) / 2):.3f},{y_axis - 0.5:.3f}) "
             r"{Paired difference in $\rho$ (MPUP $-$ dense), 95\% CI};")
    L.append(f"  \\node[anchor=east, inner sep=2pt] at (0,{ROW * 0.6 + 0.15:.3f}) "
             r"{\textit{Held-out dataset (queries)}};")
    for i, (label, mean, lo, hi) in enumerate(rows):
        y = -(i + 0.5) * ROW
        marker = "fill=black" if lo > 0 else "fill=white"
        L.append(f"  \\node[anchor=east, inner sep=2pt] at (0,{y:.3f}) {{{label}}};")
        L.append(f"  \\draw[thick] ({x(lo):.3f},{y:.3f}) -- ({x(hi):.3f},{y:.3f});")
        L.append(f"  \\draw[{marker}, thick] ({x(mean):.3f},{y:.3f}) circle (0.07);")
    mean, lo, hi = pooled
    L.append(f"  \\draw[gray!70, thin] ({x(X_MIN):.3f},{y_pooled + ROW * 0.55:.3f}) -- ({x(X_MAX):.3f},{y_pooled + ROW * 0.55:.3f});")
    L.append(f"  \\node[anchor=east, inner sep=2pt] at (0,{y_pooled:.3f}) {{\\textbf{{All ({nested['n_queries']})}}}};")
    L.append(f"  \\draw[very thick] ({x(lo):.3f},{y_pooled:.3f}) -- ({x(hi):.3f},{y_pooled:.3f});")
    L.append(f"  \\draw[fill=black] ({x(mean) - 0.1:.3f},{y_pooled:.3f}) -- ({x(mean):.3f},{y_pooled + 0.1:.3f}) -- "
             f"({x(mean) + 0.1:.3f},{y_pooled:.3f}) -- ({x(mean):.3f},{y_pooled - 0.1:.3f}) -- cycle;")
    L.append(r"\end{tikzpicture}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT.relative_to(ROOT)}")
