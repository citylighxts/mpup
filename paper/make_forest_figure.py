"""Write paper/fig_forest.tex: MPUP minus dense similarity, per test dataset, as a TikZ forest plot.

  python paper/make_forest_figure.py

Reads the single evaluation of the larger study (results/big_final.json, trained on the development questions
of all datasets), so the figure matches Table VII (`tab:big`) of the paper. Per dataset, the circle is the mean
paired difference in within-query Spearman correlation r_s between MPUP and dense similarity and the line is its 95% t
interval over the test queries (mean +- t * sd / sqrt(n)); a filled circle means the interval excludes zero.
The open square is the same difference for MPUP without the additions (the 14 features), a point without an
interval because only per-dataset means are stored for it. The last row is the difference over all rankable test
queries together with its 95% percentile bootstrap interval.
"""
import json
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "big_final.json"
OUT = Path(__file__).resolve().parent / "fig_forest.tex"

NAMES = {
    "2wikimultihopqa": "2WikiMultiHopQA", "ambigqa": "AmbigQA", "entityquestions": "EntityQuestions",
    "hotpotqa": "HotpotQA", "musique": "MuSiQue", "nq": "NQ", "popqa": "PopQA", "squad": "SQuAD",
    "triviaqa": "TriviaQA", "webquestions": "WebQuestions",
}
X_MIN, X_MAX, X_PER_UNIT = -0.05, 0.30, 15.0  # data units, and cm per 1.0 of rho difference (1.5 cm per 0.1)
ROW = 0.38  # cm between rows


def rows_and_pooled(result: dict):
    same = result["same_datasets"]
    tests = same["per_dataset_tests"]["dense"]
    rows = []
    for key, t in tests.items():
        n, mean, sd = t["n"], t["mean_diff"], t["sd"]
        half = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5
        sp = same["per_dataset"][key]["spearman"]
        # the stored per-dataset means must agree with the paired differences, or the plot would mix two things
        assert abs((sp["mpup+"] - sp["dense"]) - mean) < 1e-6 if "mpup+" in sp else True
        old = sp["mpup (paper)"] - sp["dense"]
        rows.append((f"{NAMES[key]} ({n})", mean, mean - half, mean + half, old))
    rows.sort(key=lambda r: -r[1])
    return rows, same["paired_ci"]["dense"], same["n_queries"]


def build() -> str:
    rows, pooled, n_all = rows_and_pooled(json.loads(SOURCE.read_text()))
    x = lambda v: (v - X_MIN) * X_PER_UNIT  # noqa: E731
    n = len(rows)
    y_pooled = -(n + 0.6) * ROW
    y_axis = y_pooled - 0.7 * ROW
    L = [
        r"\begin{tikzpicture}[x=1cm, y=1cm, font=\scriptsize, >=Stealth]",
        f"  \\draw[gray!70, thin] ({x(0):.3f},{ROW * 0.6:.3f}) -- ({x(0):.3f},{y_axis:.3f});",
        f"  \\draw[thin] ({x(X_MIN):.3f},{y_axis:.3f}) -- ({x(X_MAX):.3f},{y_axis:.3f});",
    ]
    for t in [0.0, 0.1, 0.2, 0.3]:
        L.append(f"  \\draw[thin] ({x(t):.3f},{y_axis:.3f}) -- ({x(t):.3f},{y_axis - 0.08:.3f}) "
                 f"node[below, inner sep=1.5pt] {{${t:+.1f}$}};".replace("+0.0", "0.0"))
    L.append(f"  \\node[anchor=north, inner sep=0pt] at ({x((X_MIN + X_MAX) / 2):.3f},{y_axis - 0.5:.3f}) "
             r"{Paired difference in $r_s$ (MPUP $-$ dense), 95\% CI};")
    L.append(f"  \\node[anchor=east, inner sep=2pt] at (0,{ROW * 0.6 + 0.15:.3f}) "
             r"{\textit{Test dataset (queries)}};")
    for i, (label, mean, lo, hi, old) in enumerate(rows):
        y = -(i + 0.5) * ROW
        marker = "fill=black" if lo > 0 else "fill=white"
        L.append(f"  \\node[anchor=east, inner sep=2pt] at (0,{y:.3f}) {{{label}}};")
        L.append(f"  \\draw[thick] ({x(lo):.3f},{y:.3f}) -- ({x(hi):.3f},{y:.3f});")
        L.append(f"  \\draw[{marker}, thick] ({x(mean):.3f},{y:.3f}) circle (0.07);")
        L.append(f"  \\draw[fill=white, thick] ({x(old) - 0.06:.3f},{y - 0.06:.3f}) rectangle ({x(old) + 0.06:.3f},{y + 0.06:.3f});")
    mean, lo, hi = pooled
    L.append(f"  \\draw[gray!70, thin] ({x(X_MIN):.3f},{y_pooled + ROW * 0.55:.3f}) -- ({x(X_MAX):.3f},{y_pooled + ROW * 0.55:.3f});")
    L.append(f"  \\node[anchor=east, inner sep=2pt] at (0,{y_pooled:.3f}) {{\\textbf{{All ({n_all})}}}};")
    L.append(f"  \\draw[very thick] ({x(lo):.3f},{y_pooled:.3f}) -- ({x(hi):.3f},{y_pooled:.3f});")
    L.append(f"  \\draw[fill=black] ({x(mean) - 0.1:.3f},{y_pooled:.3f}) -- ({x(mean):.3f},{y_pooled + 0.1:.3f}) -- "
             f"({x(mean) + 0.1:.3f},{y_pooled:.3f}) -- ({x(mean):.3f},{y_pooled - 0.1:.3f}) -- cycle;")
    L.append(r"\end{tikzpicture}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    rows, pooled, n_all = rows_and_pooled(json.loads(SOURCE.read_text()))
    print("datasets:", len(rows), "| all intervals above zero:", all(r[2] > 0 for r in rows), "| pooled:", [round(v, 3) for v in pooled], "| n:", n_all)
    OUT.write_text(build())
    print(f"wrote {OUT.relative_to(ROOT)}")
