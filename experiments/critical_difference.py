"""Critical-difference (CD) comparison of several methods over many blocks, after Demsar (2006).

  python -m experiments.critical_difference --scores per_query_rs.npz --out paper/fig_cd.tex

`--scores` is an .npz with one array per method: the within-query Spearman r_s of the method on each test
query, the same queries in the same order for every method. Queries are the blocks. The Friedman test says whether
the methods differ at all; the Nemenyi critical difference CD = q_alpha * sqrt(k (k + 1) / (6 N)) says how far apart
two average ranks must be to differ at level alpha. Methods whose average ranks differ by less than CD are joined
by a bar in the diagram. A published number for another method cannot be placed in the diagram: every method has
to be run on the same queries, so a method of another paper first has to be re-run under this protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

# Upper critical values of the studentised range / sqrt(2) (Demsar 2006, Table 5), for k = 2 ... 10 methods
Q = {
    0.05: [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
}


def average_ranks(scores: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean rank of every method over the blocks; rank 1 is the highest score, ties share the mean rank."""
    names = list(scores)
    M = np.stack([np.asarray(scores[n], float) for n in names], axis=1)  # (blocks, methods)
    ranks = stats.rankdata(-M, axis=1)
    return {n: float(r) for n, r in zip(names, ranks.mean(axis=0))}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    if not 2 <= k <= 10:
        raise ValueError(f"no critical value for {k} methods (2 to 10 are supported)")
    return Q[alpha][k - 2] * float(np.sqrt(k * (k + 1) / (6 * n)))


def friedman_pvalue(scores: dict[str, np.ndarray]) -> float:
    return float(stats.friedmanchisquare(*[np.asarray(v, float) for v in scores.values()]).pvalue)


def cliques(ranks: dict[str, float], cd: float) -> list[list[str]]:
    """Maximal groups of two or more methods whose average ranks all lie within `cd` of each other."""
    order = sorted(ranks, key=ranks.get)
    groups = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and ranks[order[j + 1]] - ranks[order[i]] < cd:
            j += 1
        if j > i:
            groups.append(order[i : j + 1])
    return [g for g in groups if not any(g != h and set(g) <= set(h) for h in groups)]


def to_tikz(ranks: dict[str, float], cd: float, width_cm: float = 8.0) -> str:
    """Horizontal rank axis, best method on the left; names on alternating sides; bars join the cliques."""
    k = len(ranks)
    lo, hi = 1.0, float(k)
    x = lambda r: (r - lo) / (hi - lo) * width_cm  # noqa: E731
    L = [r"\begin{tikzpicture}[x=1cm, y=1cm, font=\scriptsize]",
         f"  \\draw[thick] ({x(lo):.3f},0) -- ({x(hi):.3f},0);"]
    for t in range(1, k + 1):
        L.append(f"  \\draw ({x(t):.3f},0) -- ({x(t):.3f},0.1) node[above, inner sep=1pt] {{{t}}};")
    order = sorted(ranks, key=ranks.get)
    for i, name in enumerate(order):
        depth = -(0.55 + 0.45 * (i // 2))
        side = "east" if i % 2 == 0 else "west"
        off = -0.15 if i % 2 == 0 else 0.15
        L.append(f"  \\draw ({x(ranks[name]):.3f},0) -- ({x(ranks[name]):.3f},{depth:.3f}) -- ({x(ranks[name]) + off:.3f},{depth:.3f}) "
                 f"node[anchor={side}, inner sep=1pt] {{{name} ({ranks[name]:.2f})}};")
    for g, grp in enumerate(cliques(ranks, cd)):
        y = 0.42 + 0.14 * g  # bars above the axis, clear of the method names below it
        L.append(f"  \\draw[very thick] ({x(ranks[grp[0]]) - 0.03:.3f},{y:.3f}) -- ({x(ranks[grp[-1]]) + 0.03:.3f},{y:.3f});")
    L.append(f"  \\draw[<->] ({x(lo):.3f},1.0) -- ({x(lo) + x(lo + cd):.3f},1.0) node[right, inner sep=3pt] {{CD = {cd:.2f}}};")
    L.append(r"\end{tikzpicture}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help=".npz, one array of per-query r_s per method")
    ap.add_argument("--out", default="paper/fig_cd.tex")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    z = np.load(args.scores)
    scores = {k: z[k] for k in z.files}
    n = len(next(iter(scores.values())))
    ranks, cd = average_ranks(scores), nemenyi_cd(len(scores), n, args.alpha)
    print(f"{len(scores)} methods, {n} blocks, Friedman p = {friedman_pvalue(scores):.3g}, CD = {cd:.3f}")
    for name in sorted(ranks, key=ranks.get):
        print(f"  {name:20s} average rank {ranks[name]:.3f}")
    print("not significantly different:", cliques(ranks, cd) or "none")
    Path(args.out).write_text(to_tikz(ranks, cd))


if __name__ == "__main__":
    main()
