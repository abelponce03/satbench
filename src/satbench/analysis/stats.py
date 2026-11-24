"""Nonparametric comparison of solvers on paired per-instance data.

Conventions (kept identical to the thesis so its numbers can be reproduced):

* A *run* is one (solver, instance, seed) execution. PAR-2 of a run is its time when it
  solved the instance (SAT/UNSAT) and ``2 * timeout`` otherwise.
* Paired tests operate on one value per (solver, instance): the **median PAR-2 across seeds**.
* Pairs are reported in a fixed order ``(A, B)``; Cliff's delta ``delta_AB`` is
  ``P(X_A > X_B) - P(X_A < X_B)``. Lower PAR-2 is better, so ``delta_AB < 0`` favours ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

SOLVED = ("SAT", "UNSAT")


def par2(
    result: pd.Series, time: pd.Series, timeout: float, legacy_unknown: bool = False
) -> pd.Series:
    """PAR-2 per run: the runtime if solved, ``2 * timeout`` otherwise.

    Any run without a verified answer (timeout, memory-out, crash: ``UNKNOWN``) is penalised.
    ``legacy_unknown=True`` credits ``UNKNOWN`` runs with their observed time instead, as an
    earlier pipeline did; it exists only to reproduce that pipeline's numbers.
    """
    credited = result.isin(SOLVED) | (legacy_unknown & result.eq("UNKNOWN"))
    return pd.Series(np.where(credited, time, 2.0 * timeout), index=result.index)


def instance_matrix(
    runs: pd.DataFrame,
    timeout: float,
    agg: str = "median",
    complete: bool = True,
    legacy_unknown: bool = False,
) -> pd.DataFrame:
    """Instances x solvers matrix of seed-aggregated PAR-2.

    ``runs`` needs the columns ``solver``, ``instance``, ``result`` and ``time``. With
    ``complete=True`` only the instances with a value for every solver are kept (paired design).
    """
    runs = runs.assign(par2=par2(runs["result"], runs["time"], timeout, legacy_unknown))
    m = runs.groupby(["instance", "solver"])["par2"].agg(agg).unstack("solver")
    return m.dropna() if complete else m


# --------------------------------------------------------------------------- omnibus


@dataclass(frozen=True)
class FriedmanResult:
    chi2: float
    p_chi2: float
    F_ID: float
    p_ID: float
    df1: int
    df2: int
    n_instances: int


def friedman_iman_davenport(matrix: pd.DataFrame) -> FriedmanResult:
    """Friedman test with the Iman–Davenport F correction (Demšar 2006, eq. 2)."""
    n, k = matrix.shape
    chi2, p_chi2 = stats.friedmanchisquare(*(matrix[c].to_numpy() for c in matrix.columns))
    denom = n * (k - 1) - chi2
    F = (n - 1) * chi2 / denom if denom > 0 else np.inf
    df1, df2 = k - 1, (k - 1) * (n - 1)
    p_id = float(stats.f.sf(F, df1, df2)) if np.isfinite(F) else 0.0
    return FriedmanResult(float(chi2), float(p_chi2), float(F), p_id, df1, df2, n)


# --------------------------------------------------------------------------- effect sizes


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta P(X > Y) - P(X < Y) over all cross pairs, in O((n+m) log(n+m))."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ys = np.sort(y)
    greater = np.searchsorted(ys, x, side="left").sum()  # y < x
    less = (len(ys) - np.searchsorted(ys, x, side="right")).sum()  # y > x
    return float((greater - less) / (len(x) * len(y)))


def cliff_magnitude(delta: float) -> str:
    """Romano et al. (2006) thresholds."""
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def bootstrap_cliff_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for Cliff's delta, resampling *instances* (pairs kept together)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = cliffs_delta(x[idx], y[idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of x - y (zero differences dropped)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


# --------------------------------------------------------------------------- pairwise


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_wilcoxon(
    matrix: pd.DataFrame,
    order: list[str] | None = None,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    """All pairs: Wilcoxon signed-rank + Holm, Cliff's delta with bootstrap CI.

    ``zero_method`` is passed to :func:`scipy.stats.wilcoxon`. The thesis used ``"zsplit"``
    (zero differences split between both rank sums), which matters because many instances time
    out for both solvers and give exact ties.
    """
    solvers = order or list(matrix.columns)
    rows = []
    for a, b in combinations(solvers, 2):
        xa, xb = matrix[a].to_numpy(), matrix[b].to_numpy()
        if np.allclose(xa, xb):
            W, p = np.nan, 1.0
        else:
            res = stats.wilcoxon(xa, xb, zero_method=zero_method)
            W, p = float(res.statistic), float(res.pvalue)
        delta = cliffs_delta(xa, xb)
        lo, hi = bootstrap_cliff_ci(xa, xb, n_boot=n_boot, seed=seed)
        rows.append(
            dict(
                solver_a=a, solver_b=b, W=W, p_raw=p, r_rb=rank_biserial(xa, xb),
                cliff_delta=delta, cliff_ci_lo=lo, cliff_ci_hi=hi,
                cliff_magnitude=cliff_magnitude(delta),
            )
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    out["significant_holm"] = out["p_holm"] < alpha
    return out


def kw_scores(pairwise: pd.DataFrame, matrix: pd.DataFrame) -> pd.Series:
    """Pairwise-tournament score used in the thesis as the 'Kendall–Wei' ranking.

    For every pair: a significant (Holm) difference gives 1 point to the solver with the lower
    median PAR-2 and 0 to the other; a non-significant pair gives 0.5 to each. With k solvers the
    scores sum to k(k-1)/2.
    """
    score = {s: 0.0 for s in matrix.columns}
    med = matrix.median()
    for row in pairwise.itertuples():
        a, b = row.solver_a, row.solver_b
        if row.significant_holm:
            better = a if row.cliff_delta < 0 or (row.cliff_delta == 0 and med[a] < med[b]) else b
            score[better] += 1.0
        else:
            score[a] += 0.5
            score[b] += 0.5
    return pd.Series(score).sort_values(ascending=False)


def bootstrap_ranking(
    matrix: pd.DataFrame, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap over instances of the tournament score: mean, 95 % CI and P(rank 1)."""
    rng = np.random.default_rng(seed)
    n = len(matrix)
    draws = []
    for _ in range(n_boot):
        sample = matrix.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        pw = pairwise_wilcoxon(sample, alpha=alpha, n_boot=1)
        draws.append(kw_scores(pw, sample))
    df = pd.DataFrame(draws)
    top = df.eq(df.max(axis=1), axis=0)
    return pd.DataFrame(
        dict(
            mean=df.mean(), ci_lo=df.quantile(0.025), ci_hi=df.quantile(0.975),
            p_top=top.mean(),
        )
    ).sort_values("mean", ascending=False)


# --------------------------------------------------------------------------- power


def mde_paired(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable standardised effect d for a two-sided paired t-test."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n))


def d_to_cliff(d: float) -> float:
    """Approximate conversion of Cohen's d to the Cliff's-delta scale used in the thesis."""
    return float(d / np.sqrt(d * d + 4))
