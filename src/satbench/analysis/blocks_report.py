"""Analysis of blocked-design results, implementing study/analysis_plan.md.

Input: the JSON-lines records written by ``satbench.blocks.run_job`` (kinds: run, calibration,
block). Every function takes the loaded DataFrame, so the same code runs on smoke tests and on
the full study.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .stats import holm, rank_biserial

SOLVED = ("SAT", "UNSAT")
SEEN = ("aesat_train", "comp2425")
UNSEEN = ("clean2026",)


def load(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line truncated by a killed job
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ host quality control


def host_qc(df: pd.DataFrame, tolerance: float = 0.25) -> pd.DataFrame:
    """Per job: CPU model, calibration time (start/end, both cores) and the exclusion flags of
    the plan (§4): minority CPU model; job calibration > tolerance from the median; drift
    start→end > tolerance."""
    cal = df[df.kind == "calibration"]
    g = cal.groupby(["job", "when"]).cpu_time.mean().unstack()
    q = pd.DataFrame({"cal_start": g.get("start"), "cal_end": g.get("end")})
    q["cal_mean"] = q[["cal_start", "cal_end"]].mean(axis=1)
    q["cpu_model"] = cal.groupby("job").host_cpu_model.first()
    majority = q.cpu_model.mode().iat[0] if len(q) else None
    med = q.cal_mean.median()
    q["minority_cpu"] = q.cpu_model != majority
    q["slow_or_fast_host"] = (q.cal_mean / med - 1).abs() > tolerance
    q["drift"] = (q.cal_end / q.cal_start - 1).abs() > tolerance
    q["exclude"] = q.minority_cpu | q.slow_or_fast_host | q.drift
    return q


# ------------------------------------------------------------------ paired outcomes


def paired_table(df: pd.DataFrame, timeout: float) -> pd.DataFrame:
    """One row per block: PAR-2 and solved status of member a (base) and b (AI variant)."""
    runs = df[df.kind == "run"].copy()
    runs["solved"] = runs.status.isin(SOLVED)
    runs["par2"] = np.where(runs.solved, runs.cpu_time, 2 * timeout)
    keep = ["block_id", "unit", "instance_set", "family", "variant", "origin_hash", "job"]
    wide = runs.pivot_table(index=keep, columns="member", values=["par2", "solved", "cpu_time"],
                            aggfunc="first")
    wide.columns = [f"{v}_{m}" for v, m in wide.columns]
    return wide.reset_index()


def _stratified_boot_r(d: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """95 % CI of the rank-biserial of (a - b), resampling instances within families."""
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in d.groupby("family")]
    diff = (d.par2_a - d.par2_b).to_numpy()
    pos = {i: k for k, i in enumerate(d.index)}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        x = diff[[pos[i] for i in idx]]
        vals[b] = rank_biserial(x, np.zeros_like(x))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def unit_set_effects(paired: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Per unit × instance set × variant: n, solved counts, Wilcoxon (Pratt), r_rb (positive =
    AI variant b better) with stratified bootstrap CI, McNemar exact p on solved status."""
    out = []
    for (unit, iset, variant), d in paired.groupby(["unit", "instance_set", "variant"]):
        d = d.reset_index(drop=True)
        diff = (d.par2_a - d.par2_b).to_numpy()
        if np.allclose(diff, 0):
            p_w = 1.0
        else:
            p_w = float(stats.wilcoxon(d.par2_a, d.par2_b, zero_method="pratt").pvalue)
        r = rank_biserial(diff, np.zeros_like(diff))
        lo, hi = _stratified_boot_r(d, n_boot, seed) if len(d) > 1 else (np.nan, np.nan)
        b_only = int((d.solved_b & ~d.solved_a).sum())
        a_only = int((d.solved_a & ~d.solved_b).sum())
        p_mc = float(stats.binomtest(b_only, a_only + b_only).pvalue) if a_only + b_only else 1.0
        out.append(dict(unit=unit, instance_set=iset, variant=variant, n=len(d),
                        solved_a=int(d.solved_a.sum()), solved_b=int(d.solved_b.sum()),
                        wilcoxon_p=p_w, r_rb=r, r_lo=lo, r_hi=hi,
                        only_a=a_only, only_b=b_only, mcnemar_p=p_mc))
    return pd.DataFrame(out)


def generalisation_gap(paired: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """H1: Δ = r(seen) − r(unseen) per unit (originals), one-sided bootstrap p, Holm-adjusted."""
    rng = np.random.default_rng(seed)
    orig = paired[paired.variant == "original"]
    rows = []
    for unit, d in orig.groupby("unit"):
        seen = d[d.instance_set.isin(SEEN)].reset_index(drop=True)
        unseen = d[d.instance_set.isin(UNSEEN)].reset_index(drop=True)
        if len(seen) < 2 or len(unseen) < 2:
            continue

        def r_of(x: pd.DataFrame) -> float:
            v = (x.par2_a - x.par2_b).to_numpy()
            return rank_biserial(v, np.zeros_like(v))

        delta = r_of(seen) - r_of(unseen)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            s = seen.groupby("family", group_keys=False).sample(frac=1, replace=True,
                                                                random_state=int(rng.integers(2**31)))
            u = unseen.groupby("family", group_keys=False).sample(frac=1, replace=True,
                                                                  random_state=int(rng.integers(2**31)))
            boots[b] = r_of(s) - r_of(u)
        rows.append(dict(unit=unit, n_seen=len(seen), n_unseen=len(unseen), delta=delta,
                         lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)),
                         p_one_sided=float((boots <= 0).mean())))
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm(res.p_one_sided.to_numpy())
    return res
