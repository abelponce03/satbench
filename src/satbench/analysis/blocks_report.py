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


# ------------------------------------------------------------------ Amendment 1 (plan §6.2)


def host_ips(df: pd.DataFrame) -> pd.Series:
    """Instructions per CPU-second of each job's host, from its solo calibration runs."""
    cal = df[(df.kind == "calibration") & df.when.isin(["start", "end"])].dropna(subset=["instructions"])
    cal = cal[cal.cpu_time > 0]
    return (cal.instructions / cal.cpu_time).groupby(cal.job).median()


def instruction_budget(df: pd.DataFrame, timeout: float, safety: float = 0.95,
                       jobs: pd.Index | None = None) -> float:
    """B = safety × timeout × min IPS over (retained) hosts: a budget every host can reach."""
    ips = host_ips(df)
    if jobs is not None:
        ips = ips[ips.index.isin(jobs)]
    return float(safety * timeout * ips.min())


def instruction_par2(paired_runs: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Host-invariant outcome (§6.2.1): solved within B instructions, instruction-PAR-2.

    Expects one row per run with ``status`` and ``instructions``. Runs without a count
    (counter unavailable) get NaN and are dropped by the paired analyses.
    """
    r = paired_runs.copy()
    solved = r.status.isin(SOLVED) & (r.instructions <= budget)
    r["solved_b"] = solved
    r["ipar2"] = np.where(solved, r.instructions, 2 * budget)
    r.loc[r.instructions.isna(), "ipar2"] = np.nan
    return r


def horizon_curve(paired: pd.DataFrame, horizons: np.ndarray) -> pd.DataFrame:
    """r(T') per unit: PAR-2 recomputed at every horizon T' <= T from the same runs (§6.2.5).

    Needs ``cpu_time_a``/``cpu_time_b`` and ``solved_a``/``solved_b`` from :func:`paired_table`.
    """
    out = []
    orig = paired[paired.variant == "original"]
    for unit, d in orig.groupby("unit"):
        for h in horizons:
            pa = np.where(d.solved_a & (d.cpu_time_a <= h), d.cpu_time_a, 2 * h)
            pb = np.where(d.solved_b & (d.cpu_time_b <= h), d.cpu_time_b, 2 * h)
            diff = pa - pb
            out.append(dict(unit=unit, horizon=float(h), n=len(d),
                            r_rb=rank_biserial(diff, np.zeros_like(diff))))
    return pd.DataFrame(out)


def encoding_variance(runs: pd.DataFrame, timeout: float) -> pd.DataFrame:
    """Within- vs between-instance variance of log CPU time over order encodings (§6.2.2).

    Uses, per solver, instances run in the original and in >= 2 rcl encodings. One-way random
    effects ANOVA estimator of the ICC; also the within-instance SD (the "encoding noise floor")
    on the log scale. Times are capped at the timeout (censored runs enter at the cap, which
    underestimates within variance; reported as such).
    """
    r = runs[(runs.kind == "run") & (runs.variant.isin(["original"]) | runs.variant.str.endswith("-rcl"))]
    r = r.assign(logt=np.log(np.clip(r.cpu_time.astype(float), 0.01, timeout)))
    out = []
    for solver, d in r.groupby("solver"):
        g = d.groupby("origin_hash").logt
        k = g.size()
        d = d[d.origin_hash.isin(k[k >= 3].index)]
        if d.origin_hash.nunique() < 3:
            continue
        g = d.groupby("origin_hash").logt
        n_i, means = g.size(), g.mean()
        within = ((d.logt - d.origin_hash.map(means)) ** 2).sum() / (len(d) - len(n_i))
        n0 = (len(d) - (n_i**2).sum() / len(d)) / (len(n_i) - 1)
        between = max(((n_i * (means - d.logt.mean()) ** 2).sum() / (len(n_i) - 1) - within) / n0, 0.0)
        out.append(dict(solver=solver, instances=len(n_i), runs=len(d),
                        sd_within_log=float(np.sqrt(within)), sd_between_log=float(np.sqrt(between)),
                        icc=float(between / (between + within)) if between + within > 0 else np.nan))
    return pd.DataFrame(out)


def flip_rate(paired: pd.DataFrame) -> pd.DataFrame:
    """Share of instances whose paired winner (a vs b) differs between the original and an
    order-perturbed encoding: the 'encoding noise floor' of the unit's comparison (§6.2.2)."""
    w = paired.assign(winner=np.sign(paired.par2_a - paired.par2_b))
    piv = w.pivot_table(index=["unit", "origin_hash"], columns="variant", values="winner")
    rows = []
    for unit, d in piv.groupby(level="unit"):
        for v in [c for c in d.columns if c.endswith("-rcl")]:
            both = d[["original", v]].dropna()
            decided = both[(both.original != 0) | (both[v] != 0)]
            if len(decided):
                rows.append(dict(unit=unit, variant=v, n=len(decided),
                                 flip_rate=float((decided.original != decided[v]).mean())))
    return pd.DataFrame(rows)
