import pytest
import numpy as np
import pandas as pd

from satbench.analysis.blocks_report import generalisation_gap, host_qc, unit_set_effects


def synthetic(n=60, seed=0):
    """AI variant b is 30 % faster on seen instances and identical (noise only) on unseen ones."""
    rng = np.random.default_rng(seed)
    rows = []
    for iset, speedup in (("comp2425", 0.7), ("clean2026", 1.0)):
        for i in range(n):
            base = rng.uniform(10, 1000)
            rows.append(dict(block_id=f"{iset}{i}", unit="base|ai", instance_set=iset,
                             family=f"f{i % 5}", variant="original", origin_hash=f"{iset}{i}",
                             job=i, par2_a=base, par2_b=base * speedup * rng.uniform(0.95, 1.05),
                             solved_a=True, solved_b=True))
    return pd.DataFrame(rows)


def test_effects_sign_convention():
    eff = unit_set_effects(synthetic(), n_boot=200)
    seen = eff[eff.instance_set == "comp2425"].iloc[0]
    unseen = eff[eff.instance_set == "clean2026"].iloc[0]
    assert seen.r_rb > 0.9 and seen.wilcoxon_p < 1e-6  # positive = AI variant better
    assert abs(unseen.r_rb) < 0.4


def test_generalisation_gap_detected():
    gap = generalisation_gap(synthetic(), n_boot=300)
    row = gap.iloc[0]
    assert row.delta > 0.5 and row.lo > 0 and row.p_one_sided < 0.01


def test_host_qc_flags():
    cal = []
    for job, (model, t0, t1) in enumerate([("EPYC", 60, 61), ("EPYC", 59, 60), ("EPYC", 61, 90),
                                           ("Xeon", 60, 60), ("EPYC", 100, 101)]):
        for when, t in (("start", t0), ("end", t1)):
            cal.append(dict(kind="calibration", job=job, when=when, cpu_time=t, host_cpu_model=model))
    q = host_qc(pd.DataFrame(cal))
    assert list(q.exclude) == [False, False, True, True, True]  # drift, minority CPU, slow host


def test_encoding_variance_recovers_components():
    from satbench.analysis.blocks_report import encoding_variance

    rng = np.random.default_rng(1)
    rows = []
    for i in range(300):  # between-instance SD 1.0, within (encoding) SD 0.5, on the log scale
        mu = rng.normal(3.0, 1.0)
        for v in ("original", "s1-rcl", "s2-rcl", "s3-rcl"):
            rows.append(dict(kind="run", solver="x", origin_hash=f"h{i}", variant=v,
                             cpu_time=float(np.exp(mu + rng.normal(0, 0.5)))))
    ev = encoding_variance(pd.DataFrame(rows), timeout=1e9).iloc[0]
    assert abs(ev.sd_within_log - 0.5) < 0.05 and abs(ev.sd_between_log - 1.0) < 0.1
    assert abs(ev.icc - 0.8) < 0.05


def test_instruction_budget_and_horizons():
    from satbench.analysis.blocks_report import horizon_curve, instruction_budget, instruction_par2

    cal = pd.DataFrame([dict(kind="calibration", when="start", job=j, cpu_time=60.0,
                             instructions=ips * 60) for j, ips in enumerate([2e9, 3e9, 2.5e9])])
    assert instruction_budget(cal, timeout=100, safety=1.0) == pytest.approx(2e11)
    runs = pd.DataFrame(dict(status=["SAT", "SAT", "TIMEOUT"], instructions=[1e11, 3e11, 5e11]))
    ip = instruction_par2(runs, budget=2e11)
    assert list(ip.solved_b) == [True, False, False] and list(ip.ipar2) == [1e11, 4e11, 4e11]
    p = synthetic().assign(cpu_time_a=lambda d: d.par2_a, cpu_time_b=lambda d: d.par2_b)
    hc = horizon_curve(p, np.array([5.0, 2000.0]))
    short = hc[hc.horizon == 5.0].r_rb.iloc[0]
    long = hc[hc.horizon == 2000.0].r_rb.iloc[0]
    assert short == 0.0 and long > 0.3  # nothing solved within 5 s: no difference visible
