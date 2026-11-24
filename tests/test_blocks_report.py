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
