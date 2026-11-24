import numpy as np
import pytest

from satbench.analysis import stats as S
from satbench.analysis import survival as V


def brute_cliff(x, y):
    return np.mean([np.sign(a - b) for a in x for b in y])


@pytest.mark.parametrize("seed", range(5))
def test_cliffs_delta_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 20, 40).astype(float)  # many ties on purpose
    y = rng.integers(0, 20, 35).astype(float)
    assert S.cliffs_delta(x, y) == pytest.approx(brute_cliff(x, y))


def test_holm_known_values():
    adj = S.holm(np.array([0.01, 0.04, 0.03, 0.005]))
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_mde_and_cliff_conversion():
    assert S.mde_paired(109) == pytest.approx(0.2683431962195634)
    assert S.d_to_cliff(0.2683431962195634) == pytest.approx(0.1330, abs=1e-4)


def test_km_and_rmst_without_censoring_equal_mean():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.ones(4, bool)
    ts, s = V.kaplan_meier(t, e)
    assert s == pytest.approx([0.75, 0.5, 0.25, 0.0])
    assert V.rmst(t, e, tau=10) == pytest.approx(t.mean())


def test_logrank_identical_groups_is_null():
    t = np.array([1, 2, 3, 4, 5.0])
    e = np.array([1, 1, 0, 1, 1], bool)
    chi2, p = V.logrank(t, e, t, e)
    assert chi2 == pytest.approx(0.0) and p == pytest.approx(1.0)
