"""Right-censored runtime analysis: Kaplan–Meier, log-rank and RMST.

Timeouts are right-censored at the time limit. In SAT the censoring is *informative* (whether
a run times out depends on instance hardness), so these are descriptive tools, not unbiased
population estimates.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the event times and S(t) just after each of them."""
    time, event = np.asarray(time, float), np.asarray(event, bool)
    ts = np.unique(time[event])
    surv, s = np.empty(len(ts)), 1.0
    for i, t in enumerate(ts):
        at_risk = np.sum(time >= t)
        deaths = np.sum((time == t) & event)
        s *= 1.0 - deaths / at_risk
        surv[i] = s
    return ts, surv


def rmst(time: np.ndarray, event: np.ndarray, tau: float) -> float:
    """Restricted mean survival time: the area under the KM curve on [0, tau]."""
    ts, surv = kaplan_meier(time, event)
    keep = ts <= tau
    grid = np.concatenate([[0.0], ts[keep], [tau]])
    levels = np.concatenate([[1.0], surv[keep]])
    return float(np.sum(levels * np.diff(grid)))


def logrank(
    time_a: np.ndarray, event_a: np.ndarray, time_b: np.ndarray, event_b: np.ndarray
) -> tuple[float, float]:
    """Two-sample log-rank test; returns (chi2, p)."""
    time = np.concatenate([time_a, time_b]).astype(float)
    event = np.concatenate([event_a, event_b]).astype(bool)
    group = np.concatenate([np.zeros(len(time_a)), np.ones(len(time_b))]).astype(bool)
    obs_minus_exp, var = 0.0, 0.0
    for t in np.unique(time[event]):
        at_risk = time >= t
        n, n1 = at_risk.sum(), (at_risk & ~group).sum()
        d = ((time == t) & event).sum()
        d1 = ((time == t) & event & ~group).sum()
        obs_minus_exp += d1 - d * n1 / n
        if n > 1:
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if var == 0:
        return 0.0, 1.0
    chi2 = obs_minus_exp**2 / var
    return float(chi2), float(stats.chi2.sf(chi2, 1))
