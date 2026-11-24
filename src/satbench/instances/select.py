"""Deterministic instance selection and train/test partition.

Sampling is driven by a keyed hash rank, ``sha256(f"{seed}:{instance_hash}")``, rather than by a
random generator over a list. The selection therefore does not depend on database row order or on
the library version, and adding new instances to GBD never reshuffles the ones already chosen.

Each instance also gets an ``era`` label: ``recent`` if it appeared in a competition track from
``cutoff_year`` on, ``legacy`` otherwise. For the study of automatically evolved solvers this
separates the competition distribution their heuristics were optimised on from the rest of the
industrial families.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def rank_key(seed: int, h: str) -> str:
    return hashlib.sha256(f"{seed}:{h}".encode()).hexdigest()


def label_era(instances: pd.DataFrame, cutoff_year: int) -> pd.Series:
    recent = instances["last_year"].fillna(0).astype(int) >= cutoff_year
    return recent.map({True: "recent", False: "legacy"})


def sample(
    instances: pd.DataFrame,
    per_family: int,
    seed: int,
    cutoff_year: int,
    min_recent_share: float = 0.0,
) -> pd.DataFrame:
    """Pick ``per_family`` instances per family by hash rank.

    ``min_recent_share`` reserves at least that share of each family's quota for ``recent``
    instances when enough exist, so both eras are represented in every family.
    """
    df = instances.assign(era=label_era(instances, cutoff_year))
    df["rank"] = [rank_key(seed, h) for h in df["hash"]]
    picked = []
    for _, fam in df.groupby("family"):
        fam = fam.sort_values("rank")
        n_recent = min(int(round(per_family * min_recent_share)), (fam.era == "recent").sum())
        recent = fam[fam.era == "recent"].head(n_recent)
        rest = fam.drop(recent.index).head(per_family - len(recent))
        picked.append(pd.concat([recent, rest]))
    return pd.concat(picked).sort_values(["family", "rank"]).reset_index(drop=True)


def split_train_test(selection: pd.DataFrame, test_share: float, seed: int) -> pd.Series:
    """Stratified (family x era) split; returns a 'train'/'test' label per row.

    The ``seed`` must differ from the sampling seed, so the split is not simply the first half of the
    sampling order.
    """
    labels = pd.Series(index=selection.index, dtype=object)
    for _, g in selection.groupby(["family", "era"]):
        order = sorted(g.index, key=lambda i: rank_key(seed, selection.at[i, "hash"]))
        n_test = int(round(len(order) * test_share))
        for j, i in enumerate(order):
            labels[i] = "test" if j < n_test else "train"
    return labels
