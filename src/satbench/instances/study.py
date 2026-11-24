"""Instance sets of the study, defined by *exposure*: what solver developers could have seen.

* ``clean2026``: instances that first appeared in 2026 (submitted together with the solvers, so
  no developer could tune on them). Instances isomorphic (equal GBD ``isohash``) to any earlier
  instance are excluded: a renamed duplicate is not new.
* ``aesat_train``: the 60 instances AESAT was trained on (exact list from its repository).
* ``comp2425``: other instances of the SAT Competition 2024/2025 main tracks. These are the
  development data of most AI-tuned solvers of 2025/2026.
* ``legacy``: the thesis' industrial families, last used in a competition before 2024.

Sampling within each set is by keyed hash rank (see ``select.rank_key``), capped per family so
that no single family dominates a set.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .gbd import MetaDB
from .select import rank_key


def _isohashes(db: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        return pd.read_sql("select hash, isohash from features", c).drop_duplicates("hash")


def _cap(df: pd.DataFrame, n: int, per_family: int, seed: int) -> pd.DataFrame:
    df = df.assign(rank=[rank_key(seed, h) for h in df.hash]).sort_values("rank")
    df = df.groupby("family", group_keys=False).head(per_family)
    return df.head(n)


def select_study(meta_db: Path, cfg: dict, aesat_train: list[str], seed: int) -> pd.DataFrame:
    meta = MetaDB(meta_db)
    allf = meta.families().index.tolist()
    inst = meta.instances(allf).merge(_isohashes(meta_db), on="hash", how="left")
    tracks = inst.tracks.fillna("")
    first = inst.first_year.fillna(0).astype(int)

    old_iso = set(inst.loc[(first > 0) & (first < 2026), "isohash"].dropna())
    clean = inst[(first == 2026) & ~inst.isohash.isin(old_iso)]
    train = inst[inst.hash.isin(set(aesat_train))]
    comp = inst[tracks.str.contains("main_2024|main_2025") & ~inst.hash.isin(set(aesat_train))
                & (first < 2026)]
    legacy_fams = cfg["legacy"]["families"]
    legacy = inst[inst.family.isin(legacy_fams) & (inst.last_year.fillna(0).astype(int) < 2024)
                  & (first > 0)]

    sets = [
        ("clean2026", _cap(clean, cfg["clean2026"]["n"], cfg["clean2026"]["per_family"], seed)),
        ("aesat_train", train),
        ("comp2425", _cap(comp, cfg["comp2425"]["n"], cfg["comp2425"]["per_family"], seed)),
        ("legacy", _cap(legacy, cfg["legacy"]["n"], cfg["legacy"]["per_family"], seed)),
    ]
    out = pd.concat([df.assign(instance_set=name) for name, df in sets], ignore_index=True)
    out["perturb"] = out.instance_set.map(lambda s: cfg[s].get("perturb", True))
    out.attrs["excluded_isomorphic_2026"] = int(((first == 2026) & inst.isohash.isin(old_iso)).sum())
    return out.drop_duplicates("hash")[
        ["hash", "family", "result", "tracks", "first_year", "last_year", "isohash",
         "instance_set", "perturb"]
    ]
