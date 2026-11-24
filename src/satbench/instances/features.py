"""Structural features of CNF instances using the official GBD extractor (``gbdc``).

The standard GBD base features (clause-size histogram, Horn fractions, variable/clause-graph
degree statistics, ...) replace the 22 ad-hoc features of the thesis: they are defined and
versioned by GBD, so reviewers can look up or recompute every column.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def _extract(path: str) -> dict:
    import gbdc

    try:
        feats = dict(gbdc.extract_base_features(path))
        feats["feature_error"] = ""
    except Exception as exc:  # reported per instance, never silently dropped
        feats = {"feature_error": f"{type(exc).__name__}: {exc}"}
    feats["path"] = path
    return feats


def extract_all(paths: list[Path], workers: int = 4) -> pd.DataFrame:
    from importlib.metadata import version

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_extract, [str(p) for p in paths], chunksize=4))
    df = pd.DataFrame(rows)
    df["gbdc_version"] = version("gbdc")
    return df
