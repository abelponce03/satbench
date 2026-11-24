"""Access to the Global Benchmark Database (GBD) metadata (Iser & Jabs, SAT 2024).

The ``meta`` database is a single SQLite file published at ``/getdatabase/meta``. It maps each
instance hash to its family, author, competition tracks, known result and original filename. A
snapshot is downloaded once and its SHA-256 recorded, so instance selection is reproducible even if
the live database changes later.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GBD_URL = "https://benchmark-database.de"
USER_AGENT = "satbench/0.1 (academic benchmarking study)"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def http_get(url: str, dest: Path, timeout: float = 300.0) -> None:
    """Download ``url`` to ``dest`` atomically (write to ``.part`` then rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    tmp.replace(dest)


def fetch_meta(dest_dir: Path) -> Path:
    """Download a snapshot of the GBD meta database and write a provenance record next to it."""
    db = dest_dir / "meta.db"
    http_get(f"{GBD_URL}/getdatabase/meta", db)
    record = {
        "url": f"{GBD_URL}/getdatabase/meta",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_file(db),
        "bytes": db.stat().st_size,
    }
    (dest_dir / "meta.provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return db


_YEAR = re.compile(r"_(\d{4})$")


@dataclass(frozen=True)
class MetaDB:
    path: Path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def families(self) -> pd.Series:
        with self._connect() as c:
            df = pd.read_sql("select family, count(*) n from features group by family", c)
        return df.set_index("family")["n"].sort_values(ascending=False)

    def instances(self, families: list[str]) -> pd.DataFrame:
        """One row per instance of the given families, with its tracks and first/last track year.

        Columns: hash, family, author, result (GBD-known answer or 'unknown'), filename, tracks
        (sorted, ';'-joined), first_year, last_year.
        """
        marks = ",".join("?" * len(families))
        with self._connect() as c:
            base = pd.read_sql(
                f"select f.hash, f.family, f.author, f.result, n.value as filename "
                f"from features f left join filename n on n.hash = f.hash "
                f"where f.family in ({marks})",
                c,
                params=families,
            )
            tracks = pd.read_sql(
                f"select t.hash, t.value as track from track t join features f on f.hash = t.hash "
                f"where f.family in ({marks}) and t.value != 'None'",
                c,
                params=families,
            )
        base = base.drop_duplicates("hash")
        tracks["year"] = tracks["track"].str.extract(_YEAR, expand=False).astype("Int64")
        agg = tracks.groupby("hash").agg(
            tracks=("track", lambda s: ";".join(sorted(set(s)))),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        out = base.merge(agg, left_on="hash", right_index=True, how="left")
        out["result"] = out["result"].replace({"None": "unknown"}).fillna("unknown")
        return out.sort_values(["family", "hash"]).reset_index(drop=True)
