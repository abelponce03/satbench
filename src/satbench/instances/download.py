"""Download GBD instances and verify every file before it can enter an experiment.

A file is accepted only if (1) its xz stream decompresses completely and (2) the GBD hash of its
content equals the hash it was requested under. A damaged or truncated download is
re-fetched or reported, never silently run.
"""

from __future__ import annotations

import lzma
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .gbd import GBD_URL, http_get, sha256_file


@dataclass
class DownloadRecord:
    hash: str
    family: str
    path: str
    status: str  # ok | corrupt | hash_mismatch | error
    sha256: str = ""
    bytes: int = 0
    attempts: int = 0
    detail: str = ""


def xz_is_intact(path: Path) -> bool:
    try:
        with lzma.open(path) as f:
            while f.read(1 << 22):
                pass
        return True
    except (lzma.LZMAError, EOFError, OSError):
        return False


def gbd_hash(path: Path) -> str:
    import gbdc  # official GBD implementation; imported lazily (C extension)

    return gbdc.gbdhash(str(path))


def instance_path(root: Path, family: str, h: str) -> Path:
    return root / family / f"{h}.cnf.xz"


def verify(path: Path, expected_hash: str) -> tuple[str, str]:
    if not xz_is_intact(path):
        return "corrupt", "xz stream does not decompress"
    got = gbd_hash(path)
    if got != expected_hash:
        return "hash_mismatch", f"gbdhash={got}"
    return "ok", ""


def fetch_one(root: Path, h: str, family: str, retries: int = 8) -> DownloadRecord:
    dest = instance_path(root, family, h)
    rec = DownloadRecord(hash=h, family=family, path=str(dest), status="error")
    if dest.exists():
        status, detail = verify(dest, h)
        if status == "ok":
            rec.status, rec.attempts = "ok", 0
        else:
            dest.unlink()
    while rec.status != "ok" and rec.attempts < retries:
        rec.attempts += 1
        try:
            http_get(f"{GBD_URL}/file/{h}?context=cnf", dest)
            rec.status, rec.detail = verify(dest, h)
            if rec.status != "ok":
                dest.unlink(missing_ok=True)
        except Exception as exc:  # network errors are retried, then reported
            rec.status, rec.detail = "error", f"{type(exc).__name__}: {exc}"
        if rec.status != "ok":
            time.sleep(min(30.0 * 2 ** (rec.attempts - 1), 600.0))  # link drops for minutes
    if rec.status == "ok":
        rec.sha256, rec.bytes = sha256_file(dest), dest.stat().st_size
    return rec


def fetch_all(
    selection: pd.DataFrame, root: Path, manifest: Path, workers: int = 4
) -> pd.DataFrame:
    """Download every (hash, family) of ``selection``; write and return the manifest.

    A small worker pool keeps the load on the public GBD server polite. The manifest records the
    SHA-256 of every accepted file, so the exact bytes used can be audited later.
    """
    records: list[DownloadRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_one, root, r.hash, r.family) for r in selection.itertuples()
        ]
        for fut in as_completed(futures):
            records.append(fut.result())
    df = pd.DataFrame([asdict(r) for r in records]).sort_values(["family", "hash"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    return df
