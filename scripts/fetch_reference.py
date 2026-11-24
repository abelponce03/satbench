"""Download the reference documents and data the study relies on into data/reference/.

Every file is stored with its URL, retrieval time and SHA-256 in data/reference/PROVENANCE.json;
a copy of that record is kept in study/reference-provenance.json (committed). Files already present
with a recorded hash are not downloaded again.

Usage: uv run python scripts/fetch_reference.py
"""

import hashlib
import json
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "reference"
UA = "satbench/0.1 (academic benchmarking study)"

FILES = {
    "sc2026_scores.csv": "https://satcompetition.github.io/2026/downloads/scores.csv",
    "sc2026_slides.pdf": "https://satcompetition.github.io/2026/downloads/satcomp26slides.pdf",
    "sc2026_proceedings.pdf": "https://repositum.tuwien.at/bitstream/20.500.12708/232006/2/"
    "Fazekas-2026-Proceedings%20of%20SAT%20Competition%202026%20%20Solver%20and%20Benchmark%20De...-vor.pdf",
    "sc2026_track_main.uri": "https://satcompetition.github.io/2026/downloads/track_main_2026.uri",
    "sc2025_slides.pdf": "https://satcompetition.github.io/2025/satcomp25slides.pdf",
    "aesat_ijcai2026.pdf": "https://www.ijcai.org/proceedings/2026/0257.pdf",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    prov_file = DEST / "PROVENANCE.json"
    prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
    for name, url in FILES.items():
        dst = DEST / name
        if dst.exists() and prov.get(name, {}).get("sha256") == sha256(dst):
            print(f"have {name}")
            continue
        if dst.exists() and name not in prov:  # complete file from an interrupted earlier run
            pass
        else:
            download(url, dst)
        prov[name] = dict(url=url, sha256=sha256(dst), bytes=dst.stat().st_size,
                          retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        print(f"got  {name}  {prov[name]['sha256'][:16]}", flush=True)
        save(prov, prov_file)  # after every file: an interruption loses nothing already fetched


def save(prov: dict, prov_file: Path) -> None:
    text = json.dumps(prov, indent=2) + "\n"
    prov_file.write_text(text)
    (ROOT / "study" / "reference-provenance.json").write_text(text)


def download(url: str, dst: Path, attempts: int = 8) -> None:
    """Atomic download with exponential back-off (the link drops for minutes at a time)."""
    tmp = dst.with_name(dst.name + ".part")
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dst)
            return
        except OSError as exc:
            wait = min(30 * 2**k, 600)
            print(f"  retry {k + 1}/{attempts} for {dst.name} in {wait}s ({exc})", flush=True)
            time.sleep(wait)
    raise SystemExit(f"could not download {url}")


if __name__ == "__main__":
    main()
