"""Download (verified) and perturb the instances of some jobs into a persistent cache.

Needed before running jobs offline (scripts/local_job.sh): solvers get no network.
Resumable: verified files already in the cache are kept.

Usage: uv run python scripts/prefetch_jobs.py DESIGN BLOCKS JOBS [CACHE]
  e.g. uv run python scripts/prefetch_jobs.py data/smoke/design.toml study/blocks.csv 0,1,2
"""

import sys
from pathlib import Path

import pandas as pd

from satbench.blocks import Design, InstanceCache


def main() -> None:
    design, blocks, jobs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    cache = InstanceCache(Path(sys.argv[4] if len(sys.argv) > 4 else "data/instances"))
    d = Design.load(design)
    b = pd.read_csv(blocks)
    wanted = b[b.job.isin([int(j) for j in jobs.split(",")])]
    cache.get(d.calibration_instance, "calibration", "original")
    print("calibration ready", flush=True)
    for r in wanted.drop_duplicates(["origin_hash", "variant"]).itertuples():
        path, _ = cache.get(r.origin_hash, r.family, r.variant)
        print(f"job {r.job}  {r.variant:8s} {path.name}", flush=True)
    print("ALL INSTANCES READY", flush=True)


if __name__ == "__main__":
    main()
