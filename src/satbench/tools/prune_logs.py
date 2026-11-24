"""Keep solver stdout/stderr only for runs whose outcome needs inspection.

Usage: python -m satbench.tools.prune_logs RESULTS.jsonl LOGDIR

Logs of SAT/UNSAT/TIMEOUT runs are deleted (their statistics are already in the JSON record);
logs of WRONG, CRASH, MEMOUT and ERROR runs are kept for diagnosis. This keeps CI artifacts small.
"""

import json
import sys
from pathlib import Path

KEEP = {"WRONG", "CRASH", "MEMOUT", "ERROR"}


def main() -> None:
    results, logdir = Path(sys.argv[1]), Path(sys.argv[2])
    keep = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in KEEP:
                keep.add(rec["run_id"])
    removed = 0
    for f in logdir.glob("*") if logdir.exists() else []:
        if f.stem not in keep:
            f.unlink()
            removed += 1
    print(f"kept logs of {len(keep)} runs, removed {removed} files")


if __name__ == "__main__":
    main()
