"""One orchestration cycle (run by .github/workflows/orchestrator.yml every 20 minutes).

1. Ingest: download the artifacts of every finished `blocks` run not yet ingested into the
   `results` branch (results/wave-NNN/run-ID/...), and note the run id in ingested.txt.
2. Dispatch: if fewer than `max_waves_in_flight` waves are running, write the next wave of pending
   blocks (not completed, not in a running wave), push it, and dispatch `blocks.yml` for it.
3. Report: write STATUS.md (progress, fleet composition, outcome counts).

The `results` branch is checked out at RES; the main branch at MAIN. Uses the `gh` CLI with the
workflow token (GH_TOKEN). Idempotent: running it twice in a row does no harm.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TITLE = re.compile(r"^blocks wave (\d{3}) \[([\w-]+)\]$")


def sh(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True).stdout


def git_push(res: Path, message: str, branch: str = "results") -> None:
    sh("git", "add", "-A", cwd=res)
    if not sh("git", "status", "--porcelain", cwd=res).strip():
        return
    sh("git", "commit", "-q", "-m", message, cwd=res)
    for _ in range(5):  # another writer may have pushed in between
        if subprocess.run(["git", "push", "-q", "origin", f"HEAD:{branch}"], cwd=res).returncode == 0:
            return
        sh("git", "pull", "-q", "--rebase", "origin", branch, cwd=res)
    raise SystemExit("could not push the results branch")


def runs(branch: str) -> list[dict]:
    out = sh("gh", "run", "list", "--workflow", "blocks.yml", "--limit", "300",
             "--json", "databaseId,status,conclusion,displayTitle")
    return [r for r in json.loads(out) if TITLE.match(r["displayTitle"])
            and TITLE.match(r["displayTitle"]).group(2) == branch]


def ingest(res: Path, all_runs: list[dict]) -> list[str]:
    ledger = res / "ingested.txt"
    done = set(ledger.read_text().split()) if ledger.exists() else set()
    new = []
    for r in all_runs:
        rid = str(r["databaseId"])
        if r["status"] != "completed" or rid in done:
            continue
        wave = TITLE.match(r["displayTitle"]).group(1)
        dest = res / "results" / f"wave-{wave}" / f"run-{rid}"
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gh", "run", "download", rid, "--pattern", f"wave-{wave}-job-*",
                        "--dir", str(dest)], check=False)
        new.append(rid)
    if new:
        with open(ledger, "a") as f:
            f.write("".join(f"{rid}\n" for rid in new))
    return new


def status_report(res: Path, main: Path, cfg: dict, active: list[str]) -> str:
    import pandas as pd

    recs = []
    for p in (res / "results").rglob("*.jsonl"):
        for line in p.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    blocks = pd.read_csv(main / cfg["blocks"])
    done = {r["block_id"] for r in recs if r.get("kind") == "block" and r.get("block_complete")}
    n_done = len(done & set(blocks.block_id))
    fleet = Counter((r.get("host_cpu_model"), r.get("kind") == "host_rejected")
                    for r in recs if r.get("kind") in ("host_rejected", "calibration")
                    and r.get("when", "start") == "start")
    statuses = Counter(r.get("status") for r in recs if r.get("kind") == "run")
    lines = [
        "# Study progress", "",
        f"Updated {datetime.now(timezone.utc).isoformat(timespec='minutes')} by the orchestrator.", "",
        f"- Blocks completed: **{n_done} / {len(blocks)}** ({100 * n_done / len(blocks):.1f} %)",
        f"- Waves running: {', '.join(active) or 'none'}",
        f"- Run outcomes so far: {dict(statuses)}", "",
        "| host CPU | accepted job starts | rejected |", "|---|---|---|",
    ]
    models = sorted({m for m, _ in fleet})
    for m in models:
        lines.append(f"| {m} | {fleet.get((m, False), 0)} | {fleet.get((m, True), 0)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--res", type=Path, required=True)
    a = ap.parse_args()
    cfg = tomllib.loads((a.main / "study" / "orchestrator.toml").read_text())
    branch = cfg.get("results_branch", "results")
    if branch not in ("results", "results-test"):
        raise SystemExit("results_branch must be 'results' or 'results-test'")

    all_runs = runs(branch)
    new = ingest(a.res, all_runs)
    if new:
        git_push(a.res, f"ingest runs {' '.join(new)}", branch)

    active = sorted({TITLE.match(r["displayTitle"]).group(1) for r in all_runs
                     if r["status"] != "completed"})
    dispatched = None
    if cfg.get("enabled", False) and len(active) < cfg["max_waves_in_flight"]:
        waves_dir = a.res / "waves"
        waves_dir.mkdir(exist_ok=True)
        numbers = [int(p.stem.split("-")[1]) for p in waves_dir.glob("wave-*.csv")]
        nxt = f"{(max(numbers) + 1) if numbers else 0:03d}"
        out = waves_dir / f"wave-{nxt}.csv"
        summary = json.loads(sh(
            "satbench", "blocks-wave", "--design", str(a.main / cfg["design"]),
            "--blocks", str(a.main / cfg["blocks"]), "--results-dir", str(a.res / "results"),
            "--max-jobs", str(cfg["max_jobs_per_wave"]), "--out", str(out),
            "--inflight", *[str(waves_dir / f"wave-{w}.csv") for w in active],
        ).strip().splitlines()[-1])
        print("wave plan:", summary)
        if summary["wave_blocks"]:
            git_push(a.res, f"wave {nxt}: {summary['wave_blocks']} blocks in {summary['wave_jobs']} jobs", branch)
            image = (a.main / "study" / "image-digest.txt").read_text().strip()
            sh("gh", "workflow", "run", "blocks.yml", "-f", f"wave={nxt}", "-f", f"image={image}",
               "-f", f"max_parallel={cfg['max_parallel_per_wave']}", "-f",
               f"design_file={cfg['design']}", "-f", f"results_branch={branch}")
            dispatched = nxt
            active.append(nxt)
    (a.res / "STATUS.md").write_text(status_report(a.res, a.main, cfg, active))
    git_push(a.res, f"status{f' (dispatched wave {dispatched})' if dispatched else ''}", branch)
    print("ingested:", new, "dispatched:", dispatched, "active:", active)


if __name__ == "__main__":
    main()
