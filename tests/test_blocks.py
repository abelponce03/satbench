import json
import lzma
from pathlib import Path

import pandas as pd
import pytest

from satbench.blocks import Design, InstanceCache, block_id, parse_variant, plan_blocks, run_job
from satbench.instances.download import gbd_hash, instance_path

DESIGN = {
    "timeout": 5, "memory_mb": 300, "blocks_per_job": 2, "job_budget_seconds": 600,
    "selection_seed": 1, "packing_seed": 2, "perturbations": ["s1-rcl", "s1-f"],
    "units": [{"a": "base", "b": "ai", "role": "primary"},
              {"a": "ref1", "b": "ref2", "role": "reference"}],
    "calibration": {"solver": "base", "instance": "X"},
    "instance_sets": {"seen": {"perturb_fraction": 0.5, "context_units": True},
                      "old": {"perturb_fraction": 0.0, "context_units": False}},
}


def design(**over):
    raw = {**DESIGN, **over}
    return Design(units=raw["units"], timeout=raw["timeout"], memory_mb=raw["memory_mb"],
                  blocks_per_job=raw["blocks_per_job"], job_budget_seconds=raw["job_budget_seconds"],
                  calibration_solver=raw["calibration"]["solver"],
                  calibration_instance=raw["calibration"]["instance"],
                  perturbations=raw["perturbations"], raw=raw)


def test_variant_tags_roundtrip():
    for tag in ("s1-rcl", "s1-f", "s7-rfcl"):
        assert parse_variant(tag).tag == tag


def test_plan_rules():
    inst = pd.DataFrame(dict(hash=[f"{i:032x}" for i in range(8)], family="f",
                             instance_set=["seen"] * 4 + ["old"] * 4))
    b = plan_blocks(inst, design())
    seen, old = b[b.instance_set == "seen"], b[b.instance_set == "old"]
    assert set(old.role) == {"primary"} and set(old.variant) == {"original"}
    assert (seen[seen.role == "reference"].variant == "original").all()
    pert = seen[(seen.role == "primary") & (seen.variant != "original")]
    assert pert.origin_hash.nunique() == 2 and set(pert.variant) == {"s1-rcl", "s1-f"}
    assert b.block_id.is_unique
    assert (b.groupby("job").size() <= 2).all()
    assert block_id(("base", "ai"), "h", "original") == block_id(("base", "ai"), "h", "original")


# ------------------------------------------------------------------ end-to-end with fake solvers

CNF = "p cnf 3 2\n1 -2 0\n2 3 0\n"


def fake_competition_solver(root: Path, name: str, sleep: float) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "run.sh").write_text(
        "#!/bin/bash\n"
        f"python3 -c 'import time; time.sleep({sleep})'\n"
        "python3 - \"$1\" <<'EOF'\n"
        "import sys, itertools\n"
        "cls=[[int(x) for x in l.split()[:-1]] for l in open(sys.argv[1]) if l[0] not in 'cp' and l.strip()]\n"
        "n=max(abs(x) for c in cls for x in c)\n"
        "for bits in itertools.product([1,-1], repeat=n):\n"
        "    if all(any(bits[abs(x)-1]*x>0 for x in c) for c in cls):\n"
        "        print('s SATISFIABLE'); print('v', *[(i+1)*b for i,b in enumerate(bits)], 0); sys.exit(10)\n"
        "print('s UNSATISFIABLE'); sys.exit(20)\n"
        "EOF\n"
    )


def test_run_job_end_to_end(tmp_path):
    solvers = tmp_path / "solvers"
    for name in ("base", "ai"):
        fake_competition_solver(solvers, name, 0.2)
    (solvers / "build-report.json").write_text(json.dumps(
        {n: {"mode": "competition"} for n in ("base", "ai")}))
    # pre-seed the instance cache, so no network is needed
    tmp_cnf = tmp_path / "toy.cnf.xz"
    with lzma.open(tmp_cnf, "wt") as f:
        f.write(CNF)
    h = gbd_hash(tmp_cnf)
    cache_root = tmp_path / "cache"
    for fam in ("toy", "calibration"):
        dst = instance_path(cache_root, fam, h)
        dst.parent.mkdir(parents=True)
        dst.write_bytes(tmp_cnf.read_bytes())

    d = design(units=[{"a": "base", "b": "ai", "role": "primary"}],
               calibration={"solver": "base", "instance": h}, blocks_per_job=3,
               instance_sets={"seen": {"perturb_fraction": 1.0, "context_units": True}})
    inst = pd.DataFrame(dict(hash=[h], family="toy", instance_set="seen", result="sat"))
    blocks = plan_blocks(inst, d)
    results = tmp_path / "results.jsonl"
    for job in sorted(blocks.job.unique()):
        run_job(blocks, int(job), d, solvers, InstanceCache(cache_root), results, tmp_path / "logs")
    recs = [json.loads(line) for line in results.read_text().splitlines()]
    runs = [r for r in recs if r["kind"] == "run"]
    cal = [r for r in recs if r["kind"] == "calibration"]
    assert len(runs) == 2 * len(blocks) and {r["status"] for r in runs} == {"SAT"}
    assert {r["variant"] for r in runs} == {"original", "s1-rcl", "s1-f"}
    # start + end on each core, plus one concurrent pair at the start
    assert len(cal) == 6 * blocks.job.nunique()
    assert sum(c["when"] == "start-concurrent" for c in cal) == 2 * blocks.job.nunique()
    feats = [r for r in recs if r["kind"] == "features"]
    assert {f["variant"] for f in feats} == {"original", "s1-rcl", "s1-f"}
    assert all(f["features"].get("clauses") == 2.0 for f in feats)  # perturbation keeps size
    assert all("process_cpu" in r and "solver_comments" in r for r in runs)
    by_block = pd.DataFrame(runs).groupby("block_id")
    assert (by_block.cpus.apply(lambda c: len({tuple(x) for x in c})) == 2).all()  # two cores
    assert all(r["host_cpu_model"] for r in recs if r["kind"] != "block")
    # resumability: rerunning a job does not redo completed blocks
    before = len(runs)
    run_job(blocks, int(blocks.job.iloc[0]), d, solvers, InstanceCache(cache_root), results,
            tmp_path / "logs")
    again = [json.loads(l) for l in results.read_text().splitlines()]
    assert sum(r["kind"] == "run" for r in again) == before
