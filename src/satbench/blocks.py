"""Blocked experimental design for shared, possibly heterogeneous machines (CI runners, clusters).

The quantities of interest are paired: an AI-derived solver against its human base, on the same
instance. The design therefore makes every comparison *within one machine at one time*:

* A **unit** is a pair of solvers, e.g. (kissat-mab-hypre, kissat-mab-hypre-evolve).
* A **block** is one unit on one instance variant (original, or a perturbation seed). Both
  solvers run concurrently, each pinned to its own physical core of the same host, and the core
  assignment is randomised per block. Host effects (CPU model, noisy neighbours, thermal state)
  hit both members alike and cancel in the paired difference.
* A **job** is a sequence of blocks on one host, bracketed by a fixed **calibration** run (same
  solver, same instance) at its start and end. The calibration time measures the host's speed and
  any drift within the job, so slow or unstable hosts can be detected and their blocks re-run.

Every run records the host's CPU model. The analysis keeps blocks from the majority CPU model and
reports the others separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .runner.batch import physical_cores
from .runner.execute import Limits
from .runner.run import RunSpec, cpu_model, run


@dataclass(frozen=True)
class Design:
    units: list[dict]  # {"a", "b", "role", "note"}
    timeout: float
    memory_mb: float
    blocks_per_job: int
    job_budget_seconds: float
    calibration_solver: str
    calibration_instance: str  # GBD hash
    perturbations: list[str]  # variant tags "s<seed>-<kinds>", kinds from r,f,c,l (see perturb.py)
    raw: dict

    @staticmethod
    def load(path: Path) -> "Design":
        d = tomllib.loads(Path(path).read_text())
        return Design(
            units=d["units"], timeout=d["timeout"], memory_mb=d["memory_mb"],
            blocks_per_job=d["blocks_per_job"], job_budget_seconds=d["job_budget_seconds"],
            calibration_solver=d["calibration"]["solver"],
            calibration_instance=d["calibration"]["instance"],
            perturbations=d.get("perturbations", []), raw=d,
        )


def parse_variant(tag: str):
    """'s3-rcl' -> Perturbation(seed=3, rename, shuffle clauses, shuffle literals; no flip)."""
    from .instances.perturb import Perturbation

    seed, _, kinds = tag.removeprefix("s").partition("-")
    return Perturbation(int(seed), rename="r" in kinds, flip="f" in kinds,
                        shuffle_clauses="c" in kinds, shuffle_literals="l" in kinds)


def block_id(unit: tuple[str, str], origin_hash: str, variant: str, replicate: int = 0) -> str:
    key = f"{unit[0]}|{unit[1]}|{origin_hash}|{variant}" + (f"|rep{replicate}" if replicate else "")
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def plan_blocks(instances: pd.DataFrame, design: Design) -> pd.DataFrame:
    """One row per block, with a job number.

    Rules (see study/design.toml): primary units run on every set and, on the fixed perturbation
    subsample, on every perturbed variant; secondary/reference units run on the originals of sets
    with ``context_units``. Job packing is randomised, not sorted by family, so each host mixes
    families, units and variants and no host effect lines up with a factor of interest.
    """
    from .instances.select import rank_key

    sets = design.raw.get("instance_sets", {})
    rows = []
    for name, grp in instances.groupby("instance_set"):
        cfg = sets.get(name, {})
        ranked = sorted(grp.hash, key=lambda h: rank_key(design.raw.get("selection_seed", 0) + 1, h))
        perturbed_list = ranked[: int(round(len(ranked) * cfg.get("perturb_fraction", 0.0)))]
        perturbed = set(perturbed_list)
        # extra perturbation seeds on a sub-subsample: within-instance encoding variance
        extra_cfg = design.raw.get("extra_perturbations", {})
        extra = set(perturbed_list[: int(round(len(perturbed_list) * extra_cfg.get("fraction_of_perturbed", 0.0)))])
        for r in grp.itertuples():
            for u in design.units:
                if u["role"] != "primary" and not cfg.get("context_units", False):
                    continue
                variants = ["original"]
                if u["role"] == "primary" and r.hash in perturbed:
                    variants += design.perturbations
                    if r.hash in extra:
                        variants += extra_cfg.get("variants", [])
                for v in variants:
                    key = (u["a"], u["b"])
                    rows.append(dict(
                        block_id=block_id(key, r.hash, v), solver_a=u["a"], solver_b=u["b"],
                        role=u["role"], origin_hash=r.hash, family=r.family, variant=v,
                        known_result=getattr(r, "result", "unknown"), instance_set=name,
                        replicate_of=""))
    blocks = pd.DataFrame(rows)
    # Test-retest replicates: the same block again, on a different (random) host.
    rep_frac = design.raw.get("replication", {}).get("fraction", 0.0)
    if rep_frac:
        ranked_blocks = sorted(blocks.block_id, key=lambda b: rank_key(design.raw.get("selection_seed", 0) + 3, b))
        chosen = set(ranked_blocks[: int(round(len(blocks) * rep_frac))])
        reps = blocks[blocks.block_id.isin(chosen)].copy()
        reps["replicate_of"] = reps.block_id
        reps["block_id"] = [block_id((a, b), h, v, 1) for a, b, h, v in
                            zip(reps.solver_a, reps.solver_b, reps.origin_hash, reps.variant)]
        blocks = pd.concat([blocks, reps], ignore_index=True)
    seed = design.raw.get("packing_seed", 0)
    order = random.Random(seed).sample(range(len(blocks)), len(blocks))
    blocks = blocks.iloc[order].reset_index(drop=True)
    blocks["job"] = blocks.index // design.blocks_per_job
    return blocks


# ------------------------------------------------------------------------------ execution


def host_info() -> dict:
    flags = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags = line
                break
    except OSError:
        pass
    return dict(
        host=socket.gethostname(), cpu_model=cpu_model(), logical_cpus=os.cpu_count(),
        avx2="avx2" in flags, avx512f="avx512f" in flags,
        ci_run=os.environ.get("GITHUB_RUN_ID", ""), ci_job=os.environ.get("GITHUB_JOB", ""),
        runner=os.environ.get("RUNNER_NAME", ""),
        harness_commit=os.environ.get("GITHUB_SHA", ""),
        hw_counters=_counters_available(),
    )


def _counters_available() -> bool:
    from .runner.perfcount import available

    try:
        return available()[0]
    except Exception:
        return False


class InstanceCache:
    """Fetch an instance from GBD (verified) and derive perturbed variants on demand."""

    def __init__(self, root: Path):
        self.root = root

    def get(self, origin_hash: str, family: str, variant: str) -> tuple[Path, str]:
        from .instances.download import fetch_one
        from .instances.perturb import perturb_file

        rec = fetch_one(self.root, origin_hash, family)
        if rec.status != "ok":
            raise RuntimeError(f"instance {origin_hash}: {rec.status} {rec.detail}")
        if variant == "original":
            return Path(rec.path), origin_hash
        p = parse_variant(variant)
        if p.tag != variant:
            raise ValueError(f"non-canonical variant tag {variant!r} (expected {p.tag!r})")
        dst = self.root / "perturbed" / family / f"{origin_hash}.{variant}.cnf.xz"
        info_file = dst.with_suffix("").with_suffix(".json")
        if info_file.exists():
            info = json.loads(info_file.read_text())
        else:
            info = perturb_file(Path(rec.path), dst, p)
            info_file.write_text(json.dumps(info))
        if info["status"] != "ok":
            raise RuntimeError(f"perturbation {origin_hash}/{variant} failed verification")
        return dst, info["hash"]


def _spec(name: str, solvers_dir: Path, path: Path, h: str, row, variant: str,
          context: str) -> RunSpec:
    """``context`` (block id or calibration tag) enters the run id, so a solver that appears in
    two units, or in every calibration, gets distinct records and log files."""
    report = json.loads((solvers_dir / "build-report.json").read_text())[name]
    common = dict(known_result=row.known_result, origin_hash=row.origin_hash, variant=variant,
                  config_id=context)
    if report["mode"] == "competition":
        return RunSpec(name, "competition", solvers_dir / name, path, h, row.family, 0, **common)
    return RunSpec(name, "kissat", solvers_dir / name / report["binary"], path, h, row.family, 1,
                   **common)


def _done(results: Path) -> set[str]:
    if not results.exists():
        return set()
    out = set()
    for line in results.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "block" and rec.get("block_complete"):
            out.add(rec["block_id"])
    return out


def run_job(blocks: pd.DataFrame, job: int, design: Design, solvers_dir: Path, cache: InstanceCache,
            results: Path, logdir: Path, calibration_family: str = "calibration") -> int:
    """Execute every pending block of ``job``; returns the number of blocks completed now."""
    t_start = time.monotonic()
    cores = physical_cores()
    if len(cores) < 2:
        raise RuntimeError("the blocked design needs two physical cores")
    core_a, core_b = {min(cores[0])}, {min(cores[1])}
    host = host_info()
    limits = Limits(cpu_seconds=design.timeout, memory_mb=design.memory_mb)
    lock = threading.Lock()
    results.parent.mkdir(parents=True, exist_ok=True)

    def emit(rec: dict) -> None:
        rec.update(job=job, **{f"host_{k}": v for k, v in host.items()})
        with lock, open(results, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def calibrate(when: str) -> None:
        path, h = cache.get(design.calibration_instance, calibration_family, "original")
        row = pd.Series(dict(family=calibration_family, known_result="unknown",
                             origin_hash=design.calibration_instance))
        for k, cpus in enumerate((core_a, core_b)):  # a slow core as well as a slow host
            spec = _spec(design.calibration_solver, solvers_dir, path, h, row, "original",
                         context=f"calibration:{job}:{when}:{k}")
            rec = run(spec, limits, logdir, cpus=cpus)
            rec.update(kind="calibration", when=when)
            emit(rec)

    def calibrate_concurrent() -> None:
        """Both cores at once: the slowdown versus the solo runs measures how much two
        concurrent runs on this host interfere (shared cache, memory bandwidth)."""
        path, h = cache.get(design.calibration_instance, calibration_family, "original")
        row = pd.Series(dict(family=calibration_family, known_result="unknown",
                             origin_hash=design.calibration_instance))
        recs: list[dict] = [{}, {}]

        def go(k: int, cpus: set[int]) -> None:
            spec = _spec(design.calibration_solver, solvers_dir, path, h, row, "original",
                         context=f"calibration:{job}:concurrent:{k}")
            recs[k] = run(spec, limits, logdir, cpus=cpus)

        ts = [threading.Thread(target=go, args=(k, c)) for k, c in enumerate((core_a, core_b))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for rec in recs:
            rec.update(kind="calibration", when="start-concurrent")
            emit(rec)

    featured: set[str] = set()

    def emit_features(path: Path, h: str, row) -> None:
        """GBD base features of the exact file run (once per file and job)."""
        if h in featured:
            return
        featured.add(h)
        try:
            import gbdc

            feats = {k: float(v) for k, v in dict(gbdc.extract_base_features(str(path))).items()}
        except Exception as exc:
            feats = {"error": f"{type(exc).__name__}: {exc}"}
        emit(dict(kind="features", instance_hash=h, origin_hash=row.origin_hash,
                  variant=row.variant, family=row.family, features=feats))

    calibrate("start")
    calibrate_concurrent()
    done = _done(results)
    mine = blocks[blocks.job == job]
    completed = 0
    for row in mine.itertuples():
        if row.block_id in done:
            continue
        remaining = design.job_budget_seconds - (time.monotonic() - t_start)
        if remaining < 1.5 * design.timeout + 300:  # leave room for the final calibration
            break  # the block stays pending and is picked up by a later job
        path, h = cache.get(row.origin_hash, row.family, row.variant)
        emit_features(path, h, row)
        specs = [_spec(s, solvers_dir, path, h, row, row.variant, context=row.block_id)
                 for s in (row.solver_a, row.solver_b)]
        rng = random.Random(row.block_id)
        pins = [core_a, core_b]
        rng.shuffle(pins)  # randomised core assignment per block
        recs: list[dict] = [{}, {}]

        def go(i: int) -> None:
            with tempfile.TemporaryDirectory(prefix="satbench-") as scratch:
                recs[i] = run(specs[i], limits, logdir, cpus=pins[i], scratch=Path(scratch))

        threads = [threading.Thread(target=go, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, rec in enumerate(recs):
            rec.update(kind="run", block_id=row.block_id, unit=f"{row.solver_a}|{row.solver_b}",
                       member="a" if i == 0 else "b", instance_set=row.instance_set,
                       replicate_of=getattr(row, "replicate_of", "") or "")
            emit(rec)
        emit(dict(kind="block", block_id=row.block_id, block_complete=True))
        completed += 1
    calibrate("end")
    return completed
