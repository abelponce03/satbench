"""Command-line entry point: ``satbench <command>``.

Typical pipeline:

    satbench gbd-meta     --dir data/gbd
    satbench select       --meta data/gbd/meta.db --families miter,planning,... --per-family 220 \
                          --sample-seed 2027 --split-seed 9001 --out data/selection.csv
    satbench download     --selection data/selection.csv --root data/instances
    satbench features     --manifest data/instances/manifest.csv --out data/features.csv
    satbench plan         --selection data/selection.csv --manifest data/instances/manifest.csv \
                          --solvers-dir /opt/solvers --split test --seeds 1 --out data/plan.csv
    satbench run          --plan data/plan.csv --cpu 5000 --mem 16000 --results data/runs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def cmd_gbd_meta(a):
    from .instances.gbd import fetch_meta

    print(fetch_meta(Path(a.dir)))


def cmd_select(a):
    from .instances.gbd import MetaDB
    from .instances.select import sample, split_train_test

    inst = MetaDB(Path(a.meta)).instances(a.families.split(","))
    sel = sample(inst, a.per_family, a.sample_seed, a.cutoff_year, a.min_recent_share)
    sel["split"] = split_train_test(sel, a.test_share, a.split_seed)
    sel.to_csv(a.out, index=False)
    print(sel.groupby(["family", "era", "split"]).size().unstack(fill_value=0).to_string())


def cmd_download(a):
    from .instances.download import fetch_all

    sel = pd.read_csv(a.selection)
    root = Path(a.root)
    man = fetch_all(sel, root, root / "manifest.csv", workers=a.workers)
    print(man.status.value_counts().to_string())
    if (man.status != "ok").any():
        raise SystemExit("some instances failed verification; see manifest.csv")


def cmd_features(a):
    from .instances.features import extract_all

    man = pd.read_csv(a.manifest)
    man = man[man.status == "ok"]
    feats = extract_all([Path(p) for p in man.path], workers=a.workers)
    feats.merge(man[["path", "hash", "family"]], on="path").to_csv(a.out, index=False)


def _perturb_one(args):
    from .instances.perturb import Perturbation, perturb_file

    path, family, out, seed = args
    dst = Path(out) / family / f"{Path(path).name.removesuffix('.cnf.xz')}.s{seed}.cnf.xz"
    try:
        info = perturb_file(Path(path), dst, Perturbation(seed))
    except Exception as exc:
        info = dict(path=str(dst), status=f"error: {type(exc).__name__}: {exc}", variant=f"s{seed}")
    info["family"] = family
    return info


def cmd_perturb(a):
    from concurrent.futures import ProcessPoolExecutor

    man = pd.read_csv(a.manifest).query("status == 'ok'")
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(r.path, r.family, a.out, s) for r in man.itertuples() for s in seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(_perturb_one, jobs, chunksize=2))
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out) / "perturbations.csv", index=False)
    print(df.status.value_counts().to_string())
    if (df.status != "ok").any():
        raise SystemExit("some perturbations failed verification; see perturbations.csv")


def cmd_plan(a):
    sel = pd.read_csv(a.selection)
    man = pd.read_csv(a.manifest).query("status == 'ok'")
    sel = sel[sel.split == a.split].merge(man[["hash", "path"]], on="hash")
    sel["origin_hash"], sel["variant"] = sel["hash"], "original"
    if a.perturbations:
        pert = pd.read_csv(a.perturbations).query("status == 'ok'")
        extra = sel.drop(columns=["hash", "path", "variant"]).merge(
            pert[["origin_hash", "hash", "path", "variant"]], on="origin_hash")
        sel = pd.concat([sel, extra], ignore_index=True)
    report = json.loads((Path(a.solvers_dir) / "build-report.json").read_text())
    wanted = a.solvers.split(",") if a.solvers else [n for n, r in report.items() if r["ok"]]
    rows = []
    for name in wanted:
        r = report[name]
        competition = r["mode"] == "competition"
        binary = Path(a.solvers_dir) / name
        if not competition:
            binary = binary / r["binary"]
        seeds = [0] if competition else list(range(1, a.seeds + 1))
        for inst in sel.itertuples():
            for seed in seeds:
                rows.append(dict(
                    solver=name, solver_family="competition" if competition else a.binary_family,
                    binary=str(binary), instance=inst.path, instance_hash=inst.hash,
                    family=inst.family, era=inst.era, seed=seed,
                    known_result=getattr(inst, "result", "unknown"),  # preserved by perturbation
                    origin_hash=inst.origin_hash, variant=inst.variant,
                ))
    plan = pd.DataFrame(rows)
    plan.to_csv(a.out, index=False)
    print(f"{len(plan)} runs: {plan.solver.nunique()} solvers x {plan.instance_hash.nunique()} instances")


def cmd_run(a):
    from .runner.batch import run_batch
    from .runner.execute import Limits
    from .runner.run import RunSpec

    plan = pd.read_csv(a.plan)
    if a.shard is not None:  # for cluster array jobs: --shard i --shards n
        plan = plan.iloc[a.shard :: a.shards]
    specs = [
        RunSpec(r.solver, r.solver_family, Path(r.binary), Path(r.instance), r.instance_hash,
                r.family, int(r.seed), known_result=r.known_result,
                origin_hash=getattr(r, "origin_hash", r.instance_hash),
                variant=getattr(r, "variant", "original"))
        for r in plan.itertuples()
    ]
    limits = Limits(cpu_seconds=a.cpu, memory_mb=a.mem)
    n = run_batch(specs, limits, Path(a.results), Path(a.logs), workers=a.workers,
                  scratch=Path(a.scratch) if a.scratch else None, leave_free=a.leave_free)
    print(f"executed {n} runs")


def cmd_blocks_plan(a):
    from .blocks import Design, plan_blocks
    from .instances.study import select_study

    design = Design.load(Path(a.design))
    train = [l.strip() for l in Path(a.aesat_train).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    inst = select_study(Path(a.meta), design.raw["instance_sets"], train,
                        seed=design.raw["selection_seed"])
    inst = inst[inst.hash != design.calibration_instance]  # calibration is not a study instance
    blocks = plan_blocks(inst, design)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    inst.to_csv(Path(a.out).with_name("instances.csv"), index=False)
    blocks.to_csv(a.out, index=False)
    print(blocks.groupby(["role", "variant"]).size().unstack(fill_value=0).to_string())
    print(f"{len(blocks)} blocks in {blocks.job.nunique()} jobs")


def cmd_blocks_run(a):
    from .blocks import Design, InstanceCache, run_job

    design = Design.load(Path(a.design))
    if design.calibration_instance == "TO_BE_SET":
        raise SystemExit("set [calibration].instance in the design first (blocks pick-calibration)")
    blocks = pd.read_csv(a.blocks)
    cache = InstanceCache(Path(a.instances))
    for job in [int(j) for j in a.jobs.split(",")]:
        n = run_job(blocks, job, design, Path(a.solvers_dir), cache, Path(a.results), Path(a.logs))
        print(f"job {job}: {n} blocks completed")


def cmd_blocks_pending(a):
    """Re-pack blocks not yet completed (e.g. a job ran out of time) into new job numbers."""
    from .blocks import Design, _done

    design = Design.load(Path(a.design))
    blocks = pd.read_csv(a.blocks)
    done = set()
    for f in Path(a.results_dir).rglob("*.jsonl"):
        done |= _done(f)
    pending = blocks[~blocks.block_id.isin(done)].reset_index(drop=True)
    start = blocks.job.max() + 1
    pending["job"] = start + pending.index // design.blocks_per_job
    pending.to_csv(a.out, index=False)
    print(f"{len(done)} blocks done, {len(pending)} pending in {pending.job.nunique()} new jobs")


def cmd_blocks_wave(a):
    """Next wave: pending blocks (not completed, not in an in-flight wave), in plan order."""
    from .blocks import Design, _done

    design = Design.load(Path(a.design))
    blocks = pd.read_csv(a.blocks)
    done: set[str] = set()
    for f in Path(a.results_dir).rglob("*.jsonl") if Path(a.results_dir).exists() else []:
        done |= _done(f)
    inflight: set[str] = set()
    for w in a.inflight or []:
        if Path(w).exists():
            inflight |= set(pd.read_csv(w).block_id)
    pending = blocks[~blocks.block_id.isin(done | inflight)].reset_index(drop=True)
    wave = pending.head(a.max_jobs * design.blocks_per_job).copy()
    wave["job"] = wave.index // design.blocks_per_job
    summary = dict(total=len(blocks), done=len(done & set(blocks.block_id)),
                   inflight=len(inflight), pending=len(pending), wave_blocks=len(wave),
                   wave_jobs=int(wave.job.nunique()) if len(wave) else 0)
    if len(wave):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        wave.to_csv(a.out, index=False)
    print(json.dumps(summary))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="satbench")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("gbd-meta", help="download a GBD meta.db snapshot with provenance")
    p.add_argument("--dir", default="data/gbd")
    p.set_defaults(fn=cmd_gbd_meta)

    p = sub.add_parser("select", help="deterministic instance selection and train/test split")
    p.add_argument("--meta", required=True)
    p.add_argument("--families", required=True)
    p.add_argument("--per-family", type=int, required=True)
    p.add_argument("--sample-seed", type=int, required=True)
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--cutoff-year", type=int, default=2023)
    p.add_argument("--min-recent-share", type=float, default=0.0)
    p.add_argument("--test-share", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_select)

    p = sub.add_parser("download", help="download and verify the selected instances")
    p.add_argument("--selection", required=True)
    p.add_argument("--root", default="data/instances")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("features", help="GBD base features of the downloaded instances")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_features)

    p = sub.add_parser("perturb", help="semantics-preserving scrambled copies of the instances")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="1", help="comma-separated perturbation seeds")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_perturb)

    p = sub.add_parser("plan", help="expand solvers x instances x seeds into a run plan")
    p.add_argument("--perturbations", help="perturbations.csv to add scrambled variants")
    p.add_argument("--selection", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--solvers-dir", required=True)
    p.add_argument("--solvers", help="comma-separated subset (default: every built solver)")
    p.add_argument("--split", default="test")
    p.add_argument("--seeds", type=int, default=1, help="seeds per binary-mode solver")
    p.add_argument("--binary-family", default="kissat")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="execute a run plan (resumable)")
    p.add_argument("--plan", required=True)
    p.add_argument("--cpu", type=float, required=True, help="CPU-time limit per run (s)")
    p.add_argument("--mem", type=float, required=True, help="memory limit per run (MB)")
    p.add_argument("--results", required=True)
    p.add_argument("--logs", required=True)
    p.add_argument("--workers", type=int)
    p.add_argument("--leave-free", type=int, default=1)
    p.add_argument("--scratch")
    p.add_argument("--shard", type=int)
    p.add_argument("--shards", type=int)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("blocks-plan", help="select the study instances and plan all blocks")
    p.add_argument("--meta", default="data/gbd/meta.db", help="GBD meta.db snapshot")
    p.add_argument("--design", default="study/design.toml")
    p.add_argument("--aesat-train", default="solvers/exposure/aesat_train_hashes.txt")
    p.add_argument("--out", default="study/blocks.csv")
    p.set_defaults(fn=cmd_blocks_plan)

    p = sub.add_parser("blocks-run", help="execute one or more jobs of the blocked design")
    p.add_argument("--design", default="study/design.toml")
    p.add_argument("--blocks", default="study/blocks.csv")
    p.add_argument("--jobs", required=True, help="comma-separated job numbers")
    p.add_argument("--solvers-dir", required=True)
    p.add_argument("--instances", default="data/instances",
                   help="persistent cache of verified and perturbed CNFs (never /tmp)")
    p.add_argument("--results", default="data/runs/local/results.jsonl")
    p.add_argument("--logs", default="data/runs/local/logs")
    p.set_defaults(fn=cmd_blocks_run)

    p = sub.add_parser("blocks-pending", help="re-pack unfinished blocks into new jobs")
    p.add_argument("--design", default="study/design.toml")
    p.add_argument("--blocks", default="study/blocks.csv")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_blocks_pending)

    p = sub.add_parser("blocks-wave", help="write the next wave of pending blocks")
    p.add_argument("--design", default="study/design.toml")
    p.add_argument("--blocks", default="study/blocks.csv")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--inflight", nargs="*", help="wave files still running")
    p.add_argument("--max-jobs", type=int, default=256)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_blocks_wave)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
