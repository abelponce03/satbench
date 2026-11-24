# satbench

Reproducible benchmarking of SAT solvers, built for one question: **do AI-tuned and AI-generated
SAT solvers keep their advantage outside the instances they were developed on?**

The study compares AI-derived solvers from the SAT Competitions 2025–2026 with their declared
human-designed bases. Instance sets are split by *exposure*, i.e. what each solver's developers
could have seen. Every instance is also run in semantics-preserving perturbed forms.

## What is in here

| Path | Content |
|---|---|
| `study/analysis_plan.md` | **Pre-registered analysis plan.** Its first public commit fixes hypotheses, outcomes and exclusions before any measurement. |
| `study/design.toml` | Experimental design: solver pairs, instance sets, limits, seeds |
| `study/blocks.csv`, `study/instances.csv` | The complete block plan and instance list derived from it |
| `study/*provenance.json` | URL, date and SHA-256 of every external input (GBD snapshot, competition results, papers) |
| `docs/exposure.md` | Which instances each solver could have been tuned on, with sources |
| `solvers/manifest.toml` | Every solver, pinned by SHA-256 of its official competition archive |
| `solvers/Dockerfile` | Offline, digest-pinned build of all solvers, plus the runtime image |
| `src/satbench/` | Instance selection and verification, perturbations, runner, blocked design, statistics |
| `.github/workflows/` | Image build and blocked execution on GitHub-hosted runners |

## Design in one paragraph

A **block** is one solver pair on one instance variant. Both solvers run concurrently on the same
host, each pinned to its own physical core, with the core assignment randomised. Host effects
therefore cancel in the paired comparison. Every job (one host) starts and ends with a fixed
calibration run on both cores, so slow, drifting or unusual hosts are detected and their blocks
re-run. CPU time is measured by a runsolver-style monitor:
- memory is read from VmHWM, not `ru_maxrss`, which is inflated by the forking harness;
- SAT answers are model-checked, and answers are checked against verified known results;
- aborted or wrong runs are never counted as solved.

## Quick start

```bash
uv sync
uv run pytest                                    # 80+ tests, including fake-solver end-to-end runs
uv run python -m satbench.solvers.build fetch --manifest solvers/manifest.toml --cache solvers/archives
docker build -f solvers/Dockerfile --target solvers -t satbench-solvers .
uv run python scripts/prefetch_jobs.py study/design.toml study/blocks.csv 0   # instances for job 0
scripts/local_job.sh study/design.toml study/blocks.csv 0 data/instances data/runs/local
```

Solvers are third-party code. They are only ever executed inside the container, with no network.

## Data sources

- Instances: Global Benchmark Database (Iser & Jabs, SAT 2024).
- Solvers: official SAT Competition 2025/2026 source archives.
- Reference results: SAT Competition 2026 `scores.csv`.

## License

Code: MIT. Each solver keeps its own license, as stated in its archive.
