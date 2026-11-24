"""One (solver, instance, seed, configuration) run → one self-describing JSON record.

Status values:

* ``SAT`` / ``UNSAT``: answer within limits; SAT answers have been model-checked.
* ``TIMEOUT``: CPU (or wall) limit reached.
* ``MEMOUT``: memory limit reached.
* ``WRONG``: claimed SAT with an assignment that falsifies a clause, or an answer contradicting the
  known result. Never credited as solved.
* ``CRASH``: process ended without an answer and without hitting a limit (e.g. abort, segfault).
* ``ERROR``: the harness could not start the run.

Only SAT and UNSAT count as solved. Every other status receives the PAR-2 penalty.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import platform
import shutil
import socket
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .execute import Limits, execute
from .solvers import SPECS, _competition, canonical
from .verify import check_model, parse_competition_output, parse_minisat_result_file

EXIT_CODES = {10: "SAT", 20: "UNSAT"}


@dataclass(frozen=True)
class RunSpec:
    solver: str  # name in the study, e.g. "kissat-4.0.4" or "ae-kissat-mab"
    solver_family: str  # key of SPECS
    binary: Path
    instance: Path  # verified .cnf.xz
    instance_hash: str
    family: str
    seed: int
    params: dict = field(default_factory=dict)
    config_id: str = "default"
    known_result: str = "unknown"  # from GBD: sat | unsat | unknown
    origin_hash: str = ""  # GBD hash of the unperturbed instance ("" = same as instance_hash)
    variant: str = "original"  # perturbation tag, e.g. "s1-rfcl"

    @property
    def run_id(self) -> str:
        key = json.dumps(
            [self.solver, self.config_id, self.instance_hash, self.seed, sorted(self.params.items())]
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def decompress(src: Path, workdir: Path) -> Path:
    """Decompress outside the timed region so that no solver pays for xz decoding."""
    dst = workdir / src.name.removesuffix(".xz")
    with lzma.open(src) as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 22)
    return dst


def comment_excerpt(text: str, head: int = 60, tail: int = 60, width: int = 200) -> list[str]:
    """First and last comment lines ('c ...') of the solver output, truncated.

    Solvers report configuration, preprocessing results, dispatch decisions and statistics in
    comment lines. Keeping an excerpt in every record preserves them even when full logs are
    discarded to keep CI artifacts small. Model lines ('v ...') are never kept.
    """
    lines = [ln[:width] for ln in text.splitlines() if ln.startswith("c ")]
    if len(lines) <= head + tail:
        return lines
    return [*lines[:head], f"c [satbench: {len(lines) - head - tail} lines omitted]", *lines[-tail:]]


def link_farm(solver_dir: Path, workdir: Path) -> Path:
    """A per-run directory of symlinks to the solver's top-level entries.

    Competition scripts resolve helpers relative to their own location or the working directory,
    and some write temporary files there. With a link farm every run gets a private, writable
    directory, the installed solver can stay read-only (as in a container image), and parallel
    runs never collide.
    """
    farm = workdir / "solver"
    farm.mkdir()
    for entry in solver_dir.iterdir():
        (farm / entry.name).symlink_to(entry.resolve())
    return farm


def run(spec: RunSpec, limits: Limits, logdir: Path, cpus: set[int] | None = None,
        scratch: Path | None = None) -> dict:
    solver = SPECS[spec.solver_family]
    logdir.mkdir(parents=True, exist_ok=True)
    out, err = logdir / f"{spec.run_id}.out", logdir / f"{spec.run_id}.err"
    record = dict(
        run_id=spec.run_id, solver=spec.solver, solver_family=spec.solver_family,
        config_id=spec.config_id, params=spec.params, instance_hash=spec.instance_hash,
        family=spec.family, seed=spec.seed, known_result=spec.known_result,
        origin_hash=spec.origin_hash or spec.instance_hash, variant=spec.variant,
        limits=asdict(limits), host=socket.gethostname(), cpu_model=cpu_model(),
        cpus=sorted(cpus) if cpus else None,
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    with tempfile.TemporaryDirectory(dir=scratch, prefix="satbench-") as tmp:
        try:
            cnf = decompress(spec.instance, Path(tmp))
            result_file = Path(tmp) / "result.txt"
            cwd = None
            if spec.solver_family == "competition":
                # spec.binary is the installed solver directory containing run.sh
                cwd = link_farm(spec.binary, Path(tmp))
                outdir = Path(tmp) / "out"
                outdir.mkdir()
                # Proofs are generated (same CPU cost as in the competition) but discarded,
                # so multi-GB UNSAT proofs add neither disk I/O variance nor storage.
                (outdir / "proof.out").symlink_to("/dev/null")
                cmd = _competition(cwd, cnf, outdir)
            else:
                cmd = solver.build_cmd(spec.binary, cnf, spec.seed, spec.params)
                if solver.result_file:
                    cmd.append(str(result_file))
            record["cmd"] = cmd
            m = execute(cmd, limits, out, err, cpus=cpus, cwd=cwd)
        except Exception as exc:
            record.update(status="ERROR", detail=f"{type(exc).__name__}: {exc}")
            return record

        text = out.read_text(errors="replace")
        if solver.result_file and result_file.exists():
            answer, model = parse_minisat_result_file(result_file.read_text(errors="replace"))
        else:
            answer, model = parse_competition_output(text)
        if not answer and m.returncode in EXIT_CODES:
            answer = EXIT_CODES[m.returncode]

        record.update(asdict(m))
        record["solver_comments"] = comment_excerpt(text)
        record["stderr_tail"] = err.read_text(errors="replace").splitlines()[-20:]
        stats = solver.parse_stats(text)
        record["metrics"] = canonical(stats, solver.aliases)
        record["raw_stats"] = stats

        if m.limit_hit in ("cpu", "wall"):
            record["status"] = "TIMEOUT"
        elif m.limit_hit == "memory":
            record["status"] = "MEMOUT"
        elif answer == "SAT":
            ok, detail = (check_model(cnf, model) if model is not None
                          else (False, "no model printed"))
            record["model_check"] = detail
            record["status"] = "SAT" if ok else "WRONG"
        elif answer == "UNSAT":
            record["status"] = "UNSAT"
        else:
            record["status"] = "CRASH"

    if record["status"] in ("SAT", "UNSAT") and spec.known_result in ("sat", "unsat"):
        if record["status"].lower() != spec.known_result:
            record["detail"] = f"contradicts GBD known result '{spec.known_result}'"
            record["status"] = "WRONG"
    record["solved"] = record["status"] in ("SAT", "UNSAT")
    record["time"] = record.get("cpu_time")
    return record
