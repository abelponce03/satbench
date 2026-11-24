"""How to invoke each solver family and how to read its statistics.

Command conventions differ (``--seed=N`` vs ``--random N`` vs ``-rnd-seed=N``), and so do the
statistics formats. Every parser maps solver-specific counter names onto a small canonical set,
so internal metrics can be compared across solvers. The raw counters are kept as well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CANONICAL = ("conflicts", "decisions", "propagations", "restarts", "rephased", "walks", "chrono")

# "c conflicts:      12345    123.4 per second"   (Kissat, CaDiCaL)
_COLON_STAT = re.compile(r"^c\s+([A-Za-z][\w.-]*):\s+(-?\d+(?:\.\d+)?)")
# "c conflicts                : 1234  (...)"     (CryptoMiniSat)
# "conflicts             : 1234  (12 /sec)"      (MiniSat, no 'c' prefix)
_SPACED_STAT = re.compile(r"^(?:c\s+)?([A-Za-z][A-Za-z _/-]*?)\s*:\s+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_colon_stats(text: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    for line in text.splitlines():
        if m := _COLON_STAT.match(line):
            stats.setdefault(_norm(m.group(1)), float(m.group(2)))
    return stats


def parse_spaced_stats(text: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    for line in text.splitlines():
        if m := _SPACED_STAT.match(line):
            stats.setdefault(_norm(m.group(1)), float(m.group(2)))
    return stats


def canonical(raw: dict[str, float], aliases: dict[str, str]) -> dict[str, float]:
    out = {}
    for canon in CANONICAL:
        key = aliases.get(canon, canon)
        if key in raw:
            out[canon] = raw[key]
    return out


@dataclass(frozen=True)
class SolverSpec:
    family: str  # kissat | cadical | cryptominisat | minisat
    build_cmd: Callable[[Path, Path, int, dict], list[str]]
    parse_stats: Callable[[str], dict[str, float]]
    aliases: dict[str, str] = field(default_factory=dict)
    result_file: bool = False  # MiniSat writes its answer and model to a file


def _kissat(binary: Path, cnf: Path, seed: int, params: dict) -> list[str]:
    return [str(binary), f"--seed={seed}", *(f"--{k}={v}" for k, v in params.items()), str(cnf)]


def _cadical(binary: Path, cnf: Path, seed: int, params: dict) -> list[str]:
    return [str(binary), f"--seed={seed}", *(f"--{k}={v}" for k, v in params.items()), str(cnf)]


def _cms(binary: Path, cnf: Path, seed: int, params: dict) -> list[str]:
    args = [str(binary), "--random", str(seed), "--verb", "1"]
    for k, v in params.items():
        args += [f"--{k}", str(v)]
    return [*args, str(cnf)]


def _minisat(binary: Path, cnf: Path, seed: int, params: dict) -> list[str]:
    # MiniSat's -rnd-seed only matters when -rnd-freq > 0; both are recorded with the run.
    return [str(binary), "-verb=1", f"-rnd-seed={float(seed)}",
            *(f"-{k}={v}" for k, v in params.items()), str(cnf)]


def _competition(solver_dir: Path, cnf: Path, outdir: Path) -> list[str]:
    """SAT Competition convention: ``run.sh <instance> <output-dir>`` (no seed, no parameters)."""
    return ["bash", str(solver_dir / "run.sh"), str(cnf), str(outdir)]


SPECS: dict[str, SolverSpec] = {
    "kissat": SolverSpec(
        "kissat", _kissat, parse_colon_stats,
        aliases={"rephased": "rephased", "walks": "walks", "chrono": "chronological"},
    ),
    "cadical": SolverSpec(
        "cadical", _cadical, parse_colon_stats,
        aliases={"rephased": "rephased", "chrono": "chronological"},
    ),
    "cryptominisat": SolverSpec("cryptominisat", _cms, parse_spaced_stats),
    "minisat": SolverSpec("minisat", _minisat, parse_spaced_stats, result_file=True),
    # Competition packages are mostly Kissat/CaDiCaL derivatives; when they print statistics
    # (many run with -q), the Kissat-style parser applies.
    "competition": SolverSpec(
        "competition", lambda *a: [], parse_colon_stats,
        aliases={"rephased": "rephased", "walks": "walks", "chrono": "chronological"},
    ),
}
