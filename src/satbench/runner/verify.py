"""Answer extraction and independent model checking.

A claimed SAT answer is accepted only if the reported assignment satisfies every clause of the
original CNF. This matters more than usual here: automatically evolved solvers are new code whose
correctness has only been exercised on competition instances.
"""

from __future__ import annotations

import lzma
from pathlib import Path

import numpy as np


def parse_competition_output(text: str) -> tuple[str, np.ndarray | None]:
    """Return (answer, model) from SAT-competition style output ('s ...' and 'v ...' lines)."""
    answer, lits = "", []
    for line in text.splitlines():
        if line.startswith("s "):
            tag = line[2:].strip()
            answer = {"SATISFIABLE": "SAT", "UNSATISFIABLE": "UNSAT"}.get(tag, tag)
        elif line.startswith("v "):
            lits.extend(int(t) for t in line[2:].split())
    model = np.array([v for v in lits if v != 0], dtype=np.int64) if lits else None
    return answer, model


def parse_minisat_result_file(text: str) -> tuple[str, np.ndarray | None]:
    """MiniSat writes 'SAT' / 'UNSAT' / 'INDET' and the model to its result file."""
    lines = text.split("\n", 1)
    head = lines[0].strip()
    if head == "SAT":
        lits = np.array(lines[1].split(), dtype=np.int64) if len(lines) > 1 else np.array([])
        return "SAT", lits[lits != 0]
    if head == "UNSAT":
        return "UNSAT", None
    return "", None


def _open_cnf(path: Path):
    return lzma.open(path, "rt") if str(path).endswith(".xz") else open(path)


def check_model(cnf: Path, model: np.ndarray) -> tuple[bool, str]:
    """True iff ``model`` satisfies every clause of ``cnf``.

    Variables missing from the model are treated as unassigned: a clause they alone could satisfy
    counts as falsified, which is the conservative reading.
    """
    nvars = int(np.abs(model).max()) if len(model) else 0
    value = np.zeros(nvars + 1, dtype=np.int8)  # +1 true, -1 false, 0 unassigned
    value[np.abs(model)] = np.sign(model)
    clause: list[int] = []
    n_clauses = 0
    with _open_cnf(cnf) as f:
        for line in f:
            if not line or line[0] in "cp%\n":
                continue
            for tok in line.split():
                lit = int(tok)
                if lit == 0:
                    n_clauses += 1
                    if not any(
                        abs(v) <= nvars and value[abs(v)] == (1 if v > 0 else -1) for v in clause
                    ):
                        return False, f"clause {n_clauses} falsified"
                    clause = []
                else:
                    clause.append(lit)
    return True, f"{n_clauses} clauses satisfied"
