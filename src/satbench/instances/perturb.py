"""Semantics-preserving perturbations of CNF formulas ("scrambling").

A perturbed formula is the same logical problem written differently:

* **variable renaming**: a random permutation of 1..n,
* **polarity flipping**: each variable is replaced by its negation with probability ½,
* **clause shuffling**: a random clause order,
* **literal shuffling**: a random literal order inside each clause.

Satisfiability is preserved exactly, and a model of the perturbed formula maps back to a model of
the original one (:func:`unmap_model`). A solver that generalises should be largely insensitive to
these changes. A solver tuned to surface features of particular encodings (literal order,
variable numbering, structural fingerprints used by instance matchers) need not be. The idea
follows the scrambling used to discourage benchmark-specific tuning in SAT evaluations (e.g.
Biere's ``scranfilize``).

Every output is checked twice. Its GBD ``isohash`` (invariant under exactly these
transformations) must equal the original's, and its ``gbdhash`` must differ, so the solver really
sees a new file.
"""

from __future__ import annotations

import lzma
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Perturbation:
    seed: int
    rename: bool = True
    flip: bool = True
    shuffle_clauses: bool = True
    shuffle_literals: bool = True

    @property
    def tag(self) -> str:
        parts = [p for p, on in (("r", self.rename), ("f", self.flip), ("c", self.shuffle_clauses),
                                 ("l", self.shuffle_literals)) if on]
        return f"s{self.seed}-{''.join(parts) or 'id'}"


def read_cnf(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (n_vars, literals, clause_start) with clause i = literals[start[i]:start[i+1]].

    Literals are stored without the terminating zeros. The variable count is taken as the
    maximum of the header and the largest variable that actually occurs.
    """
    opener = lzma.open if str(path).endswith(".xz") else open
    header_vars, chunks = 0, []
    with opener(path, "rb") as f:
        for line in f:
            if line[:1] == b"c" or not line.strip():
                continue
            if line[:1] == b"p":
                header_vars = int(line.split()[2])
                continue
            chunks.append(line)
    # C-level whitespace-separated integer parsing (newlines count as separators)
    tokens = np.fromstring(b"".join(chunks).decode("ascii"), dtype=np.int64, sep=" ")
    zeros = np.flatnonzero(tokens == 0)
    lits = tokens[tokens != 0]
    lengths = np.diff(np.concatenate([[-1], zeros])) - 1
    start = np.concatenate([[0], np.cumsum(lengths)])
    n = max(header_vars, int(np.abs(lits).max()) if len(lits) else 0)
    return n, lits, start


def write_cnf(path: Path, n: int, lits: np.ndarray, start: np.ndarray, comment: str = "") -> None:
    """Write DIMACS; ``.xz`` output uses preset 1 (≈10× faster than the default 6).

    The compression level changes only file size: content is identical, and instances are
    decompressed outside the timed region of a run.
    """
    m = len(start) - 1
    if str(path).endswith(".xz"):
        ctx = lzma.open(path, "wt", preset=1)
    else:
        ctx = open(path, "w")
    with ctx as f:
        if comment:
            f.write(f"c {comment}\n")
        f.write(f"p cnf {n} {m}\n")
        # Vectorised serialisation in blocks of clauses (bounded memory): insert the terminators,
        # render every token once, and make each terminator end its line.
        block = 1_000_000
        for c0 in range(0, m, block):
            c1 = min(c0 + block, m)
            part = lits[start[c0] : start[c1]]
            tokens = np.insert(part, start[c0 + 1 : c1 + 1] - start[c0], 0)
            text = tokens.astype(str).astype(object)
            text[tokens == 0] = "0\n"
            f.write(" ".join(text).replace("\n ", "\n"))


def perturb_arrays(n: int, lits: np.ndarray, start: np.ndarray, p: Perturbation):
    """Apply ``p``; return (lits', start', perm, sign).

    ``perm[v]`` is the new name of variable v, and ``sign[v]`` is -1 if v was negated.
    """
    rng = np.random.default_rng(p.seed)
    perm = np.arange(n + 1)
    if p.rename:
        perm[1:] = rng.permutation(np.arange(1, n + 1))
    sign = np.ones(n + 1, dtype=np.int64)
    if p.flip:
        sign[1:] = rng.choice(np.array([-1, 1]), size=n)
    var = np.abs(lits)
    new = np.sign(lits) * sign[var] * perm[var]

    m = len(start) - 1
    clause_id = np.repeat(np.arange(m), np.diff(start))
    order_of_clause = rng.permutation(m) if p.shuffle_clauses else np.arange(m)
    # One sort key: integer part = new clause position, fractional part = random rank inside the
    # clause. float64 keeps ~28 fractional bits even with 10^7 clauses: ample for clause lengths.
    if p.shuffle_literals:
        frac = rng.random(len(new))
    else:
        frac = (np.arange(len(new)) - start[clause_id]) / (np.diff(start)[clause_id] + 1.0)
    idx = np.argsort(order_of_clause[clause_id] + frac, kind="stable")
    new = new[idx]
    lengths = np.diff(start)[np.argsort(order_of_clause)]
    new_start = np.concatenate([[0], np.cumsum(lengths)])
    return new, new_start, perm, sign


def unmap_model(model: np.ndarray, perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Translate a model of the perturbed formula into a model of the original one."""
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    var = inverse[np.abs(model)]
    return np.sign(model) * sign[var] * var


def perturb_file(src: Path, dst: Path, p: Perturbation) -> dict:
    """Write the perturbed formula and an ``.npz`` mapping next to it; verify with GBD hashes."""
    import gbdc

    n, lits, start = read_cnf(src)
    new, new_start, perm, sign = perturb_arrays(n, lits, start, p)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_cnf(dst, n, new, new_start, comment=f"satbench perturbation {p.tag}")
    np.savez_compressed(dst.with_suffix("").with_suffix(".map.npz"), perm=perm, sign=sign)
    iso_src, iso_dst = gbdc.isohash(str(src)), gbdc.isohash(str(dst))
    h_src, h_dst = gbdc.gbdhash(str(src)), gbdc.gbdhash(str(dst))
    ok = iso_src == iso_dst and (h_src != h_dst or p.tag.endswith("-id"))
    return dict(
        origin_hash=h_src, hash=h_dst, variant=p.tag, path=str(dst), isohash=iso_dst,
        status="ok" if ok else "isohash_mismatch", variables=n, clauses=len(start) - 1,
    )
