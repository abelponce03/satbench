import itertools

import numpy as np
import pytest

from satbench.instances.perturb import (
    Perturbation, perturb_arrays, perturb_file, read_cnf, unmap_model, write_cnf,
)
from satbench.runner.verify import check_model


def random_cnf(rng, n=6, m=14, k=3):
    clauses = []
    for _ in range(m):
        vs = rng.choice(np.arange(1, n + 1), size=rng.integers(1, k + 1), replace=False)
        clauses.append([int(v) * int(rng.choice([-1, 1])) for v in vs])
    lits = np.array([lit for c in clauses for lit in c], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum([len(c) for c in clauses])])
    return n, lits, start


def models(n, lits, start):
    """All satisfying assignments, by brute force (n <= ~12)."""
    clauses = [lits[start[i]:start[i + 1]] for i in range(len(start) - 1)]
    out = []
    for bits in itertools.product([False, True], repeat=n):
        val = np.array([False, *bits])
        if all(any(val[abs(x)] == (x > 0) for x in c) for c in clauses):
            out.append(np.array([v if b else -v for v, b in zip(range(1, n + 1), bits)]))
    return out


@pytest.mark.parametrize("seed", range(25))
def test_satisfiability_and_model_count_are_preserved(seed):
    rng = np.random.default_rng(seed)
    n, lits, start = random_cnf(rng)
    new, new_start, perm, sign = perturb_arrays(n, lits, start, Perturbation(seed))
    orig_models, new_models = models(n, lits, start), models(n, new, new_start)
    assert len(orig_models) == len(new_models)  # a bijection, not just equisatisfiability
    orig_set = {tuple(sorted(m.tolist())) for m in orig_models}
    for m in new_models:  # every perturbed model maps back to an original model
        assert tuple(sorted(unmap_model(m, perm, sign).tolist())) in orig_set


def test_clause_multiset_is_a_relabelling(tmp_path):
    rng = np.random.default_rng(1)
    n, lits, start = random_cnf(rng, n=8, m=30)
    p = Perturbation(7)
    new, new_start, perm, sign = perturb_arrays(n, lits, start, p)
    relabel = lambda c: frozenset(int(np.sign(x) * sign[abs(x)] * perm[abs(x)]) for x in c)
    before = sorted(sorted(relabel(lits[start[i]:start[i + 1]])) for i in range(len(start) - 1))
    after = sorted(sorted(frozenset(new[new_start[i]:new_start[i + 1]].tolist()))
                   for i in range(len(new_start) - 1))
    assert before == after


def test_file_roundtrip_hashes_and_model_check(tmp_path):
    rng = np.random.default_rng(3)
    n, lits, start = random_cnf(rng, n=10, m=20)
    src = tmp_path / "orig.cnf.xz"
    write_cnf(src, n, lits, start)
    assert read_cnf(src)[0] == n
    info = perturb_file(src, tmp_path / "pert.cnf.xz", Perturbation(5))
    assert info["status"] == "ok" and info["hash"] != info["origin_hash"]
    # any model of the perturbed file, mapped back, satisfies the original file
    n2, l2, s2 = read_cnf(tmp_path / "pert.cnf.xz")
    found = models(n2, l2, s2)
    maps = np.load(tmp_path / "pert.map.npz")
    for m in found[:5]:
        assert check_model(src, unmap_model(m, maps["perm"], maps["sign"]))[0]


def test_seed_determinism():
    rng = np.random.default_rng(0)
    n, lits, start = random_cnf(rng)
    a = perturb_arrays(n, lits, start, Perturbation(11))[0]
    b = perturb_arrays(n, lits, start, Perturbation(11))[0]
    c = perturb_arrays(n, lits, start, Perturbation(12))[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)
