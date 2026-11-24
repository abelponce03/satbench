"""Runner behaviour with fake solvers: every status path must be reachable and correct."""

import lzma
import sys
import textwrap
from pathlib import Path

import pytest

from satbench.runner.execute import Limits
from satbench.runner.run import RunSpec, run
from satbench.runner.solvers import parse_colon_stats, parse_spaced_stats

CNF = "p cnf 3 2\n1 -2 0\n2 3 0\n"  # satisfied by 1 2 3


@pytest.fixture
def instance(tmp_path):
    p = tmp_path / "toy.cnf.xz"
    with lzma.open(p, "wt") as f:
        f.write(CNF)
    return p


def fake_solver(tmp_path: Path, name: str, body: str) -> Path:
    exe = tmp_path / name
    exe.write_text(f"#!{sys.executable}\nimport sys, os, time\n" + textwrap.dedent(body))
    exe.chmod(0o755)
    return exe


def do_run(tmp_path, instance, body, limits=Limits(cpu_seconds=5, memory_mb=200), known="unknown"):
    exe = fake_solver(tmp_path, "solver", body)
    spec = RunSpec("fake", "kissat", exe, instance, "h" * 32, "toy", 1, known_result=known)
    return run(spec, limits, tmp_path / "logs")


def test_correct_sat_is_model_checked(tmp_path, instance):
    r = do_run(tmp_path, instance, """
        print("c conflicts: 7 1.0 per second")
        print("s SATISFIABLE"); print("v 1 2 3 0"); sys.exit(10)
    """)
    assert r["status"] == "SAT" and r["solved"]
    assert r["metrics"]["conflicts"] == 7
    assert r["model_check"] == "2 clauses satisfied"


def test_wrong_model_is_rejected(tmp_path, instance):
    r = do_run(tmp_path, instance, """
        print("s SATISFIABLE"); print("v -1 2 -3 0"); sys.exit(10)
    """)
    assert r["status"] == "WRONG" and not r["solved"]


def test_abort_without_answer_is_a_crash_not_a_solve(tmp_path, instance):
    """An aborted run (SIGABRT, no answer) must never be credited with its runtime."""
    r = do_run(tmp_path, instance, "os.abort()\n")
    assert r["status"] == "CRASH" and not r["solved"]
    assert r["signal"] == 6


def test_memory_limit(tmp_path, instance):
    r = do_run(tmp_path, instance, """
        x = bytearray(400 * 2**20)
        for i in range(0, len(x), 4096): x[i] = 1
        time.sleep(30)
    """)
    assert r["status"] == "MEMOUT" and not r["solved"]


def test_cpu_limit(tmp_path, instance):
    r = do_run(tmp_path, instance, "while True: pass\n", limits=Limits(cpu_seconds=1, memory_mb=200))
    assert r["status"] == "TIMEOUT"
    assert 1.0 <= r["cpu_time"] < 3.0


def test_answer_contradicting_known_result(tmp_path, instance):
    r = do_run(tmp_path, instance, 'print("s UNSATISFIABLE"); sys.exit(20)\n', known="sat")
    assert r["status"] == "WRONG"


def test_peak_memory_excludes_harness(tmp_path, instance):
    r = do_run(tmp_path, instance, 'print("s UNSATISFIABLE"); time.sleep(0.5); sys.exit(20)\n')
    assert r["peak_rss_mb"] < 60  # a bare Python interpreter, not the harness's ~100 MB


def test_competition_mode_uses_private_writable_dir(tmp_path, instance):
    solver_dir = tmp_path / "pkg"
    (solver_dir / "bin").mkdir(parents=True)
    helper = solver_dir / "bin" / "helper"
    helper.write_text("#!/bin/sh\necho 's SATISFIABLE'\necho 'v 1 2 3 0'\nexit 10\n")
    helper.chmod(0o755)
    (solver_dir / "run.sh").write_text(
        '#!/bin/bash\n'
        'echo tmp > "$(basename "$1").temporary.cnf"\n'  # writes into its cwd, like zheng/*
        'echo proof > "$2/proof.out"\n'
        'exec ./bin/helper\n'
    )
    solver_dir.chmod(0o555)  # installed solver is read-only, as in a container image
    spec = RunSpec("pkg", "competition", solver_dir, instance, "h" * 32, "toy", 0)
    r = run(spec, Limits(cpu_seconds=5, memory_mb=200), tmp_path / "logs")
    solver_dir.chmod(0o755)
    assert r["status"] == "SAT", r
    assert not any(solver_dir.glob("*.temporary.cnf"))


def test_stat_parsers():
    kissat = "c conflicts:      8643     6194.64 per second\nc restarts:  257  34 interval\n"
    assert parse_colon_stats(kissat) == {"conflicts": 8643, "restarts": 257}
    cms = "c conflicts                : 1234   (12.0 / sec)\n"
    assert parse_spaced_stats(cms) == {"conflicts": 1234}
    minisat = "conflicts             : 98           (1000 /sec)\n"
    assert parse_spaced_stats(minisat) == {"conflicts": 98}
