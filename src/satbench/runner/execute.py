"""Run one solver process under CPU-time, wall-time and memory limits.

The monitor follows the design of ``runsolver`` (Roussel, JSAT 2011): it samples the whole process
tree, sums CPU time and resident memory, and kills the tree when a limit is crossed. Memory is
limited by *measured RSS*, not by ``RLIMIT_AS``: an address-space limit makes ``malloc`` fail inside
the solver and turns a memory-out into an ``abort()``. Such runs end without an
answer and must be reported as memory-outs, not mistaken for crashes or, worse, for solves.

Final CPU time comes from ``wait4`` (exact kernel accounting of the child and its waited
descendants). It is cross-checked against the sampled value, which also covers non-waited
grandchildren. Peak memory is the per-address-space high-water mark ``VmHWM``, sampled from /proc.
A peak reached and released within the last sampling interval before exit can be missed.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class Limits:
    cpu_seconds: float
    memory_mb: float
    wall_seconds: float | None = None  # default: 1.5 x cpu limit + 60 s
    poll_seconds: float = 0.1

    @property
    def wall(self) -> float:
        return self.wall_seconds if self.wall_seconds else 1.5 * self.cpu_seconds + 60


@dataclass
class Measurement:
    limit_hit: str  # "" | "cpu" | "wall" | "memory"
    cpu_time: float  # user + system, seconds
    wall_time: float
    peak_rss_mb: float
    returncode: int | None
    signal: int | None


def _tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def _vm_hwm_kib(pid: int) -> int:
    """Peak resident set of the process's *current* address space (reset by execve).

    ``ru_maxrss`` cannot be used here: the kernel carries the pre-exec peak across ``execve``, so
    it reports the RSS of the forking Python harness (~100 MB) for every solver.
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _sample(procs: list[psutil.Process]) -> tuple[float, float, float]:
    """(CPU seconds, current RSS MB, sum of per-process peak RSS MB) over the tree."""
    cpu = rss = hwm = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            cpu += t.user + t.system
            rss += p.memory_info().rss
            hwm += _vm_hwm_kib(p.pid) * 1024
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            pass
    return cpu, rss / 2**20, hwm / 2**20


def _kill_group(pgid: int) -> None:
    for sig, wait in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        time.sleep(wait)


def execute(
    cmd: list[str],
    limits: Limits,
    stdout: Path,
    stderr: Path,
    cpus: set[int] | None = None,
    cwd: Path | None = None,
) -> Measurement:
    """Run ``cmd`` with the limits; stdout and stderr go to files (never to pipes that could fill)."""
    hard_cpu = int(limits.cpu_seconds) + 30  # kernel backstop only; the monitor acts first

    def preexec() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (hard_cpu, hard_cpu))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if cpus:
            os.sched_setaffinity(0, cpus)

    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL, cwd=cwd,
            start_new_session=True, preexec_fn=preexec,
        )
        ps = psutil.Process(proc.pid)
        peak_rss, sampled_cpu, limit_hit = 0.0, 0.0, ""
        status = rusage = None
        while True:
            pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                status, rusage = st, ru
                break
            cpu, rss, hwm = _sample(_tree(ps))
            sampled_cpu, peak_rss = max(sampled_cpu, cpu), max(peak_rss, rss, hwm)
            elapsed = time.monotonic() - start
            if not limit_hit:
                if sampled_cpu > limits.cpu_seconds:
                    limit_hit = "cpu"
                elif rss > limits.memory_mb:
                    limit_hit = "memory"
                elif elapsed > limits.wall:
                    limit_hit = "wall"
                if limit_hit:
                    _kill_group(proc.pid)
            time.sleep(limits.poll_seconds)
        wall = time.monotonic() - start
        proc.returncode = os.waitstatus_to_exitcode(status)  # keep Popen consistent

    try:  # descendants that outlived the leader
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    cpu_time = max(rusage.ru_utime + rusage.ru_stime, sampled_cpu)
    if not limit_hit and cpu_time > limits.cpu_seconds:
        limit_hit = "cpu"  # finished between two samples, but after the limit
    rc = proc.returncode
    return Measurement(
        limit_hit=limit_hit,
        cpu_time=cpu_time,
        wall_time=wall,
        peak_rss_mb=peak_rss,
        returncode=rc if rc is not None and rc >= 0 else None,
        signal=-rc if rc is not None and rc < 0 else None,
    )
