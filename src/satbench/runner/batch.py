"""Run many specs in parallel: one worker per *physical* core, pinned, resumable.

Each worker is pinned to one physical core. Its hyper-thread sibling is reserved and left idle, so
concurrent runs do not share a core's execution units (oversubscribing cores distorts
runtimes). Results are appended as JSON lines, and a rerun skips run_ids
already present, so an interrupted batch (or a preempted cluster job) can simply be restarted.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from .execute import Limits
from .run import RunSpec, run


def physical_cores() -> list[set[int]]:
    """Logical-CPU sets per physical core, restricted to the CPUs this process may use."""
    allowed = __import__("os").sched_getaffinity(0)
    seen, cores = set(), []
    for cpu in sorted(allowed):
        if cpu in seen:
            continue
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        siblings = {cpu}
        if path.exists():
            for part in path.read_text().strip().split(","):
                a, _, b = part.partition("-")
                siblings |= set(range(int(a), int(b or a) + 1))
        siblings &= allowed
        seen |= siblings
        cores.append(siblings)
    return cores


def completed_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    ids = set()
    for line in results.read_text().splitlines():
        try:
            ids.add(json.loads(line)["run_id"])
        except (json.JSONDecodeError, KeyError):
            continue  # a line truncated by a kill is simply redone
    return ids


def run_batch(
    specs: list[RunSpec],
    limits: Limits,
    results: Path,
    logdir: Path,
    workers: int | None = None,
    scratch: Path | None = None,
    leave_free: int = 1,
) -> int:
    """Run all pending specs; returns the number of runs executed now.

    ``leave_free`` physical cores stay idle for the OS and the monitor threads.
    """
    done = completed_ids(results)
    pending = [s for s in specs if s.run_id not in done]
    cores = physical_cores()
    n = min(workers or len(cores) - leave_free, len(cores) - leave_free)
    if n < 1:
        raise RuntimeError(f"not enough physical cores ({len(cores)}) for leave_free={leave_free}")
    slots: Queue[set[int]] = Queue()
    for core in cores[leave_free : leave_free + n]:
        slots.put({min(core)})  # pin to one hardware thread; its sibling stays idle
    lock = threading.Lock()
    results.parent.mkdir(parents=True, exist_ok=True)

    def work(spec: RunSpec) -> None:
        cpus = slots.get()
        try:
            rec = run(spec, limits, logdir, cpus=cpus, scratch=scratch)
        finally:
            slots.put(cpus)
        with lock, open(results, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(work, pending))
    return len(pending)
