"""Hardware event counters for a process tree, via the ``perf_event_open`` syscall (no ``perf``).

Retired user-space instructions are close to a machine-independent measure of the work a solver
does: across x86 hosts they are reproducible to about 1 % (Weaver & McKee, IISWC 2008), whereas
CPU time varies with clock speed, cache and neighbours. Cycles divided by instructions (CPI)
measure how efficiently a host executed that work. Recording both lets the analysis separate
"the solver did less work" from "the host ran it faster".

Counters are attached to the child right after it starts, with ``inherit`` so that every process
and thread it creates later is counted too. Only user space is counted (``exclude_kernel``),
which is allowed for unprivileged processes up to ``perf_event_paranoid`` = 2 or with
CAP_PERFMON. If the host or the container does not allow it, every function degrades to
returning None: counters are an extra, never a reason for a run to fail.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct

_SYS_PERF_EVENT_OPEN = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
_PERF_TYPE_HARDWARE = 0
EVENTS = {"instructions": 1, "cycles": 0}  # PERF_COUNT_HW_INSTRUCTIONS / _CPU_CYCLES
_FLAG_FD_CLOEXEC = 1 << 3

# flag bits of perf_event_attr
_INHERIT, _EXCLUDE_KERNEL, _EXCLUDE_HV = 1 << 1, 1 << 5, 1 << 6


class _Attr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32), ("size", ctypes.c_uint32), ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64), ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64), ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32), ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64), ("config2", ctypes.c_uint64),
    ]


_libc = ctypes.CDLL(None, use_errno=True)


def open_counters(pid: int) -> dict[str, int]:
    """Attach the counters to ``pid`` (and its future descendants). Returns {event: fd}."""
    if _SYS_PERF_EVENT_OPEN is None:
        return {}
    fds = {}
    for name, config in EVENTS.items():
        attr = _Attr(type=_PERF_TYPE_HARDWARE, size=ctypes.sizeof(_Attr), config=config,
                     flags=_INHERIT | _EXCLUDE_KERNEL | _EXCLUDE_HV)
        fd = _libc.syscall(_SYS_PERF_EVENT_OPEN, ctypes.byref(attr), pid, -1, -1,
                           _FLAG_FD_CLOEXEC)
        if fd >= 0:
            fds[name] = fd
    return fds


def read_counters(fds: dict[str, int]) -> dict[str, int | None]:
    """Read and close; with ``inherit``, exited descendants' counts are included."""
    out: dict[str, int | None] = {name: None for name in EVENTS}
    for name, fd in fds.items():
        try:
            out[name] = struct.unpack("Q", os.read(fd, 8))[0]
        except OSError:
            pass
        finally:
            os.close(fd)
    return out


def available() -> tuple[bool, str]:
    """Probe: can this process count the instructions of a child?"""
    import subprocess

    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            paranoid = f.read().strip()
    except OSError:
        paranoid = "?"
    p = subprocess.Popen(["python3", "-c", "sum(range(3_000_000))"])
    fds = open_counters(p.pid)
    p.wait()
    vals = read_counters(fds)
    ok = bool(vals.get("instructions"))
    err = os.strerror(ctypes.get_errno()) if not fds else ""
    return ok, f"paranoid={paranoid} values={vals} {err}".strip()
