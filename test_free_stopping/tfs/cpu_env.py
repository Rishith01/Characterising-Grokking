"""CPU numerical-reproducibility environment, set before torch (and MKL/OpenMP) load.

Phase 1 found CPU runs on Molab that were not reproducible under load, while the same
configs replayed bit-exactly on an idle machine. The defaults let MKL and OpenMP pick
fewer threads than requested when the machine is busy (MKL_DYNAMIC / OMP_DYNAMIC), and
MKL may choose code paths by memory alignment unless Conditional Numerical
Reproducibility is on (MKL_CBWR). Either changes how reductions are split, and so the
low-order bits. Containers also report the host's core count through os.cpu_count(),
which oversubscribes the CPUs actually available.

This module imports nothing heavy, so entry points can call `pin()` before importing
torch. The launcher also passes the same variables to every child process.
"""
from __future__ import annotations

import os
import platform
import subprocess

PINNED = {
    "MKL_DYNAMIC": "FALSE",
    "OMP_DYNAMIC": "FALSE",
    "MKL_CBWR": "AUTO,STRICT",
}


def available_cpus() -> int:
    """CPUs this process may actually use: the affinity mask, further capped by a cgroup
    CPU quota (containers often limit CPU time rather than cores)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2: "<quota> <period>" or "max <period>"
            quota, period = f.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return n


def unpinned() -> bool:
    """Diagnostics only: TFS_CPU_UNPINNED=1 restores the Phase 1 behaviour (no pinning,
    thread share from os.cpu_count()), to reproduce the original CPU drift."""
    return os.environ.get("TFS_CPU_UNPINNED") == "1"


def pin(env: dict | None = None, threads: int | None = None) -> dict:
    """Set the reproducibility variables in `env` (default: this process's os.environ).
    `threads` additionally fixes OMP/MKL/torch thread counts."""
    env = os.environ if env is None else env
    if not unpinned():
        env.update(PINNED)
    if threads is not None:
        n = str(max(1, threads))
        env.update(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, TFS_NUM_THREADS=n)
    return env


def cpu_model() -> str:
    """Architecture plus model name. Phase 1b recorded "unknown" on Molab: its
    /proc/cpuinfo had no "model name" line (ARM and some sandboxes omit it), and
    platform.processor() returned "unknown". Try several sources, keep the first hit."""
    fields = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if ":" in line:
                    k, v = (x.strip() for x in line.split(":", 1))
                    fields.setdefault(k, v)
    except OSError:
        pass
    name = next((fields[k] for k in ("model name", "Model name", "Hardware", "cpu model", "Processor")
                 if fields.get(k)), None)
    if name is None and fields.get("CPU implementer"):
        name = f"implementer {fields['CPU implementer']} part {fields.get('CPU part', '?')}"
    if name is None:
        try:
            out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
            name = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name")), None)
        except (OSError, subprocess.SubprocessError):
            pass
    if name is None:
        proc = platform.processor()
        name = proc if proc and proc != "unknown" else "unknown"
    return f"{platform.machine()}: {name}"


def snapshot() -> dict:
    """What is recorded per run: the pinned variables as actually seen, and the CPU."""
    return {**{k: os.environ.get(k) for k in PINNED}, "cpu_model": cpu_model(),
            "available_cpus": available_cpus()}
