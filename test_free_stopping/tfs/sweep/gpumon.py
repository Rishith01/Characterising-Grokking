"""Background GPU utilization logger (nvidia-smi), for sizing sweeps.

    with GpuMonitor(csv_path, every_s=30) as mon:
        ... run things ...
    print(mon.summary())

Each row: wall time, GPU index, utilization %, memory used/total (MiB), power (W), and
the number of compute processes on the GPU. Silently does nothing without nvidia-smi.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

FIELDS = "index,utilization.gpu,memory.used,memory.total,power.draw"


class GpuMonitor:
    def __init__(self, csv_path: Path, every_s: float = 30.0):
        self.csv_path = Path(csv_path)
        self.every_s = every_s
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = shutil.which("nvidia-smi") is not None

    def __enter__(self) -> "GpuMonitor":
        if self.available:
            self._t0 = time.time()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.every_s + 5)
        self._write()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.every_s)

    def _sample(self) -> None:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return
        n_procs = len([l for l in procs.splitlines() if l.strip()])
        for line in out.strip().splitlines():
            idx, util, used, total, power = [x.strip() for x in line.split(",")]
            self.rows.append({"t_s": round(time.time() - self._t0, 1), "gpu": int(idx),
                              "util_pct": _num(util), "mem_used_mib": _num(used),
                              "mem_total_mib": _num(total), "power_w": _num(power), "n_procs": n_procs})

    def _write(self) -> None:
        if not self.rows:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)

    def summary(self) -> dict:
        busy = [r for r in self.rows if r["n_procs"] > 0]
        if not busy:
            return {"samples": len(self.rows), "note": "no samples with GPU processes running"}

        def stats(key):
            v = sorted(r[key] for r in busy if r[key] is not None)
            return {"mean": round(sum(v) / len(v), 1), "p50": v[len(v) // 2], "max": v[-1]} if v else None

        return {"samples": len(busy), "util_pct": stats("util_pct"), "mem_used_mib": stats("mem_used_mib"),
                "mem_total_mib": busy[0]["mem_total_mib"], "power_w": stats("power_w"),
                "procs": stats("n_procs")}


def _num(x: str) -> Optional[float]:
    try:
        return float(x)
    except ValueError:
        return None
