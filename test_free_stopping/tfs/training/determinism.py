"""Determinism setup and environment capture (DESIGN.md §6.5, PLAN.md §4.1).

`apply()` must run before any CUDA work in the process. Runs record an EnvRecord in
meta.json; replay refuses an incompatible environment unless explicitly overridden.
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .. import cpu_env

CUBLAS_WORKSPACE = ":4096:8"


def apply() -> dict:
    """Put torch into deterministic mode. Returns the flags actually set.

    Entry points should also call tfs.cpu_env.pin() before importing torch: OpenMP reads
    its variables when it loads. Pinning again here covers MKL, which reads its own
    lazily on first use.
    """
    cpu_env.pin()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for determinism")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    # oneDNN has no reproducibility mode; keep CPU math on the MKL/native paths.
    torch.backends.mkldnn.enabled = cpu_env.unpinned()
    if os.environ.get("TFS_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["TFS_NUM_THREADS"]))
    return current_flags()


def current_flags() -> dict:
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        # CPU reductions are partitioned by thread count, so it can change results.
        "num_threads": torch.get_num_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "cpu": cpu_env.snapshot(),
    }


@dataclass(frozen=True)
class EnvRecord:
    device: str  # "cpu" or "cuda"
    torch: str
    cuda: Optional[str]
    cudnn: Optional[int]
    gpu: Optional[str]
    driver: Optional[str]
    python: str
    numpy: str
    platform: str
    flags: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, device: str) -> "EnvRecord":
        dev = torch.device(device)
        on_cuda = dev.type == "cuda"
        return cls(
            device=dev.type,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version() if on_cuda else None,
            gpu=torch.cuda.get_device_name(dev) if on_cuda else None,
            driver=_nvidia_driver() if on_cuda else None,
            python=platform.python_version(),
            numpy=np.__version__,
            platform=platform.platform(),
            flags=current_flags(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvRecord":
        return cls(**d)

    def mismatches(self, other: "EnvRecord") -> list[str]:
        """Fields that differ in a way that can change floating-point results."""
        keys = ["device", "torch", "cuda", "cudnn", "gpu", "driver"]
        out = [k for k in keys if getattr(self, k) != getattr(other, k)]
        if self.flags.get("deterministic_algorithms") != other.flags.get("deterministic_algorithms"):
            out.append("flags.deterministic_algorithms")
        if self.device == "cpu":
            for k in ("num_threads", "mkldnn_enabled"):
                if self.flags.get(k) != other.flags.get(k):
                    out.append(f"flags.{k}")
            mine, theirs = self.flags.get("cpu") or {}, other.flags.get("cpu") or {}
            for k in ("cpu_model", *cpu_env.PINNED):
                if mine.get(k) != theirs.get(k):
                    out.append(f"flags.cpu.{k}")
        return out

    def compatible_with(self, other: "EnvRecord") -> bool:
        return not self.mismatches(other)


def _nvidia_driver() -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_version(path: Path) -> dict:
    """Git hash and dirty flag of the repo containing `path` (None outside git)."""
    def git(*args):
        r = subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--", ".")
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status) if status is not None else None}
