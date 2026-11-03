"""TraceStore: cached signal traces on disk (DESIGN.md §13.3).

Layout under root:  <signal>/<config hash>/config.json
                    <signal>/<config hash>/parts/<run_id>.parquet   (one per run)
                    <signal>/<config hash>/traces.parquet            (pack(): all runs, long format)
The same layout is uploaded to HF under phase4/<dataset>/signals/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .signal import SignalConfig


class TraceStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dir(self, cfg: SignalConfig) -> Path:
        return self.root / cfg.name / cfg.hash()

    def _part(self, cfg: SignalConfig, run_id: str) -> Path:
        return self.dir(cfg) / "parts" / f"{run_id}.parquet"

    def has(self, cfg: SignalConfig, run_id: str) -> bool:
        return self._part(cfg, run_id).exists()

    def write(self, cfg: SignalConfig, run_id: str, df: pd.DataFrame) -> Path:
        d = self.dir(cfg)
        (d / "parts").mkdir(parents=True, exist_ok=True)
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
        path = self._part(cfg, run_id)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # a crash never leaves a half-written part that has() would accept
        return path

    def read(self, cfg: SignalConfig, run_id: str) -> pd.DataFrame:
        return pd.read_parquet(self._part(cfg, run_id))

    def run_ids(self, cfg: SignalConfig) -> list[str]:
        return sorted(p.stem for p in (self.dir(cfg) / "parts").glob("*.parquet"))

    def load(self, cfg: SignalConfig, run_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        ids = self.run_ids(cfg) if run_ids is None else list(run_ids)
        frames = [self.read(cfg, r).assign(run_id=r) for r in ids]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def pack(self, cfg: SignalConfig) -> Path:
        path = self.dir(cfg) / "traces.parquet"
        self.load(cfg).to_parquet(path, index=False)
        return path

    def upload_files(self, cfgs: Iterable[SignalConfig]) -> dict[str, Path]:
        """Remote path -> local file, for the packed traces and their configs."""
        out = {}
        for cfg in cfgs:
            rel = f"signals/{cfg.name}/{cfg.hash()}"
            out[f"{rel}/traces.parquet"] = self.pack(cfg)
            out[f"{rel}/config.json"] = self.dir(cfg) / "config.json"
        return out
