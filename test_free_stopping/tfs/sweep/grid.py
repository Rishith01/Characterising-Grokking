"""Sweep YAML -> list[RunConfig] (DESIGN.md §6, "Launcher").

A sweep file is a list of blocks. Each block has a full `base` run config, an optional
list of `variants` (each one is a set of overrides producing one run), and an optional
`product` (dotted path -> list of values, crossed with every variant):

    blocks:
      - base: {task: {...}, model: {...}, optim: {...}, loss: ce, steps: 100000, seed: 0}
        variants:
          - {model.init_scale: 2.0, optim.weight_decay: 0.03}
          - {model.init_scale: 0.5}
        product:
          seed: [0, 1, 2, 3]

Override paths are dotted (`task.params.n_train`). A dict value replaces the whole
sub-tree (so `optim: {name: sgd, lr: 0.1}` swaps the optimizer completely). YAML anchors
(&base / *base) can share bases between blocks.
"""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Union

import yaml

from ..config import RunConfig


def _set(d: dict, path: str, value) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = copy.deepcopy(value)


def expand_block(block: dict) -> list[RunConfig]:
    variants = block.get("variants") or [{}]
    product = block.get("product") or {}
    paths = list(product)
    out = []
    for variant in variants:
        for combo in itertools.product(*(product[p] for p in paths)):
            d = copy.deepcopy(block["base"])
            for path, value in variant.items():
                _set(d, path, value)
            for path, value in zip(paths, combo):
                _set(d, path, value)
            out.append(RunConfig.from_dict(d))
    return out


def expand(spec: Union[str, Path, dict]) -> list[RunConfig]:
    if not isinstance(spec, dict):
        with open(spec) as f:
            spec = yaml.safe_load(f)
    configs = [cfg for block in spec["blocks"] for cfg in expand_block(block)]
    ids = [c.run_id for c in configs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"sweep produces duplicate run ids: {sorted(dupes)}")
    return configs
