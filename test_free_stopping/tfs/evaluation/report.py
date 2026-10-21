"""Plain-text reporting for notebook runs.

Phase 1b's tables did not render after a 2.6-hour marimo cell, though every JSON report
reached Hugging Face. Notebooks therefore print each table to the console as it is
produced, and upload a single summary.md alongside the JSON, so results are readable
even when the notebook UI is not.
"""
from __future__ import annotations

import json


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return "" if v is None else str(v).replace("|", "/")


def rows_to_markdown(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def summary_markdown(title: str, tables: dict[str, list[dict]], notes: list[str] = ()) -> str:
    parts = [f"# {title}\n"]
    parts += [f"- {n}" for n in notes]
    for name, rows in tables.items():
        parts.append(f"\n## {name}\n\n{rows_to_markdown(rows)}")
    return "\n".join(parts)
