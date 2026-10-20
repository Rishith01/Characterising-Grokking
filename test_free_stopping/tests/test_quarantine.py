"""Static test-data quarantine (DESIGN.md §10): measurement-side code must never reach
the oracle (test) data."""
import ast
from pathlib import Path

import pytest

TFS = Path(__file__).resolve().parents[1] / "tfs"
GUARDED = [TFS / "causal", TFS / "signals", TFS / "rules", TFS / "storage" / "reader.py",
           TFS / "training" / "replay.py", TFS / "training" / "recorder.py", TFS / "training" / "loop.py"]


def _modules():
    for g in GUARDED:
        if g.is_file():
            yield g
        elif g.is_dir():
            yield from g.rglob("*.py")


@pytest.mark.parametrize("path", list(_modules()), ids=lambda p: str(p.relative_to(TFS)))
def test_no_oracle_access(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "evaluation" in node.module:
            pytest.fail(f"{path.name} imports {node.module}")
        if isinstance(node, ast.Import) and any("evaluation" in a.name for a in node.names):
            pytest.fail(f"{path.name} imports tfs.evaluation")
        if isinstance(node, ast.Attribute) and node.attr == "ORACLE":
            pytest.fail(f"{path.name} references layout.ORACLE")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "oracle.parquet" in node.value:
            pytest.fail(f"{path.name} names the oracle file")
        if isinstance(node, ast.Attribute) and node.attr in ("x_test", "y_test"):
            pytest.fail(f"{path.name} touches test tensors ({node.attr})")
