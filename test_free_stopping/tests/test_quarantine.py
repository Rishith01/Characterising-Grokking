"""Static test-data quarantine (DESIGN.md §10): measurement-side code must never reach
the oracle (test) data."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TFS = ROOT / "tfs"
STOPPING = ROOT / "stopping"
# stopping/: everything except the evaluation-side modules heldout.py (B6, the one
# reader of test data) and pilot.py (plots against test curves).
STOPPING_EXEMPT = {STOPPING / "heldout.py", STOPPING / "pilot.py"}
GUARDED = [TFS / "causal", TFS / "storage" / "reader.py",
           TFS / "training" / "replay.py", TFS / "training" / "recorder.py", TFS / "training" / "loop.py",
           STOPPING]


def _modules():
    for g in GUARDED:
        if g.is_file():
            yield g
        elif g.is_dir():
            yield from (p for p in g.rglob("*.py") if p not in STOPPING_EXEMPT)


@pytest.mark.parametrize("path", list(_modules()), ids=lambda p: str(p.relative_to(ROOT)))
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
        # Only the tfs adapter may reach stopping.heldout (for RunHandle.heldout()).
        if path.name != "tfs_source.py" and isinstance(node, ast.ImportFrom) and node.module \
                and node.module.split(".")[-1] == "heldout":
            pytest.fail(f"{path.name} imports stopping.heldout")
