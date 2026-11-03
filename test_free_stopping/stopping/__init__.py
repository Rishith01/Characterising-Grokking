"""Phase 4-5: test-free stopping signals, rules and methods (DESIGN.md §13).

The package reads runs only through `stopping.source` (RunSource / RunHandle). The one
adapter onto the training code is `stopping.tfs_source`; nothing else here imports `tfs`
except the evaluation-side modules `heldout.py` and `pilot.py`.
"""
