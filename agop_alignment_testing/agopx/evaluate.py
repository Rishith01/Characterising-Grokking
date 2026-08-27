"""Probe scoring protocol (project.md Section 6): (probe, corpus) -> table of lead
time / false-positive rate / seed variance. Not implemented in Phase 0 -- this is
Phase 3 work, and depends on corpus.py (Phase 2) existing first.

This is also where causal probes get their future-leakage guard enforced in code
(project.md: "the harness must refuse to score a causal probe using anything past
step t" -- not by discipline, since leaking the future is the easiest mistake here
and the hardest to notice).
"""
from __future__ import annotations
