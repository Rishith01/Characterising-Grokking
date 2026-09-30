"""Probe scoring protocol (project.md Section 6): (probe, corpus) -> table of lead
time / false-positive rate / seed variance. Not implemented yet -- Phase 3 work,
blocked on the Phase 2 corpus (corpus.jsonl) being built from the deferred runs.

This is also where causal probes get their future-leakage guard enforced in code
(project.md: "the harness must refuse to score a causal probe using anything past
step t" -- not by discipline, since leaking the future is the easiest mistake here
and the hardest to notice).
"""
from __future__ import annotations
