"""Phase 5: the metrics of new_metrics.md on modular x+y and MNIST (PHASE5_PLAN.md).

Three stages with clear boundaries:
- extract (trajectory.extract): raw data per run, once. Recorded data (telemetry,
  summaries) needs no GPU; the replay pass computes AGOP, the early grid points and the
  Adam+L2 optimizer state in one pass per run.
- quantities (trajectory.quantities): every metric ID as a pure function of the raw
  data, evaluated on the common grid (trajectory.grid). Cheap; re-run whenever a
  definition changes.
- descriptors and plots: read only the curves table (not written yet).

This package is new and self-contained: it reuses low-level pieces of tfs (reader,
replay engine, hub) and stopping (data adapter, AGOP linear algebra), but none of the
old stopping-rule or scoring code.
"""
