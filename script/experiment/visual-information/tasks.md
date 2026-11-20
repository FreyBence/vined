# E1 implementation plan

Implement the requested retained-feature visual information experiment.
Production extraction, alignment, neural models and saved splits remain unchanged.

### [x] T01 — Assemble the E1 diagnostic view from retained artifacts

**Summary:** Produce the frozen train/validation source-pair and aligned-center
views, qualified labels, observability controls and support/exclusion accounting.
Verify provenance, representative saved pixels and saved interpolation.

**Out of scope:** Stimulus regeneration, extraction, neural changes, resplitting
and test-observation inspection.

### [x] T02 — Run and report the three fixed diagnostic feature variants

**Summary:** Fit fixed linear A/B/C probes with train-only preprocessing and
trial-balanced weighting. Produce metrics, baselines, paired comparisons,
diagnostic plots and a report with confounding and interpretation limits.

**Out of scope:** Neural training, nonlinear searches, new feature extraction,
new dependencies and claims about unavailable pre-normalization features.

## Dependencies

- T02 depends on T01.
