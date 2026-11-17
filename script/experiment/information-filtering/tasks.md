# Information filtering implementation tasks

### [x] T01 — Publish reproducible training-only selections

**Summary:** Score eligible neurons and complete subregions using grouped
out-of-fold visual readouts; persist fixed selections, fold identities, anatomy,
training constants and source provenance for both delays.

**Out of scope:** Selection optimization, combined filtering and causal claims.

### [x] T02 — Provide independent filtered training and compatible restoration

**Summary:** Reuse production training for four fresh 50-epoch configurations,
publish preflight artifacts and restore the same selected populations for
inference. Reference compatible Experiment 2 checkpoints without retraining.

**Out of scope:** Launching the four full training runs during implementation.

### [x] T03 — Compare fixed full-region predictions and native diagnostics

**Summary:** Evaluate validation trials, fill omitted activity with training
constants, publish paired weighted metrics, curves, plots and scientific reports
with separate native diagnostics and explicit uncertainty.

**Out of scope:** Claiming an empirical winner before the four models are trained.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.
