# Subregion aggregation implementation tasks

### [x] T01 — Support unfiltered recorded-subregion count targets

**Summary:** Preserve the regional neuron population, expose recorded anatomical
subregions with parent/source identities, and support compatible training,
checkpoint restoration and evaluation without changing existing regional behavior.

**Out of scope:** Upstream regeneration, activity filtering and transformer changes.

### [x] T02 — Provide controlled standalone Experiment 2 stages

**Summary:** Provide configuration-driven preflight, two 50-epoch training
entrypoints and native/common evaluation with reusable exact regional references.
Publish inspectable mappings, reconstruction, resolved settings and provenance.

**Out of scope:** Launching full training during implementation verification.

### [x] T03 — Provide the four-model scientific comparison workflow

**Summary:** Compare matched parent observations for all four configurations,
publish structured metrics, paired trial contrasts, native diagnostics and
visualizations, and document operation and interpretation limits.

**Out of scope:** Experiment 3 filtering, causal inference and independent test confirmation.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.
