# Training-dataset implementation plan

Produce reproducible, persistent scientific samples from the public alignment boundary, following [spec.md](spec.md) and the gaps identified in [audit.md](audit.md). Work stays inside training-dataset; preserve compatible existing persistence and loader behavior.

## Tasks

### [x] T01 — Construct faithful samples from aligned trials

**Summary:**
Consume explicitly selected generations through `alignment.load_alignment` and produce one stable sample per original session/trial identity. Preserve both modalities without count narrowing, physical timing independently of positional indices, the complete ordered neuron identities, true sequence lengths, and alignment provenance. Support variable lengths and optional temporal/channel padding with explicit validity masks; reject incompatible maxima and malformed or duplicate inputs without repair or silent omission. Keep sample construction independent of model objectives and optional neuron metadata.

**Out of scope:**
Realignment, feature normalization, segmentation/truncation policies, model embeddings, and training masks.

### [x] T02 — Assign reproducible splits by source identity

**Summary:**
Provide within-session and session-held-out train/validation/test strategies using explicit dataset configuration. Make sample ordering and assignments invariant to source enumeration order, enforce trial/session grouping without leakage, and retain strategy, ratios or explicit session assignments, seed, memberships, and any configured exclusions for persistence.

**Out of scope:**
Additional evaluation strategies, class balancing, and runtime sampling behavior.

### [x] T03 — Publish and reload complete dataset artifacts

**Summary:**
Persist samples, masks, identities, timing, split memberships, dataset configuration, and source-generation provenance with verified loading and complete publication. Reuse compatible cache integrity and publication mechanisms while removing assumptions about upstream split Hugging Face artifacts. Migrate `src/create_dataset.py` and its dataset launcher to explicit alignment inputs and dataset-owned options, with clear failures and no model/trainer configuration dependency. Demonstrate construction and reload directly, including preservation of variable lengths, neuron populations, zero spikes, and counts exceeding eight-bit range.

**Out of scope:**
Legacy artifact conversion, new acquisition/cache infrastructure, and performance benchmarking.

### [x] T04 — Expose the dataset handoff for training and evaluation

**Summary:**
Adapt the training-dataset-owned loading boundary to expose persisted splits and samples consistently, retaining compatible model-facing field names where possible without losing scientific fields or mask semantics. Verify the boundary against direct consumer calls and demonstrate loading representative samples through it. Retire or isolate superseded dataset paths only where necessary to make this handoff unambiguous, and document the completed public API, artifact format, configuration, and compatibility limits in `interface.md`.

**Out of scope:**
Changes to training, evaluation, or model internals; runtime batching redesign; and general cleanup of possibly obsolete helpers.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.
- T04 depends on T03.
