# Training implementation tasks

Implement [spec.md](spec.md) using the gaps identified in [audit.md](audit.md). Preserve compatible NEDS-derived training infrastructure and consume the existing training-dataset and model boundaries.

## Tasks

### [x] T01 — Configure reproducible training on an explicit dataset generation

**Summary:**

Make training and fine-tuning select a verified dataset generation and honor its existing split membership, sample identities, and ordered session populations. Resolve one effective configuration for the encoding objective, optional retained objectives, duration, optimizer, scheduler, masking, loss composition, validation, checkpoint selection, and seed. Apply supported CLI/search overrides consistently, correct configuration paths, and reject unsupported settings. Construct only the loaders required for optimization and configured validation; an empty test split must not block training.

**Out of scope:**

Dataset construction, resplitting, acquisition, alignment, and new training objectives or search infrastructure.

### [x] T02 — Execute validity-aware optimization with consistent scheduling

**Summary:**

Use explicit temporal and neuron validity to control the training objective while keeping stochastic training masks separate and valid zero-spike observations intact. Preserve modality and session correspondence at the existing model boundary, adapting loss handling within training where needed. Make gradient accumulation, partial batches, device/distributed execution, and scheduler steps agree with actual optimizer updates and configured duration. Stop visibly on malformed batches, incompatible dimensions/session identities, or unsupported non-finite outputs and losses. Demonstrate a bounded encoding optimization run on prepared data with padding and a partial batch.

**Out of scope:**

Model architecture changes, upstream scientific transformations, numerical recovery strategies, and performance benchmarking.

### [x] T03 — Validate all eligible observations and select checkpoints explicitly

**Summary:**

Calculate configured validation losses and metrics over valid observations without gradients, including singleton and mixed-session groups. Handle sessions absent from validation and optional plots without silently dropping eligible samples. Use the configured selection rule consistently for the selected checkpoint and reported best values, replacing the unreachable best-checkpoint branch. Keep test observations outside validation and selection, and report unusable selection metrics explicitly. Verify selection using a short training/validation run.

**Out of scope:**

Final scientific evaluation, additional research metrics, and test-driven model selection.

### [x] T04 — Produce traceable run artifacts and resume compatible training

**Summary:**

Give each run a stable identity and isolated artifact location that cannot mix interrupted or differently configured runs. Save effective model/training configuration, dataset generation and split provenance, session and ordered unit mappings, histories, and checkpoint-selection information alongside resumable model, optimizer, scheduler, epoch/step, and applicable random/sampler state. Resume from an explicitly selected checkpoint after verifying compatibility, restoring the optimization and selection trajectory. Demonstrate save/load and continuation of a short run and document the implemented checkpoint and training entry-point contracts in `interface.md`.

**Out of scope:**

Historical checkpoint reconstruction, external experiment-management infrastructure, and changes to evaluation implementation.

### [x] T05 — Adapt compatible pretrained models to an explicitly selected session

**Summary:**

Make fine-tuning load a pretrained checkpoint through training-owned orchestration and the existing model boundary, removing its dependency on evaluation and unrelated test-data loading. Distinguish adaptation from resuming the same run, verify shared model/configuration compatibility, and initialize or retain session-specific input/output parameters according to explicit session and ordered unit identity for the configured encoding direction. Reuse the configured optimization, validation, and artifact behavior from the preceding tasks. Demonstrate a bounded session-adaptation run and finalize `interface.md` with adaptation usage and downstream checkpoint compatibility.

**Out of scope:**

New model architecture or dependencies, downstream evaluation changes, and generalization benchmarks.

### [x] T06 — Run configured scenarios through a shared launcher

**Summary:**
Provide a train/eval launcher accepting only operation and scenario ID, with scenario-first precedence and module configuration defaults. Select explicit runtime neural populations without changing published data or split membership, and record the selected population for compatible checkpoint restoration and evaluation.

**Out of scope:**
Dataset/checkpoint discovery, upstream data regeneration, and changes to model architecture.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.
- T04 depends on T03.
- T05 depends on T04.
- T06 depends on T04.
