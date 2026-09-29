# Visual features implementation plan

Implement [spec.md](spec.md), using [audit.md](audit.md) as the implementation-gap snapshot and the public [visual-replay interface](../visual-replay/interface.md) as the input contract. The outcome is reusable CLIP feature sequences from reconstructed mouse-perspective observations, preserving source identity, timing, coverage, and completion semantics.

Work remains inside `visual-features`, following [component dependencies](../dependencies.md). Reuse conforming fixed-weight inference, batching, numeric validation, and staged compressed persistence. No separate architecture document currently exists; concrete preprocessing and persistence choices must satisfy the specification and be recorded in the implemented interface, without changing authoritative documents. Downstream alignment migration is separate work; expose its required information and check direct caller compatibility without silently weakening coverage or identity guarantees.

Validation belongs inside each task: inspect code, use existing checks, execute representative extraction/readback, and manually inspect prepared images where needed. Do not write tests or validation infrastructure. A full multi-session extraction, model benchmark, or training run is not required for functional completion.

## Tasks

### [x] VF01 — Consume public replay observations with preserved selection semantics

**Summary:**

Accept observations and trial outcomes through the public replay contract, including saved compressed artifacts via `ReplayArtifactReader`, without video decoding or renderer-internal reconstruction. Preserve mouse-view labeling, session/trial/table/observation identities, source-time precision and classification, validity reasons, blank flags, requested domains, coverage, and every requested trial outcome. Support the full sequence and explicitly configured source-time subsampling without losing gaps between selected observations or duplicating observations. Retain the upstream completion result for final publication. Addresses A01 and the input/selection aspects of A03–A04.

**Out of scope:**

Source acquisition, replay rendering or storage changes, MP4 adapters, neural-grid resampling, and automatic migration of legacy inputs.

### [x] VF02 — Extract reproducible CLIP features while preserving image extent

**Summary:**

Produce bounded batches of fixed-weight CLIP image features from the supplied observations using explicit, reproducible image preparation that preserves the complete field of view and spatial proportions. Record effective resize/padding, channel/value conversion, normalization, immutable encoder identity, extracted model output, and feature dimensions. Preserve actual representations of valid blank images; distinguish upstream invalidity from extraction failure and reject unusable encoder outputs. Reuse the current model and normalization where compatible, and demonstrate representative readable features and spatially complete prepared images. Addresses A02 and the encoder aspects of A03–A04.

**Out of scope:**

Encoder fine-tuning, model-family expansion, representation-quality benchmarking, stimulus geometry changes, and alignment interpolation.

### [x] VF03 — Publish and load complete feature generations through the extraction CLI

**Summary:**

Connect the extraction CLI and shell wrapper to unambiguous replay-generation selection and the completed observation/encoder path. Publish compressed variable-length feature outputs only after upstream completion and requested-trial accounting, including zero-feature trial outcomes, coverage, selected observation associations, full representation provenance, and content-associated replay/feature generation identities. Make the public loader validate and return the metadata needed for downstream compatibility and explicit reuse; incompatible legacy artifacts or failed requests must not silently become current results. Preserve bounded resource use and staged publication. Demonstrate extraction and readback from a representative saved replay, inspect direct consumer compatibility, and document the finished CLI, persistence, reader, and failure contracts in `interface.md`. Completes A01, A03–A05.

**Out of scope:**

Alignment implementation changes, recursive downstream cache invalidation, automatic legacy archive conversion, dataset construction, and training.

## Dependencies

- VF02 depends on VF01.
- VF03 depends on VF01 and VF02.
