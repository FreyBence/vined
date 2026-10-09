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

### [x] VF04 — Parallelize feature input preparation

**Summary:**

Expose configurable parallel image fitting and bounded replay read-ahead through extraction and its CLI, sharing one CLIP encoder and preserving record ordering, trial boundaries, sampling, and verified publication.

**Out of scope:**

Multiple model replicas, distributed inference, encoder precision changes, and performance benchmarking infrastructure.

### [x] VF05 — Optimize CUDA inference precision

**Summary:**

Provide automatic CUDA mixed precision with explicit float32 and float16 choices, record effective inference precision in representation provenance, and retain float32 normalized outputs and existing publication contracts. Execute representative GPU extraction and compare outputs with float32.

**Out of scope:**

GPU power or cooling control, concurrent model replicas, encoder/model changes, and benchmarking infrastructure.

### [x] VF06 — Limit sustained GPU extraction load

**Summary:**

Expose configurable CUDA batch duty-cycle pacing through the encoder and CLI, preserving feature values, ordering, and publication behavior. Demonstrate paced extraction on representative replay frames.

**Out of scope:**

System-wide GPU control, automatic temperature regulation, cooling configuration, and performance benchmarking infrastructure.

### [x] VF07 — Reuse features for identical encoder inputs

**Summary:**

Avoid repeated CLIP inference for byte-identical prepared images using within-batch deduplication and a configurable bounded cache. Preserve all observation identities and timestamps, record reuse policy, and directly verify readable normalized outputs and reuse on representative replay frames.

**Out of scope:**

Temporal subsampling, approximate image matching, persistent caches, replay storage changes, and benchmarking infrastructure.

### [x] VF08 — Accelerate CLIP vision attention

**Summary:**

Use PyTorch scaled dot-product attention with the existing fixed CLIP projections, expose an explicit original-backend option, and record the effective backend. Compare representative real-image features and inference time while preserving all frames and current preprocessing.

**Out of scope:**

Model or weight changes, temporal subsampling, dependency upgrades, cooling-policy changes, and benchmarking infrastructure.

### [x] VF09 — Use parallel replay image reading during extraction

**Summary:**

Forward the extraction worker setting to the public replay reader, retaining existing batching, inference, and completion semantics.

**Out of scope:**

Inference changes, session parallelism, and storage-format changes.

## Dependencies

- VF02 depends on VF01.
- VF03 depends on VF01 and VF02.
- VF04 depends on VF03.
- VF05 depends on VF03.
- VF06 depends on VF05.
- VF07 depends on VF05.
- VF08 depends on VF05.
- VF09 depends on VF03 and visual-replay VR08.
