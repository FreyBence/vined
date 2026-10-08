# Model implementation tasks

Implement the visual-to-neural contract in [spec.md](spec.md) and [architecture.md](architecture.md), using [audit.md](audit.md) as the implementation delta. Preserve compatible NEDS transformer infrastructure and optional modes. Work remains inside the model component; preserve the current training call boundary where feasible and report any incompatible consumer contract rather than silently expanding scope.

Validation belongs within each task: inspect affected code and directly execute representative functionality using existing project facilities. Do not write test code or validation infrastructure. Update completed tasks through `[ ] → [x]`.

## Tasks

### [x] T01 — Make session mappings and trainable components model-owned

**Summary:**  
Register all required session projections and output heads at construction, with normal device and parameter handling. Derive session embeddings from explicit configured identities, apply them consistently to singleton and mixed-session batches, and expose recoverable session/population and architectural identity for checkpoint compatibility and intentional adaptation. Keep neuron ordering explicit, reject unknown sessions, and prevent padded output channels from affecting real neural outputs. Remove confirmed unused visual projection parameters as part of this boundary correction. Addresses A02–A04, the output-mapping portion of A06, and the relevant portion of A08.

**Out of scope:**  
Training-owned checkpoint serialization or adaptation workflows, automatic migration of ambiguous historical checkpoints, and speculative removal of helpers whose checkpoint compatibility is unknown.

### [x] T02 — Provide independent encoding with explicit validity

**Summary:**  
Provide visual-only prediction from features, positions, temporal validity, and explicit session identity without requiring neural inputs, targets, loss calculation, or stochastic masking. Preserve log expected spike-count semantics and make neural channel validity available. Carry scientific validity separately from optional training-controlled corruption through attention and any retained convenience loss; include valid zero counts and isolate invalid values from real predictions. Ordinary evaluation must be deterministic, including when masking is disabled. Retain a compatible legacy call adapter where needed without making legacy objectives mandatory. Addresses A01, A05, and the validity portion of A06.

**Out of scope:**  
New training objectives, redesign of optional decoding or multimodal behavior, and changes to training or evaluation orchestration.

### [x] T03 — Make feature dimensions and temporal behavior configuration-consistent

**Summary:**  
Use configured visual width and actual runtime sequence length throughout projections, embeddings, attention positions, and output reshaping. Enforce the configured maximum explicitly, support shorter padded sequences, and make configured temporal context effective while preserving the current non-causal default. Reject malformed dimensions, masks, positions, and incompatible session mappings clearly. Resolve unused context wiring as part of this correction. Addresses A07 and the context portion of A08.

**Out of scope:**  
Upstream feature extraction or temporal alignment changes, new positional architectures, and performance optimization.

### [x] T04 — Publish the implemented model interface

**Summary:**  
Write `interface.md` from the completed implementation, describing public construction and prediction entry points, configuration, shapes, session/neuron identity, validity, log-count output semantics, errors, checkpoint compatibility identity, and optional legacy calls. Confirm compatibility with the directly connected training and evaluation boundaries, including the existing registration and singleton workarounds, without requiring consumer internals to use the model.

**Out of scope:**  
Rewriting authoritative design documents, changing downstream components, or adding implementation history and runtime reports.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.
- T04 depends on T01, T02, T03.
