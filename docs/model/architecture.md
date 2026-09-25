# Model Internal Architecture

## 1. Purpose

This document defines the internal architecture of the `model` module.

The purpose of the architecture is to preserve the important structural relationships inherited from the NEDS-derived implementation while allowing targeted corrections required by `model/spec.md`.

The primary model task is:

```text
visual → neural
```

The architecture is therefore defined primarily around the encoding path.

Legacy decoding and multimodal capabilities may remain available, but they are secondary extensions around the same shared model core.

This document defines architectural responsibilities and relationships.

It does not define:

- implementation tasks;
- source-file changes;
- bug fixes;
- migration steps;
- exact class organization;
- exact hyperparameter values.

Those belong to the later implementation audit and task planning.

---

# 2. Architectural principle

The primary architecture is organized into three stages:

```text
SESSION-SPECIFIC
visual input adaptation
        ↓
────────────────────────────
SHARED
latent temporal representation
────────────────────────────
        ↓
SESSION-SPECIFIC
neural output mapping
```

Different recording sessions may contain different neural populations and may therefore require different mappings into and out of the common latent representation.

The temporal model itself is shared.

This separation allows the model to learn common visual-temporal structure while preserving session-specific neural spaces.

---

# 3. Encoding data flow

The encoding path is conceptually:

```text
aligned visual features
        +
    session identity
        +
 temporal validity
        │
        ▼
session-specific
visual projection
        │
        ▼
common latent width
        │
        ├── modality information
        ├── temporal position information
        └── session context
        │
        ▼
shared temporal transformer
        │
        ▼
shared latent sequence
        │
        ▼
session-specific
neural output mapping
        │
        ▼
predicted neural activity
```

For encoding, the temporal sequence entering the shared model is derived from the visual modality only.

No neural placeholder sequence is required.

The neural modality appears as the prediction space, not as an additional model input.

---

# 4. Visual input representation

The model receives an already aligned visual feature sequence:

```text
X_visual ∈ R^(B × T × D_visual)
```

where:

- `B` is batch size;
- `T` is aligned temporal length;
- `D_visual` is visual feature dimension.

The visual features already correspond to the neural temporal grid.

The model therefore performs no:

- visual resampling;
- interpolation;
- temporal alignment;
- scientific preprocessing.

---

# 5. Session-specific visual projection

Before entering the shared temporal model, visual features are projected into the common model dimension.

Conceptually:

```text
X_visual
    │
    ▼
P_visual(session_id)
    │
    ▼
X_content ∈ R^(B × T × D_model)
```

The projection is selected using explicit session identity.

Therefore different sessions may use different visual projection parameters while producing the same latent dimensionality.

This preserves the existing NEDS concept of session-specific input adaptation.

The projection layer is responsible only for mapping feature representations into the common latent space.

It must not:

- alter temporal correspondence;
- infer session identity;
- perform temporal pooling;
- reconstruct missing features.

---

# 6. Common latent dimension

All session-specific visual projections produce the same internal feature width:

```text
D_model
```

The shared temporal model operates only on this common representation.

This creates a clear architectural boundary:

```text
session-specific input space
        ↓
common latent space
        ↓
session-specific output space
```

The exact value of `D_model` is configuration rather than architecture.

---

# 7. Content and contextual embeddings

Projected visual content and contextual embeddings represent different concepts and remain logically separate.

The model combines:

```text
projected visual content
+
modality information
+
temporal position information
+
session context
```

before shared temporal processing.

Conceptually:

```text
z_t =
    content_t
  + modality_embedding
  + position_embedding_t
  + session_embedding
```

The exact mathematical implementation may differ, but these responsibilities remain distinct.

---

# 8. Modality representation

The inherited model supports multiple modalities.

Modality information identifies the semantic origin of a latent token.

For the primary encoding path all input tokens belong to the visual modality.

Modality representation remains part of the shared infrastructure because optional decoding and multimodal modes may reuse the same latent model.

Encoding must not require additional modalities merely because modality-aware infrastructure exists.

---

# 9. Temporal position representation

The model represents temporal order explicitly.

The temporal coordinates originate from the aligned dataset and correspond to the neural-grid positions:

```text
0, 1, 2, ..., T - 1
```

These coordinates are model positions rather than physical time in seconds.

The architecture may use:

- learned positional representations;
- rotary positional representations;
- or another explicitly configured mechanism.

The existing NEDS design may continue to use both learned position information and rotary attention positions.

The architecture requires only that:

- aligned ordering is preserved;
- temporal coordinates remain deterministic;
- padded positions do not acquire scientific meaning;
- positional representation does not redefine alignment.

---

# 10. Session representation

Session identity appears in two different architectural mechanisms.

## 10.1 Session-specific projections

These determine how representations are mapped between:

```text
session-specific feature/neural spaces
```

and:

```text
shared latent space
```

They perform an actual feature-space transformation.

## 10.2 Session embeddings

Session embeddings provide contextual information to the shared latent representation.

They do not replace session-specific projection.

Conceptually:

```text
session projection
    = mapping between spaces

session embedding
    = contextual identity inside shared space
```

These mechanisms serve separate purposes and must remain conceptually distinct.

Every sample must receive the correct session context independent of batch composition.

---

# 11. Shared temporal model

After session-specific visual adaptation, all samples are processed by a shared temporal model.

Conceptually:

```text
[B, T, D_model]
        ↓
shared temporal processing
        ↓
[B, T, D_model]
```

The current architecture uses a transformer encoder stack.

The shared transformer is responsible for modeling relationships across visual temporal positions.

Its parameters are shared across:

- sessions;
- neural populations;
- compatible model modes.

The transformer must not contain session-specific neural output dimensionality.

Session-specific specialization occurs outside the shared temporal core.

---

# 12. Transformer structure

The inherited temporal core follows a residual transformer structure.

Conceptually:

```text
x
 │
 ├── normalization
 │      ↓
 │   self-attention
 │      ↓
 └──── residual
        │
        ├── normalization
        │      ↓
        │     MLP
        │      ↓
        └──── residual
```

The exact:

- number of layers;
- number of heads;
- hidden width;
- feed-forward width;
- dropout;
- normalization implementation;

are configuration.

The architectural requirement is the presence of a shared temporal processing core operating in the common latent space.

---

# 13. Temporal context

The temporal model may use information across multiple valid stimulus positions.

The current inherited architecture is non-causal.

Therefore a prediction associated with one temporal position may depend on visual information from other positions in the aligned stimulus interval.

Whether temporal processing is causal or non-causal must remain explicit configuration.

It must not change implicitly between training and evaluation.

---

# 14. Scientific validity

Scientific validity describes whether a token corresponds to a real aligned observation.

Validity originates upstream.

The model receives this information and must propagate it to every operation where invalid or padded positions could influence valid positions.

Conceptually:

```text
aligned observation
        ↓
validity = true
        ↓
may participate in temporal computation
```

while:

```text
padding / invalid position
        ↓
validity = false
        ↓
must not influence valid observations
```

Validity is therefore part of the model's temporal-computation boundary, not merely a loss-selection mechanism.

---

# 15. Training corruption

Training-time corruption is independent from scientific validity.

A valid token may deliberately be hidden:

```text
valid observation
        +
training mask
        ↓
masked training token
```

This observation remains scientifically valid.

Therefore the architecture distinguishes:

```text
scientific validity
```

from:

```text
training corruption
```

Mask-token replacement or inherited masking mechanisms may remain part of the model, but they must never redefine whether an observation exists.

The primary visual-to-neural encoding path does not require stochastic masking to operate.

---

# 16. Shared latent sequence

The transformer produces one latent representation for each valid visual temporal position:

```text
Z ∈ R^(B × T × D_model)
```

For encoding:

```text
visual token t
        ↓
latent token t
        ↓
neural prediction at t
```

There are no neural placeholder tokens in the required encoding sequence.

The latent representation is shared across sessions in dimensional structure and temporal processing.

Its interpretation may still contain session context through session embeddings and the surrounding session-specific projections.

---

# 17. Neural output mapping

The shared latent sequence is mapped into the neural space of the corresponding recording session.

Conceptually:

```text
Z
 │
 ▼
P_neural(session_id)
 │
 ▼
neural prediction
```

This mapping is session-specific.

Different sessions may therefore use separate output parameters while sharing the same temporal transformer.

The mapping preserves:

- sample ordering;
- temporal ordering;
- session identity;
- neural-channel ordering.

---

# 18. Neural output space

The neural output conceptually has shape:

```text
[B, T, N_session]
```

or a padded equivalent representation when a common runtime width is required.

`N_session` represents the neural channels belonging to the selected session.

A neural channel position is meaningful only together with:

```text
session_id
+
neuron ordering
```

The model must not treat the same numeric channel index across sessions as the same biological neuron.

---

# 19. Neural output semantics

The primary neural output represents the parameter used to predict spike counts in one aligned neural time bin.

For the current encoding objective, the output semantics are:

```text
log expected spike count
```

per:

```text
(time bin, neuron)
```

Conceptually:

```text
prediction = log(λ)
```

and:

```text
λ = expected spike count
```

for the corresponding neural bin.

This statistical interpretation is part of the model-output contract.

The optimization policy using this representation belongs to `training`.

---

# 20. Mixed-session batches

A runtime batch may contain samples from multiple sessions.

Session-specific components therefore operate through explicit dispatch.

Conceptually:

```text
session A visual
    ↓ projection A ─┐

session B visual
    ↓ projection B ─┼→ shared transformer

session C visual
    ↓ projection C ─┘
                      │
                      ├→ neural mapping A
                      ├→ neural mapping B
                      └→ neural mapping C
```

Dispatch must preserve original batch ordering.

A sample must always return to the same logical batch position after session-specific processing.

---

# 21. Session adaptation boundary

The architecture separates:

```text
session-specific adaptation
```

from:

```text
shared learned temporal representation
```

This boundary provides the natural adaptation point for new compatible sessions.

Conceptually:

```text
new session
    ↓
new / adapted input projection
    ↓
preserved shared temporal core
    ↓
new / adapted neural output mapping
```

Fine-tuning may therefore adapt session-specific mappings while reusing compatible shared model parameters.

The exact fine-tuning procedure is not defined by this architecture.

---

# 22. Encoding-only path

The required encoding path must remain independently executable:

```text
visual
    ↓
visual projection
    ↓
shared temporal model
    ↓
neural output mapping
    ↓
neural prediction
```

It must not require execution of:

- neural input projection;
- visual reconstruction;
- neural-to-visual decoding;
- multimodal reconstruction;
- self-supervised reconstruction objectives.

Optional inherited components may still be constructed or available where this does not affect encoding semantics.

---

# 23. Optional decoding path

The inherited architecture may retain a decoding path:

```text
neural
    ↓
neural input projection
    ↓
shared temporal model
    ↓
visual output mapping
```

This path is optional compatibility functionality.

It must not constrain the scientific contract of visual-to-neural encoding.

---

# 24. Optional multimodal path

The inherited architecture may also retain multimodal operation.

Conceptually:

```text
neural tokens ─┐
               ├→ shared transformer
visual tokens ─┘
```

Multiple modality sequences may share:

- latent dimensionality;
- temporal representation;
- transformer layers.

Multimodal operation is not required by the primary encoding architecture.

Its presence must not introduce additional required objectives into encoding mode.

---

# 25. Shared core and optional branches

The architectural relationship between modes is therefore:

```text
                   ┌─ visual input → encoding
                   │
session adapters ──┼─ neural input → decoding
                   │
                   └─ neural + visual → multimodal
                              │
                              ▼
                    shared temporal core
                              │
                   ┌──────────┼──────────┐
                   ▼          ▼          ▼
                 neural     visual     optional
                 output     output      MM heads
```

The shared core may be reused.

The branches remain direction-specific.

---

# 26. Model and loss boundary

The architectural forward computation produces predictions.

Loss calculation may remain colocated with the model implementation for compatibility, but it is not part of the essential architectural dependency graph.

Conceptually:

```text
model architecture
      ↓
prediction
      ↓
training objective
      ↓
loss
```

The model output must therefore remain interpretable independently of whether a loss is calculated in the same function call.

Optimization behavior belongs to `training`.

---

# 27. Model and masking boundary

The trainer may request training corruption.

The model may implement the actual latent replacement.

Conceptually:

```text
training
    ↓
masking policy / request
    ↓
model
    ↓
token corruption
```

The model must not independently decide the scientific training objective.

Mask execution and mask policy therefore remain separate responsibilities.

---

# 28. Trainable component ownership

Every trainable component used by the required encoding path belongs to the model parameter hierarchy.

This includes:

- visual projection;
- additive embeddings;
- temporal transformer;
- neural output mapping;
- other trainable components participating in encoding.

The architectural requirement is that all required learnable components participate consistently in:

- optimization;
- device movement;
- checkpoint persistence;
- parameter traversal.

The concrete PyTorch container used to achieve this is an implementation detail.

---

# 29. Checkpoint identity

A model checkpoint represents more than transformer weights.

Its interpretation depends on architectural identity including:

- visual representation dimension;
- common latent dimension;
- temporal-model configuration;
- positional representation;
- session identities;
- session-specific mappings;
- neural channel dimensionality and ordering;
- output semantics.

These relationships must remain recoverable when loading or adapting a checkpoint.

The exact checkpoint serialization belongs to `training`.

---

# 30. Current implementation correspondence

The existing NEDS-derived implementation approximately maps the architecture to:

```text
visual feature
    ↓
EncoderEmbedding
    ↓
EncoderEmbeddingLayer
    ↓
StitchEncoder
    ↓
modality / position / session embeddings
    ↓
MultiModal temporal encoder
    ↓
EncoderLayer stack
    ↓
StitchDecoder
    ↓
neural prediction
```

These class names describe the current implementation.

They are not themselves architectural requirements.

Future code organization may change while preserving the relationships defined in this document.

---

# 31. Architectural invariants

The model architecture must preserve the following.

## Session-specific input adaptation

Visual representations are mapped into the common latent space through the mapping associated with their session.

## Shared temporal core

Temporal representation learning is shared across sessions.

## Session-specific neural output

The shared latent representation is mapped into the correct neural space for each session.

## Separate session mechanisms

Session embeddings and session-specific projections remain conceptually different mechanisms.

## Visual-only encoding

The required encoding path operates from visual tokens without neural placeholder tokens.

## Explicit temporal representation

Aligned temporal positions remain represented consistently inside the model.

## Validity reaches temporal mixing

Invalid or padded positions must not influence real observations through temporal operations.

## Validity and corruption remain separate

Training masks do not redefine scientific validity.

## Explicit neural-output semantics

Encoding output represents log expected spike counts for the configured neural bin.

## Optional modes remain optional

Decoding and multimodal capabilities may reuse the shared infrastructure without becoming dependencies of encoding.

## Required trainable components belong to the model

Every learnable component participating in encoding is included in normal parameter and checkpoint handling.

---

# 32. Responsibilities outside this architecture

The model architecture does not determine:

### Scientific temporal alignment

Owned by `alignment`.

### Sample construction and padding policy

Owned by `training-dataset`.

### Optimization

Owned by `training`.

### Active training objective

Owned by `training`.

### Checkpoint selection

Owned by `training`.

### Final scientific metrics

Owned by evaluation.

### Specific implementation corrections

Determined by the implementation audit.

---

# 33. Audit boundary

The later implementation audit must compare the existing NEDS-derived implementation against this architecture and `model/spec.md`.

Implementation-specific behavior must not automatically become intended architecture.

Examples of issues that belong to the audit rather than this document include:

- incorrect parameter registration;
- incomplete attention validity handling;
- batch-cardinality-dependent session behavior;
- fragile session-identity lookup;
- inconsistent configuration sources;
- fine-tuning assumptions about available modalities;
- checkpoint omissions;
- unnecessary coupling between predictions and loss calculation.

The audit determines whether such behavior violates the intended architecture and what targeted changes are necessary.

---

# 34. Summary

The model architecture preserves the central NEDS-derived structure:

```text
session-specific visual adaptation
        ↓
shared latent temporal model
        ↓
session-specific neural prediction
```

Visual features are projected into a common latent dimension using session-aware mappings.

Modality, temporal-position, and session context are added without changing the scientific alignment of the sequence.

A shared transformer models temporal relationships.

The resulting latent sequence is mapped into the session-specific neural space and interpreted as log expected spike counts.

Scientific validity is propagated independently from optional training corruption.

Decoding and multimodal paths may remain as optional extensions around the shared core.

The architecture intentionally defines stable relationships rather than concrete class layout, allowing the existing NEDS implementation to be preserved where compatible while leaving implementation defects and targeted corrections to the later audit.