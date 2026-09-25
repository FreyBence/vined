# Model Module Specification

## 1. Purpose

The `model` module defines the neural network architecture responsible for predicting neural activity from temporally aligned visual representations.

The primary task of the current project is:

```text
visual representation
        ↓
       model
        ↓
predicted neural activity
```

or:

```text
visual → neural
```

The module defines the learnable transformation between model inputs and predictions.

It does not prepare scientific source data and does not control the optimization process.

---

# 2. Responsibilities

The module is responsible for:

- accepting model-ready visual sequences;
- representing temporal visual information in a learnable latent space;
- applying the configured neural network architecture;
- preserving temporal correspondence;
- supporting session-aware neural prediction;
- mapping shared latent representations into session-specific neural spaces;
- producing neural predictions with explicit output semantics;
- respecting temporal and neural validity masks;
- exposing trainable parameters to the training module;
- supporting deterministic inference behavior.

The module must not:

- reconstruct visual stimuli;
- extract CLIP features;
- load or bin neural recordings;
- perform temporal alignment;
- create dataset splits;
- perform batching;
- optimize parameters;
- select checkpoints;
- perform final scientific evaluation.

---

# 3. Primary model contract

The required model functionality is neural encoding:

```text
visual features
      ↓
shared model representation
      ↓
session-specific neural prediction
```

The required input modality is:

```text
visual
```

and the required prediction modality is:

```text
neural
```

Legacy NEDS support for:

- decoding;
- multimodal reconstruction;
- self-supervised modality reconstruction;
- other masking objectives;

may be preserved where it remains compatible with the implementation.

These modes are not required by the primary model contract.

---

# 4. Input contract

The model consumes runtime tensors derived from `training-dataset`.

Conceptually:

```text
ModelInput
    visual_features
    temporal_positions
    temporal_mask

    session_id

    optional training_mask
```

Additional metadata may be supplied where required by the implementation.

The model must not rely on reconstructing scientific metadata that was discarded before the forward call.

---

# 5. Visual input

The visual input represents the aligned visual feature sequence.

Conceptually:

```text
visual_features.shape = [B, T, D_visual]
```

where:

- `B` = batch size;
- `T` = aligned temporal sequence length;
- `D_visual` = visual feature dimension.

For the current pipeline, `D_visual` may correspond to the projected global CLIP image embedding dimension.

The model contract must not permanently hard-code a specific visual feature dimension when it can be supplied through configuration.

---

# 6. Temporal semantics

Every visual token corresponds to one neural-grid temporal position produced by `alignment`.

The model must preserve this ordering.

The model must not:

- interpolate observations;
- change the temporal grid;
- shift visual features relative to neural targets;
- create additional scientific time points.

Temporal transformation inside attention or other neural-network operations does not alter the source temporal identity of the tokens.

---

# 7. Physical time and model position

The model may use positional indices such as:

```text
0, 1, 2, ..., T - 1
```

for positional embeddings or other internal temporal representations.

These are model coordinates.

They must remain conceptually distinct from physical timestamps such as:

```text
stimOn + 10 ms
stimOn + 30 ms
stimOn + 50 ms
...
```

The model does not redefine physical time.

---

# 8. Temporal validity

The model must receive an explicit temporal validity mask whenever padded temporal positions are present.

Conceptually:

```text
temporal_mask.shape = [B, T]
```

where:

```text
1 = real aligned observation
0 = padding / invalid runtime position
```

Padding must not be inferred from tensor values.

---

# 9. Padding isolation

Padded temporal positions must not influence predictions for valid observations.

Where the architecture uses attention, recurrence, pooling, or another operation allowing information exchange between temporal positions, padded positions must be excluded appropriately.

Replacing padding with a learned token alone is not sufficient if that token can subsequently influence valid positions.

The model must therefore respect temporal validity during internal computation where required.

---

# 10. Training masks versus validity masks

A training-time mask and a validity mask represent different concepts.

```text
validity mask
```

means:

> whether a real scientific observation exists.

```text
training mask
```

means:

> whether an otherwise valid observation is deliberately hidden or corrupted for a training objective.

The model must keep these concepts distinct.

An invalid or padded position cannot become scientifically valid through training-time masking.

---

# 11. Visual feature projection

The model may transform the source visual feature dimension into a shared internal model dimension.

Conceptually:

```text
R^D_visual
    ↓
visual projection
    ↓
R^D_model
```

The projection is learnable unless explicitly configured otherwise.

Its purpose is model representation, not modification of the persisted scientific visual features.

---

# 12. Shared latent representation

The architecture must provide an internal representation capable of modeling temporal relationships in the visual sequence.

Conceptually:

```text
visual features
      ↓
visual projection
      ↓
temporal model
      ↓
latent sequence
```

The exact internal architecture may preserve the existing NEDS transformer-based implementation.

The specification does not require replacing that implementation with a different architecture.

---

# 13. Temporal modeling

The model must be capable of using information across the aligned temporal sequence.

For the current encoding task, predictions at one temporal position may depend on visual information from other valid temporal positions where permitted by the configured architecture.

Any causal or non-causal temporal restriction must be explicit model configuration.

The model must not silently change between causal and non-causal behavior.

---

# 14. Session identity

Session identity is a required model input whenever neural output spaces are session-specific.

The model must not assume that:

```text
neuron_index = n
```

has the same biological meaning across different sessions.

Neural prediction is interpreted in the context of:

```text
session_id
```

and its associated neuron ordering.

---

# 15. Session-specific neural spaces

Different sessions may contain different neuron populations.

Therefore:

```text
N_session_A != N_session_B
```

is valid.

The model must support mapping a shared latent representation into the neural output space associated with the corresponding session.

Conceptually:

```text
shared latent
     ↓
session-specific mapping
     ↓
neural prediction for session S
```

The existing NEDS stitcher-style mechanism may be preserved where it satisfies this requirement.

The specification does not require a particular class or implementation pattern.

---

# 16. Session dispatch

For every sample, the model must use the neural mapping associated with its explicit session identity.

Session dispatch must not rely on:

- batch position;
- tensor dimensionality alone;
- ordering of unrelated external files without validation;
- implicit global state.

If multiple sessions occur within one batch, predictions must still use the correct session-specific mapping for every sample.

---

# 17. Mixed-session batches

The model may support batches containing samples from multiple sessions.

For such batches, it must correctly:

1. identify samples belonging to each session;
2. apply the corresponding session-specific neural mapping;
3. restore predictions to the original batch ordering.

Samples from different sessions must not accidentally share session-specific parameters unless this sharing is explicitly part of the architecture.

---

# 18. Neural output

The primary model output is predicted neural activity.

Conceptually:

```text
neural_prediction.shape = [B, T, N]
```

where `N` corresponds to the configured neural output representation.

If neural dimensionality is padded across sessions, an explicit neuron-validity representation must accompany the output or be available from the runtime batch.

---

# 19. Neural output semantics

Every predicted neural value must have an explicit statistical meaning.

For the current Poisson-based encoding objective, the natural interpretation is a parameter representing expected spike count for one:

```text
(neural bin, neuron)
```

observation.

The current implementation may represent this quantity in log-space.

The exact parameterization must be explicit and compatible with the configured training objective.

The model must not expose an ambiguous scalar whose interpretation changes between training and evaluation.

---

# 20. Uniform temporal exposure

All aligned neural bins provided to the model represent the same configured temporal duration.

Therefore the model may interpret each output temporal position under the same neural-bin exposure.

The model is not required to compensate for partial final bins because these are removed by `alignment`.

---

# 21. Zero-spike targets

A target value of zero at a valid neural position represents a real biological observation:

> no detected spike occurred during that neural bin.

The model and downstream objective must not interpret valid zero targets as:

- padding;
- missing data;
- invalid neurons.

Validity must be determined through masks, not spike-count values.

---

# 22. Neural channel validity

Where neural representations are padded to a common dimensionality, the model must preserve the distinction between:

```text
real neuron
```

and:

```text
neural-dimension padding
```

Padded neural channels must not be interpreted as synthetic neurons.

A neural validity mask or equivalent explicit representation must determine which output channels correspond to real neural units.

---

# 23. Output ordering

Predicted neural channels must follow the neuron ordering defined for the corresponding session.

The model must not reorder neural units without preserving an explicit reversible mapping.

The output must remain compatible with:

- target neural activity;
- evaluation metrics;
- neuron identity metadata.

---

# 24. Unknown sessions

A session-specific model cannot silently predict an unknown neural population.

If the model encounters a session for which no compatible neural mapping exists, it must fail explicitly unless an explicit adaptation or initialization mechanism has been configured.

Matching tensor dimensions alone are not sufficient evidence of session compatibility.

---

# 25. Cross-session evaluation

Cross-session evaluation is valid only when the model contains or obtains an explicitly defined neural mapping for the evaluation session.

A held-out session must not silently reuse the neural head, stitcher, or session embedding of another recording session.

If evaluation of entirely unseen sessions requires adaptation, that procedure belongs to an explicitly defined training/fine-tuning workflow.

---

# 26. Model output structure

The forward result should conceptually provide:

```text
ModelOutput
    neural_prediction

    optional latent_representation

    optional diagnostic_outputs
```

The primary scientific output is `neural_prediction`.

Intermediate representations may be exposed for analysis or diagnostics but must not be required for ordinary training unless explicitly configured.

---

# 27. Loss boundary

The model defines prediction semantics.

The training system defines how predictions are optimized.

Therefore the model contract must not require optimization policy to be embedded into the architectural forward computation.

A legacy implementation may expose convenience loss calculations, but:

- the loss definition must remain explicit;
- training configuration must determine the active objective;
- unused legacy objectives must not become active implicitly.

For the primary task:

```text
visual → neural
```

only the configured neural prediction objective is required.

---

# 28. Masking support

The model may support training-time masking for inherited or explicitly configured objectives.

When masking is used:

- masking must operate only on valid observations;
- masking policy must be controlled by training configuration;
- masked inputs must remain distinguishable from padded inputs;
- output-target selection must be explicit.

The primary visual-to-neural encoding contract does not require self-supervised masking to be active.

---

# 29. Encoding-only operation

The model must support a clean encoding-only configuration.

In this configuration:

```text
input modality  = visual
output modality = neural
```

Legacy decoding or multimodal reconstruction paths must not be required to execute the encoding forward pass.

Their existence must not introduce additional losses or prediction requirements.

---

# 30. Optional inherited capabilities

Existing NEDS capabilities may remain available, including:

- neural → visual decoding;
- multimodal input;
- modality reconstruction;
- masking-based objectives.

These capabilities are considered optional compatibility features.

They must not alter the behavior of the required encoding configuration.

---

# 31. Model configuration

Architectural behavior must be explicitly configurable.

Relevant configuration may include:

- visual input dimension;
- internal model dimension;
- number of temporal layers;
- number of attention heads;
- feed-forward dimensions;
- dropout;
- positional representation;
- causal/non-causal temporal behavior;
- session embedding behavior;
- session-specific projection configuration;
- neural output parameterization;
- supported sequence length.

Scientifically relevant architectural assumptions must not be hidden in unrelated source code.

---

# 32. Temporal sequence length

The model may impose a maximum supported sequence length.

If such a constraint exists, it must be explicit model configuration.

The model must not silently truncate valid input sequences.

If:

```text
T_input > T_supported
```

the runtime must either:

- reject the input explicitly; or
- use an explicitly configured segmentation mechanism outside the model.

---

# 33. Variable sequence lengths

Where supported by the architecture, shorter valid sequences may be represented through temporal padding and validity masks.

The model must use validity information correctly so padding does not influence real predictions.

A fixed maximum tensor size must not be interpreted as a fixed scientific stimulus duration.

---

# 34. Positional representation

The model may use:

- learned positional embeddings;
- rotary positional representations;
- another explicit temporal encoding.

Its positional representation operates over aligned model positions.

The chosen mechanism must be part of the model configuration and compatible with the supported sequence length.

---

# 35. Session embeddings

Session embeddings may be used to represent recording-session-specific context.

If enabled:

- their mapping to session identity must be explicit;
- every sample from the same session must receive the same session identity;
- a session represented by a single sample in a batch must behave consistently with a session represented by multiple samples.

Batch cardinality must not change whether session information is applied.

---

# 36. Deterministic inference

When the model is in evaluation mode, identical:

- model parameters;
- input tensors;
- masks;
- session identities;

must produce equivalent outputs within the deterministic limits of the numerical framework.

Training-only stochastic operations such as dropout must follow framework train/eval semantics.

---

# 37. Parameter registration

Every trainable module participating in the required forward path must be registered as part of the model.

Required trainable components must therefore:

- appear in model parameter traversal;
- participate in optimizer construction;
- appear in model state persistence.

A trainable component used during encoding must not exist only in an unregistered container that prevents normal parameter or checkpoint discovery.

---

# 38. Checkpoint compatibility

The model must expose enough architectural identity for checkpoint compatibility to be verified.

Relevant compatibility properties may include:

- model dimension;
- layer structure;
- visual input dimension;
- neural output mappings;
- supported sessions;
- neuron dimensionality/order;
- positional representation;
- session embeddings.

A checkpoint must not be considered compatible solely because some parameter tensor shapes match.

---

# 39. Fine-tuning and adaptation

Fine-tuning may initialize a model from a compatible pretrained checkpoint.

Adaptation may replace or initialize session-specific components where explicitly required.

Such operations must preserve clear distinction between:

```text
resume training
```

and:

```text
initialize from pretrained weights
```

The model module must expose components in a way that makes intentional adaptation possible without relying on undocumented parameter mutation.

---

# 40. No scientific preprocessing

The model must not:

- normalize temporal coverage;
- recreate missing stimulus features;
- interpolate visual features;
- bin spikes;
- alter trial boundaries;
- infer session identity;
- change neuron identity.

These are upstream responsibilities.

The model operates on the scientific representation it receives.

---

# 41. Relationship to `training-dataset`

`training-dataset` determines:

> what samples and masks exist.

The model determines:

> how a sample is transformed into a prediction.

The model must respect dataset-provided:

- temporal validity;
- neural validity;
- session identity;
- temporal ordering.

---

# 42. Relationship to `training`

`training` determines:

- optimization;
- active objective;
- optimizer;
- learning rate;
- training-time masking policy;
- validation;
- checkpoint selection.

The model provides:

- trainable architecture;
- forward computation;
- predictions;
- optional intermediate representations.

---

# 43. Relationship to evaluation

Evaluation interprets model predictions using the original neural targets and scientific metadata.

The model must therefore preserve output semantics and neuron ordering sufficiently for metrics to be calculated correctly.

The model itself does not determine the final scientific evaluation protocol.

---

# 44. Failure conditions

The model must fail explicitly when required invariants are violated.

Examples include:

- unsupported visual feature dimension;
- unknown session identity;
- incompatible session-specific neural mapping;
- malformed temporal mask;
- unsupported sequence length;
- mismatching batch dimensions;
- missing required input modality;
- incompatible checkpoint architecture.

These conditions must not be silently repaired through unrelated fallback behavior.

---

# 45. Compatibility principle

The model is substantially derived from the existing NEDS implementation.

Existing:

- transformer layers;
- modality projections;
- positional representations;
- session embeddings;
- stitcher-style session mappings;
- encoding output paths;

should be preserved where they satisfy this specification.

The model must not be rewritten solely to create a cleaner architecture.

Changes are justified when required to:

- provide correct visual-to-neural encoding;
- isolate padding from valid computation;
- guarantee session-correct neural outputs;
- register all required trainable components;
- remove unintended legacy-objective coupling;
- provide explicit output semantics;
- support reproducible checkpoint compatibility.

The later implementation audit determines which existing components already satisfy these requirements and where targeted modifications are necessary.

---

# 46. Core invariants

The model must guarantee the following.

### Encoding is the required task

```text
visual → neural
```

must operate independently of optional legacy objectives.

### Temporal alignment is preserved

The model does not alter scientific correspondence established upstream.

### Padding cannot affect real observations

Invalid temporal positions are excluded appropriately from internal computation.

### Session identity is authoritative

Session-specific neural mappings are selected using explicit session identity.

### Neural identities remain session-specific

Neuron index alone has no cross-session biological meaning.

### Output semantics are explicit

Every predicted neural scalar has a defined interpretation.

### Zero spikes remain valid targets

Zero activity is not confused with missing data or padding.

### Training masks and validity masks remain distinct

Artificial corruption cannot redefine scientific validity.

### Required trainable components are registered

Every learnable component in the required forward path participates in normal parameter and checkpoint handling.

### Unknown sessions are not silently mapped

Cross-session behavior requires an explicit compatible mapping or adaptation procedure.

### Model and training remain separate

The model defines predictions; training defines optimization.

---

# 47. Current project configuration

The current project uses:

```text
input:
    aligned visual feature sequence

primary output:
    predicted neural activity sequence
```

Visual and neural observations already share a common neural-bin temporal grid before entering the model.

The current default alignment resolution is:

```text
20 ms per temporal position
```

but the model contract must not require this specific duration where temporal length and positional representation can remain configuration-driven.

The visual representation is currently based on CLIP features.

The model should preserve the existing NEDS-derived transformer infrastructure wherever compatible with this encoding task.

---

# 48. Summary

The `model` module implements the learnable transformation from aligned visual representations to neural activity predictions.

It operates on already synchronized scientific data.

Its required task is visual-to-neural encoding.

The architecture may preserve the existing NEDS transformer and session-specific stitcher mechanisms, while ensuring that:

- temporal padding cannot contaminate valid observations;
- session identity selects the correct neural output space;
- neuron identity remains interpretable;
- output semantics are explicit;
- all required trainable components are correctly registered;
- optional inherited NEDS objectives remain isolated from encoding-only operation.

Optimization, data preparation, alignment, and final scientific evaluation remain outside the model module.