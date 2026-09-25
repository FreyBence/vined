# Training Module Specification

## 1. Purpose

The `training` module is responsible for executing model training using datasets produced by `training-dataset`.

Its responsibility is to coordinate the training process while keeping scientific data preparation separate from model optimization.

Conceptually:

```text
visual-replay
    ↓
visual-features ─┐
                 ├── alignment
neural-data ─────┘
                      ↓
               training-dataset
                      ↓
                   training
                      ↓
             trained checkpoint
```

The module consumes already prepared and split datasets.

It must not reconstruct stimuli, process raw neural recordings, perform temporal alignment, or redefine dataset samples.

---

# 2. Responsibilities

The module is responsible for:

- loading training, validation, and test-ready dataset samples;
- constructing runtime batches;
- initializing or restoring the configured model;
- defining the configured training objective;
- applying training-time masking or corruption where required;
- executing forward and backward passes;
- calculating training losses;
- updating model parameters;
- running validation during training;
- managing optimizers and learning-rate scheduling;
- saving and restoring checkpoints;
- tracking training state;
- recording training configuration and metrics;
- ensuring deterministic and reproducible execution where technically possible.

The module must not:

- generate or reconstruct visual stimuli;
- extract CLIP features;
- bin raw neural spikes;
- realign modalities;
- change dataset split membership;
- silently alter scientific sample content;
- perform final scientific evaluation on the test set.

---

# 3. Input contract

The module consumes dataset samples produced by `training-dataset`.

A sample may conceptually contain:

```text
TrainingSample
    sample_id

    session_id
    trial_id

    neural
    visual

    physical_timestamps
    temporal_positions

    temporal_mask

    neuron_identity
    neuron_mask

    sequence_length
    split

    metadata
```

The training module may transform this representation into model-specific runtime tensors.

Such transformations must not change the scientific meaning of the source data.

---

# 4. Dataset split usage

The training module must respect dataset split assignments.

The intended roles are:

```text
train       → parameter optimization
validation  → training-time model selection and monitoring
test        → reserved for final evaluation
```

The training process must never optimize model parameters using validation or test samples.

The test set must not be used for:

- gradient updates;
- hyperparameter selection;
- early stopping;
- checkpoint selection;
- threshold selection;
- training decisions.

Final test evaluation belongs to the downstream evaluation stage.

---

# 5. Primary training task

The primary task of the current project is:

```text
visual stimulus representation
        ↓
      model
        ↓
predicted neural activity
```

or conceptually:

```text
visual → neural
```

Training therefore optimizes the model to predict neural activity from visual representations.

The training module must support this encoding objective explicitly.

Support for additional objectives may remain possible, but they must not change the semantics of the primary encoding task unless explicitly configured.

---

# 6. Model boundary

The training module uses a configured model but does not define the scientific architecture of that model.

The model is responsible for:

- modality-specific projections;
- latent representations;
- attention or other internal computation;
- neural prediction heads;
- architectural forward behavior.

The training module is responsible for invoking the model and optimizing its trainable parameters.

Conceptually:

```text
training
    ↓
model.forward(...)
    ↓
prediction
    ↓
loss
    ↓
backpropagation
```

Model architecture and training orchestration must remain separate concerns.

---

# 7. Runtime batching

Training may group dataset samples into batches.

Batch construction may perform runtime operations such as:

- stacking tensors;
- applying dataset-provided masks;
- session-aware grouping where required;
- device transfer;
- runtime layout conversion.

Batching must preserve:

- sample identity;
- session identity;
- temporal validity;
- neuron validity;
- modality correspondence.

Batch construction must not redefine temporal alignment.

---

# 8. Padding handling

Padding introduced by `training-dataset` must remain distinguishable from valid observations.

Training must use the corresponding validity masks when calculating losses or applying model operations where padded positions must not contribute.

A padded zero must never be interpreted as a valid zero-spike observation.

For neural data:

```text
value = 0
mask  = 1
```

means:

> valid bin with zero detected spikes

while:

```text
value = 0
mask  = 0
```

means:

> padded / unavailable position

These cases must remain semantically distinct.

---

# 9. Session-aware neural representation

Different sessions may contain different neurons and different neural dimensionalities.

Training must preserve session identity when applying session-specific model components.

The module must not assume that neural channel index:

```text
n
```

represents the same neuron across different sessions.

If the model uses session-specific projections, stitchers, output heads, or equivalent mechanisms, the correct component must be selected using explicit session identity.

---

# 10. Training objective

The training objective must be explicitly configured.

For the encoding task, the objective compares:

```text
predicted neural activity
```

with:

```text
observed neural activity
```

over valid temporal positions and valid neural channels.

The loss function must not include:

- padded temporal positions;
- padded neural channels;
- explicitly masked-out targets unless required by the configured objective.

The exact loss formulation is model/training configuration and must be recorded as provenance.

---

# 11. Loss composition

If multiple loss components are used, the total objective must be explicit.

Conceptually:

```text
L_total =
    λ1 × L_primary
  + λ2 × L_auxiliary_1
  + ...
```

Every enabled loss component and its weight must be part of the training configuration.

No hidden or implicitly enabled loss contribution is allowed.

For encoding-only training, components unrelated to the configured encoding objective must not become active merely because they exist in the inherited NEDS implementation.

---

# 12. Training-time masking

Training-time masking may be applied when required by the configured objective.

Examples may include:

- temporal masking;
- embedding masking;
- modality masking.

Masking belongs to training and must not modify the persisted source dataset.

Given the same source sample, different training iterations may therefore produce different masked runtime inputs where stochastic masking is enabled.

The masking configuration must be explicit.

---

# 13. Masking and scientific validity

Training masks must remain conceptually separate from dataset validity masks.

These represent different concepts:

```text
dataset validity mask
```

means:

> whether a real scientific observation exists

while:

```text
training mask
```

means:

> whether an otherwise valid observation is temporarily hidden for a training objective

Training logic must not merge these meanings.

Invalid or padded observations must never become valid through training-time masking logic.

---

# 14. Optimizer

The optimizer must be explicitly configured.

Relevant configuration may include:

- optimizer type;
- learning rate;
- weight decay;
- optimizer-specific parameters.

Optimizer state must be included in resumable checkpoints when applicable.

The specification does not prescribe a specific optimizer.

---

# 15. Learning-rate scheduling

Learning-rate scheduling may be used.

If enabled, the scheduler configuration must be explicit and reproducible.

Relevant settings may include:

- scheduler type;
- warm-up;
- decay policy;
- step frequency;
- minimum learning rate.

Scheduler state must be restored when resuming training.

---

# 16. Epochs and training steps

Training duration must be explicitly defined.

This may be expressed through:

- number of epochs;
- number of optimization steps;
- or another explicit stopping criterion.

The configured training duration must be recorded.

The module must not depend on an implicit hard-coded number of epochs inherited from previous experiments.

---

# 17. Validation

Validation may be executed periodically during training.

Validation must:

- use only the validation split;
- disable parameter updates;
- preserve the same scientific input semantics as training;
- use deterministic evaluation behavior where possible;
- calculate explicitly configured validation metrics and losses.

Validation frequency must be configurable.

---

# 18. Checkpoint selection

A checkpoint intended as the trained model output must be selected using an explicit rule.

Possible rules may include:

- lowest validation loss;
- best configured validation metric;
- final training epoch;
- explicitly requested checkpoint.

The rule must be known before test-set evaluation.

The test set must not determine which checkpoint is selected.

---

# 19. Checkpoint contents

A resumable checkpoint should preserve sufficient state to continue the same training run.

Where applicable, this includes:

```text
model parameters
optimizer state
scheduler state
epoch / step
training configuration
dataset identity
random state / seed information
validation history
session/model mapping metadata
```

A checkpoint used only for inference may contain a reduced subset, but its model configuration and provenance must remain recoverable.

---

# 20. Resume training

Training must support continuation from a compatible checkpoint where the inherited implementation provides this capability.

Resume behavior must restore the training state rather than silently starting a new optimization trajectory from model weights only.

Compatibility must be verified before restoration.

Relevant incompatibilities may include:

- model architecture mismatch;
- neural dimensionality mismatch;
- session mapping mismatch;
- incompatible optimizer state;
- incompatible configuration.

Such mismatches must not be silently ignored.

---

# 21. Session compatibility

A checkpoint may contain session-specific parameters.

Training must explicitly verify whether the sessions represented by the current dataset are compatible with the checkpoint.

A session-specific parameter mapping must not be applied to a different session merely because tensor dimensions happen to match.

Session identity is authoritative.

---

# 22. Reproducibility

Training configuration must support explicit random seeds.

Where technically possible, the following sources of randomness should be controlled:

- dataset shuffling;
- masking;
- parameter initialization;
- stochastic model operations;
- framework random generators.

Identical inputs, configuration, software environment, and random seeds should produce equivalent training behavior within the deterministic limitations of the underlying numerical stack.

---

# 23. Data shuffling

Training samples may be shuffled between epochs.

Shuffling must:

- affect only ordering;
- preserve split membership;
- preserve sample identity;
- use controlled randomness.

Validation and test ordering should remain deterministic unless a different behavior is explicitly required.

---

# 24. Metrics during training

Training may calculate diagnostic metrics in addition to the optimization loss.

These metrics are intended for:

- monitoring convergence;
- comparing checkpoints;
- detecting training instability.

Training-time metrics must not be confused with the final scientific evaluation.

The final evaluation module may calculate more extensive or differently aggregated metrics.

---

# 25. Encoding metrics

For the current encoding task, training and validation may expose suitable neural-prediction metrics when useful.

Examples may include quantities derived from:

- prediction loss;
- neural prediction accuracy;
- correlation;
- explained variance;
- likelihood-based metrics.

The specification does not require every final evaluation metric to be calculated during training.

Metrics used for checkpoint selection must be explicitly configured.

---

# 26. Logging

Training must expose sufficient information to understand the progress and outcome of a run.

Logging should include at least:

- run identity;
- epoch / step;
- training loss;
- validation loss when evaluated;
- configured validation metrics;
- learning rate;
- checkpoint events;
- relevant warnings or failures.

Logging must not alter training behavior.

---

# 27. Run identity

Every training run must have a stable identity.

The identity must allow association between:

```text
training configuration
dataset
checkpoint
logs
metrics
```

Separate runs using different configurations must remain distinguishable.

---

# 28. Training configuration

Training behavior must be driven by explicit configuration.

Relevant configuration may include:

```text
model configuration

training mode / objective

epochs
batch size

optimizer
learning rate
weight decay

scheduler

masking configuration

loss configuration

validation frequency
checkpoint frequency

checkpoint-selection rule

random seed

resume checkpoint
```

Values affecting scientific or optimization behavior must not be hidden in unrelated source code where they cannot be reproduced.

---

# 29. Configuration provenance

Every checkpoint intended for later evaluation must be traceable to:

- model configuration;
- training configuration;
- training dataset identity;
- split configuration;
- source sessions;
- random seed;
- training objective;
- loss configuration.

A checkpoint without sufficient provenance must not be treated as a fully reproducible experiment result.

---

# 30. Failure handling

Training must fail explicitly when required invariants are violated.

Examples include:

- malformed batch structure;
- incompatible model and dataset dimensions;
- unknown session identity;
- missing required modality;
- incompatible checkpoint;
- invalid loss configuration;
- NaN or non-finite loss where recovery is not explicitly supported.

Scientific inconsistencies must not be silently repaired inside the training loop.

---

# 31. Numerical failures

Non-finite model outputs or losses must be detectable.

The module must not silently continue optimization when:

```text
loss = NaN
```

or:

```text
loss = ±Inf
```

unless an explicitly documented recovery strategy exists.

The failure must be visible in training logs.

---

# 32. Test-set isolation

The test set is outside normal model training.

The training module may verify that a test split exists, but it must not use test observations to influence optimization or model selection.

Conceptually:

```text
training
    ├── train
    └── validation

evaluation
    └── test
```

This separation must remain explicit.

---

# 33. Output

The primary output of the training module is a trained model checkpoint together with sufficient metadata to reproduce and evaluate it.

Conceptually:

```text
TrainingResult
    run_id

    checkpoint
    model_configuration
    training_configuration

    dataset_identity

    training_history
    validation_history

    selected_checkpoint_info

    provenance
```

The exact persistence format is not prescribed by this specification.

---

# 34. Relationship to `training-dataset`

`training-dataset` defines:

> what scientific samples exist.

`training` defines:

> how those samples are used to optimize a model.

Training must not compensate for errors that belong to dataset construction.

For example:

- temporal misalignment → `alignment`;
- missing visual features → upstream visual pipeline;
- incorrect split → `training-dataset`;
- wrong batch device → `training`;
- incorrect optimizer behavior → `training`.

---

# 35. Relationship to evaluation

Training-time validation and final scientific evaluation are distinct.

Training validation answers questions such as:

> Is optimization improving?

> Which checkpoint should be retained?

Final evaluation answers questions such as:

> How well does the trained model predict neural activity?

> How does performance generalize across trials, neurons, or sessions?

The `training` module must therefore preserve all information required by the later evaluation stage without performing that final analysis itself.

---

# 36. Compatibility principle

The current implementation is derived substantially from NEDS.

Existing trainer, checkpoint, batching, masking, and optimization logic should be preserved where they satisfy this specification.

The module must not be rewritten solely to create a cleaner internal design.

Changes are justified when required to:

- support the current visual-to-neural encoding objective;
- enforce dataset validity masks correctly;
- preserve session identity;
- prevent test-set leakage;
- provide reproducible configuration;
- create compatible checkpoints;
- remove assumptions belonging to obsolete NEDS objectives or modalities.

The later implementation audit determines which existing components already satisfy this contract and which require targeted modification.

---

# 37. Core invariants

The training module must guarantee the following.

### Training uses only training data for optimization

Validation and test observations do not contribute gradients.

### Test data does not influence model selection

Checkpoint or hyperparameter decisions must not depend on final test results.

### Dataset semantics are preserved

Training does not redefine alignment, trial identity, or scientific validity.

### Padding is not data

Padded observations do not contribute as real neural or visual observations.

### Zero spikes are valid data

Valid neural zero values remain distinguishable from padding.

### Session identity is authoritative

Session-specific model parameters are never implicitly shared between unrelated sessions.

### Training masks are not validity masks

Artificial training corruption remains distinct from real data availability.

### Configuration is explicit

Scientifically or numerically relevant training behavior is reproducible from recorded configuration.

### Checkpoints are traceable

A trained model remains associated with the dataset and configuration that produced it.

### Failures are visible

Invalid scientific or numerical states are not silently repaired.

---

# 38. Current project objective

The current project primarily trains an encoding model:

```text
visual representation
        ↓
 neural activity prediction
```

The visual representation originates from reconstructed stimulus observations and their extracted visual features.

The neural target originates from temporally aligned, uniformly binned neural activity.

Both representations are already synchronized before entering the training module.

Training therefore operates on the aligned scientific representation rather than establishing correspondence itself.

---

# 39. Summary

The `training` module orchestrates optimization of the neural encoding model.

It consumes reproducible samples from `training-dataset`, constructs runtime batches, invokes the configured model, calculates explicit objectives, updates model parameters, validates progress, and produces traceable checkpoints.

Scientific data preparation remains upstream.

Final test evaluation remains downstream.

The module preserves the existing NEDS-derived training infrastructure wherever compatible, while requiring targeted changes where inherited behavior conflicts with the current visual-to-neural encoding task, session identity, explicit validity handling, or reproducibility requirements.