# Training Dataset Module Specification

## 1. Purpose

The `training-dataset` module is responsible for transforming temporally aligned neural and visual data into a deterministic dataset representation suitable for downstream model training and evaluation.

It consumes the output of the `alignment` module and organizes it into training samples while preserving the scientific meaning, identity, timing, and provenance of the aligned observations.

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
              model-ready samples
                      ↓
                 training
```

The module does not perform model inference, optimization, masking objectives, neural or visual encoding, or loss calculation.

---

# 2. Responsibilities

The module is responsible for:

- consuming valid aligned trial representations;
- defining dataset samples;
- preserving trial and session boundaries;
- producing deterministic train / validation / test assignments;
- representing variable temporal sequence lengths;
- applying padding where required for dataset/model compatibility;
- exposing temporal validity masks for padded positions;
- preserving neuron identity and session identity;
- maintaining reproducible sample ordering;
- persisting dataset-level metadata and provenance.

The module must not:

- reconstruct visual stimuli;
- extract visual features;
- bin raw neural spikes;
- realign modalities;
- interpolate missing data;
- redefine the temporal grid;
- perform model-specific embedding;
- construct prediction targets through learned transformations;
- calculate losses;
- train a model.

---

# 3. Input contract

The module consumes aligned trial representations produced by `alignment`.

Each aligned trial must provide at least:

```text
AlignedTrial
    session_id
    trial_id

    stim_on
    stim_off

    aligned_start
    aligned_end

    bin_size
    bin_count

    bin_start_times
    bin_center_times
    bin_end_times

    neural_activity
    neuron_identity

    visual_features

    alignment_metadata
```

For every input trial:

```text
neural_activity.shape[time]
==
visual_features.shape[time]
==
bin_count
```

The module must not attempt to repair an input that violates the alignment contract.

---

# 4. Dataset sample unit

The default atomic dataset sample is one aligned trial.

Conceptually:

```text
DatasetSample
    session_id
    trial_id

    neural
    visual

    timestamps
    neuron_identity

    sequence_length
    temporal_mask

    metadata
```

Trial boundaries must remain explicit.

Observations from different trials must not be implicitly concatenated into a single training sample.

If future experiments require multi-trial or sub-trial samples, such transformations must be explicitly configured and must preserve source trial identity.

---

# 5. Trial integrity

A source trial must remain an atomic identity throughout dataset construction.

The same trial must not appear in more than one dataset split.

For example:

```text
trial 42 → train
```

must make the following invalid:

```text
trial 42 → validation
trial 42 → test
```

This requirement applies even if the trial is later represented through padded or transformed tensors.

---

# 6. Session identity

Session identity must remain available for every sample.

The module must never assume that:

- all sessions contain the same neurons;
- neuron indices have equivalent biological meaning across sessions;
- trials from different sessions belong to a shared continuous timeline.

Session-specific neural identities must remain distinguishable.

A neural channel index is only meaningful together with the corresponding session and neuron / cluster identity.

---

# 7. Temporal representation

The temporal representation produced by `alignment` is authoritative.

The `training-dataset` module must not:

- create a new physical time grid;
- change neural bin duration;
- interpolate visual features;
- extend the aligned interval;
- recreate discarded partial bins.

For each sample, the module must preserve:

- physical bin-center timestamps;
- sequence length;
- neural bin size;
- stimulus-relative temporal origin.

Physical timestamps must remain distinct from model positional indices.

---

# 8. Variable sequence length

Different trials may contain different numbers of aligned temporal bins.

Therefore:

```text
T_trial1 != T_trial2
```

is valid.

The stored scientific representation must preserve the true sequence length of every trial.

A trial must not be shortened merely because another trial is shorter.

---

# 9. Padding

Padding may be applied when downstream batching or model architecture requires equal sequence lengths.

Padding is a representation operation and must not alter the scientific meaning of the original observations.

For a trial with:

```text
sequence_length = T
```

and dataset/model length:

```text
T_max
```

where:

```text
T < T_max
```

the representation may become:

```text
valid data | padding
0 ... T-1  | T ... T_max-1
```

Every padded representation must include an explicit temporal validity mask.

Conceptually:

```text
temporal_mask =
[1, 1, 1, ..., 1, 0, 0, ..., 0]
```

where:

```text
1 = real aligned observation
0 = padding
```

Padding values themselves must never be relied upon to indicate validity.

---

# 10. Padding semantics

Padded positions must not be interpreted as:

- zero neural activity;
- blank visual stimulus;
- missing biological activity;
- additional stimulus duration.

A padded neural value of zero is fundamentally different from a valid neural bin containing zero spikes.

Therefore validity must always be determined from the temporal mask rather than tensor values.

This distinction is mandatory.

---

# 11. Sequence truncation

The module must not silently truncate valid aligned data.

If a requested fixed model sequence length is shorter than an input trial:

```text
T_trial > T_max
```

one of the following must occur:

- dataset construction fails with an explicit configuration error; or
- an explicitly configured segmentation/truncation policy is used.

Implicit removal of valid temporal observations is not allowed.

Any truncation policy must be recorded in dataset provenance.

---

# 12. Visual representation

Visual feature values received from `alignment` must be preserved.

The `training-dataset` module must not:

- re-run CLIP;
- interpolate visual features;
- renormalize features unless explicitly required by a documented dataset transformation;
- infer missing visual representations.

The current aligned visual representation may consist of normalized CLIP feature vectors, but the dataset contract should remain independent of a permanently fixed feature dimension.

Conceptually:

```text
visual.shape = [T, D_visual]
```

---

# 13. Neural representation

Neural activity received from `alignment` must preserve its temporal and neural-channel interpretation.

Conceptually:

```text
neural.shape = [T, N_session]
```

where:

- `T` is the number of aligned temporal bins;
- `N_session` is the number of retained neurons / clusters for that session.

A value of:

```text
neural[t, n] = 0
```

at a valid temporal position means zero detected spikes in that neural bin.

It is not padding and not missing data.

---

# 14. Cross-session neural dimensionality

Different sessions may contain different numbers of neural channels:

```text
N_session_A != N_session_B
```

This is valid.

The persistent dataset representation must not require neuron index `n` to refer to the same biological unit across sessions.

If downstream batching requires neural-dimension padding, that operation must preserve:

- the number of real neurons;
- neuron identity;
- a neural-channel validity mask where required.

Cross-session padding must never create synthetic neurons.

---

# 15. Model-task independence

The dataset must preserve both aligned modalities without deciding which modality is the prediction target.

For example, the same dataset representation may support:

```text
visual → neural
```

or other model objectives without rebuilding the scientific source data.

For the current project, neural activity prediction from visual stimulus is the primary task, but the dataset format must not encode neural activity as a fundamentally different class of stored data solely because it is currently the prediction target.

Model and trainer configuration determine:

- model inputs;
- prediction targets;
- masking objectives;
- training losses.

---

# 16. No training-time masking

Masking strategies used for model training must not be permanently applied during dataset construction.

Examples include:

- temporal masking;
- embedding masking;
- modality masking;
- masked-token objectives.

These belong to the training pipeline.

The persisted dataset must represent the complete available aligned observations.

---

# 17. Dataset splitting

The module is responsible for assigning samples to:

- training;
- validation;
- test.

Split assignment must be:

- deterministic;
- reproducible;
- explicit;
- stored with the dataset or its metadata.

The split process must operate on sample identities rather than tensor positions.

---

# 18. Supported split scopes

The module must support at least two conceptually different evaluation strategies.

## 18.1 Within-session split

Trials from the same session may be divided between:

```text
train
validation
test
```

while ensuring that an individual trial occurs in exactly one split.

This evaluates generalization to unseen trials within a known recording session.

## 18.2 Session-held-out split

Entire sessions may be reserved for validation or testing.

For example:

```text
session A → train
session B → train
session C → test
```

No trial from a held-out session may occur in the training set.

This evaluates cross-session generalization.

The selected strategy must be explicit configuration and part of dataset provenance.

---

# 19. Split reproducibility

If split selection uses randomized assignment, the random seed must be explicit and persisted.

Given identical:

- source samples;
- split strategy;
- ratios;
- grouping constraints;
- random seed;

the resulting split must be identical.

---

# 20. Split stability

Split assignment must be based on stable sample identity.

Reordering source files or changing iteration order must not silently move trials between train, validation, and test sets.

Where practical, stable identity such as:

```text
(session_id, trial_id)
```

should form the basis of deterministic assignment.

---

# 21. Data leakage prevention

Dataset construction must prevent direct sample leakage between splits.

At minimum:

- a trial may appear in only one split;
- a held-out session may not contribute training trials;
- padded or transformed versions of the same source trial may not be distributed across different splits.

Any future transformation that creates multiple samples from one source trial must maintain a shared grouping identity so all derived samples remain in the same split.

---

# 22. Sample ordering

Sample ordering must be deterministic.

Ordering may follow stable identifiers such as:

```text
session_id
trial_id
```

or another explicit deterministic policy.

Dataset correctness must not depend on filesystem enumeration order or unordered container iteration.

---

# 23. Invalid inputs

The dataset module should consume only successfully aligned trials.

Trials that fail the upstream alignment contract must not be silently repaired.

If explicit upstream validity metadata is provided, exclusion must be deterministic and traceable.

The dataset should record sufficient information to determine why expected source trials are absent where such exclusion information is available.

---

# 24. Dataset output

A model-facing sample should conceptually expose:

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
    neuron_mask          # where required

    sequence_length
    split

    metadata
```

The exact serialization and tensor container are implementation details and are not prescribed by this specification.

---

# 25. Sample identity

Each sample must have a stable identity derived from its scientific source.

For the default one-trial-per-sample representation, the identity must uniquely correspond to:

```text
(session_id, trial_id)
```

plus any transformation configuration that would produce a distinct sample representation.

Dataset ordering must not define identity.

---

# 26. Physical timestamps and model positions

The dataset must preserve the distinction between:

```text
physical timestamp
```

and:

```text
model temporal position
```

Example:

```text
physical:
stimOn + 10 ms
stimOn + 30 ms
stimOn + 50 ms

position:
0
1
2
```

Model-relative positions may be generated for downstream convenience, but they must not replace physical timestamps in persisted scientific metadata.

---

# 27. Dataset configuration

Dataset construction configuration may include:

- source sessions;
- inclusion/exclusion policy;
- split strategy;
- train/validation/test ratios;
- split seed;
- fixed sequence length where required;
- temporal padding policy;
- neural-dimension padding policy where required;
- optional explicit segmentation policy.

Any configuration that changes sample content or split membership must be preserved in provenance.

---

# 28. Provenance

The dataset must remain traceable to its aligned source data.

At minimum, provenance should identify:

- source alignment dataset/version;
- included sessions;
- source trials;
- alignment configuration identity;
- dataset construction configuration;
- split configuration;
- random seed where applicable;
- padding configuration;
- any exclusions or transformations affecting sample content.

---

# 29. Determinism

Given identical:

- aligned source data;
- dataset configuration;
- split configuration;
- random seed;

the module must produce equivalent dataset content, sample identity, sample ordering, and split assignments.

Dataset creation must not depend on:

- wall-clock time;
- uncontrolled randomness;
- filesystem enumeration order;
- unordered data structures.

---

# 30. Persistence

The dataset representation must support persistent storage so preprocessing does not need to be repeated for every training run.

Persistence must preserve:

- feature values;
- neural values;
- masks;
- sample identities;
- timestamps;
- neuron identities;
- split assignments;
- configuration;
- provenance.

The specification does not prescribe a concrete storage format.

Existing NEDS persistence formats may be retained where they satisfy this contract.

---

# 31. Relationship to the training pipeline

The `training-dataset` module provides scientific samples.

The training pipeline consumes those samples and determines how they are used.

Conceptually:

```text
training-dataset
      ↓
TrainingSample
      ↓
data loader / batching
      ↓
model input preparation
      ↓
model
      ↓
loss
      ↓
optimization
```

The boundary between dataset construction and training must remain explicit.

---

# 32. Responsibilities outside the module

## `alignment`

Responsible for:

- temporal correspondence between modalities;
- common neural temporal grid;
- visual resampling;
- physical aligned timestamps;
- removal of pre-stimulus neural data;
- removal of partial final neural bins.

## Model/data loader

Responsible for runtime:

- batching;
- device transfer;
- model-specific tensor layout;
- dynamic model masks;
- model-specific embeddings.

Dataset-level padding may be persisted when required, but runtime batching behavior is not part of this module.

## Trainer

Responsible for:

- optimization;
- loss calculation;
- masking objectives;
- choice of prediction target;
- evaluation scheduling;
- checkpointing.

---

# 33. Core invariants

The module must guarantee the following.

### One aligned trial is one default sample

Trial boundaries remain explicit.

### No split leakage

A source trial belongs to exactly one dataset split.

### Session identity is preserved

Neural channel identity is never interpreted independently of session identity.

### Physical timing is preserved

Dataset preparation does not replace physical timestamps with positional indices.

### Padding is explicit

Padding positions are identified through masks.

### Zero neural activity is real data

A valid zero-spike neural bin must never be confused with padding.

### No implicit truncation

Valid aligned observations are not silently discarded to satisfy model shape requirements.

### No temporal realignment

The temporal relationship produced by `alignment` is authoritative.

### No scientific data synthesis

The module does not invent neural or visual observations.

### Deterministic construction

Identical inputs and configuration produce equivalent datasets and splits.

---

# 34. Compatibility principle

The module should preserve existing NEDS dataset and loader behavior where that behavior satisfies this specification.

Existing implementations should not be replaced solely to achieve a cleaner internal design.

Changes are justified when required to:

- satisfy the explicit data contract;
- preserve identity or provenance;
- prevent data leakage;
- distinguish padding from valid zero activity;
- preserve physical timing;
- support the required dataset split strategy.

The later implementation audit is responsible for determining where the current NEDS-derived implementation already satisfies these requirements and where targeted changes are necessary.

---

# 35. Summary

The `training-dataset` module converts aligned neural and visual trial data into reproducible scientific training samples.

Its primary unit is an aligned trial.

The module preserves:

- neural and visual observations;
- physical time;
- session and trial identity;
- neuron identity;
- true sequence length;
- padding validity;
- split membership;
- provenance.

Variable trial durations are supported.

Padding may be introduced for model compatibility but must remain explicitly distinguishable from real observations.

Zero-spike neural bins remain valid biological observations.

Dataset splitting is deterministic and prevents source-trial leakage.

The resulting dataset remains independent of a particular prediction objective, while being directly consumable by the downstream model and training pipeline.