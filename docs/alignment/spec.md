# Alignment Module Specification

## 1. Purpose

The `alignment` module is responsible for temporally aligning the neural activity representation produced by `neural-data` with the visual feature representation produced by `visual-features`.

Its responsibility is to create a shared temporal representation in which neural and visual observations refer to the same stimulus interval and temporal positions.

The module operates only on already prepared modality-specific data.

Conceptually:

```text
visual-replay
    ↓
visual-features ───────────┐
                           │
                           ▼
                       alignment
                           │
                           ▼
                  aligned multimodal data
                           ▲
                           │
neural-data ───────────────┘
```

---

## 2. Responsibilities

The module is responsible for:

- establishing the aligned stimulus interval;
- defining the common temporal grid;
- mapping visual features onto neural temporal positions;
- preserving explicit physical timestamps;
- preserving session and trial identity;
- validating temporal coverage;
- producing synchronized neural and visual representations;
- preserving the alignment configuration and relevant temporal provenance.

The module must not:

- reconstruct visual stimuli;
- generate missing visual observations;
- load raw electrophysiological recordings;
- infer missing neural activity;
- perform model-specific projection or tokenization;
- perform train/validation/test splitting;
- construct batches.

---

## 3. Input modalities

The module consumes two independently prepared representations:

- neural activity from `neural-data`;
- visual features from `visual-features`.

Both modalities must belong to the same session and trial before they can be aligned.

Alignment between different sessions or unrelated trials is invalid.

---

# 4. Neural input contract

The neural representation must provide:

- session identity;
- trial identity;
- neural activity values;
- neuron / cluster identity;
- neural bin size;
- temporal reference sufficient to reconstruct the physical interval of every neural bin.

Neural activity is interpreted as observations over temporal bins.

A neural bin with zero detected spikes is a **valid neural observation**.

```text
spike_count = 0
```

means that no detected action potential occurred within that neural bin.

It must never be interpreted as:

- missing data;
- invalid data;
- unavailable recording.

Within the declared alignment interval, neural recording coverage is assumed to be complete.

---

# 5. Visual input contract

The visual representation must provide:

- session identity;
- trial identity;
- visual feature vectors;
- explicit timestamps in the session temporal reference;
- sufficient temporal coverage for alignment.

The visual feature stream is expected to be complete over the requested stimulus interval.

Missing expected visual observations are considered an **upstream preprocessing or stimulus-reconstruction failure**.

The `alignment` module must not repair missing visual observations by:

- extrapolation;
- interpolation across missing expected observations;
- synthetic feature generation;
- repetition of unrelated observations.

Temporal interpolation is used for **resampling**, not for missing-data reconstruction.

---

# 6. Alignment interval

The aligned representation begins at stimulus onset.

```text
aligned_start = stimOn
```

Neural activity occurring before `stimOn` is outside the aligned representation.

It must not be exposed to downstream model components as invalid, masked, padded, or otherwise present temporal data.

From the perspective of the aligned dataset:

```text
time = 0
```

corresponds to stimulus onset.

Thus:

```text
raw neural timeline

────── pre-stimulus ──────│──── stimulus-related neural data ────>
                          │
                       stimOn


aligned timeline

                          │0────1────2────3────...──>
                          │
                       stimOn
```

The neural and visual aligned representations therefore share the same temporal origin.

---

# 7. Alignment end

The nominal end of the alignment interval is the stimulus offset:

```text
stimOff
```

However, aligned neural observations must use a uniform neural bin duration.

If the stimulus interval is not an integer multiple of the neural bin size, only complete neural bins are retained.

Let:

```text
D  = stimOff - stimOn
Δt = neural bin size
```

The number of aligned bins is:

```text
T = floor(D / Δt)
```

and:

```text
aligned_end = stimOn + T × Δt
```

Therefore:

```text
aligned_end <= stimOff
```

and the discarded trailing duration is:

```text
tail = stimOff - aligned_end
```

with:

```text
0 <= tail < Δt
```

For the current default neural bin size:

```text
Δt = 1/60 second ≈ 16.67 ms
```

the maximum discarded trailing duration is therefore strictly less than 1/60 second (approximately 16.67 ms).

---

# 8. Partial final bins

A final neural interval shorter than the configured neural bin size must not be included as an ordinary neural observation.

For example:

```text
stimulus duration = 1013 ms
bin size          = 1/60 second ≈ 16.67 ms
```

produces:

```text
60 complete bins = 1000 ms
discarded tail   =   13 ms
```

Conceptually:

```text
| 16.67 | 16.67 | ... | 16.67 | 13 |
                              ↑
                           discarded
```

The partial bin is excluded because its neural spike count would represent a different observation duration from all other bins.

Uniform bin duration is therefore a required invariant of the aligned neural representation.

The exact `stimOff` must still be preserved as metadata.

---

# 9. Common temporal grid

The neural bin grid defines the common temporal grid.

The alignment module must not create an independent temporal grid unrelated to the neural representation.

For a bin beginning at:

```text
bin_start[k]
```

with duration:

```text
Δt
```

the common query timestamp is its center:

```text
t[k] = bin_start[k] + Δt / 2
```

For stimulus-relative alignment:

```text
t[k] = stimOn + (k + 0.5) × Δt
```

for:

```text
k = 0, 1, ..., T - 1
```

The physical timestamps must remain distinct from model-relative positional indices.

For example:

```text
physical timestamp:
stimOn + 1/120 second (≈ 8.33 ms)
stimOn + 3/120 seconds (25 ms)
stimOn + 5/120 seconds (≈ 41.67 ms)
...

model position:
0
1
2
...
```

These concepts must not be represented as if they were equivalent timestamps.

---

# 10. Neural temporal semantics

Each aligned neural observation represents neural activity over exactly one complete neural bin:

```text
[start, end)
```

with:

```text
end - start = Δt
```

The associated alignment timestamp is the bin center.

For the current default configuration:

```text
Δt = 1/60 second ≈ 16.67 ms
```

the first bins are:

```text
bin 0: [stimOn,          stimOn + 1/60 second)
        center = stimOn + 1/120 second (≈ 8.33 ms)

bin 1: [stimOn + 1/60 second, stimOn + 2/60 seconds)
        center = stimOn + 3/120 seconds (25 ms)
```

and so on.

---

# 11. Visual temporal resampling

Visual features may have a lower sampling frequency than the neural grid.

They are therefore mapped onto neural bin-center timestamps through temporal resampling.

The required visual representation at a neural query timestamp is calculated using normalized linear interpolation between the two temporally adjacent visual feature observations.

For adjacent visual features:

```text
v_i     at t_i
v_i+1   at t_i+1
```

and query:

```text
t_i <= q <= t_i+1
```

the interpolated feature is conceptually:

```text
v(q) = normalize(
    (1 - α) × v_i
    +
    α × v_i+1
)
```

where `α` is determined by the temporal position of `q` between the two source timestamps.

The resulting vector is L2-normalized.

---

# 12. Meaning of visual interpolation

Linear interpolation is used only to map a regularly sampled visual feature representation onto the denser neural temporal grid.

It must be interpreted as:

> a temporal estimate in visual feature space.

It must not be interpreted as:

- a newly observed visual frame;
- an independently extracted CLIP feature;
- evidence that the visual stimulus was sampled at the neural sampling frequency.

For example, visual features sampled at 5 Hz and resampled onto a 60 Hz neural grid remain derived from the original 5 Hz visual observations.

---

# 13. Visual continuity

A valid visual stream must provide sufficient temporally consecutive feature observations to evaluate every required neural bin-center timestamp.

Conceptually:

```text
visual features

●──────●──────●──────●──────●
```

is valid.

An unexpected missing observation such as:

```text
●──────●─────────────●──────●
              ↑
          missing data
```

is not considered a normal alignment case.

It is an upstream data-generation failure.

The visual reconstruction and feature extraction pipeline is responsible for guaranteeing temporal continuity over the requested alignment interval.

---

# 14. No visual extrapolation

The alignment module must not extrapolate visual features outside valid visual temporal coverage.

Every neural bin-center timestamp included in the aligned output must have sufficient visual source coverage for the configured interpolation operation.

If this requirement is not satisfied, the input does not satisfy the alignment contract.

The solution must occur upstream rather than by synthesizing data in the alignment module.

---

# 15. Stimulus-end coverage

The visual pipeline must preserve sufficient representation of the final visible stimulus state to support alignment of all complete neural bins before `aligned_end`.

Visual feature generation may therefore include a terminal representation when necessary to guarantee temporal coverage.

This does not extend the neural alignment interval beyond available neural data and does not introduce neural observations after `stimOff`.

---

# 16. Trial boundaries

Alignment is performed within individual trials.

Temporal interpolation must never cross trial boundaries.

Visual observations belonging to one trial must not be used to produce aligned values for another trial even when session timestamps are adjacent.

Trial identity forms part of the alignment key.

---

# 17. Session consistency

Both modalities must originate from the same recording session.

The module must reject alignment when:

```text
visual.session_id != neural.session_id
```

unless the mismatch has been explicitly resolved before entering the module.

Cross-session temporal alignment is outside the scope of this module.

---

# 18. Output representation

The aligned output must conceptually provide:

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
    discarded_tail_duration

    bin_start_times
    bin_center_times
    bin_end_times

    neural_activity
    neuron_identity

    visual_features

    alignment_metadata
```

The specification does not prescribe the concrete serialization or in-memory data structure.

---

# 19. Output shape relationship

For every aligned trial:

```text
neural_activity.shape[time]
==
visual_features.shape[time]
==
number_of_complete_neural_bins
```

Thus both modalities expose one aligned temporal observation per neural bin.

Their feature dimensions may differ.

Conceptually:

```text
neural:
[T, N]

visual:
[T, D_visual]
```

where:

- `T` is the number of complete neural bins inside the aligned interval;
- `N` is the number of neural channels / neurons;
- `D_visual` is the visual feature dimension.

---

# 20. Temporal ordering

Output observations must be strictly ordered in time.

For consecutive bins:

```text
center_time[k] < center_time[k + 1]
```

and:

```text
start_time[k + 1] = end_time[k]
```

for the uniform aligned neural grid.

Ambiguous or non-monotonic source timestamps are invalid input.

---

# 21. Identity preservation

The alignment operation must preserve all identities required to interpret the resulting data.

At minimum:

- session identity;
- trial identity;
- neuron / cluster identity;
- visual source identity where available.

Alignment must not merge data belonging to semantically different trials, sessions, or neural channels.

---

# 22. Provenance

The output must preserve enough information to reproduce the alignment.

Relevant provenance includes at least:

- source session;
- source trial;
- `stimOn`;
- exact `stimOff`;
- neural bin size;
- aligned start;
- aligned end;
- discarded trailing duration;
- visual resampling policy;
- relevant source dataset or preprocessing identity.

---

# 23. Determinism

Given identical:

- neural input;
- visual input;
- trial timing;
- alignment configuration;

the module must produce identical aligned output.

Alignment must not depend on:

- execution order;
- wall-clock time;
- random sampling;
- unordered iteration.

---

# 24. Error conditions

The following are considered alignment or input-contract errors:

- session mismatch;
- trial mismatch;
- malformed or non-monotonic timestamps;
- missing required visual observations;
- insufficient visual temporal coverage;
- unsupported temporal representation;
- inconsistent neural bin definitions;
- variable neural bin duration inside a single aligned representation.

These conditions must not be silently repaired by generating synthetic observations.

---

# 25. Responsibilities outside the module

## `visual-replay`

Responsible for:

- stimulus reconstruction;
- stimulus timing;
- stimulus state;
- perspective transformation;
- production of a temporally complete visual replay.

## `visual-features`

Responsible for:

- image preprocessing;
- feature extraction;
- temporal feature sampling;
- explicit visual timestamps;
- complete visual feature coverage required by alignment.

## `neural-data`

Responsible for:

- neural recording access;
- neural channel / cluster selection;
- spike processing;
- construction of binned neural activity;
- neural temporal metadata.

## Downstream dataset/model pipeline

Responsible for:

- padding;
- batching;
- sequence-length normalization where required;
- model positional encoding;
- modality projection;
- masking for model objectives;
- train/validation/test splitting;
- definition of prediction targets.

---

# 26. Core invariants

The alignment module must guarantee the following.

### Shared start

```text
aligned_start = stimOn
```

No pre-stimulus neural data is included.

### Neural grid authority

The common temporal grid is defined by complete neural bins.

### Uniform neural duration

Every included neural observation represents the same temporal duration.

### No partial final bin

A trailing interval shorter than one neural bin is excluded.

### Exact stimulus timing preserved

The original `stimOff` remains available even when:

```text
aligned_end < stimOff
```

### Zero spikes are valid data

A zero-valued neural bin represents absence of detected spikes, not missing data.

### Complete visual coverage

The visual input must cover all aligned temporal positions.

### No missing-data interpolation

Visual interpolation is temporal resampling between consecutive valid observations, not gap repair.

### No extrapolation

Visual features must not be extrapolated beyond their supported temporal range.

### Trial isolation

Temporal mapping must never cross trial boundaries.

### Explicit physical time

Session timestamps and model positional indices must remain distinct concepts.

---

# 27. Current default temporal configuration

The current pipeline uses:

```text
neural bin size = 1/60 second ≈ 16.67 ms
```

corresponding to a nominal neural temporal resolution of:

```text
60 Hz
```

The previously used 20 ms bin size remains the reference temporal resolution.
Candidate sizes for this experiment must remain within ±5 ms (15–25 ms) to
avoid substantially changing neural spike-count sparsity. The selected 1/60-second
bin is approximately 3.33 ms below the reference and satisfies this constraint
while matching the confirmed 60 Hz visual projection and replay rate. This is
the selection rationale, not a measured claim that sparsity is unchanged.

Replay timestamps and neural bin centers have different offsets even at the
same rate; physical timestamp resampling remains required.

This value is a pipeline configuration, not a permanent architectural constant.

The alignment contract is defined in terms of the configured neural bin size.

Changing the bin size therefore changes the common alignment grid without changing the semantics of the module.

---

# 28. Summary

The `alignment` module establishes a shared representation of stimulus-related neural and visual information.

Its central rule is:

> The stimulus defines the usable temporal interval, while complete neural bins define the common temporal grid.

The aligned timeline begins at stimulus onset.

Visual features are resampled onto neural bin centers through normalized linear interpolation.

Neural bins containing zero spikes remain valid observations.

Missing visual observations are upstream failures rather than alignment gaps.

Only complete neural bins are retained. Any final interval shorter than one neural bin is discarded while the exact stimulus offset remains preserved as metadata.

The resulting representation therefore provides synchronized, uniform-duration neural and visual observations suitable for downstream multimodal modeling.

## 29. Trial eligibility and session completion

A session may contain individually unusable trials. Alignment must publish the
usable subset with original session/trial/unit identities preserved, and record
every excluded original trial ID, validation stage, and specific reason. Retained
and excluded trials must partition the requested trial set exactly. No damaged
trial is repaired, renumbered, or published as a placeholder. Completion means
all requested trials are accounted for, with at least one usable trial; it does
not assert that every requested trial is usable.

Artifact corruption, contradictory source identities/domains, unsupported
session representations, and malformed schemas remain fatal session errors.
Unknown or insufficient support may exclude individual trials; if no usable
trials remain the session fails. A strict diagnostic mode may reject the first
unusable trial. Downstream dataset construction uses retained trials and preserves
upstream exclusion accounting in provenance. Store complete session provenance
once and expose compact trial-specific source references to avoid memory growth
from repeatedly copying session-sized definitions into every trial.
