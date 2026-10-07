# Visual Replay Specification

## Purpose

The `visual-replay` component reconstructs the visual stimulus presented during an IBL behavioral session from recorded session data and historically relevant task/rendering information.

Its output represents the project's reconstructed visual modality.

The reconstruction is intended to preserve the meaningful visual and temporal properties of the stimulus sufficiently for downstream visual representation and neural encoding/decoding experiments.

The goal is **source-derived functional reconstruction**, not proof of pixel-identical reproduction of the original display.

---

## Component role

At project level:

```text
session-data
     │
     ▼
visual-replay
     │
     ▼
visual-features
```

`visual-replay` consumes session source information through `session-data`.

It reconstructs visual stimulus frames and associated timing/provenance.

`visual-features` consumes those reconstructed visual outputs and converts them into model-oriented visual representations such as CLIP embeddings.

---

# Input responsibility

The component consumes source information required to reconstruct the visual task.

Depending on availability and session/task version, this may include:

* trial identity;
* stimulus side;
* contrast;
* stimulus onset;
* stimulus offset;
* stimulus freeze/response timing;
* wheel position;
* wheel timestamps;
* stimulus parameters;
* task protocol/version metadata;
* renderer/task configuration;
* display-related metadata;
* raw task evidence where available.

Source-data acquisition itself belongs to `session-data`.

`visual-replay` must not independently own normal ONE/Alyx download, cache, collection, or revision policy.

---

# Trial identity

Every reconstructed trial must preserve the identity of the original source trial.

Filtering, skipped trials, failed reconstruction, or output generation must not silently renumber later trials.

At minimum, a replay remains traceable through:

```text
EID + original trial identity
```

A missing or invalid trial must remain distinguishable from a valid reconstructed trial.

---

# Stimulus reconstruction

The component must reconstruct the visual stimulus according to the task behavior applicable to the source session.

The reconstruction should be derived from:

1. session-specific evidence where available;
2. the historically relevant task and renderer implementation;
3. documented task behavior;
4. explicit project assumptions only where required information is unavailable.

Project assumptions must not silently replace known source information.

---

## Stimulus form

The visual task stimulus is a Gabor-like visual pattern consisting of a sinusoidal carrier modulated by a Gaussian envelope.

The reconstruction must preserve meaningful stimulus properties where supported by source information, including:

* side / initial location;
* contrast;
* spatial frequency;
* orientation;
* Gaussian envelope or equivalent aperture behavior;
* phase;
* stimulus position over time;
* visibility interval.

Parameters must retain their physical or source-defined meaning where possible rather than being replaced by arbitrary pixel-space constants.

---

## Contrast

Stimulus contrast must affect the reconstructed visual stimulus.

A zero-contrast trial is valid and must not be interpreted as missing stimulus information.

Different source contrast values must remain meaningfully distinguishable in the rendered stimulus.

---

## Position and movement

Stimulus position must be reconstructed from the task's source behavior and session wheel data.

The component must preserve:

* initial stimulus side;
* wheel-dependent closed-loop movement;
* correct movement direction;
* inward movement;
* outward/error movement where present;
* movement termination or freeze behavior;
* final visibility until stimulus offset where appropriate.

Movement must not be artificially constrained to successful trajectories.

Wheel movement must be applied once.

If session-specific gain or mapping information is available, it takes precedence over generic fallback values.

---

# Temporal reconstruction

Replay timing must remain tied to the source session clock.

Relevant source events may include:

* stimulus onset;
* display freeze;
* response;
* stimulus offset;
* wheel timestamps;
* other version-specific task events.

The component must preserve the semantic distinction between different events.

For example:

```text
response
feedback
stimulus freeze
stimulus offset
```

must not be treated as interchangeable merely because one is easier to access.

---

## Visibility interval

The reconstructed stimulus must appear only during the source-defined visible interval.

The component must not invent arbitrary minimum trial durations or extend the stimulus beyond available source timing merely to satisfy an output format.

Known blank intervals must remain conceptually different from unavailable visual information.

---

## Freeze behavior

Where the task freezes stimulus motion before stimulus offset, the reconstructed stimulus must preserve that behavior.

The appropriate freeze event should follow the session/task evidence.

Fallback between timing events must be explicit and semantically justified.

---

# Frame timing

Every generated visual frame must have a meaningful mapping to source session time.

Video playback time alone is insufficient as the authoritative timing representation.

The replay must preserve enough timing information for downstream components to associate visual observations with neural activity.

Frame generation may use:

* recorded display timing;
* reconstructed display timing;
* regular sampling derived from source timing;

depending on available source evidence and architecture.

The timing source must remain identifiable.

## Experimental projection cadence

The researcher has confirmed that projection during the experiments operated at
60 Hz. Regular replay frame generation must therefore use 60 Hz (one frame
every 1/60 second), with optional constant-rate video encoded at 60 FPS when
storing one frame per observation. This is the experimental replay cadence;
downstream feature sampling is a separate operation.

When individual recorded display timestamps are unavailable, reconstruct frame
times from stimulus onset at this cadence within the source visibility interval.
Keep these timestamps classified as reconstructed: the confirmed projection
rate does not establish the measured time of each display refresh or dropped
frames. Explicit alternative sampling requests must retain their configured
cadence and must not be represented as the experimental display cadence.

---

## Irregular and regular timing

The source timing representation and the encoded video timing representation may differ.

If irregular source observations are stored in a constant-frame-rate video, the relationship between:

```text
source session time
↕
generated frame
↕
encoded video presentation time
```

must remain recoverable.

Frame repetition, dropping, or resampling must not silently change the scientific time axis.

---

# Historical source usage

The historical IBL task/rendering implementation is an important behavioral source.

The component should use the relevant historical implementation to determine:

* stimulus geometry;
* projection behavior;
* carrier construction;
* Gaussian/aperture behavior;
* parameter units;
* contrast application;
* coordinate transformations;
* clipping behavior;
* task state transitions;
* wheel coupling;
* display clearing/background behavior.

The project is not required to execute the original historical renderer in production.

A source-derived implementation is acceptable when it reproduces the behavior required for the project.

---

# Reconstruction fidelity

The project does not claim exact reproduction of the original sensory input received by the animal.

The replay is a reconstruction.

Exact reconstruction may be impossible because session-specific information such as the following may be incomplete or unavailable:

* exact physical display calibration;
* luminance;
* gamma;
* ambient illumination;
* complete renderer/package revision;
* eye position;
* head position;
* retinal optics;
* raw display-frame capture;
* exact session-specific hardware state.

These unknowns must not be silently invented and presented as measured session properties.

---

## Required fidelity level

The component must preserve the visual properties that materially define the experimental stimulus and its relationship to the neural recording.

Required correctness includes:

* correct stimulus type;
* meaningful source-derived geometry;
* contrast;
* side;
* task-relevant motion;
* task timing;
* freeze behavior;
* trial identity;
* session-relative frame timing.

The component does **not** require:

* pixel-identical reproduction of the historical framebuffer;
* execution of the original Bonsai/BonVision environment;
* lossless reference captures from the original renderer;
* reconstructed room illumination;
* photometrically exact monitor luminance;
* retinal-image simulation.

Those may be useful research evidence but are not required for functional completion of the ViNED replay pipeline.

---

# Parameter resolution

Visual parameters may come from different evidence levels.

The preferred evidence order is:

```text
session-specific source data
        ↓
session/task-version-specific historical implementation
        ↓
documented protocol/task configuration
        ↓
explicit project fallback
```

Higher-confidence evidence should replace lower-confidence fallback assumptions when available.

---

## Missing parameters

When a required parameter is unavailable, the component may use a documented fallback if that fallback is sufficient for the intended research pipeline.

The fallback must:

* have an explicit value or derivation;
* have a known source or rationale;
* remain distinguishable from recovered session-specific information;
* not be described as measured session truth.

Missing information must not trigger arbitrary hidden values.

---

## Session and task versions

Different sessions may originate from different task software versions.

Where those differences materially affect replay behavior, the reconstruction must support selecting or resolving the appropriate source behavior.

Exact package/commit recovery is desirable when available but is not required when the relevant rendering behavior can be established sufficiently from other evidence.

---

# Display reconstruction

The stimulus must first be reconstructable in its display-space representation.

Display reconstruction is responsible for applying the visual transformation required to determine what appears on the experimental display.

This includes the relevant source mapping between task-space stimulus parameters and displayed stimulus location/geometry.

The component must avoid replacing known source projection behavior with unrelated arbitrary linear mappings merely because they are easier to implement.

When the source mapping cannot be fully recovered, the approximation must be explicit.

---

# Mouse-perspective scene

The project visual modality should represent the stimulus from an approximate mouse-centered viewpoint rather than treating the display framebuffer as if it were the complete sensory scene.

The replay therefore supports a schematic physical scene containing the display.

Conceptually:

```text
source stimulus
      ↓
display-space reconstruction
      ↓
physical/schematic display
      ↓
mouse-perspective projection
      ↓
replay visual frame
```

---

## Viewpoint purpose

The mouse-perspective scene exists to provide a more meaningful visual input for downstream visual feature extraction.

It is not intended to reconstruct the exact retinal image.

---

## Viewpoint requirements

The scene should use a fixed, documented viewpoint approximating the animal's position relative to the experimental display.

It should preserve:

* screen geometry;
* screen aspect ratio;
* approximate screen distance;
* approximate visual coverage;
* stimulus position on the screen;
* visibility of the complete relevant screen area.

Surrounding scene content should remain minimal and neutral.

Unsupported rig details should not be invented.

---

## Projection separation

Display reconstruction and mouse-perspective projection are separate conceptual transformations.

A display mapping that has already been applied must not accidentally be applied a second time during scene projection.

The component must preserve this distinction:

```text
task coordinates
    ↓
display mapping
    ↓
display image
    ↓
scene/camera projection
```

---

# Background and unknown environment

The reconstructed display background should follow the best available source evidence.

Physical room illumination and other environmental appearance outside the display are not known sufficiently to reconstruct faithfully.

The schematic environment may therefore use a neutral project-defined representation.

This representation must not be described as measured experimental illumination.

---

# Output

The component must provide reconstructed visual observations suitable for downstream visual processing.

The output may include:

* individual frames;
* frame sequences;
* per-trial videos;
* timing sidecars;
* replay metadata;
* validity information;
* provenance information.

The exact technical representation belongs to the component interface and architecture.

The scientific boundary is:

```text
visual-replay
    → reconstructed visual observations + source timing
```

not:

```text
visual-replay
    → CLIP embeddings
```

CLIP and other learned visual representations belong to `visual-features`.

---

# Output identity

Generated replay artifacts must remain traceable to:

* source EID;
* original trial identity;
* replay generation/configuration;
* relevant source parameters;
* timing source;
* significant fallback assumptions.

Artifact naming or storage layout is an implementation/interface decision.

Identity preservation is a specification requirement.

---

# Validity and incomplete reconstruction

The component must distinguish:

* successfully reconstructed frames/trials;
* known blank visual content;
* unavailable visual information;
* invalid source data;
* failed reconstruction.

Unavailable or invalid visual information must not silently become an apparently valid generated observation.

A failed trial does not require failure of every other valid trial unless the architecture explicitly requires atomic session publication.

---

# Provenance

The replay must preserve sufficient provenance to understand how the visual reconstruction was produced.

Relevant provenance may include:

* EID;
* original trial identity;
* source dataset identity;
* task/version evidence;
* effective visual parameters;
* parameter source;
* renderer implementation/version;
* timing source;
* reconstruction assumptions;
* scene/viewpoint profile;
* known unresolved properties.

Provenance should describe what is known and what was assumed.

It should not imply a stronger level of reconstruction fidelity than the available evidence supports.

---

# Determinism

Given identical:

* source session data;
* replay configuration;
* explicit fallback parameters;
* renderer implementation;

the reconstruction should be reproducible.

Where a historically random stimulus property such as phase cannot be recovered and a synthetic fallback is required, the fallback should be reproducible for the same replay configuration.

Synthetic values must remain identifiable as synthetic rather than recovered.

---

# Failure behavior

The component must fail or mark reconstruction invalid when continuing would produce misleading visual data.

Examples include:

* unusable trial identity;
* missing essential timing;
* contradictory stimulus side information;
* insufficient wheel coverage for required movement reconstruction;
* invalid source parameter units;
* unsupported source behavior with no justified fallback.

The component should not fail merely because optional high-fidelity evidence is unavailable.

For example, absence of:

* exact gamma calibration;
* original framebuffer captures;
* historical runtime environment;

does not by itself make a functional replay invalid.

---

# Validation expectation

Validation should establish that the reconstructed pipeline behaves according to this specification.

Relevant functional validation includes demonstrating that:

* supported sessions can produce replay output;
* generated visual artifacts are readable;
* stimulus side and contrast affect the output correctly;
* wheel movement affects stimulus position correctly;
* error/outward movement is representable;
* freeze and offset behavior are respected;
* source trial IDs are preserved;
* generated frames have source-session timing;
* required mouse-perspective projection is applied;
* downstream visual processing can consume the result.

Pixel-by-pixel comparison against the original historical renderer is not required unless a future research claim explicitly depends on such fidelity.

Validation must remain proportional to the claim made by the project.

---

# Relationship to `session-data`

`visual-replay` depends on `session-data`.

`session-data` owns:

* session identity resolution;
* ONE/Alyx access;
* dataset discovery/acquisition;
* source cache behavior;
* collection/revision resolution;
* factual session metadata.

`visual-replay` owns:

* interpretation of source data as visual task behavior;
* historical renderer/task semantics;
* parameter resolution for rendering;
* stimulus state reconstruction;
* visual rendering;
* mouse-perspective projection;
* replay artifact generation.

The intended boundary is:

```text
session-data
    ↓
source evidence

visual-replay
    ↓
interpreted visual reconstruction
```

---

# Relationship to `visual-features`

`visual-features` is a downstream consumer.

It owns:

* CLIP loading;
* image preprocessing required by CLIP;
* frame sampling for feature extraction;
* embedding generation;
* visual feature serialization.

`visual-replay` must provide enough frame timing, identity, validity, and provenance for this processing without depending on CLIP.

The replay renderer must not be designed around one specific visual-feature model.

---

# Out of scope

The following are outside `visual-replay`:

* ONE/Alyx acquisition infrastructure;
* general source-session cache management;
* neural spike processing;
* neural binning;
* visual-neural alignment;
* CLIP embedding extraction;
* model dataset creation;
* training-time masking;
* model training;
* evaluation metrics;
* exact retinal simulation;
* eye tracking reconstruction;
* detailed mouse anatomy;
* reconstructed room lighting;
* photometric monitor calibration where no session evidence exists;
* pixel-identical historical renderer certification.

---

# Required behavioral properties

The component must provide the following guarantees:

1. Visual replay remains traceable to the original session and trial.
2. Source session timing remains recoverable for generated frames.
3. Contrast, side, motion, and visibility affect the reconstruction according to the applicable task behavior.
4. Wheel-driven motion supports both successful and error trajectories.
5. Freeze/response/offset events remain semantically distinct.
6. Source-derived parameters take precedence over project assumptions.
7. Fallback assumptions remain explicit and reproducible.
8. Missing high-fidelity calibration does not prevent otherwise valid functional reconstruction.
9. Missing essential source information does not silently produce apparently valid visual data.
10. Display-space rendering and mouse-perspective scene projection remain conceptually separate.
11. The output can be consumed independently of CLIP or any specific visual-feature model.
12. Source-data acquisition remains behind the `session-data` boundary.
13. The component does not claim exact reconstruction beyond the available evidence.
14. Functional replay correctness takes priority over unnecessary historical pixel-fidelity validation.
