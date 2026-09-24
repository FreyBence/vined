# ViNED project instructions

## Project identity

**ViNED — Visual–Neural Encoding and Decoding** is an MSc research project derived from **NEDS (Neural Encoding and Decoding at Scale)**.

The repository name is **`vined`**.

The project investigates the relationship between visual stimuli presented to a mouse during IBL experiments and neural population activity recorded with Neuropixels electrodes.

The two main research directions are:

* **Encoding:** predict neural population activity from visual stimulus representations.
* **Decoding:** predict visual stimulus representations from neural population activity.

The project adapts the inherited NEDS multimodal architecture to visual and neural data, with support for multiple recording sessions and future extension to additional task-related modalities.

---

## Current research representation

### Visual modality

The visual representation used by the learning pipeline is a sequence of **CLIP image embeddings**.

The current CLIP representation has width:

```text
768
```

Visual preparation may use generated stimulus frames or videos as intermediate artifacts, but the model consumes CLIP feature vectors rather than reconstructed image pixels.

Do not treat image or video reconstruction as part of the model output unless explicitly introduced by a separate feature or experiment.

### Neural modality

The neural input consists of already processed spike-count sequences derived from electrophysiological recordings.

A new spike-sorting pipeline is not part of ViNED.

Neuron and recording metadata may vary between sessions, so session-specific neural populations must not be assumed to share identical channel or cluster identities.

---

## Active modalities

The current primary modalities are:

```text
spike
vision-clip
```

Historical NEDS modalities such as wheel speed, whisker motion, choice, or block variables are not part of the current primary two-modality pipeline unless explicitly reintroduced.

Task-event modalities may be added separately in future work, but their representation and supervision must not be assumed until implemented.

---

## Relationship to NEDS

ViNED inherits major architectural ideas and code from NEDS, including:

* multimodal transformer processing;
* session-specific input/output projections;
* session stitching across different neural populations;
* masking-based training;
* shared latent representations;
* neural prediction and evaluation infrastructure.

ViNED extends this with visual stimulus processing and CLIP-based representations.

Key conceptual differences include:

| Area                   | NEDS                                | ViNED                                                |
| ---------------------- | ----------------------------------- | ---------------------------------------------------- |
| Primary modalities     | Neural and behavioral signals       | Neural activity and visual CLIP features             |
| Visual preparation     | Not present                         | Stimulus replay/frame generation and CLIP extraction |
| Visual representation  | Not present                         | 768-dimensional CLIP vectors                         |
| Neural representation  | Spike-based                         | Spike-based                                          |
| Cross-session handling | Session-specific neural projections | Retained                                             |
| Visual loss            | Not present                         | CLIP-oriented prediction loss                        |
| Neural loss            | Spike prediction                    | Retained/adapted                                     |

Preserve NEDS attribution, citation, and licensing requirements.

Do not assume historical NEDS documentation describes the current ViNED implementation exactly. Verify implementation-specific architectural claims against the current source code.

---

## High-level pipeline

The active research pipeline is conceptually:

```text
IBL session data
    ↓
visual stimulus reconstruction / replay
    ↓
visual frames
    ↓
CLIP feature extraction
    ↓
visual feature sequences
    ↓
time alignment with neural activity
    ↓
dataset preparation
    ↓
multimodal model
    ↓
encoding / decoding evaluation
```

Each stage should preserve the identities and timing required by downstream stages.

---

## Code map

### Visual replay

Primary components include:

```text
src/visual_stim_gen.py
src/utils/stimulus_parameters.py
```

These components are responsible for reconstructing or generating visual stimulus representations from available session information.

Rendering behavior may depend on the historical task/software version associated with a recording.

Do not silently apply the newest rendering behavior to historical sessions when version-dependent behavior is known or relevant.

Detailed visual-replay behavior belongs in the corresponding visual specification and task documents rather than this repository-wide instruction file.

### Visual feature extraction

Primary component:

```text
src/prepare_visual_stim.py
```

This stage:

* obtains visual frames;
* applies the configured CLIP image pipeline;
* stores per-trial feature sequences;
* preserves trial identities and relevant timestamps.

The model currently expects 768-dimensional visual features.

If the CLIP model or representation changes, update all dependent tensor dimensions and interfaces consistently.

### Neural and visual alignment

Primary components include:

```text
src/prepare_data.py
src/utils/ibl_data_utils.py
```

This stage aligns visual and neural representations onto the time structure expected by the learning pipeline.

Alignment correctness is important because visual and neural samples must continue to represent the same trial and physical time interval.

Do not treat independent array lengths or indices as sufficient evidence of alignment.

### Dataset construction

Primary components include:

```text
src/create_dataset.py
src/utils/dataset_utils.py
src/loader/
```

A prepared sample should preserve:

* trial identity;
* session identity;
* temporal ordering;
* neural population structure;
* visual feature structure;
* validity or padding information where applicable.

Typical unbatched representations are conceptually:

```text
spike:  [T, N]
vision: [T, 768]
```

where:

* `T` is the temporal dimension;
* `N` is the session-specific number of neural channels or clusters.

Batched tensors add a leading batch dimension.

### Training and adaptation

Primary components include:

```text
src/train.py
src/finetune.py
src/multi_modal/
src/models/stitcher.py
src/trainer/
```

The architecture supports session-specific projections so recordings with different neural populations can participate in a shared representation.

Do not assume cluster IDs or neuron indices are globally shared between sessions.

### Evaluation

Primary components include:

```text
src/eval.py
src/utils/eval_utils.py
```

Evaluation may operate on visual predictions, neural predictions, or both depending on the configured model mode.

Ensure evaluation is interpreted according to the actual prediction direction and loaded checkpoint configuration.

---

## Identity and alignment conventions

### Trial identity

Preserve original trial identities throughout:

```text
session data
→ replay
→ visual features
→ aligned data
→ cached dataset
→ evaluation
```

Filtering or missing artifacts must not silently renumber later trials.

### Session identity

Session identity is part of the model and data interpretation.

A trial or neuron identifier from one session must not automatically be interpreted as equivalent to the same numerical identifier from another session.

### Time

Keep recorded, reconstructed, resampled, and model-grid timestamps conceptually distinct when they are different representations.

Conversions between time representations should be explicit.

The visual and neural modalities must refer to the intended same physical interval before they are combined into a training sample.

---

## Historical software and data compatibility

IBL recordings may originate from different versions of acquisition software, task definitions, renderer behavior, preprocessing, or configuration.

When historical behavior can affect generated project data:

1. use session-specific evidence when available;
2. use a compatible historical source/configuration when available;
3. use documented defaults or assumptions only when required;
4. record important fallbacks when they materially affect the generated representation.

Do not silently treat the newest available implementation as historically correct for all sessions.

Detailed evidence handling and scope decisions are governed by the repository-wide scope rules.

---

## Model conventions

### Visual features

The current visual model input is based on CLIP image embeddings.

Keep the following consistent across preprocessing, loaders, projections, model heads, training, and evaluation:

* feature width;
* normalization assumptions;
* temporal dimensions;
* masks;
* prediction direction.

### Neural outputs

Neural prediction operates on spike-based representations.

Changes to output activation, normalization, or loss interpretation must remain compatible with the mathematical assumptions of the selected neural loss.

### Masking

Padding, unavailable observations, and deliberately masked targets are distinct concepts.

Do not silently treat padded or unavailable values as valid observations.

Masks must remain aligned with the data they describe.

---

## Evaluation concepts

Visual and neural predictions use different evaluation concepts.

Common visual evaluation may include similarity or reconstruction-oriented metrics.

Common neural evaluation may include:

* trial-level R²;
* PSTH R²;
* bits per spike (BPS).

Interpret metrics according to their definitions and baselines.

A high visual similarity score alone does not automatically establish that the model recovered all task-relevant stimulus information.

Do not treat historical reported values as current reproduced results unless they were generated by the current experiment configuration.

---

## Experiment boundaries

Distinguish clearly between:

* held-out trials from a known recording session;
* held-out recording sessions;
* fine-tuning or adaptation to a new session;
* evaluation of a session already present during training.

These represent different forms of generalization and should not be reported interchangeably.

---

## Generated data and artifacts

Generated research artifacts should remain separate from source changes unless explicitly intended to be version-controlled.

Examples include:

* replay videos;
* generated stimulus frames;
* CLIP feature archives;
* aligned datasets;
* cached datasets;
* checkpoints;
* runtime logs;
* temporary inspection outputs;
* Python bytecode;
* generated package metadata.

Use project-configured data/output locations rather than embedding machine-specific absolute paths where avoidable.

---

## Repository and runtime conventions

ViNED is a checkout-based application rather than an installable Python distribution.

The active repository should be run from its checkout root where required by relative configuration references.

Runtime paths should prefer repository-relative paths or configured environment variables such as:

```text
VINED_DATA_DIR
VINED_VISUAL_DIR
VINED_REPLAY_DIR
```

Use the project-local Python environment defined by the active environment documentation.

Do not assume archived NEDS setup instructions describe the current ViNED runtime.

Platform-specific launchers or cluster wrappers may exist, but they should not define the conceptual architecture of the project.

---

## Documentation roles

Keep documentation responsibilities separated.

Use:

* `AGENT.md` for repository-wide agent entry instructions;
* `.agent/rules/scope.md` for scope and proportional-validation decisions;
* `.agent/rules/planning.md` for planning and task tracking;
* this file for stable project and architecture context;
* specifications for detailed feature behavior;
* task files for planned work and completion state;
* experiment or research reports for measured results and scientific interpretation.

Do not use this file as a backlog, progress report, issue list, or experiment log.

---

## Naming

Use:

```text
ViNED
```

for the project name in user-facing documentation.

Use:

```text
vined
```

for the repository slug and checkout directory.

Preserve attribution to NEDS for inherited architecture, code, and research ideas.
