# ViNED: Visual–Neural Encoding and Decoding

## Project identity and goal

ViNED is Frey Bence's MSc research project, derived from **NEDS (Neural Encoding and Decoding at Scale)**. The chosen repository name is **`vined`**. The name describes the two directions of the research:

- **Encoding:** predict neural population activity from visual stimulus representations.
- **Decoding:** predict visual stimulus representations from neural population activity.

The central goal is to learn the relationship between what a mouse sees during an IBL decision-making task and the activity recorded by implanted Neuropixels electrodes. Extend the inherited multimodal transformer so it can learn jointly from visual stimuli, neural activity, and eventually task events, with adaptation across recording sessions.

The current visual output is a sequence of **CLIP embeddings**, not reconstructed image pixels or playable video. Image/video reconstruction would require an additional method and evaluation. The current neural inputs are already processed spike counts; a new spike-sorting pipeline is not implemented here.

## Evidence and interpretation

This description was prepared from the working-tree changes relative to commit `61d2ef6`, including the new visual preparation scripts, and these Hungarian research documents:

- [Research report](docs/Frey_Bence_ITLNUH_Beszámoló.pdf): methodology on pages 15–20, evaluation and results on pages 21–22, and future objectives on page 23.
- [Presentation](docs/Frey_Bence_ITLNUH_prezentáció.pptx), dated 2026-06-04: research questions on slide 2, the adapted model on slide 5, metrics/results on slides 7–8, and future work on slide 9.

The report's abstract and introduction still describe wheel movement and behavioral decision prediction. Its methodology, final objectives, the presentation, and the current code support the more specific visual–neural direction above. Treat the earlier behavioral wording as background requiring reconciliation, rather than the current primary target.

The report also describes temporal convolution, patch tokens, and a separate alignment loss. Do not assume these descriptions match implemented components: the current extractor saves one pooled CLIP feature vector per frame, the inspected neural embedding path uses linear projections, and the model computes modality losses under task masks. Verify architectural claims against source code before repeating them.

## What changed from NEDS

| Area | Inherited NEDS behavior | Current adaptation |
| --- | --- | --- |
| Modalities | Spikes, wheel speed, whisker motion, choice, and block | Spikes (`ap` mapped to `spike`) and `vision-clip`; static modality lists are empty |
| Visual preparation | No stimulus-to-CLIP preparation path | Synthetic task replay generation and frozen CLIP feature extraction |
| Dataset representation | Scalar dynamic behavioral targets and static labels | Trial sequences of 768-dimensional visual features, alongside spike counts and neuron metadata |
| Model | Multimodal transformer with session-specific projections | Visual projections/output heads and normalization adapted to CLIP vectors; shared transformer and session stitching retained |
| Loss and masking | Behavioral regression/classification plus spike prediction | Cosine loss for vision, Poisson negative log-likelihood for spikes; `self-vision` replaces `self-behavior` |
| Evaluation | Behavioral and neural metrics | Visual cosine similarity/MSE paths plus neural R² and bits-per-spike metrics |
| Execution | Original cluster/data configuration | Local dataset loading, Windows paths, revised session lists, smaller batches and revised training schedules |

Additional architectural edits include query/key normalization in attention and normalization in decoder heads. These are experiments to validate, not established improvements. Changes to bytecode and package metadata are generated artifacts, not research contributions.

## Current pipeline and code map

1. **Generate stimulus replays** — `src/visual_stim_gen.py` creates per-trial grating videos under `ibl_task_replay/<eid>/`, using IBL trial timing, stimulus side, and wheel position. These are approximate synthetic replays; rendering fidelity needs validation against task parameters.
2. **Extract visual features** — `src/prepare_visual_stim.py` samples frames (default 5 FPS), runs frozen `openai/clip-vit-large-patch14`, normalizes its image features, and writes `<eid>_visual_clip.npz` with per-trial `trial_ids`, `times`, and `features`. The current model expects feature width 768.
3. **Prepare aligned data** — `src/prepare_data.py` and `src/utils/ibl_data_utils.py` load IBL spikes and visual features. Neural preprocessing uses 20 ms bins in a two-second window from -0.5 to +1.5 seconds relative to `stimOn_times` (100 bins). The visual resampling currently needs correction to represent that same physical time window.
4. **Create cached datasets** — `src/create_dataset.py`, `src/utils/dataset_utils.py`, and `src/loader/` package, pad, and load trial data. A sample should preserve spike shape `[T, N]`, vision shape `[T, 768]`, session identity, and neuron metadata. Batched tensors add a leading batch dimension.
5. **Train/adapt** — `src/train.py`, `src/finetune.py`, `src/multi_modal/`, `src/models/stitcher.py`, and `src/trainer/base.py` contain single-session, multi-session, and fine-tuning paths. Session stitching maps recordings with different neuron populations into a shared representation. These paths exist but were not executed during this review.
6. **Evaluate** — `src/eval.py` and `src/utils/eval_utils.py` evaluate visual and neural predictions. Check task routing and checkpoint restoration before interpreting results.

Session selections live in `data/eids.txt`, `data/train_eids.txt`, and `data/test_eids.txt`. Trial preprocessing currently uses a seeded 70%/10%/20% train/validation/test split. Distinguish held-out trials within a session from held-out recording sessions when reporting generalization.

## Research status and evaluation

The report and presentation give the following **historical single-session results**, which were not reproduced in this review:

| Metric | Reported value |
| --- | ---: |
| CLIP cosine similarity | 0.99175 |
| Trial-level R² | -0.00389 |
| PSTH R² | -0.67825 |
| Bits per spike (BPS) | -0.01476 |

These results motivate improving neural prediction. High CLIP cosine similarity alone does not establish successful decoding of stimulus identity or motion: compare with a training-set mean embedding and shuffled neural/visual pairings, particularly for visually similar grating frames. For neural predictions, retain trial R², peri-stimulus time histogram (PSTH) R², and BPS against their defined baselines. BPS is a likelihood improvement per spike, not a score bounded by one.

## Development priorities

### 1. Establish a trustworthy visual–neural baseline

Resolve the following findings from static inspection before treating current evaluation scores as reliable:

- **Time alignment:** `bin_behaviors` interpolates each video's first-to-last frame into a fixed number of bins, while spikes use the stimulus-aligned -0.5 to +1.5 second window. Use a shared physical time grid with an explicit policy for missing/pre-stimulus frames and padding.
- **Trial identity and validity:** the visual extractor assigns trial IDs by enumerating filenames, which can shift IDs when trial videos are missing. Preserve original trial IDs through filtering and splits. In `align_data`, the `trials_mask` branch combines `beh_mask` again instead of applying the supplied trial mask.
- **Checkpoint restoration:** `load_model_data_local` skips all keys containing `stitcher_dict`, `project_dict`, or `stitch_decoder_dict`. Verify that evaluation restores the trained session-specific layers; distinguish deliberate transfer to a new session from evaluation of an existing trained session.
- **Task routing:** visual evaluation in `src/eval.py` is enabled for `mm`/`encoding`; review this against the intended decoding task and available output modalities.
- **Attention and outputs:** `CrossAttention.forward` now derives keys and values from `x` rather than `context`; check affected call paths. Also audit normalization changes on neural decoder outputs for compatibility with Poisson log-rate prediction.
- **Masks:** exclude invalid and padded time steps from visual losses and metrics, and ensure masked reconstruction cannot access its target modality through unintended inputs.

These are code-review findings and validation priorities, not experimentally established explanations for the reported scores.

### 2. Add task-event modalities

The report's final objectives explicitly call for learning relationships among visual stimuli, neural activity, and task events. Define an event representation, timestamps, masks, prediction targets, and losses. Stimulus onset/offset, go cue, and feedback are candidate events available in the preparation code; the final event set is still a design decision. Evaluate whether adding events improves the corrected two-modality baseline.

### 3. Extend neural prediction targets

The report and slides propose prediction of neural recording clusters and each neuron's brain region. Existing region and cluster metadata are a starting point, not completed prediction heads. Define what “cluster prediction” means, the label space across sessions, and the supervision/evaluation protocol before implementing it. Do not equate session-local cluster IDs with globally shared classes.

### 4. Demonstrate generalization and reproducibility

First validate a single-session experiment, then test multi-session pretraining and adaptation to held-out sessions. Record session/trial splits, seed, preprocessing version, CLIP checkpoint, masking scheme, model configuration, restored checkpoint keys, and per-session metrics. Compare visual-only, neural-only, joint, and event-augmented experiments where applicable. The desired outcome is reproducible improvement over explicit baselines, with evidence for each claimed prediction direction.

## Guidance for future work

- Preserve NEDS attribution, its citation, and the existing license; identify ViNED-specific changes separately.
- Keep the goal, implemented behavior, and proposed extensions distinct in documentation and experiment reports.
- Keep modality names and tensor dimensions consistent across preprocessing, loaders, embeddings, heads, training, and evaluation. A different CLIP checkpoint may require coordinated dimension changes.
- Verify alignment and checkpoint loading before spending time on long training runs. For relevant code changes, use small checks covering trial identity, temporal alignment, padding, cross-modal masking, and checkpoint round trips.
- Bash wrappers resolve the checkout and use `.venv` (`VENV_DIR` override); they do not source user shell initialization. Slurm wrappers remain Linux/site-specific. Read [environment setup](docs/environment.md) and [validation status](docs/environment-validation.md) before running them. The archived NEDS README is historical, not active setup guidance.
- Keep datasets, replay videos, checkpoints, logs, bytecode, and generated package metadata out of source changes. Existing tracked generated files require a separate cleanup.

## Naming and migration status

Use **ViNED** in project-facing documentation and **`vined`** as the repository slug. The working directory is still `NEDS_new`; the Python distribution remains `neds`. The environment is now project-local `.venv`, created with standalone Python 3.10 and pip.

Runtime checkout paths are repository-relative or configurable through `VINED_DATA_DIR`, `VINED_VISUAL_DIR`, `VINED_REPLAY_DIR`, and launcher output overrides. Run direct Python commands from the checkout root because model YAML includes still use relative paths. Recreate `.venv` after moving the checkout. Package renaming and machine-wide Conda removal are separate tasks; preserve NEDS attribution.
