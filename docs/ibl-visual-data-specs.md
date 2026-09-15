# IBL visual data: specifications and project parameters

Research date: **2026-09-15**. Scope: the visual stimulus used by **ViNED**, its IBL source data, synthetic replay, CLIP features, and alignment with spikes.

This reference distinguishes **documented IBL behavior**, **current project implementation**, and **recommended requirements**. Web documentation and local source were inspected; session files were not downloaded or replay fidelity experimentally validated for this document.

## 1. Which IBL data does this project use?

ViNED reads public International Brain Laboratory sessions through ONE at `https://openalyx.internationalbrainlab.org`. The inherited NEDS instructions identify the repeated-site electrophysiology dataset. IBL calls the corresponding release **2024 Reproducible Ephys**, with release tag `RepeatedSite`; its documentation describes 91 released Neuropixels recording sessions across 12 laboratories. The historical NEDS instruction to prepare 84 sessions is a project selection, not the release size. [Release documentation][S1]; [archived NEDS README](README_NEDS.md).

The current checkout lists **20 EIDs**, with **18 training and 2 test EIDs**. These text files, rather than a hard-coded publication count, define the configured selection. Membership of every listed EID in `RepeatedSite`, its task version, and its dataset revision remain to be verified against session metadata. [All sessions](../data/eids.txt); [training sessions](../data/train_eids.txt); [test sessions](../data/test_eids.txt); [preparation entry point](../src/prepare_data.py).

**The visual modality is a locally generated task replay.** The renderer loads trial events and wheel positions, creates `trial_XXXX.mp4`, and the extractor converts frames to CLIP vectors. These files are derived project artifacts. IBL's recorded animal-camera videos are a separate data stream. [Renderer](../src/visual_stim_gen.py); [extractor](../src/prepare_visual_stim.py); [IBL data organization][S2].

## 2. IBL stimulus and physical parameters

The following are reference protocol values, not measurements of the 20 configured sessions. Recover session-specific parameters before claiming a faithful replay.

| Parameter | Documented reference | Source |
| --- | --- | --- |
| Stimulus | Gabor: a sinusoidal grating with a Gaussian envelope | [Behavior paper][S3] |
| Orientation / spatial frequency | Vertical; 0.1 cycles per visual degree | [Behavior paper, Methods][S3] |
| Carrier phase | Random phase; do not assume the same phase across trials | [Behavior paper][S3] |
| Initial horizontal position | Left −35° or right +35°; correct target 0° | [Position guide][S4] |
| Incorrect threshold | 35° away from the initial location, toward −70° or +70° | [Position guide][S4] |
| Screen placement | Approximately 8 cm from the animal; approximately 102° horizontal visual coverage | [Behavior paper][S3] |
| Display hardware reference | LG LP097QX1 LCD, approximately 246 mm active diagonal | [Behavior paper][S3] |
| Wheel radius / typical gain | 31 mm; 4 visual degrees per mm of wheel-surface displacement | [Position guide][S4] |
| Training exception | Initial gain can be 8°/mm | [Behavior paper][S3] |
| Contrast values | Task utilities accept fractions `0, 0.0625, 0.125, 0.25, 0.5, 1`; inspect each session's actual set | [Trial utilities][S5] |
| Side prior | `probabilityLeft`, commonly 0.5, 0.2, or 0.8 in the biased task | [Training utilities][S6] |

**Not established here:** session-specific Gaussian sigma, pixel resolution, refresh rate, gamma/luminance calibration, projection geometry, and phase values. In particular, the position guide's compact patch-size notation does not provide a sufficiently unambiguous Gaussian sigma definition for this specification. Do not substitute ViNED's pixel radius for IBL's angular patch size.

IBL's raw visual-stimulus loader documents `trial_num`, `stim_pos_init`, `stim_contrast`, `stim_freq`, `stim_angle` (0 means vertical), `stim_gain`, `stim_sigma`, `stim_phase`, and `bns_ts`. These are candidates for recovering actual rendering parameters, subject to availability. Its gain comment uses mm/degree whereas the position guide uses degrees/mm: verify units and the task version before conversion. Bonsai timestamps also require synchronization before combination with ALF event times. [Raw-data loader API][S7].

## 3. Input data dictionary

ALF is IBL's analysis data organization. A session is identified by its EID; datasets are organized into collections and may have revisions. The current renderer requests the following exact files from collection `alf`. Other releases/loaders may expose trial attributes through a combined object or table. [Data organization][S2]; [trial loading guide][S8]; [renderer](../src/visual_stim_gen.py).

Let `N` be the number of original trials and `M` the number of wheel samples.

| Dataset | Logical shape / contents | Project use |
| --- | --- | --- |
| `_ibl_trials.table.pqt` | Parquet table, `N` rows | Contrast side, stimulus onset, feedback time/type |
| `_ibl_trials.stimOff_times.npy` | `[N]`, seconds | Replay end |
| `_ibl_trials.goCueTrigger_times.npy` | `[N]`, seconds | Reference for empirical gain estimation |
| `_ibl_wheel.position.npy` | `[M]`, radians | Wheel trajectory |
| `_ibl_wheel.timestamps.npy` | `[M]`, seconds | Wheel interpolation |

File names and access above are verified in the [renderer](../src/visual_stim_gen.py); wheel units and sampling conventions are documented by [IBL's wheel extractor][S9] and [position guide][S4].

| Trial attribute | Meaning / handling | Source |
| --- | --- | --- |
| `contrastLeft`, `contrastRight` | Fractional contrast on the assigned side; the absent side is `NaN`. Zero is a valid contrast, not missing data. | [Contrast extractor][S10], [trial utilities][S5] |
| `stimOn_times`, `stimOff_times` | Extracted display onset/offset in seconds; ephys extraction uses frame2ttl events. | [Ephys extractor][S11] |
| `goCueTrigger_times`, `goCue_times` | Keep command trigger and extracted go-cue time distinct; cue/display delays depend on hardware. | [Trial loader][S8], [task timing QC][S12] |
| `response_times` | Response event used by the position guide to end wheel-coupled motion | [Position guide][S4] |
| `feedback_times` | Feedback event; do not silently substitute it for response time | [Task timing QC][S12] |
| `stimFreeze_times` | Extracted display freeze event where available; not necessarily a separately saved array | [Ephys extractor][S11], [trial extractor][S10] |
| `feedbackType` | Standard correct `+1`, error/no-go `−1`; extractor variants can include other outcomes | [Trial extractor][S10] |
| `choice` | `+1` clockwise, `−1` counterclockwise, `0` no-go; on correct trials it is the negative of the initial position's sign | [Choice extractor][S10] |
| `probabilityLeft` | Trial's side-prior probability; task context, not a directly rendered patch property | [Training utilities][S6] |
| `intervals` | Trial start/end, logical `[N, 2]`; table representations may use separate columns | [Position guide][S4], [trial loader][S8] |

**Recommended validation:** retain original row IDs, require exactly one finite contrast side, reject invalid required events, and check wheel timestamps are increasing and cover the requested interval. Record which clock and revision each array uses. Synchronize raw device times before combining them with processed ephys times. Missing data should produce an explicit validity flag, not an invented sensory observation.

## 4. Wheel-to-stimulus mapping

An algebraic restatement of the [IBL position guide][S4], during the wheel-coupled interval, is:

```text
delta_mm(t) = radius_mm * (wheel_rad(t) - wheel_rad(t_reference))
azimuth_deg(t) = initial_azimuth_deg - gain_deg_per_mm * delta_mm(t)
```

The guide uses stimulus onset as the reference. With radius 31 mm and gain 4°/mm, the coefficient is 124°/radian: a 35° displacement requires approximately 0.282 radians. Positive wheel displacement moves the patch toward decreasing azimuth. These numbers are derived from the guide, not fitted to ViNED sessions. [Position guide][S4].

Use the actual closed-loop timing for the session protocol. Keep the patch fixed after the response/display-freeze event until offset, and render background outside its visible interval. The guide has an apparent example-code error: `idx_off` searches `response_times` although the accompanying text specifies `stimOff_times`. Follow the event meaning when implementing this boundary. [Position guide][S4]; [display event extraction][S11].

## 5. Current ViNED replay specification

All values in this section describe [src/visual_stim_gen.py](../src/visual_stim_gen.py), not IBL hardware specifications.

| Parameter | Implemented value or rule |
| --- | --- |
| Storage | `<replay_dir>/<eid>/trial_<original row index, minimum 4 digits>.mp4` |
| Default root | `ibl_task_replay/`; `VINED_REPLAY_DIR` override |
| Video | 720 × 720 pixels, 30 FPS, `mp4v`, grayscale converted to three-channel BGR |
| Background | Integer intensity 128 |
| Patch | 240 × 240 pixels; radius constant 120 pixels |
| Coordinates | Vertical center 360; initial horizontal center 120 or 600; target 360 |
| Carrier | `sin(2*pi*3*x)`, with `x` spanning −1 to +1: six cycles across the patch coordinate span |
| Envelope | `exp(-(x*x+y*y)/(2*0.25**2))`; sigma 0.25 in normalized coordinates, approximately 30 pixels |
| Intensity | `(grating*envelope + 1)/2`, scaled to 8-bit; no per-trial contrast multiplier |
| Phase / orientation | Fixed sine phase and vertical bars |
| Gain | 240 pixels divided by absolute wheel displacement from go-cue trigger to feedback in the first usable successful trial; fallback 500 pixels/radian |
| Motion | Relative to wheel position at stimulus onset; side-dependent sign; clamped to 0–240 pixels toward center |
| Freeze | Wheel position sampled at `feedback_times` after feedback |
| Duration | `max(stimOff-stimOn, 1.0)` seconds; missing offset replaced by onset +2 seconds |
| Render query times | Inclusive `linspace(stimOn, stimOff, int(duration*30))` |

Path defaults are defined in [src/utils/paths.py](../src/utils/paths.py).

### Consequences for fidelity

The following are findings from source inspection, not measured reconstruction errors:

1. **Contrast is lost.** Contrast selects side, but its magnitude never scales the patch. Even a valid zero-contrast trial gets a visible grating.
2. **Errors cannot be replayed faithfully.** Movement away from the center is clamped away. The real task permits outward error trajectories.
3. **The right patch can disappear at its starting position.** Its right edge is 720, but the paste guard requires `x1 < VIDEO_WIDTH`; equality fails.
4. **Calibration is empirical and event-dependent.** Fitting one successful trial's displacement does not establish the physical screen/wheel conversion; gain fitting and rendering also use different reference events.
5. **Time is distorted.** `linspace` includes offset, whereas video timestamps are frame index/30. The one-second minimum makes the mismatch much larger for short trials. No pre-stimulus frames are generated.
6. **Patch geometry and phase are approximations.** The code has no measured conversion from pixels to visual degrees, session sigma, random trial phase, or display calibration.

Evidence: [renderer](../src/visual_stim_gen.py). Physical interpretation: [behavior methods][S3], [IBL movement model][S4], [raw stimulus fields][S7].

## 6. CLIP representation and output schema

This section describes [src/prepare_visual_stim.py](../src/prepare_visual_stim.py) and [load_visual_stimulus / bin_behaviors](../src/utils/ibl_data_utils.py).

| Setting | Current value |
| --- | --- |
| Model | `openai/clip-vit-large-patch14`, loaded with `CLIPModel` and its `CLIPProcessor` |
| Extraction | Evaluation mode, no gradients, `get_image_features`; one pooled vector per frame |
| Normalization | L2 normalization along feature dimension |
| Sampling | Default 5 FPS; stride `max(int(native_fps/sample_fps), 1)`; 30 FPS input gives every sixth frame |
| Time coordinate | `frame_index / native_fps`, relative to the beginning of the encoded video |
| Batch / device | Default 32 frames; CUDA if available, otherwise CPU |
| Feature width | Current project expects 768; verify after any checkpoint change |
| Output | `<output_dir>/<eid>_visual_clip.npz` |
| Downstream default root | `datasets/vis_stim/`, configurable through `VINED_VISUAL_DIR` |

Let `K` be the saved trial count and `F_i` the sampled frame count of trial `i`:

| NPZ key | Logical schema |
| --- | --- |
| `trial_ids` | `K` IDs, stored using `dtype=object` |
| `times` | `K` sequences, each `[F_i]`, values converted to float32 |
| `features` | `K` sequences, each `[F_i, 768]`, values converted to float32 |

The writer wraps all three arrays with `dtype=object`; equal-length sequences can yield extra physical array dimensions. Treat the table as the logical schema, and validate loaded shapes. The loader uses `allow_pickle=True` and renames `features` to `values` internally. The archive contains feature vectors, not patch tokens or reconstructed pixels.

**Trial identity issue:** extraction enumerates sorted video filenames instead of parsing their original trial numbers. Missing files can shift IDs. Also, default multi-session extraction slices `eids[9:n_sessions]`, while replay generation contains its own explicit session list. Record the actual processed EIDs and original trial IDs for each run. [Extractor](../src/prepare_visual_stim.py); [renderer](../src/visual_stim_gen.py).

## 7. Alignment with neural data

The [preparation configuration](../src/prepare_data.py) uses `stimOn_times`, a window of **−0.5 to +1.5 seconds**, and **20 ms bins**: 100 time bins per trial. Expected aligned shapes are spikes `[100, N_neurons]` and vision `[100, 768]`.

Currently, [bin_behaviors](../src/utils/ibl_data_utils.py) stretches each video's first-to-last sampled feature time into 100 positions. Equal array lengths therefore do not establish equal physical times. Linear interpolation also does not create new observations between the approximately 200 ms-spaced source features.

**Recommended contract, not yet implemented:**

- Use shared bin edges `stimOn - 0.5 + k*0.02`, `k=0..100`, and an explicit visual sampling convention, such as bin centers.
- Save the actual session timestamp of every rendered frame; video playback time alone is insufficient for the current renderer.
- Preserve `(eid, original_trial_id)` through extraction, filtering, splitting, and batching.
- Distinguish observed/reconstructed blank background from unavailable frames. Encode known blank frames through the same processor; mask unavailable intervals.
- Keep a time-level validity mask; exclude padding and unavailable features from losses and metrics.
- Record whether interpolated features are renormalized; interpolation of unit vectors does not generally preserve unit length.

## 8. Minimum provenance and acceptance checks

The following are proposed project requirements based on the gaps above:

| Record | Required contents |
| --- | --- |
| IBL origin | EID, subject/session metadata, release membership, task/rig version, collection, dataset revision and file identifiers/hashes |
| Stimulus | Side, contrast, phase, spatial frequency, orientation, sigma with units, calibrated screen mapping, wheel radius/gain |
| Events | Onset, response/freeze, offset, cue and feedback; clock/synchronization method and missing-value policy |
| Replay | Renderer revision, canvas/FPS/codec, frame session timestamps, assumptions and validity masks |
| Features | Checkpoint and revision, processor configuration, package versions, sampling rule, dtype, shape and normalization |
| Training alignment | Window, edges/centers, interpolation policy, original trial IDs, split membership and masks |

Before calling a replay faithful, check zero/low/high contrast trials on both sides, inward and outward movements, short trials, missing events, and right-edge visibility. Compare reconstructed positions with the IBL physical mapping and available raw stimulus parameters. Confirm feature timestamps fall on the intended neural window and that removing a video does not renumber later trials. These checks have **not been run** as part of this documentation task.

## Sources

All external sources below were consulted on **2026-09-15**. Inline citations identify where they are used. IBL's live documentation may change; preserve dataset/software revisions with experiments.

1. [IBL: 2024 Reproducible Ephys release][S1].
2. [IBL: datasets and folder structure][S2].
3. [International Brain Laboratory et al. (2021): Standardized and reproducible measurement of decision-making in mice, eLife 10:e63711][S3].
4. [IBL: Computing the stimulus position using the wheel][S4].
5. [IBL: brainbox.task.trials source][S5].
6. [IBL: brainbox.behavior.training source][S6].
7. [IBL: raw data loader API and stimulus fields][S7].
8. [IBL: Loading Trials Data][S8].
9. [IBL: training wheel extraction and units][S9].
10. [IBL: training trial extractors][S10].
11. [IBL: electrophysiology FPGA event extraction][S11].
12. [IBL: task timing quality-control metrics][S12].

Local implementation sources: [replay generator](../src/visual_stim_gen.py), [CLIP extractor](../src/prepare_visual_stim.py), [data preparation](../src/prepare_data.py), [IBL/visual loading and interpolation](../src/utils/ibl_data_utils.py), [path configuration](../src/utils/paths.py), [all EIDs](../data/eids.txt), [train EIDs](../data/train_eids.txt), [test EIDs](../data/test_eids.txt), and [archived NEDS documentation](README_NEDS.md).

[S1]: https://docs.internationalbrainlab.org/notebooks_external/2024_data_release_repro_ephys.html
[S2]: https://docs.internationalbrainlab.org/notebooks_external/data_structure.html
[S3]: https://elifesciences.org/articles/63711
[S4]: https://docs.internationalbrainlab.org/notebooks_external/docs_wheel_screen_stimulus.html
[S5]: https://docs.internationalbrainlab.org/_modules/brainbox/task/trials.html
[S6]: https://docs.internationalbrainlab.org/_modules/brainbox/behavior/training.html
[S7]: https://docs.internationalbrainlab.org/_autosummary/ibllib.io.raw_data_loaders.html
[S8]: https://docs.internationalbrainlab.org/notebooks_external/loading_trials_data.html
[S9]: https://docs.internationalbrainlab.org/_modules/ibllib/io/extractors/training_wheel.html
[S10]: https://docs.internationalbrainlab.org/_modules/ibllib/io/extractors/training_trials.html
[S11]: https://docs.internationalbrainlab.org/_autosummary/ibllib.io.extractors.ephys_fpga.html
[S12]: https://docs.internationalbrainlab.org/_autosummary/ibllib.qc.task_metrics.html
