# ViNED repository audit

**Reviewed:** 2026-09-15

**Source revision:** `d308b04` (`docs: record checkout rename and environment validation`)

**Scope:** data and execution pipelines, correctness, research validity, performance, security risks, unused code, and optimization opportunities.

## Summary

The highest-priority problems affect whether training and evaluation measure the intended visual–neural relationship. Visual and neural timestamps use different grids; trial filtering ignores its supplied validity mask; standalone encoding/decoding heads are not registered with PyTorch; evaluation discards trained session projections; and evaluation task routing is reversed. Fixing these should precede long training runs or interpretation of current scores.

This report records findings from the reviewed source, including small synthetic reproductions. It extends and rechecks the research concerns in [AGENT.md](../AGENT.md). Findings describe the revision above unless a dated follow-up records a resolution.

The report contains **64 numbered findings and opportunities**, plus a separate dead-code inventory.

### Contents

- [Data preparation and stimulus pipeline](#1-data-preparation-and-stimulus-pipeline)
- [Model, masks, and tensor contracts](#2-model-masks-and-tensor-contracts)
- [Training, fine-tuning, and distributed execution](#3-training-fine-tuning-and-distributed-execution)
- [Evaluation and scientific interpretation](#4-evaluation-and-scientific-interpretation)
- [Optional LFP path](#5-optional-lfp-path)
- [Pipeline, packaging, operational and security risks](#6-pipeline-packaging-operational-and-security-risks)
- [Coding defects and dead-code inventory](#7-coding-defects-and-dead-code-inventory)
- [Performance and optimization opportunities](#8-performance-and-optimization-opportunities)
- [Verification and recommended regression checks](#9-verification-performed-and-recommended-regression-checks)

### Reading the findings

| Label | Meaning |
| --- | --- |
| **P0** | Blocks trustworthy training, data pairing, or evaluation on an active path. |
| **P1** | Significant correctness, reproducibility, reliability, or resource issue; fix before using the affected workflow. |
| **P2** | Maintainability issue, conditional defect, or optimization opportunity. |
| **P3** | Lower-priority cleanup or inactive legacy functionality. |
| **Reproduced** | Demonstrated using local synthetic inputs against the current code. |
| **Confirmed** | Directly established by source/control-flow inspection; production execution was not needed to establish the defect. |
| **Risk** | The implementation creates a plausible problem whose practical impact needs measurement or infrastructure validation. |
| **Candidate** | No active in-repository caller was found; external use must be checked before removal. |

Priorities describe the affected path, not a claim that every experiment encounters every defect. Source links include line anchors for the reviewed revision. No finite review can establish that all possible defects have been found.

### Current resolution status (2026-09-16)

Status labels beneath each numbered finding describe the current implementation. The original evidence and actions remain below them as historical context. **FIXED** means the reported implementation defect is addressed; it does not claim real-session regeneration or production training/evaluation has been completed. Recent implementation references: `ca95114` (renderer), `f3a4370` (visual pipeline), and `7173291` (repository rule).

| Status | Findings |
| --- | --- |
| **Fixed** | D01, D02, D03, D04, D06, D07, D11, D12, D13, M03, P02, P05, F06 |
| **Partially fixed** | D05, M05, M06, M09, T05, T06, S01, S02, F02, F05 |
| **Superseded by repository policy** | P01 |
| **Open** | All other numbered findings |

### Suggested order

1. Correct stimulus rendering, trial identity, time alignment, and validity filtering: **D01–D06, D08–D09**.
2. Make training heads, masks, singleton batches, and optimizer schedules correct: **M01–M06, T01–T03, T05–T06**.
3. Restore complete checkpoints and evaluate the right task over every trial: **E01–E06**.
4. Version datasets/checkpoints and add regression checks: **D12–D13, T08, P01**.
5. Establish meaningful baselines, then profile and optimize: **E08, F01–F09**.
6. Validate optional LFP and distributed workflows; remove or isolate legacy code after checking its callers.

## Visual pipeline follow-up (2026-09-16)

D02, D01, D03, and D11 have been addressed in the current implementation:

- **D02:** extraction parses original trial IDs and checks videos against the replay manifest and sidecars. IDs are mapped explicitly during binning and saved with intervals through dataset splits, loader batches, and caches.
- **D01:** schema-v2 numeric feature archives carry float64 session timestamps and frame validity. Visual features are interpolated at `stimOn - 0.5 + (k + 0.5)*0.02`, using the neural bin centers. No extrapolation or interpolation across invalid source samples is allowed; interpolated vectors are renormalized. Unavailable bins use zero placeholders plus false masks. Onset-to-offset replays do not establish pre-onset/post-offset background observations.
- **D03:** alignment applies the actual trial mask, checks its length and supplied index order, and excludes trials with no usable visual bins. Partial coverage is retained with per-bin masks.
- **D11:** missing feature files lead to an explicit no-coverage session skip; malformed archives raise errors rather than being silently treated as missing. Numeric archives validate IDs, offsets, timestamp ordering, feature width/finite values, EID, clock, and masks without pickle. Interrupts propagate. The visual validity mask reaches model input masking, losses, trainer visual metrics, and standalone visual cosine scoring.

**Migration:** regenerate CLIP features from the corrected replay videos and sidecars, then rebuild aligned datasets and caches in fresh directories. Legacy feature archives and caches are rejected because they lack trustworthy identity/time/validity metadata. Do not mix old and new cached samples. General cache publication/versioning remains D13.

Validation uses syntax compilation, source inspection, and runtime import checks; no tests were written under the repository rule. Real-session feature regeneration, training, and evaluation have not been performed for this change. Separate evaluation findings (including last-batch-only scoring and truncated visual prediction exports) remain open; the validity-mask integration does not resolve those findings.

## 1. Data preparation and stimulus pipeline

### D01 — Visual features and spikes do not share a physical time grid

**Status (2026-09-16): FIXED** - Session timestamps now drive neural-bin-center interpolation; unavailable visual bins retain validity masks through training and visual metrics. Existing artifacts must be rebuilt.

**P0 · Confirmed.** [Neural window](../src/prepare_data.py#L48); [visual interpolation](../src/utils/ibl_data_utils.py#L414).

Spikes cover `stimOn_times + [-0.5, 1.5)` with 20 ms bins. Visual interpolation stretches each video's first-to-last sampled frame over 100 bins, regardless of its duration. Replay time zero is stimulus onset; it has no pre-stimulus frames. Equal array lengths therefore do not establish temporal correspondence, and later video frames can be paired with earlier neural bins.

**Action:** define one stimulus-relative time grid, preserve original frame timestamps, and apply an explicit missing-frame/background policy with validity masks. Verify event alignment on known synthetic timestamps before rebuilding features and datasets.

### D02 — Filename enumeration replaces original trial identity

**Status (2026-09-16): FIXED** - Original IDs are parsed and checked against replay metadata, mapped explicitly during alignment, and preserved through splits and caches.

**P0 · Confirmed.** [Feature extraction](../src/prepare_visual_stim.py#L145); [trial lookup](../src/utils/ibl_data_utils.py#L425).

The extractor saves the enumeration index of sorted filenames as `trial_ids`, rather than parsing `trial_XXXX.mp4`. A missing file shifts subsequent IDs; for example, `trial_0000.mp4` and `trial_0002.mp4` become IDs 0 and 1. Additional MP4s in the directory also participate. `bin_behaviors` then uses these IDs as array positions. Its optional filtered DataFrame path also assumes original IDs equal compact positions.

**Action:** carry explicit original trial IDs through replay, extraction, filtering, and splitting; validate uniqueness and map IDs to rows explicitly. Include a missing-middle-trial case.

### D03 — The supplied trial validity mask is ignored

**Status (2026-09-16): FIXED** - Alignment applies the supplied trial mask and validates its length and supplied index order.

**P0 · Reproduced.** [align_data](../src/utils/ibl_data_utils.py#L539).

The `trials_mask` branch converts the mask and then combines `target_mask` with `beh_mask` again. A synthetic `[True, False]` trial mask retained both trials when visual data existed. Trials excluded for missing events, reaction time, or choice can therefore enter training and evaluation.

**Action:** combine the actual trial mask after checking length and identity. Verify that invalid neural trials remain excluded even when visual features are complete.

### D04 — A stationary right-side stimulus is not rendered

**Status (2026-09-16): FIXED** - The renderer clips patches to valid canvas slices, including exact right-edge coordinates.

**P1 · Reproduced.** [render_trial_frame](../src/visual_stim_gen.py#L193).

For a right stimulus at zero wheel displacement, `x1 == VIDEO_WIDTH`. The strict `x1 < VIDEO_WIDTH` condition rejects the otherwise valid slice. The synthetic right frame contained zero non-background pixels, while the equivalent left frame contained 54,356. Right-side stimuli can remain absent until movement brings the patch inside the boundary.

**Action:** use bounds consistent with array slicing, and compare stationary and moving stimuli on both sides, including exact-edge coordinates.

### D05 — Replay omits stimulus contrast and constrains movement

**Follow-up (2026-09-19):** Added an explicit unit-normalized session/trial parameter manifest and strict completeness mode. Rendering now accepts supplied phase, angular sigma, frequency, orientation, signed starting position, wheel radius/gain and field of view; checks side/contrast against ALF; records effective parameters, source evidence and manifest hash; and scales patch support with sigma. Unknown/missing IDs and malformed parameters fail rather than silently mispairing trials. Python compilation and code inspection passed; no tests were written. **Empirical validation remains open:** no measured session manifest was available, raw-unit/phase/clock conversions are not inferred, and replay fidelity is still explicitly unverified. See the D05 manifest contract in the stimulus specification.

**Status (2026-09-16): PARTIALLY FIXED** - Contrast, outward motion, physical-reference gain, and recorded replay parameters are implemented. Session-specific calibration and reconstruction fidelity remain unverified.

**P1 · Confirmed implementation; research impact needs validation.** [Grating creation](../src/visual_stim_gen.py#L126), [movement clamp](../src/visual_stim_gen.py#L219), [trial rendering](../src/visual_stim_gen.py#L283).

Contrast values select the side but never scale the patch, so zero/low/high contrast trials can receive the same grating appearance. Movement is clamped between its starting point and the center; reverse motion and overshoot are suppressed. Gain is inferred from the first successful trial, using go-cue timing, with a hardcoded fallback of 500. These choices constrain the visual targets and may erase task-relevant variation.

**Action:** parameterize and validate these assumptions against the [stimulus reference](ibl-visual-data-specs.md). Store replay parameters and distinguish approximate replay from validated stimulus reconstruction.

### D06 — Replay timing, invalid trials, and output handling are fragile

**Follow-up (2026-09-19):** Reviewed the existing staged replay and atomic feature-output changes. Extraction now also checks decoded dimensions against the manifest canvas and session timestamps against stimulus onset/offset, rejects unsupported sample rates before model initialization, and requires positive integer batch sizes. Python compilation and code inspection passed; no tests were written. Video/CLIP execution was not repeated: the local `.venv` points to an unavailable Python interpreter.

**Status (2026-09-16): FIXED** - Replay sessions are staged and published by directory rename after MP4 decoding verifies FPS, dimensions, and frame count. Input-invalid trials are distinguished from output failures. Offset-equal frames caused by floating-point rounding are excluded. Extraction checks manifest frame counts/FPS against sidecars and decoded videos, and validates feature archives before replacement using unique temporary files with cleanup. Python compilation, CLI help, code inspection, and manual short-video rendering/decoding were used; no tests were written. Full production regeneration was not run.

**P1 · Confirmed.** [Video generation](../src/visual_stim_gen.py#L294); [output directory](../src/visual_stim_gen.py#L399); [frame sampling](../src/prepare_visual_stim.py#L21).

Frame count uses `max(stim_off - stim_on, 1.0)`, while frame contents use `linspace(stim_on, stim_off, n_frames)`. Sub-second trials are stretched to at least one playback second, and endpoint-inclusive timestamps differ from fixed-FPS playback. NaN duration is printed but execution continues into `int(duration * FPS)`. The writer is not checked with `isOpened()` or released in a `finally` block. Creating the session directory omits `parents=True`, so a fresh replay root can fail. Extraction does not validate positive FPS/batch size or detect partial decoding.

**Action:** validate events and arguments, derive frame timestamps from one clock, preserve timing metadata, create parent directories, and verify completed outputs before publishing them.

### D07 — Session selection differs across preparation stages

**Follow-up (2026-09-19):** Preparation wrappers now accept the shared named selection arguments directly while preserving their positional count/EID/manifest interface. The reporting helper materializes iterable EID selections before processing so reporting cannot consume them. Selection CLI checks confirmed first-N selection and rejection of zero counts; Python compilation and wrapper Bash syntax checks passed. No production sessions were processed.

**Status (2026-09-16): FIXED** - Generation, extraction, aligned preparation, and batch caching use `utils.sessions`: `--eid`, `--eids-file` (or `VINED_EIDS_FILE`), and `--n-sessions`/`--n_sessions`. Defaults select all of `data/eids.txt`; count 1 without an EID selects its first entry. Empty/duplicate/invalid EIDs and invalid counts are rejected. Each stage reports requested, completed, skipped, failed, and unprocessed EIDs. Ordinary failures continue to the next EID and yield a nonzero final exit; interrupts propagate. The cache wrapper no longer defaults to test sessions. Manifest selection, CLI help, and Bash syntax were checked; full multi-session production was not run.

**P1 · Confirmed.** [Extraction selection](../src/prepare_visual_stim.py#L236); [data selection](../src/prepare_data.py#L37); [replay selection](../src/visual_stim_gen.py#L455); [cache wrapper](../script/run_create_dataset.sh#L11).

Extraction uses `[9:args.n_sessions]`, so 20 requests select 11 sessions and requests from 2 through 9 select none. Data preparation uses `[:n_sessions]`; replay uses a separate hardcoded list; the batch cache wrapper processes only `test_eids.txt`. Individually valid commands therefore do not define one consistent end-to-end session set.

**Action:** use an explicit session manifest or shared selection function across all stages, and report requested, completed, skipped, and failed EIDs.

### D08 — Neuron metadata can become detached from binned columns

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed conditional defect.** [Binned cluster IDs](../src/utils/ibl_data_utils.py#L236); [metadata filtering](../src/prepare_data.py#L91).

`bin_spiking_data` returns `clusters_used_in_bins`, but the caller ignores it. `keep_unit_idxs` indexes binned columns and is applied directly to metadata for all original clusters. Missing/non-spiking clusters or region selection can make those index spaces differ, assigning the wrong UUID, depth, or region to a neuron.

**Action:** map binned columns through returned cluster IDs before filtering metadata; preserve that mapping in the cache. Verify with non-contiguous cluster IDs and a cluster with no spikes.

### D09 — Merging three or more probes reuses cluster IDs

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Reproduced.** [merge_probes](../src/utils/ibl_data_utils.py#L56).

The offset is replaced by each probe's local maximum plus one instead of accumulating previous offsets. Three synthetic two-cluster probes produced only four unique IDs instead of six. The helper also changes its input spike dictionaries in place. Current two-probe use does not demonstrate the three-probe failure, but the general merge contract is incorrect.

**Action:** build cumulative, explicit cluster mappings without mutating inputs, and verify uniqueness and metadata correspondence after merging.

### D10 — Firing-rate filtering has ambiguous units and uses all trials

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed behavior; protocol risk.** [Filtering](../src/prepare_data.py#L48), [split](../src/prepare_data.py#L155).

`fr_thresh=0.2` becomes a threshold of `1 / fr_thresh`, i.e. 5 Hz. The variable name does not communicate the inversion. Selection uses all binned trials before validity filtering and train/validation/test partitioning. If the intended protocol requires training-only feature selection, held-out neural activity influences which units are retained. Sessions with zero retained units have no explicit guard.

**Action:** declare threshold units, decide whether selection is session-level QC or a fitted preprocessing step, and validate the resulting unit count. Fit on training trials when required by the chosen protocol.

### D11 — Missing and malformed visual data are not handled consistently

**Status (2026-09-16): FIXED** - Numeric schema-v2 archives validate identity, clocks, shapes, finite values, and masks. Missing files produce no coverage; malformed files fail explicitly and interrupts propagate.

**P1 · Confirmed.** [Loading](../src/utils/ibl_data_utils.py#L601); [binning](../src/utils/ibl_data_utils.py#L409); [alignment](../src/utils/ibl_data_utils.py#L552).

Loading catches `BaseException`, including interrupts, and returns `skip=True`. Binning may then return no modalities; the subsequent `align_data` trial-mask branch still reads `beh_mask`, which was never assigned. `align_data` tests only `x is not None`, so arrays containing NaNs are treated as present. The “Remove modality” message at the missing-trial threshold does not remove that modality. Negative/duplicate trial IDs, non-monotonic times, and inconsistent feature widths are not rejected explicitly.

**Action:** validate a feature schema at ingestion, separate recoverable missing data from malformed data, propagate interrupts, and define whether each failure skips a trial, a modality, or the session.

### D12 — Trial provenance and split reproducibility are incomplete

**Status (2026-09-19): FIXED (implementation)** - Splits use an EID-derived seed and keep connected overlapping physical intervals together. Aligned rows retain split membership and provenance IDs; provenance records parameters, original IDs/intervals, rejection IDs, neuron ordering, feature hashes, and source hashes. `split_independence.json` reports configured train/test session and subject overlap, with missing sessions explicitly unverified. Block/subject independence is not claimed for within-session trial splits. Real-session regeneration remains pending.

**P1 · Confirmed; independence is a research risk.** [Partitioning](../src/prepare_data.py#L155); [dataset schema](../src/utils/dataset_utils.py#L57).

Original trial IDs and the computed `intervals` are not saved. A single process-level NumPy seed drives successive session splits, so a session's split depends on which sessions were processed before it. The code offers random trial splits, without checking temporal overlap, block dependence, or subject-level independence. The checked-in EID lists currently contain 18 training and 2 test sessions with no overlap; that alone does not establish subject independence.

**Action:** persist original IDs, interval timestamps, split membership, seed, and selection criteria. Make per-session splits stable, and report separately held-out trials, sessions, and subjects. Check actual interval overlap rather than assuming leakage.

### D13 — Cache reruns can retain stale samples and mix preprocessing versions

**Status (2026-09-19): FIXED (implementation)** - Cache generation writes immutable generation directories and atomically publishes each session manifest after checking complete split membership. Readers load only manifest-listed files and validate schema, preprocessing source/package versions, options, aligned-file hashes, sample hashes, identities, and shapes. Old/unreferenced files are ignored; legacy caches must be rebuilt. Publication is atomic per session, not across a multi-session run.

**P1 · Confirmed mechanism.** [Cache writes](../src/create_dataset.py#L123); [cache discovery](../src/loader/base.py#L356); [cached item loading](../src/loader/base.py#L601).

Cache filenames contain an EID and sequential counter. Reruns overwrite matching names but do not invalidate older files beyond the new count; a smaller dataset leaves stale trials loadable. Changed shuffling or multi-session composition also changes counter assignments. There is no schema/preprocessing hash tying `.npy` caches to aligned datasets, neuron ordering, CLIP revision, or padding settings. The cached path bypasses the preprocessing options passed to `make_loader`.

**Action:** write a versioned manifest with exact files, sample IDs, shapes, and preprocessing hashes; build into a new directory and publish only after validation. Reject incompatible caches explicitly.

### D14 — Spike counts silently narrow to unsigned bytes

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Reproduced conditional defect.** [Sparse conversion](../src/utils/dataset_utils.py#L43).

`dtype=np.ubyte` restricts counts to 0–255. Synthetic counts `[256, 300]` became `[0, 44]`. Such counts may be unusual in current 20 ms single-unit bins, but bin-size changes or other inputs can silently corrupt values.

**Action:** validate integer/nonnegative range before conversion or select a suitable count dtype. Add a boundary round trip when changing the cache format.

## 2. Model, masks, and tensor contracts

### M01 — Standalone output heads are not registered with PyTorch

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P0 · Reproduced.** [init_unimodal_stitcher](../src/multi_modal/mm.py#L114).

For `encoding` and `decoding`, `mod_stitcher_proj_dict` is an ordinary dictionary of modules. In both modes, the synthetic model had **0 of 4 output-head parameter tensors** in `model.parameters()` and **zero output-head checkpoint keys**. These heads receive no optimizer updates or parent-module gradient clearing; parent `.to()`, `.train()`, and distributed management also do not traverse them. Explicitly placing them on whichever CUDA device is available adds device-selection risk.

**Action:** use registered module/parameter containers, let the parent model control placement, and verify parameter updates and checkpoint round trips. The registration distinction is documented by [PyTorch ModuleDict](https://docs.pytorch.org/docs/2.12/generated/torch.nn.ModuleDict.html).

### M02 — Training's mask CLI/search settings do not reach the model

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Argument selection](../src/train.py#L68); [model construction](../src/train.py#L255); [YAML masker](../src/configs/multi_modal/mm_single_session.yaml#L3).

`train.py` reads/logs `mask_ratio` and uses `args.mask_mode` in run names, but never writes them into `config.model.masker`. The model retains YAML defaults, including ratio 0.3 and temporal mode. A run named ratio 0.1 can therefore train with ratio 0.3; hyperparameter search also spends trials on an ineffective mask-ratio parameter.

**Action:** resolve and validate all CLI/search overrides once before model construction, persist the resolved configuration, and assert that model settings match the run metadata.

### M03 — Mixed masking reuses the first sample's mask and validity

**Status (2026-09-16): FIXED** - Mixed masking indexes each sample's mask and validity instead of reusing sample zero. Shared spike/vision random-token patterns remain an explicit design choice.

**P1 · Confirmed.** [Mixed masks](../src/multi_modal/mm.py#L302), [application](../src/multi_modal/mm.py#L358).

Every selected scheme uses `mask_map[mod][scheme][0, :, 0]` and `inputs_attn_mask[0]`. Samples sharing a scheme receive the first row's random pattern, and all samples inherit the first row's temporal validity. With different valid lengths, this can include padded targets or exclude real targets. Spike and vision also share the random-token pattern, a separate design choice that should be explicit.

**Action:** index masks and validity by sample; verify independent patterns and valid-target counts with unequal-length inputs.

### M04 — Session embeddings depend on batch composition

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Reproduced.** [Session embedding application](../src/multi_modal/encoder_embeddings.py#L122).

`argwhere(...).squeeze()` produces a scalar for one sample from an EID; `if mask.dim() > 0` then skips its session embedding. The same input received a different positional/modality/session embedding when evaluated alone versus duplicated in a batch. This affects singleton final batches and mixed-session batches.

**Action:** preserve a one-dimensional index vector and add session embeddings for every sample. Verify batch-composition invariance with dropout disabled.

### M05 — Padding is not excluded consistently from attention and metrics

**Status (2026-09-16): PARTIALLY FIXED** - Visual validity reaches input masking, losses, and visual metrics; mixed-mask indexing is fixed. Attention still lacks an explicit padding mask, and neural metric masking remains open.

**P1 · Confirmed.** [Encoder attention](../src/multi_modal/mm.py#L176); [loss masks](../src/multi_modal/mm.py#L350); [validation metrics](../src/trainer/base.py#L451); [standalone metrics](../src/utils/eval_utils.py#L475).

The encoder always receives `mask=None`, so padded input tokens remain visible to attention. Normal forward loss does intersect the target mask with time validity, and spike loss additionally excludes `-1` padding; however, validation and standalone metrics consume complete predicted/target tensors without corresponding time masks. Visual loss has no independent finite-value validity check. The mixed-mask bug in M03 further weakens the normal loss protection.

**Action:** separate input validity, reconstruction masks, and metric masks, propagate each explicitly, and ensure arbitrary padding values cannot change valid predictions or scores.

### M06 — Loader padding and small-shape handling violate their contracts

**Status (2026-09-16): PARTIALLY FIXED** - The visual loader rejects sequences above its configured maximum and pads visual validity explicitly. Other padding and singleton-shape contracts remain open.

**P1 · Partly reproduced.** [Raw sample conversion](../src/loader/base.py#L448); [padding](../src/loader/base.py#L582); [attention mask](../src/loader/base.py#L83).

Vision is padded before the assertion comparing it with unpadded spikes. A valid two-bin trial requested at length four raised `Spike/vision mismatch: 2 vs 4`. `_pad_data` does not truncate overlong input but still returns a negative padding length. Left padding of two positions in length five produces `[1,1,0,0,0]`, rather than `[0,0,1,1,1]`. Unqualified `squeeze()` also collapses one-neuron/one-time-bin axes. With `load_meta=False`, the fallback metadata has length one and can reduce the selected neuron columns to one.

**Action:** validate unpadded correspondence first, pad/truncate modalities consistently, preserve named axes, and make metadata optional without changing data selection.

### M07 — Output normalization couples neural rates to other neurons

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Risk.** [Spike decoder](../src/models/stitcher.py#L99); [shared latent normalization](../src/multi_modal/mm.py#L380).

Spike outputs pass through `LayerNorm(val)` before being interpreted as Poisson log-rates. The normalization couples each neuron's output to other neurons, including padded output columns. Shared latents are also unit-normalized whenever vision is available. These operations constrain rate predictions; they are not automatically invalid, but their effect on neural scale and likelihood is unvalidated.

**Action:** compare an unconstrained neural log-rate head against the normalized head, track firing-rate calibration and BPS, and test that adding padded neuron columns does not alter real-neuron predictions. Do not attribute historical negative BPS to this alone.

### M08 — Query/key normalization may flatten attention

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Risk.** [Attention](../src/multi_modal/mm_utils.py#L159).

Queries and keys are normalized across the entire hidden vector before splitting into heads; SDPA then applies its default inverse-square-root head-dimension scale. This can produce small logits and nearly uniform attention. The implementation differs from ordinary unnormalized scaled dot-product attention and from per-head normalization with a deliberate temperature.

**Action:** measure attention entropy and gradients, then compare validated alternatives. The default scaling is described in [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). No speed or accuracy improvement is assumed.

### M09 — Dimensions and session identity are spread across code and files

**Status (2026-09-16): PARTIALLY FIXED** - Visual feature width is validated as 768. Centralized dimension configuration and stable session vocabulary remain open.

**P1 · Confirmed configuration risk.** [Session vocabulary](../src/multi_modal/encoder_embeddings.py#L18); [feature dimensions](../src/models/stitcher.py#L7); [model dimensions](../src/multi_modal/mm.py#L83); [RoPE construction](../src/multi_modal/encoder_embeddings.py#L234).

Visual width 768, sequence length 100, modality lists, and EID vocabulary are repeated. Although extraction accepts another CLIP model, downstream dimensions remain fixed. Session embeddings use the order of text files read at import time: reordering those files can silently reinterpret checkpoint rows, and an unseen EID can raise `KeyError`. Encoder layers do not pass the configured sequence length into attention's RoPE cache constructor.

**Action:** store feature schema and EID-to-index mapping with datasets/checkpoints, validate compatibility at startup, and centralize dimension/modality definitions. Define a supported unseen-session adaptation path.

## 3. Training, fine-tuning, and distributed execution

### T01 — Scheduler steps underestimate batches

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Schedule calculation](../src/train.py#L295); [loader](../src/loader/make_loader.py#L87); [stepping](../src/trainer/base.py#L288).

Training calculates `num_train // global_batch_size`, but the loader keeps the final partial batch. Nine samples at batch size eight cause two optimizer steps per epoch, while the schedule budgets one. OneCycleLR can exhaust its configured steps before training ends. The count also comes from the aligned dataset, while actual samples come from independently maintained `.npy` caches.

**Action:** calculate optimizer steps from the actual prepared loader, accumulation policy, and actual epoch count; validate against the observed steps, including partial batches and distributed sharding.

### T02 — Configured gradient accumulation is not implemented

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed conditional defect.** [Configuration use](../src/train.py#L297), [fine-tuning schedule](../src/finetune.py#L328), [training loop](../src/trainer/base.py#L295).

The schedule divides by `gradient_accumulation_steps`, but the trainer zeros gradients and updates weights every batch. No matching Accelerator accumulation configuration/context is present. Values above one change schedule length without accumulating gradients. The current default of one avoids this specific mismatch.

**Action:** implement one consistent accumulation policy or reject unsupported values; verify parameter updates and scheduler advances over a partial accumulation group.

### T03 — Actual epochs and scheduled epochs diverge

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Encoding epochs](../src/train.py#L53), [schedule epochs](../src/train.py#L121), [trainer range](../src/trainer/base.py#L212).

Encoding sets the trainer configuration to 130 epochs, but schedules learning rate over a local `num_epochs=4000`. Multi-GPU logic modifies that local number further without changing the trainer's range. Global-batch/LR overrides also need reconciliation with actual per-rank loader sizes. In fine-tuning, a searched learning rate initializes the optimizer, while OneCycleLR still uses `config.optimizer.lr` as its peak.

**Action:** derive trainer duration, effective batch size, and schedule from one resolved run configuration and record the resulting LR curve.

### T04 — The generic best-checkpoint branch cannot run

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed.** [Best-metric updates](../src/trainer/base.py#L226).

The loop over metrics includes `eval_avg_metric` and updates its best value. The following strict `>` comparison against that same updated value is false. `model_best.pt` is never saved by that branch and `best_eval_loss` stays infinite. `model_best_avg.pt` can still be saved by the earlier loop, which is the file current evaluation requests.

**Action:** define checkpoint-selection rules once, update best values after the decision, and use consistent names. Remove the unreachable duplicate branch.

### T05 — Validation drops one-sample session groups

**Status (2026-09-16): PARTIALLY FIXED** - Singleton visual validation groups are retained. Neural validation and encoding collection still need correction.

**P1 · Confirmed.** [Neural collection](../src/trainer/base.py#L335); [visual collection](../src/trainer/base.py#L361); [encoding collection](../src/trainer/base.py#L384).

The squeezed index of a singleton EID group is treated as empty: neural code sets `num_neuron=0`, and visual code skips it. Final one-sample batches and sessions occurring once per batch disappear from metrics. Plain shuffled batching increases singleton groups during multi-session use, and some sessions can contribute no validation samples.

**Action:** keep index vectors one-dimensional and verify that collected counts equal expected valid trial counts for each session, including batch size one.

### T06 — Failed validation collection can reuse another result

**Status (2026-09-16): PARTIALLY FIXED** - The stale-result defect is fixed: missing results are skipped before metrics. Explicit missing-sample counts in metric reporting remain open.

**P1 · Confirmed.** [eval_epoch](../src/trainer/base.py#L465).

A bare `except` prints a missing-EID message, then execution continues using `_gt` and `_preds`. They may be undefined or may still contain another session/modality's tensors from the preceding iteration. T05 supplies a realistic path to empty lists and this exception.

**Action:** fail with contextual information or explicitly mark/skip the result; never continue with stale variables. Make missing sample counts visible in the reported metrics.

### T07 — Main-process-only validation can conflict with DDP collectives

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Risk requiring distributed validation.** [Accelerator setup](../src/train.py#L111), [prepared objects](../src/train.py#L344), [validation gate](../src/trainer/base.py#L216).

Only the main process evaluates, using the DDP-wrapped model, while other ranks can advance into the next training epoch. DDP forward/backward synchronization must occur in compatible order; this can hang or mismatch collectives. Non-main ranks also retain initial best metrics, yet every rank reaches `train.report`.

**Action:** choose an explicit distributed validation design: all ranks evaluate and reduce results, or synchronize ranks around evaluation of an appropriate unwrapped model. Run a two-rank job through validation and checkpointing with a timeout. [PyTorch DDP guidance](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) identifies forward/backward synchronization requirements. This is not a reproduced cluster failure.

### T08 — Checkpoints lack the information needed for a reliable resume

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Checkpoint contents](../src/trainer/base.py#L574); [resume path](../src/train.py#L315); [run naming](../src/train.py#L222).

Checkpoints contain epoch, model, optimizer, and scheduler states, but no resolved configuration, preprocessing identity, EID mapping, RNG state, or best-so-far metrics. Resume assumes a `pretrained/model_epoch.pt` path that the normal saver does not create. Resume resets best tracking and does not restore loader shuffle state. Run names omit some behavior-changing settings, including mixed-training choice, and use truncated single-session IDs.

**Action:** make checkpoint location explicit, store a run manifest and required resume state, use unique run IDs, and verify interrupted-versus-uninterrupted training. Publish checkpoints atomically to protect against interrupted writes.

### T09 — Fine-tuning can load the wrong source and fails for some modes

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Pretraining path](../src/finetune.py#L174); [session replacement](../src/finetune.py#L273).

The pretraining path hardcodes `taskVar-all`; `args.pretrain_task_var` is an unused extra `.format` argument. Multi-session fine-tuning loops over every modality while indexing `model.encoder_embeddings`, but standalone encoding/decoding only constructs input modalities, so a missing modality can raise `KeyError`. It also reconstructs source architecture from current YAML and relies on the permissive loader in E01.

**Action:** accept an explicit source checkpoint/manifest, stitch only relevant registered modules, and verify each supported source/target mode and session transition independently.

### T10 — Hyperparameter search cannot stop poor trials early and evaluation picks an arbitrary trial

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Training report](../src/train.py#L418); [fine-tuning report](../src/finetune.py#L376); [ASHA setup](../src/train.py#L470); [evaluation selection](../src/eval.py#L137).

ASHA receives a report only after the complete training run, so it has no intermediate progress to use for early stopping. Evaluation chooses the first directory returned by `os.listdir`, rather than the trial with the best validation metric. M02 makes searched mask ratios ineffective, and T03 affects the fine-tuning LR search.

**Action:** report validation results at checkpoints during each trial, persist the selected best trial/checkpoint explicitly, and evaluate that artifact without filesystem-order assumptions.

## 4. Evaluation and scientific interpretation

### E01 — Evaluation intentionally skips trained session layers

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P0 · Confirmed.** [Checkpoint filtering](../src/utils/eval_utils.py#L142).

All keys containing `stitcher_dict`, `project_dict`, or `stitch_decoder_dict` are excluded. Evaluation of a training session therefore uses newly initialized input/output projections instead of its trained projections. Shape mismatches and other missing keys are also tolerated with `strict=False`. Printing missing keys does not establish that the loaded model is equivalent to the saved model.

**Action:** require full restoration for same-session evaluation, with identical predictions before/after saving. Make transfer to a new session a separate operation with an explicit allowlist of reinitialized parameters. M01 must be fixed before standalone heads can be restored at all.

### E02 — Encoding and decoding evaluation routes are reversed

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P0 · Confirmed.** [Training definitions](../src/train.py#L98); [evaluation switches](../src/eval.py#L213); [visual setup](../src/eval.py#L276).

Training defines encoding as vision-to-spikes and decoding as spikes-to-vision. Evaluation enables visual scoring for `encoding` and spike scoring for `decoding`, requesting outputs those standalone models do not produce. Visual evaluation additionally assumes an encoder named `spike`, which encoding lacks. The routing logic is duplicated later in the file with reversed explanatory comments.

**Action:** centralize the task definition and select evaluators from actual output modalities. Run an end-to-end smoke check for all three modes.

### E03 — Evaluation mixes data roots and does not preserve multi-session selection

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Loader data roots](../src/utils/eval_utils.py#L84), [second dataset load](../src/utils/eval_utils.py#L188); [evaluation arguments](../src/eval.py#L185).

Aligned datasets are read from `config.dirs.dataset_cache_dir` (default `datasets`), while cached trials use the caller's `data_path`. `eval.py` does not pass its session count, so the loader defaults to one. The second load explicitly requests one session and the cache folder is always `ibl_mm`, even though multi-session training uses `ibl_mm_N`. Model identity, metadata, and actual trials can therefore disagree.

**Action:** derive all data locations and EIDs from the checkpoint's dataset manifest, honor overrides consistently, and validate exact session and neuron mappings before prediction.

### E04 — Standalone evaluation only predicts the last batch

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Spike loop](../src/utils/eval_utils.py#L295), [spike forward](../src/utils/eval_utils.py#L366), [vision loop](../src/utils/eval_utils.py#L426), [vision forward](../src/utils/eval_utils.py#L473).

Both paths construct `mod_dict` inside the dataloader loop but call the model after that loop, retaining only the last batch. The hardcoded batch size of 10,000 hides the bug for smaller datasets and creates memory risk. A straightforward batch-size reduction would silently reduce evaluation coverage without fixing this control flow.

**Action:** predict and aggregate inside the loop, retain trial/session IDs, and compare scores and sample counts across multiple evaluation batch sizes.

### E05 — Held-out selections are computed but not applied to inputs

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed conditional defect.** [Spike held-out mask](../src/utils/eval_utils.py#L318); [vision held-out mask](../src/utils/eval_utils.py#L428).

`mask_result` and `mask_result_dict` are calculated but never passed into the model inputs or reconstruction masks. Evaluation instead masks the entire target modality. Current all-time-step tasks largely match that behavior, but partial forward-prediction/co-smoothing requests do not honor their requested conditioning. Neural metadata and active-neuron count are also taken from the first sample, which is unsuitable for combining sessions with different neurons.

**Action:** distinguish whole-modality tasks from partial holdout tasks; apply the declared mask and evaluate each session with its own neuron metadata. Verify with an input perturbation test that held-out targets cannot influence predictions.

### E06 — Saved visual predictions lose 767 feature dimensions

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed.** [Visual export](../src/utils/eval_utils.py#L414), [feature indexing](../src/utils/eval_utils.py#L520).

The code sets `N = len(DYNAMIC_VARS)` (one), then uses that modality index on the final axis of `[B,T,768]` visual arrays. Exported `vision-clip_data.npy` therefore contains only feature zero. Whole-vector cosine is calculated earlier, so this specifically corrupts the saved prediction artifact. `r2.npy` stores a dictionary of cosine scores rather than R², and one neural completion check looks for `cosine.npy` instead of the emitted `r2.npy`.

**Action:** keep modality and feature axes distinct, preserve full arrays and IDs, name metrics/files by their contents, and verify saved shapes and completion conditions.

### E07 — Reported PSTH is no longer task-conditioned

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Confirmed semantic mismatch.** [Dummy behavior tensor](../src/utils/eval_utils.py#L298); [PSTH calculation](../src/utils/eval_utils.py#L807).

Task variables are replaced with an all-zero dummy variable. Every trial belongs to one condition, so “task” PSTH/subtraction becomes a global trial average. This remains a computable score, but it should not be presented as an analysis conditioned on stimulus side, contrast, choice, or reward. Event markers are also hardcoded or discarded instead of derived from the current 20 ms grid.

**Action:** label the current calculation accurately, or preserve the intended trial conditions and event times in the dataset and evaluate them explicitly.

### E08 — Model selection combines unlike metrics without baseline checks

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 · Research risk.** [Metric aggregation](../src/trainer/base.py#L489); [historical interpretation](../AGENT.md).

The selection metric averages visual cosine similarity with neural BPS. These scores have different scales and meanings, so an improvement in one can hide deterioration in the other. There is no implemented mean-embedding, shuffled-pairing, or explicit visual/neural-only baseline evaluation in the reviewed active path. High cosine between similar grating embeddings alone does not establish useful neural decoding.

**Action:** report per-direction metrics, valid sample counts, uncertainty, and specified baselines. Define checkpoint selection deliberately. Treat current historical numbers as unreproduced until D01–D03 and E01–E04 are resolved.

## 5. Optional LFP path

### L01 — Multiple-probe LFP handling repeatedly reads probe zero

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 for LFP · Confirmed.** [prepare_lfp](../src/utils/preprocess_lfp.py#L93).

The loop over probes always selects `pids[0], probes[0]`; the spike sorting loader used for clock conversion is also constructed only for probe zero. Appending `lfp_per_trial` occurs outside the probe loop, so the result does not merge all probes as documented. `if mask:` is additionally ambiguous for a multi-element NumPy/Pandas mask.

**Action:** process each probe with its own clock conversion and append its result inside the loop; validate channel ordering and masks with a two-probe fixture.

### L02 — LFP feature windows do not match the documented 20 ms grid

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 for LFP · Confirmed.** [featurize_lfp](../src/utils/preprocess_lfp.py#L155); [caller](../src/prepare_data.py#L127).

At 2,500 Hz, a 743-sample window is 297.2 ms and a 43-sample step is 17.2 ms, despite the 20 ms comment. A two-second, 5,000-sample signal happens to yield 100 windows, but their centers do not match the neural bins. `bin_size` is documented as milliseconds, supplied as the number of neural bins, and only used in a divisibility assertion. `lf` also iterates over global `BANDS` rather than the supplied band's keys.

**Action:** define units and window timestamps explicitly, align centers to the shared grid, and test both output shape and physical timing.

### L03 — LFP normalization fits across held-out trials and ignores supplied statistics

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1 for LFP · Confirmed.** [standardize_lfp_data](../src/utils/ibl_data_utils.py#L521); [alignment before split](../src/prepare_data.py#L143).

Normalization is performed before train/validation/test splitting. The helper recomputes means and standard deviations even when they are supplied, and estimates one value over trials and channels for each time step. It cannot currently apply a frozen training-set transform to held-out data.

**Action:** choose the intended normalization axes, fit on training data, persist statistics, and apply them unchanged elsewhere. Check that inputs are not unintentionally modified through array views.

### L04 — Preparing LFP does not make it available to the active model

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed.** [Preparation flag](../src/prepare_data.py#L32); [raw loader output](../src/loader/base.py#L519); [active modalities](../src/train.py#L28).

The `--use_lfp` flag activates an optional expensive preprocessing path through inverted `store_false` logic, but the active cache/model pathway only carries spikes and vision. LFP stored in the aligned dataset is not propagated by `_preprocess_ibl_data`. Enabling the flag therefore does not establish LFP training support.

**Action:** declare the branch preparation-only or connect a complete, validated LFP schema and model path. Keep experimental support separate from the working two-modality baseline.

## 6. Pipeline, packaging, operational and security risks

### P01 ? Validation policy

**Status (2026-09-19): SUPERSEDED BY REPOSITORY POLICY** - The repository prohibits writing tests and adding CI/CD automation; see [AGENTS.md](../AGENTS.md). Validation uses code inspection, existing local checks, and manual verification.

Existing [environment checks](../script/check_environment.py), [launcher checks](../script/check_launchers.py), and [validation notes](environment-validation.md) remain available. Their passes do not establish production trainer, checkpoint, or real-session correctness.

**Action:** use the existing local checks and manually verify relevant behavior without creating tests or automated pipelines.

### P02 — Package installation is incomplete without the checkout workflow

**Status (2026-09-16): FIXED** - Resolved by the documented checkout-only installation decision; see the existing follow-up below.

**P2 · Resolved by checkout-only decision (2026-09-16).** Historical finding: `src/setup.py`; [requirements](../requirements.txt); [import-time EID reads](../src/multi_modal/encoder_embeddings.py#L18).

At the reviewed revision, package metadata declared only Torch, NumPy, tqdm, and timm, while source imported many additional dependencies. `find_packages()` did not package top-level entry scripts, YAML files, or session lists through explicit package-data/entry-point definitions. Runtime imports also assumed session lists existed beside the checkout. The documented editable installation worked around these limitations; a standalone wheel/install was not equivalent.

**Resolution:** ViNED is a checkout-only application. Removed `src/setup.py` and `src/pyproject.toml`, including the editable-install step. The [README](../README.md#requirements-and-installation) now lists this application's prerequisites, dependencies and Windows/Linux installation commands. Core requirements cover the spike/vision workflow; the inherited LFP branch has an optional requirements file. Unused timm/torchvision requirements were removed, while Seaborn remains an IBL transitive dependency. Entry scripts use the checkout's `src/`; Bash workers receive it through `PYTHONPATH`. The source, YAML files and session lists intentionally remain together. No NEDS checkout or installed distribution is required. See [checkout validation](environment-validation.md#checkout-only-validation-2026-09-16) for checks and limitations.

### P03 — CLI/configuration contracts are inconsistent

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed.** [Train CLI](../src/train.py#L421); [evaluation CLI](../src/eval.py#L23); [configuration helpers](../src/utils/config_utils.py#L20); [dataset entry point](../src/create_dataset.py#L24).

Training defaults `--config_dir` to `configs` although the root contains `src/configs`. Training/evaluation expose `--modality` but reconstruct a hardcoded modality list; evaluation exposes `--seed` but uses configuration seed/42. Some cache CLI settings are calculated but never affect the stored representation. Includes are resolved from the working directory, and YAML schema/value validation is absent. Several entry points parse arguments and execute work at import time.

**Action:** resolve arguments into one validated configuration, remove ineffective options, resolve paths deliberately, and use import-safe entry-point functions. Test documented commands from their supported working directories.

### P04 — Cluster wrappers still need operational validation

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed gaps; runtime impact unverified.** [Ray launcher](../script/train.sh#L57); [distributed launcher](../script/train_multi_gpu.sh#L42); [evaluation wrapper](../script/eval.sh#L1).

Slurm account/partition values remain site-specific. Ray startup relies on fixed sleeps and background processes, without readiness checks or cleanup traps. Positional training/evaluation arguments are not validated consistently, and several shell expansions are unquoted. `#SBACTH --array=0` is misspelled and ignored. The mocked launcher checks do not prove real Ray/DDP scheduling, connectivity, or failure propagation.

**Action:** validate argument counts/types, use a site configuration, check service readiness, propagate worker failures, and clean up background services. Exercise one short real allocation before a search or long distributed run.

### P05 ? Dependency and model reproducibility

**Status (2026-09-19): IMPLEMENTED WITH MANUAL DEPENDENCY REVIEW** - CLIP model and processor load from one immutable Hub snapshot. Feature archives record its SHA, artifact hashes, processor configuration, sampling/normalization, package versions, and source/replay hashes. Dependency installation and upgrades use the documented platform constraints and local checks. Automated dependency management is excluded by repository policy.

**P2 ? Reproducibility controls.** [Requirements](../requirements.txt), [Windows constraints](../constraints-windows-py310.txt), [Linux constraints](../constraints-linux-py310.txt), [freeze helper](../script/freeze_environment.py), [provenance documentation](data-provenance.md).

The freeze helper excludes both legacy local distribution names, `neds` and `vined`. Installing only `requirements.txt` leaves some versions unconstrained; use the matching constraints file. Real-model extraction remains unverified, and version pinning does not establish a vulnerability-free environment or fully hash-locked installation.

**Action:** review intentional upgrades manually, keep constraints consistent, run the existing local environment checks, and supply the recorded CLIP commit SHA when reproducing extraction.

### S01 — Dataset and checkpoint deserialization assumes trusted artifacts

**Status (2026-09-16): PARTIALLY FIXED** - New visual feature archives use numeric arrays without pickle. Cached samples and checkpoints still use pickle-backed formats.

**P1 when artifacts are untrusted · Confirmed exposure, no exploit attempted.** [Cached samples](../src/loader/base.py#L603); [visual archive](../src/utils/ibl_data_utils.py#L622); [checkpoint loading](../src/utils/eval_utils.py#L142); [Tune parameters](../src/eval.py#L149).

Object-array archives require `allow_pickle=True`; checkpoints and Tune parameters also use pickle-backed loading. A supplied malicious artifact can execute code during loading. This is an artifact trust boundary, not evidence of a remotely exposed application or an existing compromise. [NumPy's version-matched documentation](https://numpy.org/doc/1.26/reference/generated/numpy.load.html) and [PyTorch's loading documentation](https://docs.pytorch.org/docs/stable/generated/torch.load) explain the underlying risk.

**Action:** use numeric arrays plus structured metadata where possible; verify artifact provenance and restrict writers to cache/checkpoint locations. Evaluate safer loading formats/options against the pinned runtime before migration.

### S02 — Output ownership, interruption safety, and external logging need explicit policy

**Status (2026-09-16): PARTIALLY FIXED** - Feature extraction validates a temporary archive before replacement; replay refuses existing artifacts. General locking, interruption-safe publication, and logging policy remain open.

**P2 · Confirmed mechanisms; impact depends on execution.** [Direct cache writes](../src/create_dataset.py#L145); [checkpoint writes](../src/trainer/base.py#L574); [W&B defaults](../src/configs/multi_modal/trainer_mm.yaml#L9); [evaluation flags](../script/eval.sh#L52).

Caches, features, results, and checkpoints write directly to final paths. Concurrent runs with the same naming scheme or interrupted writes can replace or leave incomplete artifacts; no completion manifest or locking is used. Training enables W&B in YAML, and the evaluation wrapper always requests W&B plus overwrite. Configuration, metrics, and figures can therefore be sent to the configured account when those workflows run.

**Action:** make run/output identity and logging destination explicit, support an offline workflow, and use temporary writes followed by publication after validation. The public IBL credential in the source is documented public-service access, not classified here as a leaked private secret.

## 7. Coding defects and dead-code inventory

### C01 — Legacy dataset-loading branches have broken return/path contracts

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed; inactive in the reviewed predefined path.** [load_ibl_dataset](../src/utils/dataset_utils.py#L215).

The default `session_based` branch and `random_split` branch reach a return using `val_dataset` and `meta_data` that those branches do not assign. Other branches return two or three items instead of the four expected by active callers. Some construct paths by prefixing `cache_dir` and appending `_aligned` to values that are already paths/suffixed names. Unsorted directory enumeration also makes session truncation dependent on filesystem order. During predefined loading, split datasets are appended before metadata validation; an exception can leave partially accepted data while skipping its metadata.

**Action:** narrow the public contract to supported modes or repair each mode; normalize paths, select sorted explicit EIDs, and append a session only after validating its full data and metadata.

### C02 — General metric helpers contain latent failures

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed.** [Metric implementation](../src/utils/utils.py#L116), [metrics_list](../src/utils/utils.py#L195); [duplicate implementation](../src/utils/eval_utils.py#L666).

`utils.utils.neg_log_likelihood` references undefined `logger` on the zero-rate warning branch. `metrics_list` converts inputs to NumPy for BPS, then its default combined metric list continues into tensor-only operations. The `rsquared` loop reuses its accumulator for individual results. Both BPS implementations divide by total spike count without an explicit zero-spike policy. These do not imply every current BPS-only call fails.

**Action:** consolidate metric definitions, preserve input types, define empty/zero/constant-signal behavior, and check each metric alone and in combinations against small analytic examples.

### C03 — Small utility/configuration branches fail on valid edge cases

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2/P3 · Confirmed conditional defects.** [Config conversion](../src/utils/config_utils.py#L98); [HDF5 helper](../src/utils/dataset_utils.py#L122); [sampler weights](../src/loader/base.py#L255).

`convert_to_dtype` indexes an empty string for `[]`/empty input, and `ParseKwargs` splits unrestricted `=` characters. `get_data_from_h5` references undefined `h5py` and `self`; its `if "train_truth" and "valid_truth" in h5dict` checks only the second key. Weight calculation indexes weights using raw class labels, which assumes contiguous zero-based labels. These helpers require explicit supported-input contracts if retained.

**Action:** remove unused helpers after caller review or repair them with narrowly targeted checks before exposing them as supported APIs.

### Dead-code and obsolete-code candidates

The inventory combines AST name/attribute-reference inspection with repository searches. “No caller” means no active caller found in tracked Python code; it does not establish that notebooks or external users never import the symbol. Referenced helper chains belonging solely to an inactive subsystem are listed together. LFP is optional and reachable, so it is **not** classified as dead code.

| Location | Evidence/status | Suggested disposition |
| --- | --- | --- |
| [CrossAttention](../src/multi_modal/mm_utils.py#L193) | **Candidate, broken if used:** no active caller. Keys/values come from `x`, ignoring `context`; changing only context produced identical output in a synthetic check. A context-length mask can also disagree with key length. | Remove/isolate, or repair and verify context dependence before adoption. This is not an established cause of active-model scores. |
| [FactorsProjection](../src/multi_modal/mm_utils.py#L90) | **Candidate:** no caller found. | Remove or document its intended experiment. |
| [create_context_mask](../src/multi_modal/mm_utils.py#L48) and [context configuration](../src/multi_modal/mm.py#L70) | **Inactive:** use/register/application code is commented out. | Wire up a validated attention policy or remove misleading configuration. |
| [LengthGroupedSampler, SessionSampler, WeightedSessionSampler, LengthStitchGroupedSampler](../src/loader/base.py#L188) | **Inactive subsystem:** imported but never selected by [make_loader](../src/loader/make_loader.py#L87); `weighted_sampler` is ignored. Associated grouping/weight helpers serve these classes. | Choose supported sampling behavior; remove the rest after external-caller review. |
| [calculate_weights in make_loader](../src/loader/make_loader.py#L21) | Unused duplicate of a helper in `loader/base.py`. | Consolidate if weighted sampling is retained. |
| [Wrap-padding helpers](../src/loader/base.py#L55), [_spikes_mask](../src/loader/base.py#L100), [_prepare_column_data](../src/loader/base.py#L560) | **Candidates:** no active callers found. | Remove or move into a tested legacy module. |
| Former `vision_cache` in `src/loader/base.py` | **Resolved (2026-09-16):** unused NPZ loading/handles removed. | Vision now comes from the validated dataset. |
| [Visual stitcher linear layer](../src/models/stitcher.py#L34) | **Confirmed dead parameters:** a 768-to-1536 linear layer is registered per EID, but visual forward bypasses it. Approximately 1,181,184 parameters per session. | Remove after considering checkpoint compatibility; see F03. |
| [Repeated spike linear construction](../src/models/stitcher.py#L34) | Initial layer is immediately replaced in the non-vision branch. | Construct the final layer once; this also avoids unnecessary RNG consumption. |
| [upload_dataset/download_dataset](../src/utils/dataset_utils.py#L113) | **Candidates:** no active callers; upload call in preparation is commented out. | Isolate supported Hub utilities or remove. |
| [get_data_from_h5](../src/utils/dataset_utils.py#L122), [split_both_dataset](../src/utils/dataset_utils.py#L388) | **Candidates:** unused legacy dataset paths; HDF5 helper has undefined names. | Remove/isolate, or implement a supported adapter with a consistent return contract. |
| [globalize](../src/utils/ibl_data_utils.py#L21), [create_intervals](../src/utils/ibl_data_utils.py#L149), [get_behavior_per_interval](../src/utils/ibl_data_utils.py#L262) | **Candidates:** no active callers; current visual binning has a separate implementation. | Remove or consolidate after checking inherited uses. |
| [create_behave_list](../src/utils/eval_utils.py#L739) | **Candidate:** expects removed choice/reward/block columns; no caller. | Keep only with an explicit task-event/behavior schema. |
| [viz_single_cell_unaligned](../src/utils/eval_utils.py#L1000) | **Candidate:** active evaluation rejects unaligned data; repeats the same R² calculation in a loop. | Remove/isolate or repair if unaligned evaluation is planned. |
| [Legacy plotting helpers](../src/utils/utils.py#L83), [plot_rate_and_spike / plot_avg_rate_and_spike](../src/utils/utils.py#L252) | **Candidates:** no active calls; some imports remain. | Retain only intentionally supported visualizations. |
| [Legacy result discovery and aggregation](../src/utils/utils.py#L375) | **Candidates:** old wheel/contrastive run-name conventions; no active calls to this module's `get_npy_files` or `return_spike_bps`. | Remove/isolate. The separate cache `get_npy_files` in `loader/base.py` is active. |
| [_one_hot / _std](../src/utils/utils.py#L360), [ParseKwargs](../src/utils/config_utils.py#L85) | **Candidates:** no callers found. | Remove or provide supported callers and input validation. |
| [Duplicated likelihood/BPS, R² and plotting helpers](../src/utils/utils.py#L116) versus [eval_utils](../src/utils/eval_utils.py#L666) | Some copies are active, others unused; implementations can diverge, as the missing logger shows. | Consolidate into one metric/plotting API, preserving behavior deliberately. |
| [Training test loader](../src/train.py#L196), [fine-tuning test loader](../src/finetune.py#L148) | Constructed and passed as a keyword, but `MultiModalTrainer.__init__` never stores/uses it. | Remove unnecessary construction or add a deliberate final-test stage separate from model selection. |
| [Empty static-modality branches](../src/multi_modal/mm.py#L29), [static trainer paths](../src/trainer/base.py#L520) | Inactive with `STATIC_VARS=[]`; some placeholders and nested branches cannot run. | Isolate future modality support; do not count it as implemented functionality. |
| [Unused imports/locals](../src/utils/preprocess_lfp.py#L1) across source | Pyflakes reports unused imports, unused assignments, redundant f-strings, and the undefined names described above. | Remove in a separate cleanup after functional fixes; preserve intentional re-exports. |
| [Former timm/Seaborn and torchvision declarations](../requirements.txt) | P02 follow-up: direct declarations and checker imports for unused timm/torchvision were removed. Seaborn remains an IBL transitive dependency. | Resolved in the checkout-only dependency review; see environment validation. |

## 8. Performance and optimization opportunities

These are source-supported opportunities, not measured speedup claims. Establish a correct baseline first; preserve numerical behavior unless an experiment explicitly changes it.

### F01 — Per-trial file loading serializes the input pipeline

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed mechanism; profile impact.** [Cache reads](../src/loader/base.py#L603); [DataLoader](../src/loader/make_loader.py#L87); [device transfer](../src/utils/utils.py#L37).

Every sample opens and unpickles a separate `.npy` file, every epoch. No loader workers/prefetch settings are exposed; `seed_worker` has no worker processes to seed under the current default. Pinned memory is always enabled, while device transfers do not request `non_blocking=True`.

**Opportunity:** benchmark numeric sharded storage or memory mapping, configurable workers/prefetch, and pinned asynchronous transfer on CUDA. Measure samples/second, GPU idle time, file opens, and host memory. Avoid increasing worker count blindly on Windows or shared storage.

### F02 — Data/metadata are loaded repeatedly despite cached training inputs

**Status (2026-09-16): PARTIALLY FIXED** - Unused visual-cache loading was removed. Repeated aligned-dataset loading and duplicated per-trial metadata remain.

**P2 · Confirmed.** [Aligned dataset loading](../src/utils/dataset_utils.py#L290); [training data setup](../src/train.py#L137); [double evaluation load](../src/utils/eval_utils.py#L84).

Training loads aligned datasets and materializes columns for counts/metadata, then reads cached `.npy` trials instead. Evaluation repeats dataset loading. Session metadata arrays are repeated in every trial row and then in cached samples. The unused `vision_cache` opens more files.

**Opportunity:** store compact per-session metadata plus a sample manifest, obtain counts/shapes from metadata without materializing spike columns, and use one authoritative representation per stage.

### F03 — Session-specific layers scale with the largest session and include dead weights

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed.** [StitchEncoder](../src/models/stitcher.py#L23); [StitchDecoder](../src/models/stitcher.py#L89).

Each session's spike projection uses the maximum neuron count across all sessions, and all trials are padded to that width. Each session therefore pays for layers sized for the largest recording. The unused visual linear layer adds approximately 4.51 MiB of float32 parameter storage per session, plus checkpoint/distribution overhead; unused parameters do not necessarily allocate optimizer moments because they receive no gradients.

**Opportunity:** remove dead layers and evaluate session-sized projections with explicit grouping/scattering. Compare parameter count, checkpoint size, memory, and throughput as session count grows. Preserve neuron mappings and verify mixed-session batches.

### F04 — Mask generation allocates full feature-sized tensors that are discarded

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed.** [Masker](../src/models/masker.py#L128); [mixed masking](../src/multi_modal/mm.py#L302); [model mask use](../src/multi_modal/mm.py#L351).

Masker allocates replacement/random tensors with `[B,T,N]` shape and modifies cloned inputs, but the model often consumes only the returned temporal mask's first channel. Mixed masking generates several complete masks and also runs a general mask path before overriding it. Several random tensors are created on CPU and transferred to the device.

**Opportunity:** provide a mask-only path using `[B,T]` tensors on the intended device, generate only selected schemes, and broadcast views where safe. Verify sampling semantics and reproducibility before/after changes.

### F05 — Preprocessing repeats expensive scans and per-feature interpolation

**Status (2026-09-16): PARTIALLY FIXED** - Visual interpolation is vectorized and redundant preparation-time visual loading was removed. Spike scans, repeated trial loading, worker utilization, and LFP optimizations remain open.

**P2 · Confirmed.** [Spike binning](../src/utils/ibl_data_utils.py#L180); [visual interpolation](../src/utils/ibl_data_utils.py#L440); [duplicate feature loading](../src/utils/ibl_data_utils.py#L469).

Each interval scans the full spike-time array. Visual binning constructs 768 separate interpolators per trial. Preparation loads visual features through `load_anytime_behaviors` and reloads them during binning, and loads trials twice. `n_workers` is accepted but does not parallelize these active loops. LFP segmentation additionally copies overlapping windows into a large float64 array and builds DataFrames inside nested loops.

**Opportunity:** after correcting timestamps, use sorted-time bounds for spike slicing, interpolate along an array axis, reuse loaded session data, and batch numeric LFP operations. Compare exact bin boundaries and output tolerance on fixed fixtures.

### F06 — CLIP extraction repeatedly initializes models and materializes videos

**Status (2026-09-19): FIXED (implementation); real-session benchmarking pending** - One pinned CLIP model/processor is reused across sessions. Extraction yields bounded image batches, spools numeric features/times/masks to disk, and streams them into the existing schema-v2 NPZ archive before atomic publication. The optional `--frame-source render` path reconstructs only sampled frames from current-renderer parameters and saved wheel deltas, bypassing MP4; generation supports `--no-video`. Video input remains the default and retains full decode validation. Both paths use the same sampling grid and CLIP preprocessing, but lossless-rendered and lossy-decoded pixels/features are not expected to be identical. See [F06 usage](visual-extraction.md). No measured speedup or real-session equivalence is claimed.

**P2 · Confirmed.** [Extractor lifecycle](../src/prepare_visual_stim.py#L112); [outer session loop](../src/prepare_visual_stim.py#L249); [frame buffering](../src/prepare_visual_stim.py#L34).

The CLIP model/processor are initialized for every session. All sampled images for a trial are buffered, all session features accumulate until final export, and extraction decodes every video frame while keeping only a subset. The pipeline renders 720×720 MP4s at 30 FPS and later extracts at 5 FPS. Lossy video encoding adds work and may alter pixel values before feature extraction.

**Opportunity:** reuse one encoder, stream batches and output shards, and benchmark direct frame-to-CLIP processing with optional replay videos for inspection. Preserve exact preprocessing and timestamp semantics. Validate sampling when source FPS is not divisible by requested FPS.

### F07 — Evaluation and plotting can dominate memory and runtime

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P1/P2 · Confirmed mechanism; resource impact depends on dataset.** [Evaluation batch size](../src/utils/eval_utils.py#L207); [validation accumulation](../src/trainer/base.py#L328); [plotting](../src/utils/eval_utils.py#L957).

A float32 `[10000,100,768]` tensor alone occupies about **2.86 GiB**, before spikes, clones, predictions, and transformer intermediates. Validation retains predictions/targets on the device for all sessions and loops over the loader separately for the two directions. Training accumulates tensor-valued modality losses without detaching them, retaining unnecessary graph references. Plotting creates figures without explicit closure and may perform two spectral clusterings for each plotted neuron. Training plots are computed at configured intervals even when W&B is disabled and the figures are not otherwise saved.

**Opportunity:** fix E04 before reducing evaluation batches; aggregate sufficient statistics or move detached outputs to CPU, cap diagnostic plots, close figures, and detach scalar logs. Profile peak CPU/GPU memory and plotting time separately from prediction time.

### F08 — Optional dummy workload can consume more memory than the model

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Confirmed; opt-in.** [dummy_load](../src/utils/utils.py#L19); [training flag](../src/train.py#L407).

With the CLI default size of 50,000, the dummy linear weight matrix alone has 2.5 billion float32 values, approximately **9.31 GiB**, plus inputs and outputs. It performs unrelated matrix multiplications in a thread and can cause OOM or invalidate throughput measurements. Current wrappers pass `--dummy_size` but do not pass `--dummy_load`, so supplying a size alone does not activate it.

**Opportunity:** remove it from normal research workflows, or isolate it as a bounded explicit stress utility with clear resource estimates and error propagation.

### F09 — Precision and device work should be tuned from measurements

**Status (2026-09-16): OPEN** - No resolution recorded in the current changes.

**P2 · Opportunity.** [Accelerator setup](../src/train.py#L111); [session grouping](../src/models/stitcher.py#L56); [normalization](../src/multi_modal/mm.py#L105).

There is no explicit reproducible precision policy in the run configuration. Forward paths repeatedly convert EIDs to NumPy, find unique groups, create device indices, clone tensors, and normalize visual vectors multiple times. Some choices are required by current semantics; others may be redundant.

**Opportunity:** profile a fixed correct run, then evaluate mixed precision, precomputed integer session indices, cached grouping metadata, and reduced copies/normalizations. Keep numerically sensitive likelihood and metric calculations appropriately precise. Report throughput, peak memory, loss/gradient finiteness, and prediction agreement; do not assume flash attention, compilation, or lower precision will help on every platform.

## 9. Verification performed and recommended regression checks

### Performed during this audit

| Check | Result and limit |
| --- | --- |
| Repository inventory and source inspection | Inspected tracked Python, Bash, YAML, packaging/dependency files, session lists, and relevant Markdown documentation at the stated revision. Binary research documents were not re-rendered or revalidated. |
| Python AST parse | All **33** Python files under `src/` and `script/` parsed successfully. Syntax validity does not establish functional correctness. |
| `.venv/Scripts/python.exe -B -m pyflakes src script` | Nonzero exit with unused imports/locals, undefined `h5py`/`self`/`logger`, an unused format argument, and other static warnings. No fixes applied. |
| Trial validity | Two complete synthetic trials with validity `[True, False]` retained `[True, True]`. |
| Replay boundary | Stationary left/right frames had 54,356/0 non-background pixels. No videos or datasets were written. |
| Probe merge | Three two-cluster probes resulted in four unique cluster IDs rather than six. |
| Standalone head registration | Encoding and decoding each registered 0/4 head parameter tensors; head checkpoint key count was zero. |
| Cross-attention context | Replacing context while holding `x` fixed changed output by exactly zero. This class has no active caller. |
| Singleton session embedding | Same sample alone versus duplicated in a batch produced different embedding tensors with dropout disabled. |
| Loader and dtype edge cases | Reproduced short-trial assertion failure, wrong left-padding mask, and `[256,300]` spike counts becoming `[0,44]`. |
| Session selections | Current train/test lists contain 18/2 unique EIDs and have no overlap. Subject and temporal independence were not tested. |
| Scope check | Only this Markdown document is intended to change. No research assets, production code, dependency files, or test files were changed. |

The Python executable could not start inside the restricted sandbox, so the read-only static/synthetic checks ran in the approved normal execution context with bytecode generation disabled. IBL downloads, CLIP downloads, production training, checkpoint evaluation, online logging, and Slurm/Ray jobs were not run. Existing environment validation is recorded separately in [environment-validation.md](environment-validation.md); its historical passes are not new audit results.

### Checks to add while implementing fixes

| Area | Acceptance evidence |
| --- | --- |
| Alignment and identity | Known event timestamps land in the same bins; missing filenames and invalid trials never shift pairings; original IDs survive every stage. |
| Replay fidelity | Boundary, contrast, reverse-motion, short-duration, missing-event, and fresh-directory cases have explicit expected outputs. |
| Dataset/cache | Stable split memberships; range-safe spike round trip; metadata matches cluster IDs; incompatible/stale caches are rejected. |
| Shapes and masking | Batch size one, one neuron, short/long sequences, unequal lengths, and mixed sessions behave correctly; padded values do not affect valid outputs or metrics. |
| Training | Every intended head appears in optimizer parameters, receives updates, and round-trips through checkpoints; CLI settings reach the model; actual and scheduled steps match. |
| Evaluation | Each mode evaluates its declared targets; every trial appears once; scores are invariant to evaluation batch size; full visual feature vectors are exported. |
| Resume and search | Interrupted/resumed runs preserve required state; best-trial selection is deterministic; ASHA receives intermediate results. |
| Distributed/LFP | A short two-rank run passes validation/checkpoint boundaries; two-probe LFP channels and timestamps align with the declared grid. |
| Scientific baseline | Report separate neural/visual metrics against explicit baselines, with sample counts and split provenance. |

These are future implementation acceptance criteria, not tests added by this documentation-only change.
