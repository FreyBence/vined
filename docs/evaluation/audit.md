# Evaluation audit

## Summary

The NEDS-derived evaluator contains reusable Poisson/BPS and trial/PSTH R² calculations, but does not currently satisfy the final visual-to-neural evaluation contract. The entry point selects the wrong prediction direction, cannot restore current identity-bearing checkpoints through its filtered state dictionary, and does not consume current training artifacts through their supported boundary. Inference coverage, validity filtering, session grouping, metric failure semantics, and artifact provenance also require correction.

This is a source-inspection audit against `spec.md`, `../dependencies.md`, and the model, training, and training-dataset interfaces. No evaluation architecture or interface document exists. End-to-end checkpoint inference was not executed; findings below distinguish source-established behavior from runtime reproduction.

## Findings

### A01 — Encoding and decoding select the opposite evaluation target

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Runtime

**Location:**  
`src/eval.py`: modality selection and both evaluation-dispatch blocks.

**Finding:**  
Construction correctly defines encoding as visual input and spike output, but dispatch enables vision evaluation for `encoding` and spike evaluation for `decoding`. Encoding then accesses `model.encoder_embeddings["spike"]`, which is absent from its visual-only encoder mapping. Decoding requests spike outputs despite having visual outputs. The default `mm` mode conceals this direction mismatch by exposing both modalities.

**Expected:**  
Specification §§1, 15, 31, 51: encoding evaluates neural predictions; optional decoding evaluates visual predictions using its actual checkpoint direction.

**Suggested disposition:** Refactor

---

### A02 — Checkpoint loading discards identity and trained session parameters

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Interface

**Location:**  
`src/utils/eval_utils.py`: `load_model_data_local`, filtered checkpoint restoration.

**Finding:**  
The loader reconstructs a model from caller/YAML settings and current dataset populations, skips every parameter whose name contains `stitcher_dict`, `project_dict`, or `stitch_decoder_dict`, and skips shape mismatches. It copies the remaining tensors into a plain dictionary and loads with `strict=False`. The current model contract requires identity in state-dictionary `_metadata`; a plain dictionary loses that identity and is rejected. Even without that guard, the skipped trained visual projections and neural heads would leave newly initialized session parameters in the evaluated model. Session-row and ordered-unit compatibility are not established.

**Expected:**  
Specification §§8–9, 48: restore the selected compatible model completely and reject scientific incompatibility. Training exposes `trainer.pretrained.model_from_checkpoint(path)` for strict dataset-independent restoration.

**Suggested disposition:** Replace

---

### A03 — Evaluation cannot explicitly select current checkpoint and dataset artifacts

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:**  
`src/eval.py`: argument parser, `pretrain_path`, checkpoint/configuration selection; `src/utils/eval_utils.py`: both `load_ibl_dataset` calls.

**Finding:**  
The CLI derives an old `results/<log_name>/model_best_avg.pt` location instead of accepting a selected checkpoint from training's `runs/<attempt_id>/` artifacts. It always chooses the average alias, so a supported `final` selection producing `model_last.pt` is inaccessible. Model configuration comes from fresh size-based YAML defaults or partial Ray parameters rather than the checkpoint's effective configuration. Dataset loading uses `config.dirs.dataset_cache_dir` twice; `--data_path` only reaches the loader's legacy cache option, and `dataset_path` is unused. There is no explicit dataset-generation identity selection or checkpoint/dataset compatibility comparison. Multi-session CLI requests omit `num_sessions` from the helper arguments, which defaults to one; its second dataset load additionally hard-codes one session.

**Expected:**  
Specification §§4–10, 45–46: use a fixed selected checkpoint and prepared test generation, retain selection/run identity, and verify populations, representation, and split strategy through declared upstream interfaces.

**Suggested disposition:** Wrap / centralize

---

### A04 — Only the final dataloader batch is inferred and scored

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/utils/eval_utils.py`: `co_smoothing_eval`, neural and visual inference loops.

**Finding:**  
Both loops replace `mod_dict` for each batch, but call `outputs = model(mod_dict)` after the batch loop. Predictions, targets, metrics, and saved arrays therefore cover only the last batch. The helper's fixed batch size of 10000 can hide this defect for small datasets. No count or identity coverage check exposes omitted test samples.

**Expected:**  
Specification §§3, 17, 21–22: infer every selected test sample and preserve complete coverage across batch sizes.

**Suggested disposition:** Refactor

---

### A05 — Neural metrics include invalid temporal observations

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/utils/eval_utils.py`: neural branch of `co_smoothing_eval`, `bits_per_spike`, `plot_psth`.

**Finding:**  
Neural scoring slices all configured held-out time indices and the first `N` channels, then passes those dense arrays to BPS, firing-rate selection, and R². Neither temporal validity nor per-sample neuron validity filters metric cells. Attention masking does not remove cells from these metric arrays. Invalid model outputs use zero fill; exponentiating them creates apparent expected counts of one at padded positions. PSTH averaging likewise includes padding instead of averaging real observations at each time coordinate.

**Expected:**  
Specification §§12–16, 24: score only the conjunction of dataset temporal/neuron validity, preserve valid zero counts, and use identical valid observations for predicted and observed PSTHs.

**Suggested disposition:** Refactor

---

### A06 — Population metadata and aggregation are taken from the first sample

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/utils/eval_utils.py`: `co_smoothing_eval` initial `N`, UUID/region selection, metric lists and returned means.

**Finding:**  
The evaluator obtains neuron count from the first loader sample and unit UUIDs/regions from dataset row zero. It uses that population for all subsequent targets and predictions, pooling channel index across samples without session grouping. Mixed populations can be truncated or include padded channels, and equal indices from different sessions are treated as one neuron. Outputs contain no session-level trial/neuron counts or explicit session-versus-neuron weighting. An unavailable empty test loader also falls through to an unbound `batch` rather than a clear input error.

**Expected:**  
Specification §§9–10, 14, 17, 20–21, 33–34: preserve scoped ordered unit identities, group results by session, and explicitly define cross-session aggregation.

**Suggested disposition:** Refactor

---

### A07 — Metric availability and baseline policies are implicit

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/utils/eval_utils.py`: firing-rate gate in `co_smoothing_eval`, `neg_log_likelihood`, `bits_per_spike`, `compute_R2_psth`, `compute_R2_main`.

**Finding:**  
R² is skipped using a hard-coded two-second trial duration and `mean_fr >= 1/fr_threshold`, irrespective of each sample's valid bin duration. Unavailable values are anonymous NaNs. BPS estimates each neuron's null mean from the scored test targets without recording that baseline policy, divides by total observed spikes without explicit zero-spike handling, and replaces infinite results with NaN. Subsequent `nanmean` silently drops unavailable neurons. R² delegates constant-target behavior to `r2_score` without an explicit undefined-result policy. Likelihood checks NaN predictions but not all infinities; numeric failures can abort unrelated metrics or disappear into aggregate omission. The active R² caller does pass `clip=False`, so negative R² clipping is not an active-path defect.

**Expected:**  
Specification §§16, 23, 26–28, 32, 35–37: use recorded temporal semantics, expose metric-specific invalidity/nonfinite values, retain valid negative performance, and record the reproducible baseline and permitted test-statistics policy.

**Suggested disposition:** Refactor

---

### A08 — Saved results cannot reconstruct scientific prediction identity

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/eval.py`: result naming and existing-file skip logic; `src/utils/eval_utils.py`: `spike_data.npy`, metric persistence, dummy PSTH grouping.

**Finding:**  
Neural persistence stores only `gt` and `pred`; BPS/R² files are positional arrays. They omit session/trial/sample identities, ordered units, physical timestamps, validity masks, checkpoint/training-run identity, dataset generation/split strategy, and evaluation definitions. The PSTH groups every trial using a zero-valued dummy variable without recording that unconditional grouping. Existing BPS/R² filenames cause reuse without validating artifact provenance, so different datasets/checkpoints can inherit old results under the same log name. No structured evaluation manifest connects predictions, scores, configuration, and source observations.

**Expected:**  
Specification §§17–18, 19, 40–42, 45–48: persist traceable predictions and machine-readable results with explicit grouping, aggregation, and reproducibility metadata; detect incompatible result reuse.

**Suggested disposition:** Refactor

---

### A09 — Optional visual artifacts discard most CLIP coordinates

**Priority:** Medium  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/utils/eval_utils.py`: `eval_vision` feature indexing and persistence.

**Finding:**  
After removing the singleton modality axis, visual arrays have shape `[B,T,768]`. `N = len(DYNAMIC_VARS)` is one, and indexing `ys[..., target_n_i[i]]` saves only coordinate zero as the visual ground truth/prediction artifact. Cosine similarity uses the full embedding, so it cannot be reconstructed from that saved artifact. `r2.npy` contains cosine values rather than R².

**Expected:**  
Specification §§18, 31, 41–42: retained optional decoding artifacts must preserve their actual representation and identify metrics correctly.

**Suggested disposition:** Refactor

---

### A10 — Obsolete evaluation helpers and ineffective controls obscure the active path

**Priority:** Medium  
**Confidence:** Confirmed  
**Category:** Dead code

**Location:**  
`src/utils/eval_utils.py`: `create_behave_list`, `viz_single_cell_unaligned`, unused held-out results; `src/eval.py`: repeated dispatch assignments and unused CLI controls.

**Finding:**  
Source/script caller search finds no caller for the legacy choice/reward/block builder or unaligned visualization helper; the active neural evaluator explicitly rejects unaligned data. `heldout_mask` results in both active branches are computed but never consumed, including `mask_result_dict`. The CLI ignores `--modality`, `--mixed_training`, and the requested `--seed` in its effective configuration/helper seed, while duplicating evaluation flag assignments and reversed direction comments. These paths and controls imply behavior the supported evaluator does not perform. No configuration, entry-point, or checkpoint callback reference to the two unused helpers was found.

**Expected:**  
Specification §§31, 44–45, 49: retain useful inherited functionality, remove confirmed inactive behavior, and make exposed evaluation controls effective and unambiguous.

**Suggested disposition:** Remove

## Conforming areas

- Both active inference branches call `model.eval()` and run their forward call under `torch.no_grad()`; evaluation does not update parameters.
- Neural predictions are exponentiated before count-based metrics, consistent with log expected count output semantics.
- The Poisson likelihood and mean-count BPS formula are reusable once validity, baseline provenance, and numerical policies are explicit.
- The active neural PSTH/trial R² caller disables clipping and calculates trial R² over flattened trial/time observations per neuron, separately from trial-averaged PSTH R².
- Optional visual cosine scoring uses explicit temporal/visual validity masks.
- Dataset loading requests the predefined test split; no test-based checkpoint ranking occurs in the inspected evaluator. Full split isolation still requires the explicit compatible artifact selection missing above.
