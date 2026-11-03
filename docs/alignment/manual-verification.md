# Alignment restoration and Scenario 11 manual verification

This guide separates source/configuration findings from directed runtime observations.
No automated test suite is used. Do not start training until the regenerated data
and setup summary have been reviewed. Run commands from the repository root using
`.venv/Scripts/python.exe` (or the equivalent activated project environment).

## Implemented contract and source review

**Verified by source/configuration:**

- `src/alignment/temporal.py::_visual` queries supplied neural bin midpoints,
  selects adjacent encoded source observations with `searchsorted`, calculates
  float64 linear interpolation, L2-normalizes, and stores float32 vectors.
  The identifier is the existing `L2-normalized linear interpolation at neural bin centers`,
  now accompanied by `resampling.policy_version=1`.
- Exact matches use identical indices and zero coefficient. Out-of-range queries,
  invalid/missing scheduled observations, incomplete outcomes, non-monotonic times,
  unsupported source timing and near-zero interpolated norms fail explicitly.
  Features selected at irregular intervals use their actual timestamps. Current
  replay schedule validation still requires onset/cadence evidence; arbitrary
  irregular replay schedules without completeness evidence are not newly supported.
- `_align_trial` uses the unchanged `_window` selection, supplied edges and raw
  count slice. It never shortens a trial to make interpolation work. Complete bins
  remain uniform, starting at onset; the terminal partial bin remains excluded.
- `src/alignment/artifacts.py::_validate` checks original held-state associations
  separately from new-version interpolation associations. Historical interpolation
  records without the added version keep their previous read contract. Source
  hashes, policy metadata and payload hashes determine immutable generation IDs;
  existing destination directories are refused.
- `src/training_dataset/samples.py` preserves the selected policy and associations,
  rejects unknown/contradictory identities, and validates new-version query/index/
  coefficient consistency. Dataset manifests retain source alignment generations.
- `src/trainer/setup.py::resolve_setup` accepts either known policy independently
  of temporal mode, requires one policy across consumed train/validation samples,
  and records interpolation policies in effective configuration. Historical
  held-state configurations retain their fields for exact resume comparison.
- `src/evaluation/setup.py::resolve_setup` requires the checkpoint's exact dataset
  generation, known sample policies, and the recorded interpolation policy when
  present. It also checks checkpoint/model context. A historical checkpoint cannot
  be evaluated on newly interpolated data merely because tensor sizes agree.

No scientific specification, architecture, neural binning, model architecture,
optimization setting or scenario definition was changed.

## Scenario semantics and call paths

| Property | Scenario 11 | Scenario 15 |
| --- | --- | --- |
| Direction / selection | encoding / visual_only | encoding / visual_only |
| Context | strict, 12 bins | full_trial, no bin limit |
| Input at target t | t-11 through t | all valid tokens in that trial |
| Eligible targets | t >= 11 with complete H=12 support | same H=12 scoring support |
| First eleven targets | no prediction or score in strict H=12 | predicted, excluded from score |
| Cross-trial context | forbidden | forbidden |

**Verified by source:** `src/run.py::build_command` forwards the scenario mode,
length, direction and region selection to training/evaluation. It does not select
an alignment policy. `src/training_dataset/context.py::temporal_context_view`
constructs per-row window indices and a separate fixed H=12 target mask.
`src/multi_modal/mm.py::_context_forward` revalidates the exact indices, gathers
`x[row, t-11:t+1]` before `forward_encoder`, and reads the last window token.
No full-trial Transformer representation precedes isolation; all layers process
only those gathered tokens. Projection and embedding operations before gathering
are per-token/session. Full-trial mode calls the original unrestricted encoder.

Training and validation both use `src/trainer/objective.py::forward_objective`
and the same context view. Evaluation's `_infer` in
`src/evaluation/predictions.py` constructs the same view and uses keyword encoding.
Both dictionary and keyword paths enter `_context_forward`; positional invocation
cannot bypass isolation. Padding is excluded through explicit masks, incomplete
strict windows are not gathered, and session heads preserve row identity.

Twelve bins represent 200 ms of exposure; their centers span 11/60 s (183.33 ms).
This is strict context at the aligned-token level. Interpolation may use a right
source observation later than the query; physical online causality is not claimed.

## Directed runtime evidence

`output/audit/context-restoration-inference.json` records a manually directed
CPU inference audit using the existing two-layer, width-128 architecture, fresh
strict-model initialization, and real cached CLIP tokens (32 bins, two trial rows).
No checkpoint was relabeled as strict and no optimization occurred.

- Target 20 used indices 9?20; eligible targets were 11?31.
- Recorded Transformer input shape was `[42,12,128]` (21 windows per row).
- Changing earlier indices 0?8 and future indices 21?31: maximum target change 0.
- Changing only the other trial row: maximum first-row prediction change 0.
- Changing allowed token 15: maximum target change 0.0004699081 log-count units.
- Dictionary versus keyword prediction difference: 0, with corruption disabled.
- Restoring `output/runs/scenario-15/006/model_best.pt` retained `full_trial`;
  the same outside-window perturbation changed target 20 by 0.0054907799.

These observations demonstrate this CPU diagnostic execution, not a trained
Scenario 11 result or an exhaustive device/environment certification. Training
and validation equivalence with stochastic corruption enabled is source-reviewed;
no optimizer/backward step or complete training experiment was run.

To repeat the diagnostic manually, build a fresh model using
`trainer.pretrained.build_model` with a copy of the selected effective config and
`temporal_context={mode: strict, bins: 12, target_support_bins: 12,
boundary_policy: complete}`. Keep its existing architecture and set `eval()`.
Use real features with positions 0?31 and boolean validity; obtain a context view
from `temporal_context_view(..., direction="encoding", mode="strict", context_bins=12)`.
Under `torch.inference_mode()`, compare target 20 before/after rolling feature
components by 23 only outside indices 9?20, only at index 15, and only in another
batch row. Inspect `forward_encoder` inputs to confirm length 12 and retained
coordinates. Repeat dictionary and keyword calls with identical corruption masks.
Restore Scenario 15 with `model_from_checkpoint` and a full-trial view separately;
never change its checkpoint context identity to run the strict diagnostic.

## Regeneration workflow

The selected session is `0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63`.
Its original neural generation and cached CLIP observations can be reused.
The historical dataset has 312 trials and split configuration
`within_session`, ratios `0.7 0.1 0.2`, seed `42`.

```powershell
.venv/Scripts/python.exe src/prepare_alignment.py --eid 0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63 --neural-generation output/neural/0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63/2e491dc413d9785fa8d687c504a41e4093b286ae0e8a86ad4c88e4a96e1c5ae7 --visual-features output/visual_features/0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63_visual_clip.npz --output-dir output/alignment --workers 4
```

This records unusable trials explicitly; `--strict-trials` instead fails on any
unusable trial (including already excluded upstream trials). Publication refuses
an existing identical destination. Use an already verified new generation below
instead of regenerating it merely to repeat inspection. Never delete old artifacts.

Use the printed **new interpolation alignment path** explicitly:

```powershell
.venv/Scripts/python.exe src/create_dataset.py --alignment-generation output/alignment/0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63/7e635484272fdd3e209ab442db97a391b6fc9352f31101fa9e08dc2a3145b229 --split-strategy within_session --split-ratios 0.7 0.1 0.2 --split-seed 42 --output-dir output/datasets/training-dataset
```

This command selects the already published interpolation alignment generation. For other sessions,
copy their existing split configuration and padding settings, not these defaults.
Compare `(session_id,trial_id)` membership in every split before accepting the new
dataset. Ratio-based assignments can change if trial availability changes. Stop
before dataset publication/use if the required original trial population cannot
be retained; resolve missing source coverage upstream rather than silently
accepting a different population. No new upstream generation command is supplied
for that case because its missing observations must first be identified.

Inspect alignment with `alignment.load_alignment(path)` and dataset with
`training_dataset.load_dataset(path)`; both verify hashes and schema. Check
`visual_mapping_policies`, each sample's `metadata.alignment.visual_resampling`,
`resampling.policy_version`, query times, and source alignment generation identity.
Check `bin_size=1/60`, bin centers as edge midpoints, retained IDs, exclusion
reasons, raw count equality, and identical source-bin slices against historical
alignment. Compare complete neural arrays and edges, not just their totals.
Recalculate a selected value using encoded sources addressed by provenance:

```python
alpha = (query - source_times[left]) / (source_times[right] - source_times[left])
raw = (1-alpha)*left_feature.astype("float64") + alpha*right_feature.astype("float64")
expected = (raw / numpy.linalg.norm(raw)).astype("float32")
```

For exact matches take the single source and L2-normalize without division by a
zero timestamp difference. Read original features through `FeatureArtifactReader`;
encoded source indices are mapped to full schedule indices/observation IDs in the
provenance. Never treat the coefficient as a neural-bin overlap fraction.

For the next Scenario 11 setup, set only `dataset_generation` and
`expected_dataset_generation_id` in `src/configs/run.json` to the verified **new**
dataset path/ID (or set these in Scenario 11's operation configuration). Preserve
session selection and all existing optimization settings. Current run.json still
points to the historical held-state dataset; it is intentionally not switched by
this implementation. Review setup without starting training:

```powershell
.venv/Scripts/python.exe src/run.py train 11 --setup-only
```

Require `temporal_context.mode=strict`, `bins=12`, `target_support_bins=12`,
`boundary_policy=complete`, interpolated `alignment_policies`, expected generation,
visual-only selection, region targets, and unchanged train/validation memberships.
The eventual separate experiment uses `src/run.py train 11` after review; it was
not invoked here. Evaluation of a future Scenario 11 checkpoint uses its exact
new dataset and strict context (`src/run.py eval 11 --attempt NUMBER --setup-only`
checks restoration). Historical Scenario 15 evaluation must select the original
dataset path/ID and `full_trial`; do not repoint old checkpoints at interpolation.

## Scope and remaining verification

Only the currently selected session was regenerated. Coverage and eligibility for
other cached sessions remain unverified. No missing terminal representation was
invented, and no source feature/neural generation, checkpoint or historical
alignment/dataset/evaluation was modified. Unsupported input trials remain
explicit exclusions; new interpolation support loss must be reported before
accepting any changed population. No full future Scenario 11 evaluation is
possible until that experiment produces its own compatible checkpoint.


## Published data and numerical inspection

**Verified by generated output and consumer readback:** selected session retains
312 trials (IDs 1?312), with the same pre-existing exclusions 0 and 313 because
visual outcomes are incomplete. No additional interpolation support exclusion
occurred. All 312 raw neural arrays and bin edges match historical alignment.
All train/validation/test membership records match, with counts 218/31/63.

- New alignment ID: `7e635484272fdd3e209ab442db97a391b6fc9352f31101fa9e08dc2a3145b229`.
- New dataset ID: `01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c`.

Paths are `output/alignment/<session>/<alignment-id>` and
`output/datasets/training-dataset/generations/<dataset-id>`. The full evidence,
including per-trial count/grid comparisons, is in
`output/audit/alignment-context-restoration/alignment-evidence.json`.
Existing generations remain at their original paths.

A real example is trial 1, bin 0 (trial has 161 complete bins):

| Quantity | Value |
| --- | --- |
| Left source timestamp | 79.49077411516761 s |
| Query midpoint | 79.49910744850095 s |
| Right source timestamp | 79.50744078183428 s |
| Left / right encoded indices | 0 / 1 |
| Right coefficient | 0.5000000000004263 |
| First source components | 0.005085357464849949 / 0.0048994095996022224 |
| Full interpolated vector norm before normalization | 0.9999948540450713 |
| Stored normalized first component | 0.004992409143596888 |
| Maximum component difference after independent recalculation | 0 |

Recalculate the coefficient from the three timestamps, mix all 768 components,
compute the full-vector norm, divide, and cast to float32. Normalizing only the
single displayed component would be incorrect. The right source is later than
the query; this illustrates why token isolation does not imply online availability.

## Changed files

Implementation:

- `src/alignment/temporal.py`: normalized bin-center resampling and provenance.
- `src/alignment/artifacts.py`: policy recognition and new-version association checks;
  historical held-state validation remains separate.
- `src/training_dataset/samples.py`: reject unsupported/contradictory policies and
  validate versioned interpolation provenance through construction and readback.
- `src/trainer/setup.py`: accept both known representations, enforce one consumed
  policy, and record interpolation identity without breaking held-state resume config.
- `src/evaluation/setup.py`: replace held-state-only rejection with policy checks
  and optional checkpoint-policy consistency; preserve exact generation checks.

Documentation:

- `docs/alignment/interface.md`: corrected representation, normalization, source
  associations, errors and historical compatibility.
- `docs/training-dataset/interface.md`, `docs/training/interface.md`,
  `docs/evaluation/interface.md`: affected downstream compatibility contracts.
- `docs/alignment/manual-verification.md`: this guide, evidence and regeneration.

No persistent diagnostic script was added. Directed Python commands produced
JSON evidence under ignored output paths. Other working-tree modifications
already present when this task began are outside this change list.


## Setup-only execution evidence

**Verified by direct execution:** Scenario 11's controller-generated training
command, with only the dataset path and expected ID replaced in memory by the
published interpolation generation, returned exit code 0 with `--setup-only`.
The original run.json was not edited. The summary is
`output/audit/alignment-context-restoration/scenario-11-setup.json`;
its original console output is preserved as `.log`.
It records strict mode, 12 bins, complete H=12 support, interpolation policy,
region targets, and 218 train / 31 validation batches at the unchanged batch size.
The entry point prints a legacy "Starting training" log before setup dispatch;
`src/train.py` returns from setup-only before model/optimizer/run construction.
No training was started.

Historical Scenario 15 evaluation setup also returned exit code 0, restoring
`model_best.pt` from attempt 006 on its original held-state dataset with full-trial
context and 63 test samples. Its summary is
`output/audit/scenario-15-restoration-setup.json`. Reproduce with:

```powershell
.venv/Scripts/python.exe src/eval.py --checkpoint output/runs/scenario-15/006/model_best.pt --dataset-generation output/datasets/training-dataset/generations/99e803760e5f3769a8a87755bbb9ea00005b2fcf2cb46d92c662225d8f61cc05 --context-mode full_trial --device cpu --setup-only
```

New interpolation evaluation setup remains **unverified** until a compatible new
checkpoint exists. It is source-reviewed and dataset readback is demonstrated;
no historical checkpoint was repurposed to claim this runtime observation.
