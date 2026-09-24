# Streaming visual feature extraction

The F06 implementation reuses one immutable CLIP model and processor across the selected sessions. Image batches contain at most `--batch_size` sampled frames. Features, float64 session timestamps, and validity masks are spooled to temporary numeric files, then compressed into the existing schema-v2 `<eid>_visual_clip.npz` format. Original trial IDs and offsets are preserved; downstream loaders require no format changes. Temporary files are removed on ordinary failure or interruption, and the final archive is replaced only after validation.

Memory use for images and features is bounded by the batch size; sidecar time/trajectory arrays still occupy memory for one trial. Temporary disk space must accommodate uncompressed numeric features (approximately 3 KB per sampled frame) plus the final compressed archive. Archive integrity verification streams compressed members rather than loading all session features back into memory.

## Video input (default)

Use replays regenerated with the VR06 contract described below. Unversioned replay metadata/sidecars are rejected, including previously timestamped replays:

```powershell
.venv/Scripts/python.exe src/prepare_visual_stim.py --eid EID --video_dir ./ibl_task_replay --output_dir ./datasets/vis_stim --frame-source video --sample_fps 5 --batch_size 32
```

All video frames are decoded to detect incomplete outputs, but only selected frames enter image batches. The sampling rule is unchanged: select the first native frame at or after each requested sample time, anchored at onset. For example, 30 FPS sampled at 7 FPS selects frame indices 0, 5, 9, 13, 18, 22, 26 within the first second. Batch boundaries do not reset the sampling clock.

## Direct rendering without MP4

Newly generated sidecars contain wheel displacement and declared stimulus state for every source frame. Together with the manifest's effective parameters, phase, side, and contrast, these reproduce the renderer's uncompressed frame inputs. Extraction checks the complete declared rendering dependency fingerprint and renders only selected frames. Older sidecars must be regenerated.

Use a fresh replay directory and a separate feature output directory:

```powershell
$env:VINED_REPLAY_DIR = './ibl_task_replay_direct'
.venv/Scripts/python.exe src/visual_stim_gen.py --eid EID --no-video
.venv/Scripts/python.exe src/prepare_visual_stim.py --eid EID --video_dir ./ibl_task_replay_direct --output_dir ./datasets/vis_stim_direct --frame-source render --sample_fps 5 --batch_size 32
```

Omit `--no-video` when generating replays to also save inspection videos; the same new sidecars support both extraction modes. Session selection and calibrated parameter options remain available on the generator. Removing MP4 does not resolve uncertainty in the stimulus calibration.

Direct rendering uses the same RGB conversion, CLIP processor, normalization, IDs, timestamps, and masks. Its pixels bypass lossy MP4 compression, so feature vectors can differ from video-derived vectors. Keep the two feature generations separate and rebuild dependent data/caches when changing modes. Provenance records `frame_source`, renderer hash, pinned CLIP revision, and extraction elapsed seconds; console timing also includes archive publication.

## Verification and measurement

Syntax compilation and both CLI help commands passed locally. The existing `script/check_environment.py --device cpu --clip` also passed, including cached CLIP extraction, MP4/dataset round trips, project imports, and CLI checks. These checks do not exercise the complete new streaming/direct-render pipeline. No tests or CI workflows are added for this change.

Real-session performance and feature differences have not been measured. For a manual comparison, generate one session with inspection videos, extract it through both modes into separate directories with the same immutable `--clip-revision`, sampling rate, batch size, and device, and compare recorded extraction times, peak process memory, IDs/timestamps/masks, and feature cosine differences. Include generation time when assessing the benefit of `--no-video`; exclude first-time model downloads from steady-state throughput comparisons.

## Parameter recovery (VR05)

`src/recover_stimulus_parameters.py` creates a schema-v2 normalized parameter manifest from an explicitly selected ALF trial table and optional raw evidence. It preserves every original zero-based row, including rows with invalid contrasts/events. No files are downloaded. An existing output is never overwritten.

The following command recovers the locally available UCLA015 contrasts; it does **not** recover phase, calibration or raw task settings:

```powershell
.venv/Scripts/python.exe src/recover_stimulus_parameters.py --eid 0c828385-6dd6-4842-a702-c5075f5f5e81 --alf-trials "datasets/churchlandlab_ucla/Subjects/UCLA015/2022-03-31/001/alf/#2025-03-03#/_ibl_trials.table.pqt" --output tmp/ucla015-parameters.json
```

Pass the result to `visual_stim_gen.py --stimulus-parameters tmp/ucla015-parameters.json` with a fresh `VINED_REPLAY_DIR`. Missing values use the legacy renderer's explicitly recorded approximations. Adding `--require-parameters` to recovery or generation rejects this incomplete manifest. Schema-v2 strict mode also rejects explicit `project_assumption` and `synthetic` candidates. Completeness is not evidence of renderer fidelity or calibration.

### Evidence and source priority

An optional `--evidence evidence.json` describes field candidates, applicable source defaults and raw-file interpretation. Its top-level keys are `eid`, optional `session`, `defaults`, `reference_profile`, and `raw`. Unknown fields fail. The EID must match the command.

`session` supports `wheel_radius_mm`, `gain_deg_per_mm`, and `horizontal_fov_deg`. `defaults` supports `initial_azimuth_deg`, `contrast`, `spatial_frequency_cpd`, `sigma_deg`, `orientation_deg`, `phase_rad`, and signed `gain_deg_per_mm`. Each field maps to a nonempty list of candidates with exactly these keys:

```json
{
  "value": 3.1,
  "unit": "cm",
  "source_kind": "session",
  "source": "Replace with the measured rig record location, revision and hash",
  "applicability": "Replace with evidence connecting this measurement to this EID",
  "fallback_reason": null
}
```

This is an illustrative wheel-radius candidate, **not a measured value for UCLA015**. Configured candidates can represent inspected task settings without assuming a version-specific settings-file layout. Rank is `session`, `historical_source`, `task_configuration`, `publication`, `manufacturer`, `retrospective`, `project_assumption`, then `synthetic`. Lower-ranked candidates require a nonempty fallback reason. Selection uses source rank before scope; equally ranked trial evidence overrides session/default evidence. Conflicting equally ranked candidates within one scope fail. Raw/ALF contrast disagreements and initial-position/ALF-side disagreements also fail.

Supported units are the normalized suffix units: mm, deg/mm, deg, cycles/deg, rad and fraction. Explicit conversions support cm/m to mm, radians/degrees, percent to fraction, and mm/degree to degrees/mm by reciprocal. Signed nonzero gain is preserved and applied once, including trial-specific reverse contingency. Pixel sigma, squared sigma, encoder counts, unspecified phase units, or other raw conventions require separately evidenced normalization; they are not guessed. In particular, locate the logged phase/sigma in the relevant historical graph before declaring its unit and meaning.

The optional reference profile has exactly `id: "iblrig-a0a031e2"`, a nonempty `applicability_evidence` string, and boolean `stim_reverse`. It supplies the pinned task configuration's 0.1 cycles/degree, 7-degree **task sigma**, zero task angle, signed gain magnitude 4 degrees/mm, and initial position magnitude 35 degrees using observed ALF side. It supplies no phase, physical FOV or wheel calibration. The profile is an explicit evidence declaration, not an automatic assertion that v8 applies to a 6.x recording. None of VR04's session versions establishes applicability; the UCLA015 run above therefore does not select this profile. Higher-ranked session evidence wins over it.

### Raw logs, trial identity and synchronization

Add `--raw-trial-info PATH` only together with the evidence file's `raw` section. Supported input is the original headerless nine-column whitespace-separated encoderTrialInfo `.ssv`, or a headered `.csv` with those columns in order:

```text
trial_num stim_pos_init stim_contrast stim_freq stim_angle stim_gain stim_sigma stim_phase bns_ts
```

The raw section must contain exactly:

| Key | Required value |
| --- | --- |
| `units` | Mapping for all seven `stim_*` columns to supported explicit unit strings. |
| `unit_evidence` | Nonempty reference to the inspected session/task/logging source establishing those units and that these are task inputs. |
| `trial_map` | Mapping from canonical raw trial-number strings to original integer ALF row IDs. Must be one-to-one; no guessed zero/one-based shift. |
| `mapping_evidence` | Nonempty evidence describing how raw IDs were matched to ALF rows. |
| `excluded_trials` | Mapping from raw trial-number strings to exclusion reasons; use an empty object when none are excluded. |
| `clock` | Object containing `unit`, `anchors`, and nonempty `evidence` identifying matched synchronization events. |

Every raw ID must be mapped or explicitly excluded, including an unfinished final raw trial. Duplicate raw IDs, unknown IDs, duplicate ALF targets, out-of-range rows and unexplained omissions fail. ALF rows without raw records remain in the output with their original IDs. Missing numeric cells (`""`, `nan`, `NaN`) are recorded as missing; other malformed/nonfinite values fail. Observed phase is retained whenever present, including zero.

The clock unit is `s`, `ms`, `us`, or `iso8601`. ISO timestamps require explicit UTC offsets; naive datetimes and ambiguous numeric epochs are not inferred. At least two measured anchors are required, each with `raw` and numeric `alf_seconds`; both coordinates must increase strictly. Conversion uses piecewise linear interpolation between those anchors. Backward raw timestamps, nonfinite anchors, unsupported units and extrapolation fail. Raw timestamps remain in provenance beside the converted `parameter_log_time_alf_seconds`. These are **parameter-log times**, not measured stimulus onsets or display-frame times. Do not align them to stimulus onset merely because they share a trial number. ISO parsing retains microsecond precision; sub-microsecond original text remains preserved.

This importer does not recover encoder reset/coupling events or an ALF-to-raw wheel sign transform from trial-info logs. Those missing streams remain explicitly unverified in `recovery.wheel_mapping`; the existing onset-based trajectory remains approximate until the timing/backend tasks address them.

### Manifest compatibility and replay provenance

Schema v1 remains readable with its declared normalized values and strict completeness rule. It is labeled as a legacy user declaration, not upgraded to verified per-field evidence. No file is migrated in place. For v2, retain the existing `source`, `conversion_notes`, `session`, `defaults`, and `trials` maps and add:

- `provenance`: parallel `session`, `defaults`, and `trials` maps, with exactly one evidence record per supplied numeric field. Each records `source_kind`, `source`, `original_value`, `original_unit`, `normalized_unit`, `conversion`, `fallback_reason`, and `applicability`.
- `recovery`: ALF/raw/evidence file hashes and paths, row counts, invalid contrast IDs, explicit raw trial mapping/exclusions and clock anchors, original raw values/timestamps, converted log times and unresolved mappings.

The recovery command also binds row IDs to the ALF table's ordered contents, column names and dtypes with a fingerprint. Generation rejects a different table before rendering; select the same revision or recover again. The pandas hashing version is recorded. A change in pandas hashing/dtype representation may require recovery again even when the underlying measurements are equivalent. Hand-authored v1 manifests have no such binding.

Replay records add each parameter's effective value and evidence. Synthetic phase records the phase seed, EID, original row, hash input/digest, NumPy version and random algorithm; supplied phase is never replaced by a random draw. Missing radius, gain, FOV, azimuth, frequency, orientation and sigma receive separate fallback reasons. Invalid trial records retain their resolved parameters and evidence without shifting later IDs.

Normalized v2 phase, angle and sigma describe **task inputs**. The current legacy renderer still uses a conventional positive sine phase, Gaussian pixel sigma and rotating carrier. Those differ from the pinned Bonsai shader phase, aperture/blending and fixed-angle paths traced by VR03. Replay metadata records this backend interpretation separately. This change does not implement VR09 or establish source fidelity. Regenerate sidecars for direct extraction because the renderer source hash changes.

## Replay contract (VR06)

[The shared contract module](../src/utils/replay_contract.py) defines replay schema **1**, independent of parameter manifest versions 1/2 and feature archive schema 2. Producers and consumers validate it. Newly generated `replay_metadata.json` contains `replay_contract`; each trial retains its original `trial_id`, resolved numeric parameters, per-field provenance, validity and, for an invalid trial, a reason. Valid trials declare their half-open session interval, source frame count and whether video was written. Invalid trials have no fabricated sidecar. A fingerprint binds each trial record to its sidecar and the contract.

### Backend, dependencies and output space

The contract includes:

| Field | Meaning |
| --- | --- |
| `schema_version` | Replay contract version; currently 1. Unknown versions fail. |
| `backend` | String `id`, positive integer `version`, and explicit boolean capabilities `stimulus_frames`, `scene_frames`, `direct_frames`, `capture_storage`. |
| `dependencies` | Rendering source-file SHA-256 hashes, Python/platform identity, NumPy/OpenCV versions, installed package RECORD hashes, and OpenCV build-information hash. |
| `output_space` | `stimulus` for the display image or `scene` for a projected scene image. They are separate artifact generations, never implicit alternatives. |
| `scene_profile` | Null for stimulus-only output; otherwise `{id, schema_version, geometry, fingerprint}`. The profile hash covers its ID, version and complete geometry object. |
| `capture_storage` | Null or `{format, index_path, index_sha256}` for a lossless captured-frame index. Formats reserved by this version are `png_rgb8` and `npy_rgb8`. Index paths are relative POSIX paths within the artifact root. |
| `source_clock` | `session_seconds`; source time arrays are float64. |
| `source_time_kind` | `recorded_display` or `reconstructed`, with a nonempty `timing_evidence` declaration. |
| `video_mapping` | `{policy, fps}`; supported contract policies are `identity` and `hold_previous`. |
| `fingerprint` | SHA-256 of canonical JSON for all other contract fields, including the entire scene profile. |

The legacy dependency registry covers the generator, direct-render/extraction adapter, parameter resolver, replay contract, provenance helpers, paths and session helpers. Installed package RECORD hashes identify the package artifact manifests; versions and OpenCV build information identify the runtime. This is dependency provenance, not a self-contained runtime archive. Add a new renderer's complete workflow/shader/helper dependency set when implementing that backend. Parameter evidence files are independently hashed by VR05; their effective values are bound into trial records. Upstream IBL sources are not falsely labeled as executed dependencies of the legacy renderer.

The active backend is `legacy_python_gabor`, version 1: stimulus frames and direct rendering are supported; scene frames and stored original captures are not. Its scene profile and capture storage remain null. No scene geometry is invented here: VR11 must define and validate concrete geometry before a scene backend can emit images. A future lossless capture index must identify each frame by `(eid, original trial_id, source_frame_index, output_space)` and retain its relative path, SHA-256, RGB uint8 shape and validity/reason. Implementing capture/index loading remains VR08/VR13 work; an MP4 is never original lossless capture storage.

### Frame and time arrays

NPZ files contain numeric and fixed-width Unicode arrays only, opened with `allow_pickle=False`. Scalar `schema_version`, `eid`, `trial_id`, `contract_fingerprint` and `record_fingerprint` bind the sidecar to its manifest. Per-source-frame fields are:

| Array | Type / interpretation |
| --- | --- |
| `source_frame_index` | int64 `[N]`, consecutive indices starting at zero within the original trial. |
| `frame_times` | float64 `[N]`, strictly increasing finite session seconds; retains the old name but explicitly means **source time**, classified by the contract. |
| `relative_times` | float64 `[N]`, `frame_times - stim_on`, not video playback time. |
| `valid` / `invalid_reason` | bool `[N]` and Unicode `[N]`; valid frames have an empty reason, invalid frames require a reason. Invalid frames retain their source index. |
| `stimulus_state` | uint8 `[N]`, state codes listed below. |
| `stimulus_azimuth_deg` | float64 `[N]`, horizontal source-coordinate position. |
| `stimulus_contrast` | float64 `[N]`, fractional contrast. Zero is a valid blank stimulus. |
| `stimulus_phase_rad` | float64 `[N]`, normalized task phase; backend-specific shader interpretation remains separate. |
| `wheel_delta` | Legacy adapter extension, float64 `[N]`; validated against that adapter's declared position/state. |

Other resolved parameters remain in the trial record. Valid-frame stimulus fields must be finite. Invalid-frame values may be unavailable; their masks/reasons carry that distinction. Source timestamps still need to be known to place an invalid frame in this sequence; missing timing evidence must not be fabricated. The legacy adapter emits only valid frames for valid trials and excludes the stimulus-offset boundary.

State codes are 0 hidden, 1 stationary-visible, 2 closed-loop, 3 freeze-in-place, 4 freeze-at-center and 5 terminated. The current adapter uses separate codes **240 legacy-onset-coupled** and **241 legacy-response-hold**. These describe its existing approximate trajectory and do not assert that actual IBL control events have been recovered. VR07 will implement the source state machine and its temporal evidence; VR06 does not change the current 30 Hz approximation into a 60 Hz reconstruction.

`source_time_kind=recorded_display` requires evidence for individual synchronized display frames. Stimulus onset/offset markers alone do not establish that clock. The active legacy backend always declares `reconstructed`; no recorded display times are invented.

### Encoded video mapping

Two additional arrays describe **encoded frames**, whose count M may differ from N:

- `video_pts`: float64 `[M]`, presentation seconds relative to the video start, exactly `arange(M)/fps` for this constant-FPS contract.
- `video_source_index`: int64 `[M]`, the source index used for each encoded frame. It is bounded and nondecreasing; repetitions mean held frames and skipped indices mean source frames omitted by resampling. Both arrays are empty when no video was written.

For `identity`, M=N, the mapping is `0..N-1`, and `stim_on + video_pts` matches source times. This is the only mapping emitted/consumed by the current legacy adapter.

For `hold_previous`, query session time is `stim_on + video_pts[k]`; its mapped index must be `searchsorted(frame_times, query, side="right") - 1`. No query before the first source frame or at/after `stim_off` is allowed. The final source state may be held until offset. Its validity follows the mapped source frame. Source arrays and source timestamps remain intact; presentation times never replace them. The contract validator supports this mapping, but generating/resampling videos and extracting features with repeated mapped source frames require the later publication/extraction adapters (VR13/VR14). The current extractor rejects it explicitly instead of stretching irregular times onto a uniform grid.

### Compatibility and rejection

| Artifact/input | Behavior |
| --- | --- |
| Parameter manifest v1 or v2 | Retains VR05 loading, provenance and strict-mode behavior. |
| Unversioned replay metadata/NPZ | Rejected for both video and direct extraction; regenerate into a fresh root. No in-place upgrade or guessed state/timing. |
| Contract/record/sidecar identity mismatch | Rejected, including duplicate invalid-trial IDs and mismatched state/trajectory arrays. |
| Changed local rendering dependencies/runtime | Direct rendering rejected; regenerate. Already encoded video can be decoded with its recorded provenance without matching current rendering code. |
| Capture-only backend requested in direct mode | Rejected; no switch to the legacy renderer. |
| Unknown backend, scene output or nonidentity mapping | Current extractor rejects before loading CLIP; implement the corresponding adapter first. |
| Feature archives | Schema 2 and normalized 768-wide embeddings unchanged; provenance adds contract identity, backend, output space, scene reference and time classification. |

Use the existing generation/extraction commands above with fresh output roots to regenerate metadata and sidecars together. The generator validates its completed manifest before publication. Consumers validate contract/record identities before model loading and sidecars before consuming a trial's frames. Rendering fidelity, original capture validation and complete mouse-scene/CLIP integration remain later tasks.
