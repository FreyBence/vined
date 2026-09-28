# Visual replay interface

## Preparation entry point

With `src/` on the Python path:

```python
from session_data import SessionAccess
from visual_replay import resolve_reconstruction_plan

plan = resolve_reconstruction_plan(
    SessionAccess(policy="local-only"), config["eid"], config,
    stimulus_parameters=None, trial_ids=None,
    revision=None, wheel_revision=None,
    settings_collection="raw_behavior_data", settings_revision=None,
)
```

This implemented boundary resolves reconstruction inputs. It does not generate
frames, evaluate wheel coverage, or declare a completed replay generation.
`src/visual_stim_gen.py` remains the separate legacy approximate renderer; it
does not consume these plans. The direct observation stream and its artifact
publication/readback interfaces are described below.

`trial_ids` selects unique original zero-based source-table row positions and
preserves request order; omission selects all rows. Revisions follow the
session-data contract. Auxiliary trial events use the exact table revision;
wheel arrays are jointly loaded in their own revision group. Optional task
settings come from `_iblrig_taskSettings.raw.json` in the selected collection.
No raw-clock conversion is inferred.

## Required configuration

Configuration identifies one canonical EID. There are no implicit scene,
event-proxy, cadence, or wheel-sign choices. The following illustrates an
explicit approximate setup, not recovered session calibration:

```json
{
  "eid": "54238fd6-d2d0-4408-b1a9-d19d24fd29ce",
  "profile": "iblrig-a0a031e2-choice-world",
  "applicability": {
    "kind": "project_assumption",
    "evidence": "Use this pinned active-choice reference as an explicit approximation; historical equivalence is unverified."
  },
  "scene": {
    "screen_width_mm": 400,
    "screen_height_mm": 300,
    "distance_mm": 140,
    "display_size": [800, 600],
    "image_size": [800, 600],
    "horizontal_fov_deg": 120,
    "provenance": {
      "kind": "project_assumption",
      "evidence": "Schematic frontal 4:3 screen; dimensions are not measured."
    }
  },
  "observation": {
    "cadence_hz": 30,
    "domain": "visible_interval",
    "evidence": "Configured reconstruction cadence, not measured refresh."
  },
  "movement": {
    "wheel_sign": -1,
    "max_gap_seconds": 1,
    "provenance": {
      "kind": "project_assumption",
      "evidence": "Explicit ALF sign approximation; reject interpolation across gaps greater than one second."
    }
  },
  "events": {
    "closed_loop": {
      "field": "goCue_times",
      "provenance": {"kind": "project_assumption", "evidence": "Go cue approximates activation when the closed-loop event is unavailable."}
    },
    "freeze": {
      "field": "response_times",
      "provenance": {"kind": "project_assumption", "evidence": "Response approximates freeze when recorded display freeze is unavailable."}
    }
  },
  "phase_seed": 0
}
```

The sole supported behavior reference is the pinned profile above. Applicability
must be declared separately using `kind: session` with session evidence, or
`kind: project_assumption` with its rationale. No version is selected from the
session date or a newest-version default. Known incompatible task protocols or
display workflows raise an error. Profile references describe source behavior;
they are not a claim of executed historical rendering.

Scene raster sizes are `[width, height]`. The display raster must match the
physical screen aspect ratio; the camera must contain the complete screen.
The supported schematic pose is frontal: camera at the origin looking along
positive Z, Y up, screen centered at `[0, 0, distance_mm]`, facing the camera.
The neutral surround is RGB `[128, 128, 128]`. Source ViewWindow geometry is
resolved separately from this physical scene; `STIM_TRANSLATION_Z` overrides
the reference display distance when present.

Only `visible_interval` regular reconstruction scheduling is currently resolved.
No frame support interval is implied by cadence. `wheel_sign` is `-1` or `1`;
the declared displacement interpretation is sign × wheel delta in radians ×
radius in millimeters × signed gain in degrees/millimeter. The baseline is the
closed-loop event. The positive `max_gap_seconds` configures subsequent timeline
evaluation; preparation retains the unmodified wheel samples.

Event bindings accept `closedLoop_times` or `goCue_times`, and `stimFreeze_times`
or `response_times`. The latter of each pair requires `project_assumption`.
Available finite recorded closed-loop/freeze events outrank configured proxies.
Optional event files absent at the trial-table revision are recorded as missing
evidence, allowing configured proxies to apply. No other revision is substituted;
access and integrity failures still propagate.
No-go trials do not require a freeze event. Response and feedback remain separate
source events, not interchangeable timing fields.

## Parameter evidence

`stimulus_parameters` accepts a normalized recovery JSON from
`recover_stimulus_parameters.py`, with the matching `alf_table_fingerprint`.
The existing schema-v2 evidence/units and source precedence are retained.
Trial-specific evidence wins over session defaults at the same evidence tier.
Contradictory equally ranked evidence is invalid; ALF contrast and side are
cross-checked. Missing rows in a supplied manifest are unavailable, not silently
replaced by defaults.

Session task settings supply frequency, sigma, orientation, gain/reversal, and
initial positions when available. Nominal profile values fill missing parameters
with their applicability/fallback evidence; an assumed profile's nominal values
remain project assumptions. Normalized gain represents the already signed
transmitted gain. Source ViewWindow geometry determines display FOV; a supplied
incompatible linear FOV is rejected. Unrecovered phase uses a recorded SHA256
mapping of seed, canonical EID, and original row, independent of request order.

## Returned inputs and failure semantics

`ReconstructionPlan` exposes:

- `definition`: JSON-compatible input definition with EID, profile/reference
  and applicability, effective configuration/display geometry, source identities,
  trial-table fingerprint, original trial count, requested IDs, parameter-manifest
  hash, implementation hashes, missing-source reasons, wheel hashes, and trials.
- `wheel_timestamps`: read-only float64 `[wheel_samples]`, seconds in the source
  session clock.
- `wheel_positions`: read-only float64 `[wheel_samples]`, source radians.

Every requested trial has `trial_id`, `status`, and `reason`. `prepared` means
inputs resolved, not valid rendered coverage. `unavailable` means essential
evidence is absent; `invalid` means supplied evidence is malformed or contradictory.
Prepared records include resolved parameter values and per-field evidence,
distinct events and binding evidence, side, reward/error/no-go outcome, and the
requested half-open `[onset, offset)` domain. When source `intervals_0` and
`intervals_1` are available, `trial_interval` retains those session-time bounds
for pre/post-stimulus visibility interpretation. Invalid records need not contain
parameters or events that could not be resolved. A zero contrast is valid.

Configuration errors, unsupported profiles, mismatched source identity, malformed
shared manifests, and incompatible source selection raise `ValueError` or the
public `SessionAccessError`. Access/decoding/integrity errors propagate; only
explicit source-unavailability reasons permit optional-source fallback. Trial-local
input errors retain the trial outcome and do not discard other requested rows.
Unexpected errors and interruptions propagate.

## Trial state and observation scheduling

```python
from visual_replay import prepare_trial_timelines, schedule_trial

timelines = prepare_trial_timelines(plan)
timeline = timelines[0]
schedule = schedule_trial(timeline)  # Optional explicit cadence_hz override.
for session_time in schedule.times:
    state = timeline.evaluate(session_time)
    # A renderer consumes valid state and its resolved parameters, not raw wheel.
```

`prepare_trial_timelines(plan)` returns a tuple of `TrialTimeline` objects in
requested original-trial order, including invalid/unavailable input trials.
It consumes the in-memory `ReconstructionPlan`, retains EID, original trial ID,
and source table fingerprint, and snapshots trial/configuration records. Treat
these timeline records as read-only. A changed/unsupported profile, movement
policy, or mismatch with requested trial IDs raises `ValueError`.

`timeline.evaluate(session_time)` accepts one finite time in source-session
seconds and returns `StimulusState`:

| Field | Meaning |
| --- | --- |
| `eid`, `trial_id`, `session_time` | Source observation association. |
| `profile_id`, `trial_table_fingerprint` | Behavior and source-table association checked by the display renderer. |
| `status`, `reason` | `valid`, `unavailable`, or `invalid`, with a reason for unavailable/invalid state. |
| `stage` | `stationary_visible`, `closed_loop`, `freeze_in_place`, `freeze_at_center`, `hidden`, or `unknown`. |
| `visible` | Task visibility; `None` means unknown. Visible motion may still be unavailable. |
| `azimuth_deg` | Resolved task position, or `None` when hidden/unknown/unavailable. |
| `parameters` | A copy of the resolved visual parameters, with the preparation units. |
| `known_blank` | Valid hidden content or valid zero-contrast content; never an unavailable observation. |

Evaluation is independent of request order, batches, and cadence. The supported
profile shows a stationary stimulus until closed-loop activation, applies signed
wheel displacement once, freezes errors at the exact freeze-time position, and
centers rewards at freeze. No-go trials stay coupled until recorded offset.
Offset hides the stimulus. Simultaneous transitions are applied in order
onset → closed loop → freeze → offset, so hiding wins. Angular positions wrap
to `[-180, 180)`; they are not clamped to successful trajectories.

The evaluator uses linear interpolation between eligible wheel observations.
Queries outside coverage or strictly inside gaps wider than `max_gap_seconds`
are unavailable. Exact samples on gap boundaries remain usable. Equal timestamps
retain the last acquired position; `timeline.wheel_policy` records this policy
and the number removed. Samples are never reordered. Missing coupling baseline
makes wheel-dependent motion unavailable; an unavailable error freeze remains
unavailable throughout the hold. Reward centering is a known task transition
and does not invent a wheel position inside a prior gap.

Within recorded trial bounds, times before onset or at/after offset are known
blank. Without those bounds, only the offset event itself establishes hiding;
times outside the visible interval are otherwise unavailable. No blank state
is extrapolated into a different trial.

`schedule_trial(timeline, *, cadence_hz=None)` returns `ObservationSchedule`:

- `eid`, `trial_id`, `requested_domain`: original association and the half-open
  source-time domain, or `None` if preparation could not establish one.
- `times`: read-only float64 `[observations]`, anchored at onset with configured
  cadence and excluding offset. No minimum duration or video padding is added.
- `timing`: reconstructed timing classification, session clock, precision,
  cadence/provenance, and any explicit cadence override. `support_intervals`
  remains `None`; sampling does not establish frame support durations.
- `coverage`: cadence-independent availability partition of the requested domain.
  Each `CoverageInterval` has `start`, `end`, endpoint-inclusion booleans,
  `status`, and `reason`. Open spans and separate inclusive zero-length boundary
  points distinguish gaps from their observed endpoints. These describe
  reconstruction availability, not image support intervals.
- `status`, `reason`: `available`, `partial`, or `unavailable` according to the
  full requested domain, including gaps between scheduled samples. An invalid
  preparation record retains `invalid`.

Invalid/unavailable preparation records receive empty schedules and retain their
outcomes. Prepared timelines retain scheduled times even where evaluation is
unavailable; callers inspect state validity instead of dropping or holding frames.
`timeline.coverage()` exposes the same availability partition without scheduling.
Neither schedule availability nor a valid state is a completed image generation.

## Display rendering

```python
from visual_replay import DisplayRenderer

renderer = DisplayRenderer(plan)
state = timelines[0].evaluate(schedule_trial(timelines[0]).times[0])
frame = renderer.render(state)  # Requires state.status == "valid".
rgb = frame.rgb
```

Construct one renderer per resolved plan. `render(state)` consumes the evaluator's
resolved azimuth and parameters; it never reads wheel data or reapplies movement.
It returns `DisplayFrame` with:

- `rgb`: read-only `uint8[height, width, 3]`, RGB values in `[0, 255]`, top row
  first, using `configuration.scene.display_size` (`[width, height]`).
- `eid`, `trial_id`, `session_time`: unchanged source association, seconds in
  the session clock.
- `image_space`: always `"display"`.
- `render_parameters`: effective visibility, background, renderer ID, and, for
  visible states, position, contrast, frequency, phase, envelope, orientation,
  angular extent, and opacity used to produce the image.

`renderer.provenance` is JSON-compatible and records immutable source references,
the shader reference's applicability, source display geometry, rendering policy,
implementation hash, and output format. Treat it as read-only. The renderer
rejects a different EID, unrequested trial, profile, or source-table fingerprint.
Unsupported profiles, incompatible geometry, unavailable/invalid states, and
invalid visual parameters raise `ValueError`; a non-`StimulusState` raises
`TypeError`. No failure is converted into a blank image.

Valid hidden states produce uniform RGB `[128, 128, 128]`. The selected source
also makes contrast at or below `0.01` transparent. Visible stimuli use the
pinned IBL workflow and BonVision v0.9.0 primitive reference: task sigma is
squared before the radius/aperture rescale, task phase has negative carrier
sign, and effective orientation is zero even when a different task angle was
logged. The envelope affects both intensity and alpha. Source angular mapping
and quad clipping precede display raster sampling; off-screen stimuli are not
clamped back onto the screen.

This is a functional source-derived reconstruction with analytic pixel-center
sampling. The session-installed BonVision revision is unknown; intermediate
GPU quantization, mesh tessellation, and texture filtering are not reproduced.
The sync marker is omitted because its toggle parity is not established by the
resolved state. These limitations are explicit in renderer provenance. This
API produces individual display frames. Optional scene projection and the
observation stream are described below.

## Optional mouse-view projection

```python
from visual_replay import DisplayRenderer, SceneProjector

renderer = DisplayRenderer(plan)
projector = SceneProjector(plan) if project_mouse_view else None
display = renderer.render(state)
frame = projector.project(display) if projector is not None else display
```

Projection is an explicit optional stage. Omitting it keeps the original
`DisplayFrame` with `image_space="display"`. Requesting projection produces a
distinct `SceneFrame` with `image_space="mouse_view"`; errors propagate without
substituting a display image. Retain both frames when both spaces are needed.
Scene configuration is still resolved and validated during preparation even
when the projection stage is omitted.

`SceneProjector(plan)` uses the resolved physical screen dimensions, distance,
fixed frontal camera pose, horizontal camera FOV, neutral surround, and
`scene.image_size`. It maps the completed display texture onto the physical
screen, without evaluating stimulus parameters, source angular mapping, or
wheel movement again. The camera uses square pixels and must contain the
entire screen in both dimensions; changing output aspect ratio may require
a wider horizontal FOV. Only the documented frontal pose is supported.

`project(display)` returns `SceneFrame`:

- `rgb`: read-only `uint8[height, width, 3]`, using `scene.image_size`, RGB
  values `[0, 255]`, top row first. Screen pixels use bilinear texture sampling
  at output pixel centers with clamped texture edges and final integer rounding.
  Outside the screen is neutral RGB `[128, 128, 128]`.
- `eid`, `trial_id`, `session_time`: unchanged from the display frame.
- `image_space`: `"mouse_view"`.
- `display_render_parameters`: a copy of the input frame's rendering metadata.

`projector.provenance` records the resolved scene including geometry/viewpoint
evidence and assumptions, projection and sampling policy, implementation hash,
output format, and projected screen bounds. Bounds are continuous
`[left, top, right, bottom]` pixel-edge coordinates, distinct from sampled pixel
indices. Retain this shared metadata together with display-renderer provenance;
neither is a completed generation identity. Treat provenance as read-only.

Inputs must be `DisplayFrame` objects in display space, with the plan's EID,
a requested original trial ID, finite source time, and the configured display
RGB8 shape. Invalid inputs and unsupported geometry raise `ValueError`; a
different frame type raises `TypeError`. A screen covering no output pixel
centers is rejected. Already projected frames cannot be projected again.
This is an unlit schematic screen with a project-defined surround, not measured
retinal optics, illumination, or display calibration. No CLIP processing is
performed, and image dimensions remain independent of feature-model inputs.

## Direct observations and completion

```python
from visual_replay import ReplayStream, ReplayObservation, TrialOutcome

with ReplayStream(plan, image_space="mouse_view") as replay:
    definition_id = replay.definition_id
    definition = replay.definition
    for item in replay:
        if isinstance(item, ReplayObservation):
            # Process provisionally; unavailable/failed observations have no pixels.
            metadata, rgb = item.metadata, item.rgb
        elif isinstance(item, TrialOutcome):
            trial_result = item.metadata
    completion = replay.completion
# Require completion and inspect reconstruction_status before publishing results.
```

`ReplayStream(plan, *, image_space="mouse_view")` snapshots the resolved inputs
and exposes a single-use iterator. `image_space="display"` explicitly bypasses
scene projection; any other value raises `ValueError`. No MP4, OpenCV, CLIP, or
filesystem output is required. Iteration preserves requested trial order and
source-time schedule order within each trial. The stream retains input data,
one trial's schedule, and trial summaries, but never accumulates session images.
Consumers must likewise avoid retaining every image if bounded memory is needed.

`ReplayObservation` has `metadata` and `rgb`. `rgb` is an immutable, bytes-backed
RGB8 NumPy array for `status="valid"`, or `None` for `unavailable`, `invalid`,
or `failed`. Missing pixels are never replaced with held frames or gray images.
Metadata includes:

- `kind="observation"`, `definition_id`, and `observation_id` in the form
  `trial/<original ID>/observation/<schedule index>`. The identifier is unique
  within the final generation; combine it with `generation_id` across generations.
- `eid`, original `trial_id`, `trial_table_fingerprint`, `schedule_index`, and
  `session_time` in source-session seconds. Missing records retain schedule indices.
- `timing`: reconstructed cadence, onset anchor, source-clock/precision evidence,
  half-open requested-domain convention, and `support_intervals=None`.
- `image_space`, `format="RGB8"`, `row_origin="top"`, `value_range=[0, 255]`,
  and `shape=[height, width, 3]` when pixels exist. `shape` is otherwise `None`.
- `status`, `reason`, and `known_blank`. The blank flag identifies valid hidden,
  zero-contrast, or source-opacity-suppressed content, never missing coverage.
  A false blank flag does not guarantee visible contrast (e.g. an off-screen stimulus).
- `stimulus_state`: resolved evaluator fields, or `None` if evaluation failed;
  `render_parameters`: effective display parameters, or `None` without rendering.
- `rgb_sha256`: SHA256 of C-order RGB8 bytes, or `None` without pixels.

Every requested trial ends with `TrialOutcome(metadata)`, including trials with
no usable inputs. Its metadata contains the same definition/session/trial/table
association, plus:

- `requested_domain`, `timing`, `wheel_policy`, `input_coverage`, and
  `input_coverage_status`. Coverage contains the evaluator's exact availability
  intervals and boundary points; it is not rendered-image coverage or frame support.
- `scheduled_count` and `schedule_sha256` (the source times as little-endian
  float64 bytes); `emitted_count`, `image_count`, and `observation_status_counts`.
- `unattempted_count`: remaining schedule entries after a rendering failure.
  Counts requiring a schedule are `None` if scheduling itself failed.
- `observations_sha256`: ordered digest of that trial's emitted observation
  metadata, including pixel hashes and associations.
- `status` and `reason`: `complete` means all scheduled images were produced and
  the entire requested domain has available reconstruction. `partial` means
  some images exist but domain coverage is incomplete, even if no scheduled
  sample fell inside a gap. `unavailable`/`invalid` preserve input or coverage
  failures. `failed` means scheduling, evaluation, rendering, or projection
  failed; earlier valid images remain explicitly counted.

Expected trial-local `ValueError`, `TypeError`, `ArithmeticError`, and `OSError`
are recorded as failures. An observation failure emits a failed record at that
time, stops the rest of that trial, and permits subsequent trials to proceed.
Unavailable evaluator states are emitted throughout the schedule, allowing later
valid states within the trial. Shared configuration, identity, or initialization
errors raise before streaming. Modified wheel arrays that disagree with the
resolved input hashes are rejected. Unexpected exceptions and interruptions
propagate and leave completion absent.

### Definition and generation identity

`replay.definition` returns a copy containing schema version 1, resolved source
inputs and their hashes, selected image space, display/scene provenance, stream
implementation hash, runtime versions, and digest policy. `definition_id` is the
SHA256 of its canonical JSON. It identifies the reconstruction definition, not
a completed output. Replay dependency identity excludes downstream feature code.

`replay.completion` is `None` until the iterator reaches normal exhaustion after
the final trial outcome. Merely receiving the final image or trial result is not
completion. A completed record contains:

- `definition_id`, EID, source-table fingerprint, image space, and the original
  ordered `requested_trial_ids`.
- `accounting_complete=True`, every trial result under `trials`, total
  `observation_count` and `image_count`.
- `reconstruction_status`: `success` only when every trial is `complete`;
  otherwise `partial` when any valid image exists, or `failed` when none exists.
  Complete accounting therefore does not imply successful reconstruction.
- `records_sha256`, binding all emitted observations and trial outcomes in order.
- `generation_id`, binding the completion record (excluding `generation_id`
  itself), including its definition, ordered content digest, and outcomes.

Canonical JSON uses sorted keys, compact separators, ASCII escapes, UTF-8 bytes,
and rejects NaN/infinity. Each record digest consumes an unsigned eight-byte
big-endian JSON byte length followed by those bytes. Pixel hashes are included
in observation metadata; source schedules use little-endian float64 hashes.
No completion digest is issued for a truncated stream. A digest binds content
and associations; it does not certify scientific reconstruction fidelity.

Metadata is bound before delivery; treat returned metadata as read-only.
`definition` and `completion` return defensive copies. `state` is `pending`,
`running`, `completed`, or `interrupted`; `completed` describes accounting,
not reconstruction success. Use the context manager or `close()` after stopping
early. Breaking iteration leaves completion absent; closing then marks the
stream interrupted. Exhaustion is required before consumers finalize derived
artifacts. Construct a new stream to run again.

The legacy `visual_stim_gen.py`/`utils.replay_contract` path remains separate.
Existing `visual-features` consumption has not migrated to this new contract.
CLI generation, artifact publication/readback, and video adaptation are separate
from this in-memory stream.

## Artifact publication and readback

```python
from visual_replay import ReplayStream, write_replay, ReplayArtifactReader

manifest = write_replay(
    ReplayStream(plan, image_space="mouse_view"),
    "output/replay-generation", video=False,
)
with ReplayArtifactReader("output/replay-generation") as reader:
    for item in reader:
        # Same ReplayObservation / TrialOutcome types as the direct stream.
        pass
    completion = reader.completion
```

`write_replay(replay, output, *, video=False, on_trial_published=None)` consumes a fresh, pending
`ReplayStream` into a **new directory**. Existing output raises `FileExistsError`;
there is no overwrite, resume, or fallback to an older generation. It returns the
published artifact manifest after the source stream completes. Reconstruction
failures retain their trial outcomes and may yield a completed manifest whose
`reconstruction_status` is `partial` or `failed`.

An optional `on_trial_published(outcome)` callback receives a defensive copy of
each trial outcome after its directory has been published, including optional
video processing. It is called for unavailable/failed trials too. Callback
exceptions propagate and prevent generation-manifest publication; a progress
notification is not a generation completion record.

Schema version 1 writes losslessly compressed NumPy RGB8 arrays independently of video:

```text
generation/
  definition.json
  trials/<original-trial-id>/
    observations.jsonl
    outcome.json
    images/<schedule-index>.npz
    video.mp4                 # Optional, only when encoding succeeds.
    video-mapping.jsonl        # Optional, paired with video.mp4.
  manifest.json
```

`observations.jsonl` contains the direct observation metadata, in order. An
image exists only for a valid record; it is the corresponding C-order RGB8
array stored under the sole archive key `rgb` using NumPy ZIP/DEFLATE compression,
loaded with `allow_pickle=False`. Readback and optional video encoding decompress
one frame at a time in memory without writing uncompressed image files. Missing observations keep their source
timestamps and schedule indices without an image. `outcome.json` is the direct
trial outcome, including coverage, reasons, and counts. Empty/unavailable trials
still have an empty observation file and a trial outcome.

Each trial is written under `trials/.staging-<id>` and renamed to its published
trial directory only after its images and metadata are written. The generation
manifest is staged and renamed to `manifest.json` only after stream exhaustion
and complete request accounting. Storage errors and interruptions propagate;
they leave no completed manifest for the new generation. Earlier published
trials remain inspectable, and previously completed generation directories are
untouched. Staging files may remain after failure; they are never treated as
completed artifacts. Automatic recovery/cleanup is not provided.

On Windows, publication renames retry access/sharing/lock errors for up to five
seconds to tolerate transient locks after file creation or encoding. Existing
destinations are never replaced. Persistent trial/manifest publication failures
still abort the generation and preserve staging; there is no non-atomic copy
fallback. Optional video publication failures retain the video-failure semantics
described below.

The manifest contains `schema_version=1`, `kind="replay_artifacts"`,
`format="npz_rgb8"`, writer implementation hash, `definition_id`, the unchanged
stream `completion`, and ordered trial descriptors with metadata-file hashes and
video outcomes. `artifact_id` hashes the manifest excluding itself using the
canonical JSON policy above. It binds storage/video details separately from
`generation_id`, which remains identical to the same direct scientific stream.

The reader also accepts existing `format="npy_rgb8"` manifests with raw
`images/<schedule-index>.npy` arrays. The manifest selects the storage format;
missing compressed images never fall back to raw files. Existing generations
are not converted. Compression preserves pixels, shape, dtype, observation
metadata, and pixel hashes; no CLI option is needed for new compressed output.

`ReplayArtifactReader(directory)` requires a published manifest and verifies
its artifact ID, definition ID, generation ID, and requested-trial accounting.
It is a single-use iterator yielding the same records, order, image spaces,
identities, source times, and immutable RGB arrays as `ReplayStream`. It requires
neither source-session access nor a historical rendering environment. During
iteration it verifies metadata-file hashes, each decoded image's shape/type and
pixel hash, observation associations, trial counts/digests, and the full record
digest. Available videos and their mappings are also hash-checked.

Reader `definition`, `definition_id`, `state`, `close()`, context-manager behavior,
and `completion` follow the direct-stream contract. Completion remains `None`
until successful exhaustion, including when a file is missing/corrupt or reading
stops early. `artifact_manifest` returns a defensive copy of the publication's
declarations; its embedded completion is not a substitute for verified reader
completion. Invalid data raises `ValueError` or a decoding error; missing files
raise `FileNotFoundError`. Readers reject unfinished directories and never infer
timing or trial identity from video FPS or filenames alone.

### Optional video

`video=True` additionally uses OpenCV's `mp4v` encoder for trials with
`status="complete"`. Each scheduled observation produces one frame at the
declared reconstruction cadence, without repetition or dropping. The mapping
records `encoded_index`, zero-based `presentation_time=index/fps`, original
`observation_id`, and actual `session_time`. Playback time is not source time
and does not establish scientific frame-support intervals. The writer decodes
the completed MP4 to check frame count and dimensions before publication.

Trials with incomplete reconstruction are explicitly `skipped` for video;
their lossless observations and gaps remain available. A video descriptor has
status `not_requested`, `skipped`, `available`, or `failed`, with reasons where
applicable. Available descriptors include codec, FPS, frame count, mapping
policy, and video/mapping hashes. Encoding failure is recorded as `failed` and
does not discard canonical images or alter the scientific generation identity.
MP4 requires even raster dimensions; the adapter never silently crops images.
Videos are lossy inspection outputs. Canonical readback always uses the lossless
arrays, and callers requesting video must inspect video outcomes separately.

## Preparation and generation CLI

```text
.venv/Scripts/python.exe src/prepare_replay.py --config session-config.json --output output/resolved-inputs.json --access-policy local-only
```

The CLI also accepts `--stimulus-parameters`, repeated `--trial-id`, `--revision`,
`--wheel-revision`, `--settings-collection`, and `--settings-revision`. Access
defaults to `local-only`. Output must be a new file; existing files are not
overwritten. The JSON is a diagnostic export of `definition`, without wheel
arrays, and is not a persisted replay or completion manifest. The CLI reports
input-status counts and returns normally for accounted-for invalid/unavailable
trials; callers must inspect those outcomes before using prepared inputs.

Add `--generate` to produce a new artifact directory instead of diagnostic JSON:

```text
.venv/Scripts/python.exe src/prepare_replay.py --config session-config.json --output output/replay-generation --generate --image-space mouse_view --video --trial-id 0 --trial-id 1 --access-policy local-only
```

Omit `--video` for lossless-only publication. `--image-space display` bypasses
scene projection; the default is `mouse_view`. Video and image-space selection
belong to generation mode. The same source-selection and parameter options apply
to both modes. Generation prints its ID, accounting/reconstruction status, image
count, and whether requested video failed. Exit codes are `0` for successful
reconstruction, `2` for accounted partial/failed reconstruction, and `3` if any
requested video failed (taking precedence over `2`). Configuration, I/O, or
unexpected errors raise and exit nonzero; they do not publish a completion
manifest. Argument-parser errors also use exit code `2`. Existing outputs are
never overwritten in either mode.

### Saved-image previews

```bash
bash script/preview_replays.sh EID             # All complete trials
bash script/preview_replays.sh EID 12          # Original trial ID 12 only
```

The equivalent Python command is `python src/preview_replays.py EID [TRIAL_NUMBER]`.
It reads existing artifacts from `VINED_REPLAY_DIR` (default
`output/visual_replays`), looking under `<eid>` or `<run>/<eid>` and choosing
the session directory with the newest modification time. The selected source
is printed; unfinished or invalid generations are rejected without falling
back to older runs. Both compressed NPZ and legacy NPY frames are supported.

Outputs are `<output_dir>/trial_preivew/<eid>/<original-trial-id>.mp4` and a
matching `.jsonl` source-observation/time/generation mapping. The spelling
`trial_preivew` is intentional. Existing previews for processed trials are
replaced after the source generation has been fully verified. Source artifacts
are never modified. Readback verifies the entire generation, including when
only one trial is selected. Encoding uses the recorded cadence, one frame per
observation, and verifies video frame count and dimensions before publication.

Missing replay images or an absent requested trial print a terminal error and
exit with code 1. A selected incomplete trial also fails. With no trial selector,
incomplete trials are reported and skipped; complete trials are encoded and
the command returns 2 if any trials were skipped, otherwise 0. Invalid CLI
arguments return 2. No rendering, downloading, or CLIP extraction is performed.

### Bash replay-generation entry point

```bash
# All EIDs in data/eids.txt, all trials, projection on:
bash script/generate_replay.sh

# One EID, all trials, projection off:
bash script/generate_replay.sh --eid 931a70ae-90ee-448e-bedb-9d41f3eda647 --projection off
```

The wrapper selects the project virtual environment through `environment.sh`
and invokes `prepare_replay.py --generate` without video encoding. It saves
compressed lossless frames and metadata only. Its default configuration is
`data/replay-config.json`, a reusable explicit approximation with `eid: null`,
800×600 display/scene rasters, and 30 Hz cadence. Override it with `--config`.
In the CLI only, a null configuration EID is filled from `--eid`, or the first
entry of `--eids-file` (default `data/eids.txt`, respecting `VINED_EIDS_FILE`).
Comments and blank manifest lines are ignored. A configuration with a concrete
EID retains that session; a conflicting selector is rejected rather than
silently applying another session's calibration. `--eid` and `--eids-file` are
mutually exclusive. The underlying preparation API still requires a resolved EID.

The Bash wrapper selects all EIDs from `data/eids.txt` (or `VINED_EIDS_FILE`)
unless `--eid` selects one. It always processes all trials: trial-count and
trial-ID options are rejected. `--projection on|off` defaults to `on`, mapping
to `mouse_view` or `display`. The lower-level Python CLI retains its diagnostic
trial-selection options; the wrapper never forwards them.

The wrapper delegates session coordination to `src/generate_replays.py`, which
displays two `tqdm` progress bars: accounted EIDs and published trials for the
current EID. The trial bar starts after preparation establishes the trial count,
resets for each session, and advances after image/metadata publication.
Unavailable, invalid, and failed trials count as accounted work.
Session failures advance the EID bar and allow later sessions to continue;
interruptions do not advance unfinished work. A full bar describes accounting,
not successful reconstruction. Preparation/downloads may take time before the
trial bar appears. Progress is written to stderr; terminal cursor support is
needed to display both bars in place.

Each invocation defaults to a new `VINED_REPLAY_DIR/run-<UTC timestamp>-<PID>`
directory; `--output` overrides this destination. Each EID gets a separate
subdirectory, with compressed frames under
`<eid>/trials/<original-id>/images/<schedule-index>.npz`.
Existing run directories are refused. Ordinary session failures or partial
outcomes are reported and later EIDs continue; the script returns the highest
nonzero child status, or zero when all succeed. Interrupt statuses 130/143 stop
the run. The wrapper always passes `--access-policy remote-allowed`: cached files
are reused and missing data may be acquired. Metadata/catalog lookup may still
contact IBL even when files are cached; this is not an offline guarantee.
There is no Bash access-policy option. The lower-level Python CLI retains its
explicit policy option for offline use. `--force-reload` bypasses cached HTTP
metadata responses and re-downloads each requested replay source dataset once
per session access context, including already-cached files. Size/hash validation
and decoding precede replacement; failures do not fall back to stale files.
Unrelated cached datasets and earlier replay generations are untouched. The
Python CLI also accepts `--force-reload`, requiring `--access-policy remote-allowed`.
The only other supported options are `--config`,
`--output`, and `--help`. The feature-extraction wrapper `prepare_visual_stim.sh`
remains separate.
