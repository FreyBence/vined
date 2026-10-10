# Alignment interface

## Console progress

The CLI uses stderr progress bars for visual record reading, trial pairing,
resampling, and compression. Input verification and publication messages appear
between bars. Structured results remain on stdout.

Bars show elapsed time, throughput, and remaining time when totals are known,
refreshing at most twice per second. Standalone progress messages are deferred
until all active bars close; no messages are emitted during a bar. Bars also close
on exceptions and interrupted iteration. No logging flag is required.

## Prepared input boundary

With `src/` on the Python path, `alignment.prepare_inputs` exposes the implemented
input preparation boundary. `alignment.align_trials` performs in-memory temporal
alignment. Standalone generation publication and verified loading are exposed below.

```python
from alignment import prepare_inputs

inputs = prepare_inputs(
    neural_generation_directory,
    visual_feature_archive,
    expected_neural_generation_id=neural_generation_id,
    expected_visual_generation_id=visual_generation_id,
)
```

Signature:

```python
prepare_inputs(neural, visual_path, *, request_trial_ids=None,
               expected_neural_generation_id=None,
               expected_visual_generation_id=None, skip_invalid=False) -> AlignmentInputs
```

`neural` accepts a `neural_data.NeuralCounts` or an explicit generation directory
(`str`/`Path`). Direct counts are copied and validated and have no generation ID;
an expected neural generation ID requires a directory. Persisted inputs are
loaded through `neural_data.load_generation`. `visual_path` selects one current
feature archive, read through `visual_features.FeatureArtifactReader` and fully
exhausted before any paired result is returned. Expected generation IDs, when
provided, must match exactly. No automatic generation discovery or reuse occurs.

Trial-mode counts use their original trial IDs. Absolute count requests require
`request_trial_ids`, a mapping of every request ID to a unique original trial ID.
All IDs are nonnegative int64-compatible integers. The visual and mapped neural trial sets must match exactly. With the default
`skip_invalid=False`, an unusable trial raises. `skip_invalid=True` excludes
trial-level timing, grid, or coverage failures with their original IDs and
reasons in `excluded_trials`; no trial is silently skipped or renumbered.
Integrity and unsupported-representation errors remain fatal.

## Input requirements

The current experimental neural bin size is exactly `1/60` second (approximately
16.67 ms, 60 Hz), matching visual replay cadence. Alignment reads the supplied
neural bin size; it does not rebin existing 20 ms counts. Regenerate neural counts
from source spikes and then regenerate alignment and dependent datasets for the
new grid. Explicit alternative prepared grids remain supported.

Neural bins and reconstructed replay schedules are anchored at stimulus onset.
Visual states are mapped by timestamp to whole neural intervals, without CLIP
interpolation. Equal nominal rates do not establish measured display/bin
coincidence; supplied timestamps and complete state support are checked explicitly.

- Neural counts are raw unsmoothed int64 values with session-clock seconds and
  an explicit positive bin size. Unit rows retain their supplied order and
  scoped session/recording/sorting/source-unit identities, including silent units.
- Exact onset and offset come from the public visual replay definition's trial
  `events.onset` and `events.offset`; these must agree with the visual requested
  domain. Missing, nonfinite, or reversed bounds fail.
- Supplied neural edges must cover every complete stimulus bin on an
  onset-anchored grid. Extra pre/post-stimulus bins and a shortened final source
  bin may be present; the input boundary preserves them without rebinning.
- Every complete stimulus bin/unit cell must have fully observed neural support
  (`valid=True`, coverage state 1, full observed duration). Unknown, assumed,
  partial, and invalid support fail rather than becoming valid targets. Numeric
  zero counts remain valid under the same coverage rule.
- Visual timestamps must be finite and strictly increasing within each trial.
  All visual observations, including unselected and unavailable records, and
  original trial coverage outcomes are retained. This input boundary does not
  validate held-state interval coverage or map features; `align_trials` does.

## Return contract

`AlignmentInputs` includes original `requested_trial_ids` and input-stage
`excluded_trials` records (`session_id`, `trial_id`, `stage`, `reason`), plus `session_id`, `bin_size`, copied `units` and
`recordings`, and `trials` sorted by original trial ID. It also preserves
`neural_configuration`, `neural_trial_sources`, `neural_generation_id` (or `None`
for direct counts), the complete `visual_definition`, and verified
`visual_completion` with feature and replay generation identity.

Each `PreparedTrial` contains `session_id`, `trial_id`, `stim_on`, `stim_off`,
the copied source `CountWindow` in `neural`, all `EncodedObservation` records in
`observations`, and the unchanged trial outcome metadata in `visual_outcome`.
Feature arrays are read-only. Neural counts have shape `[source_bins, units]`;
encoded features have shape `[768]`. Neither input is an aligned output tensor.
Times and bin duration are seconds in the source session clock, not model indices.

Direct in-memory counts are defensively copied; persisted counts are freshly
loaded and reused without another full-array copy. The result is caller-owned; nested metadata and neural
arrays remain caller-owned mutable objects. No acquisition, population selection,
resampling, splitting, padding, or file publication occurs.

## Errors

Invalid identities, trial membership, timing, grids, coverage, representation,
or expected generation IDs raise `ValueError`. Unsupported neural argument types
raise `TypeError`. Public loader filesystem, decoding, and verification errors
propagate. No partial paired collection is returned on failure. Incompatible
grids or missing coverage evidence must be resolved through upstream preparation.

## Temporal alignment

`alignment.align_trials(inputs: AlignmentInputs, *, workers=1) -> tuple[AlignedTrial, ...]`
returns one aligned trial per paired input, sorted by original trial ID. Use:

```python
from alignment import align_trials

aligned_trials = align_trials(inputs)
```

The operation rechecks trial/session identity and usable neural grids/coverage,
selects only complete onset-anchored source bins, and copies their counts without
rebinning or narrowing the integer dtype. Supplied edges remain authoritative;
grid coincidence uses four endpoint ULPs to accommodate source-clock floating
point arithmetic. Pre-stimulus bins, post-stimulus bins, and a shortened trailing
bin are excluded from the output arrays.

The current supported visual temporal representation is reconstructed,
onset-anchored, session-second sampling over `visible_interval`, with positive
`cadence_hz` evidence. Every scheduled observation must be accounted for in
consecutive order and valid, including unselected observations. A final terminal
observation at the exact offset is permitted if it precedes the next cadence
position. Trial outcomes must be complete with continuously valid reconstruction
coverage. Each source state is held over `[source_time, next_source_time)`, with
the final state supported until the known stimulus offset. Every neural interval
must fit entirely inside one supported state interval. Four endpoint ULPs tolerate
source-clock roundoff; an update strictly inside a neural bin raises `ValueError`.
The active observation must be encoded: an unselected update ends the preceding
state and cannot be bridged using an older feature. Source features are copied
unchanged, without interpolation or renormalization. No terminal feature is
required at offset to support the final visible state.

The shared policy is `timestamp-aware held-state over neural intervals v1`.
Reconstructed source timing remains reconstructed; alignment does not create
actual display-update evidence. Nominal schedule/bin coincidence is recorded
separately and never promoted to verified measured 1:1 display alignment. This
implements the explicitly accepted held-state requirement; the existing
`spec.md` interpolation sections have not been rewritten.

`AlignedTrial` exposes:

| Field | Contract |
| --- | --- |
| `session_id`, `trial_id` | Original paired identities. |
| `stim_on`, `stim_off` | Exact source stimulus bounds in session seconds. |
| `aligned_start`, `aligned_end` | First/last retained neural edge. |
| `bin_size`, `bin_count` | Configured duration in seconds and retained bin count `T`. |
| `discarded_tail_duration` | Offset minus aligned end; endpoint roundoff within the grid tolerance is clamped to zero. |
| `bin_start_times`, `bin_center_times`, `bin_end_times` | Float64 `[T]` physical session timestamps. Centers are midpoints of supplied edges. |
| `neural_activity` | Int64 `[T,N]` raw counts, retaining zeros and silent units. |
| `neuron_identity` | Copied full ordered unit table; row `n` identifies neural column `n`. |
| `visual_features` | Float32 `[T,768]` unchanged source features held over the matching neural intervals. |
| `alignment_metadata` | Compact trial-specific source definitions/completion identities, neural configuration/recordings/trial sources, request/generation identities, source-bin slice, visual outcome, and resampling policy/associations. |

`alignment_metadata["visual_resampling"]` records the versioned mapping policy.
The retained `resampling` container provides encoded `source_times`, observation
IDs, schedule indices, and `source_interval_end_times`. Existing downstream fields
`left_source_indices` and `right_source_indices` are identical selected-source
indices; `interpolation_weights` are zero. No arithmetic interpolation occurs.
`policy`, complete `timing_classification`, `update_bin_boundaries_coincident`,
and `verified_display_bin_alignment` distinguish representation and evidence.
The last flag is false for the currently supported reconstructed source timing.

All output cells are supported observations; no padding, missing-data masks, or
placeholder vectors are returned. Output arrays and metadata are independent
copies and may be modified by the caller. `ValueError` identifies the affected
trial for identity, timing, incomplete reconstruction, gaps, unsupported cadence,
unencoded active states, or updates inside neural bins. `TypeError` rejects
an argument other than `AlignmentInputs`. A failed trial prevents return of a
partial collection. No files or datasets are produced.

## Standalone publication and loading

```python
from alignment import generate_alignment, load_alignment

generation = generate_alignment(neural_directory, feature_archive, output_directory)
loaded = load_alignment(generation.path,
                        expected_generation_id=generation.generation_id)
trials = loaded.trials
```

- `generate_alignment(neural, visual_path, output_dir, *, workers=1, skip_invalid=True, **input_options)` calls
  verified input preparation, temporal trial processing, and the publisher.
  By default it retains usable trials and records explicit input/temporal-stage
  exclusions. `skip_invalid=False` requests strict behavior. Input options are exactly
  the keyword options of `prepare_inputs`; unknown keywords fail.
- `publish_alignment(trials, output_dir, *, workers=1, requested_trial_ids=None, excluded_trials=(), source_provenance=None)` accepts a nonempty canonical sequence
  of `AlignedTrial` objects from one canonical session UUID and one shared ordered
  unit population. It validates array shapes/dtypes, temporal consistency,
  identities, and required provenance before publishing.
- `load_alignment(path, *, expected_generation_id=None)` selects exactly one
  completed generation directory. It verifies schema, manifest identity, file
  accounting and hashes, array shapes/dtypes, unit associations, physical grids,
  normalized features, and trial accounting without acquiring sources or
  selecting another generation.

All three return `AlignmentGeneration(generation_id, path, trials, manifest)`.
`path` is absolute; `trials` is a tuple of `AlignedTrial` objects. Generation
identity is the SHA-256 fingerprint of the canonical manifest excluding its own
`generation_id`, binding artifact bytes, metadata, source provenance, implementation
hashes, and NumPy/pandas/pyarrow versions. Explicit direct counts retain a null
neural generation ID rather than inventing a prepared artifact identity.

Publication writes a temporary sibling directory, verifies it through
`load_alignment`, checks implementation stability, then renames it to the final
generation directory. Existing destinations raise `FileExistsError`; reuse is an
explicit load. A source, alignment, serialization, or verification error prevents
completed publication. Ordinary failures clean staging files; forced termination
can leave an unpublished temporary directory, which no loader selects implicitly.

### Persistence schemas 1 and 2

```text
<output_dir>/<session_id>/<generation_id>/
    manifest.json
    units.parquet
    trials/000000.npz
    trials/000001.npz
    ...
```

The ordered unit table is stored once without its incidental DataFrame index.
Each numbered NPZ corresponds to one trial in manifest order and contains only
`bin_start_times`, `bin_center_times`, `bin_end_times`, `neural_activity`, and
`visual_features`, with the dtypes/shapes above. Int64 spike counts are retained;
no eight-bit serializer or padding is used. Loading disables pickle.

New publications use `schema_version=2`. `requested_trial_ids` partitions exactly
into canonical `retained_trial_ids` and `excluded_trials`, whose records carry
original session/trial identity, input/temporal stage, and nonempty reason. No
excluded trial has a scientific payload. All trials excluded is an error.
`source_provenance` stores full visual definition/completion once per generation.
Each trial's `visual_definition` instead records `scope="trial"`, source replay
identity, representation/selection, and that trial's source record;
`visual_completion` records verified feature/replay generation identities.
Dataset samples retain these compact references rather than copying session-sized
source definitions. Schema-1 generations remain loadable with their original
full per-trial provenance and requested-equals-retained accounting.

The manifest contains `schema_version`, `kind="aligned_trials"`, `complete`,
`eid`, `requested_trial_ids`, ordered `trials`, `implementation`, `files`, and
`generation_id`, and `visual_mapping_policies`. The policy list is checked against
trial metadata and bound into generation identity. The NPZ array fields and dtypes are unchanged;
historical interpolation generations remain explicitly loadable with their old
policy/provenance and are never silently converted to held-state data. Trial
entries carry original ID, exact stimulus bounds, aligned
bounds, bin size/count, discarded tail, and complete alignment metadata. `files`
maps relative payload paths to SHA-256 digests. JSON preserves source paths as
strings, tuples as lists, dictionary keys as strings, and unavailable/nonfinite
source metadata as null. Numerical resampling timestamps/weights/indices are
restored as NumPy arrays in loaded trials; the returned manifest remains JSON data.

Missing files raise filesystem errors. Malformed, incompatible, hash-mismatched,
or inconsistently accounted generations raise `ValueError`. A supplied expected
generation ID rejects unintended selection. Completed means every requested trial is accounted for and at least one is
retained. Every retained trial has valid aligned observations; damaged trials
are recorded as exclusions, never published as placeholders.

### CLI and downstream handoff

From the checkout root with the project Python:

```text
python src/prepare_alignment.py --neural-generation PATH --visual-features PATH
```

Input paths may be omitted; session selection defaults to `eids/eids.txt`
(or `VINED_EIDS_FILE`):

```bash
# All configured sessions:
bash script/prepare_data.sh
# One session:
bash script/prepare_data.sh --eid EID --workers 4
# First two sessions from another list:
bash script/prepare_data.sh --eids-file eids/my_sessions.txt --n-sessions 2
```

`--neural-root` defaults to `<VINED_OUTPUT_DIR>/neural` (normally `output/neural`).
`--visual-root` defaults to `VINED_VISUAL_DIR` (normally `output/visual_features`).
The CLI resolves one completed neural generation under `<neural-root>/<eid>/`
and reads `<visual-root>/<eid>_visual_clip.npz`. Multiple neural generations
require `--neural-generation PATH` or `--expected-neural-generation-id ID`;
no latest generation is selected. Missing inputs fail without acquisition.
Sessions are processed in list order, with parallel trial work within each
session. The shared runner reports completed/failed/unprocessed sessions and
continues ordinary per-session failures; any failure gives a nonzero exit.
Explicit input paths override their respective defaults and remain usable without
session selection when both are supplied. Explicit paths, expected generation
IDs, and request mappings require a single selected session. Selected EIDs are
checked against both sources.

The CLI defaults to retaining usable trials. `--strict-trials` opts into failure
on the first unusable trial. Success JSON includes requested/retained counts and
the excluded trial records. Corrupt artifacts, conflicting source identities,
unsupported representations, and sessions without usable trials still fail.

`--workers N` runs trial resampling and NPZ compression in up to N shared-memory
threads (CLI default 4; `--workers 1` is sequential). The Python APIs
`align_trials`, `publish_alignment`, and `generate_alignment` accept the same
positive integer `workers` keyword, defaulting to 1 for existing callers.
Trial results and manifest entries retain canonical order regardless of worker
completion order. Input verification, pairing, and final publication/readback
remain sequential. Failed work prevents publication; progress is reported by the
calling thread. Worker count does not change scientific configuration or schemas,
and performance depends on the input and available CPU/I/O capacity.

`--output-dir` defaults to `<VINED_OUTPUT_DIR>/alignment`. Optional
`--expected-neural-generation-id` and `--expected-visual-generation-id` enforce
explicit identities. `--request-trial-ids FILE` reads a JSON object mapping
canonical decimal request-ID keys to integer original trial IDs for absolute
neural requests. Each selected session produces its own alignment generation. Success
prints generation ID, absolute path, session ID, and original trial IDs and exits
0. Input/publication failures report the error and exit 1; CLI argument errors
exit 2. No acquisition or network access is performed.

`bash script/prepare_data.sh` calls `src/prepare_alignment.py` directly.
The old positional session-selection form, acquisition flags,
fixed-window selection, LFP preparation, and automatic dataset splitting are
retired from these alignment launchers. Shared legacy helper functions remain
available for callers outside this supported boundary.

Training-dataset consumes `load_alignment(...).trials` and owns splitting,
padding, and model packaging. `src/create_dataset.py --alignment-generation PATH`
accepts published alignment generations; see the
[training-dataset interface](../training-dataset/interface.md) for creation options.
