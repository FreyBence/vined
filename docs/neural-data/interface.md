# Neural-data interface

## Console progress

The CLI uses stderr progress bars for recording loading, window counting, and
compression, including parallel processing. Selection, timing, verification, and
publication messages appear between bars.

Bars show elapsed time, throughput, and remaining time when totals are known,
refreshing at most twice per second. Standalone progress messages are deferred
until all active bars close; no messages are emitted during a bar. Bars also close
on exceptions and interrupted iteration. No logging flag is required.

## Scope and entry point

`src/neural_data` exposes validated source populations, in-memory spike counts,
and independently loadable neural generations with time, identity, and validity.

With `src/` on the Python path:

```python
from session_data import SessionAccess
from neural_data import RecordingRequest, QualitySelection, RegionSelection, load_population

population = load_population(
    SessionAccess(policy="local-only"),
    eid,
    [RecordingRequest(pname="probe01", collection="alf/probe01/pykilosort", revision="")],
    quality=QualitySelection("label", "Supplied IBL label equal to 1", values=(1,)),
    anatomy=RegionSelection(("APN",), mapping="Allen", include_descendants=False),
)
```

`load_population(access, eid, recordings, *, quality=None, anatomy=None)` uses
only the [session-data public interface](../session-data/interface.md).
`recordings` is a nonempty sequence of `RecordingRequest` objects. Each requires
a PID or probe name; supplying both requires them to agree. `collection` and
`revision` are forwarded exactly, including `revision=""` for unrevisioned
sources. `None` permits the dependency's documented source resolution, whose
effective identity is returned. Duplicate resolved recording/sorting requests
are rejected. No failed source is silently skipped or replaced.

## Selection

- `quality=None` retains units without a quality filter. `QualitySelection`
  requires `field`, a nonempty caller-supplied `interpretation`, and exactly one
  of `values` (accepted source values) or `minimum` (finite numeric lower bound).
  Supplied metrics are associated by `cluster_id`, or by original row order when
  no IDs are supplied. Missing per-unit quality does not match; a wholly absent
  required field fails. Metrics are never recomputed and labels are not converted
  to a universal good/bad code.
- `anatomy=None` retains units without an anatomical filter. `RegionSelection`
  requires nonempty Allen region `acronyms`, defaults to mapping `Allen`, and
  defaults to exact matching in that mapping. Requested acronyms must exist in
  the selected mapping. `include_descendants=True` includes their Allen-tree
  descendants, mapped into that same representation. Acronym matching does not
  select a hemisphere. Atlas identity, iblatlas version, requested mapping, and
  effective regions are recorded. Missing/unknown anatomy never matches; wholly
  unavailable required anatomy fails.

No firing-rate threshold or model population limit is applied. Optional source
unit fields and metrics are retained. Missing depths, UUIDs, labels, or anatomy
remain unknown. Source cluster anatomy takes precedence over channel enrichment;
recognized Allen IDs can supply acronyms, but contradictory supplied acronyms
fail. Unknown atlas IDs remain unselected by anatomy filters. No sampling
frequency is inferred from source timestamps.

## Returned population

`Population` exposes:

| Field | Contract |
| --- | --- |
| `eid` | Canonical session UUID. |
| `spikes["times"]` | Finite session-clock seconds, shape `[events]`, stably sorted. |
| `spikes["clusters"]` | Integer positions into `units`, shape `[events]`. These are output positions, not original cluster IDs. |
| `units` | DataFrame with one row per selected unit, including units with no spikes. |
| `recordings` | Tuple of recording records in the same canonical order used for the unit axis. |
| `selection` | Requested quality/anatomy policies, or `None` for each unfiltered policy. |
| `status` | `available` or `empty_selection`; availability here is a selected population, not confirmed temporal coverage. |

Unit order is resolved probe name, PID, collection, revision, then original ALF
cluster row. Input request order does not determine output order. `units` includes
`eid`, `pid`, `probe`, `collection`, `revision`, `source_unit_id`, and
`recording_index`. `source_unit_id` is the original ALF cluster row, scoped by
the session/recording/sorting identity. It is not a channel ID or global neuron
identity. `channels` retains the original channel reference. PID may be unknown
under the session-data offline contract. `selected_region` is present when an
anatomical policy was applied.

Recording records contain the effective `source` (including dataset identities),
`content_sha256` identifying decoded spike/cluster/channel content consumed,
`request`, `timing`, `coverage`, effective `selection`, `source_unit_count`,
`selected_unit_ids`, and `status`. Source paths retain their in-memory `Path`
type. The content digest includes source array dtypes, shapes, values, and supplied
metadata, including metric-table row identities; it is not a hash of a pathname.
All supplied event arrays are checked for matching row counts; the output spike
representation deliberately contains only times and unit assignments. Equal-time
events are retained. Supplied arrays are not mutated.

## Coverage and timing

`RecordingRequest.coverage` defaults to `Coverage()`: unknown recording coverage.
The spike-source API supplies session-clock seconds but no recording-support
intervals. Coverage is never inferred from first/last spike times.

Callers may supply `Coverage(observed=..., invalid=..., provenance=...,
qualification=...)`. Intervals are finite `[start, end)` pairs in session seconds.
`observed=None` means unknown; `observed=()` means explicitly no observed support.
Known invalid intervals can accompany unknown observed coverage. Supplied
intervals require an evidence/assumption description in `provenance`;
`qualification` is `observed` or `assumed` when observed intervals are supplied,
and `unknown` otherwise. These records preserve supplied evidence and its
qualification; source preparation does not infer coverage or compute bin masks.

Timing provenance identifies the session-data ALF timestamp contract. No clock
conversion is applied. Source errors propagate from `SessionAccess`; invalid
neural associations, policy arguments, contradictory metadata, and unresolved
required selection metadata raise `ValueError`.

## Counting intervals and trials

Omitting bin_size uses exactly 1/60 second (approximately 16.67 ms, 60 Hz).
The CLI uses the same default when the request omits this field; explicit null
retains single-bin-per-window counting. The experiment uses 20 ms as its
reference and permits candidate sizes within ±5 ms (15–25 ms); 1/60 second
satisfies this constraint and matches the confirmed visual projection cadence.
Use the full-precision JSON value below rather than rounding to 0.01667 seconds.
Existing 20 ms generations retain their recorded grids and require regeneration
from spikes before aligning on the new experimental grid.

```python
from neural_data import count_intervals, count_trials

counts = count_intervals(population, [(100., 100.05), (101., 101.07)],
                        bin_size=1/60, request_ids=[8, 19])
trials = access.load_trials(population.eid)
trial_counts = count_trials(population, trials, event="stimOn_times",
                           offsets=(-0.5, 1.5), bin_size=1/60)
```

`count_intervals(population, intervals, *, bin_size=1/60, request_ids=None,
unit_coverage=None, workers=1)` counts each `[start, end)` window independently. `intervals`
has shape `[requests, 2]` in source-session seconds. With explicit `bin_size=None`, each
window is one bin, allowing callers to request arbitrary individual physical
intervals. A positive finite bin size partitions each window and retains a
shortened final bin. The returned edges are authoritative; durations are not
assumed to equal the nominal bin size. Floating-point grid points coincident
with the final endpoint within four endpoint ULPs are folded into that endpoint.
Unrepresentable temporal resolution fails explicitly.

Request IDs are unique nonnegative int64-compatible integers; the default is
request position, which is not an original trial identity. Window order is
preserved. Nonfinite boundaries produce unavailable outcomes without inventing
an interval. Finite reversed/zero-width windows and malformed inputs raise
`ValueError`. Empty requests and empty selected populations are supported.

`count_trials(population, trials, *, event, offsets, bin_size=1/60,
unit_coverage=None, workers=1)` accepts a `LoadedTrials` from session-data. Source EIDs must
match the population. The table's unique nonnegative integer index supplies
original trial IDs and must be retained if callers subset the table. It is never
renumbered. A missing event column fails; a missing/nonfinite event value retains
that trial as unavailable. Offsets are a finite increasing pair in seconds.
Trial source records and event/offset configuration are retained. Choice,
feedback, visual coverage, and experiment splits impose no eligibility filters.

Both functions count again from the supplied source population; callers may
request another grid without interpolating existing binned counts or accessing
private helpers. Overlapping windows intentionally count the same event in each
window. Left-sided searches at every edge preserve coincident events, exclude
the right boundary, and avoid repeated full-session spike scans. Separate calls
with the same population, windows, IDs, and policies have the same numerical
results; there is no learned or batch-dependent state.

Both counting entry points accept `workers=1`, a positive integer. Above one,
independent windows run in shared-memory CPU threads, capped at the request
count. Spike inputs, recording coverage, and unit selection are validated once
before dispatch; workers read them without source acquisition or full spike-array
copies. Returned windows follow request order, retaining original IDs, unit
columns, half-open edges, and coverage semantics. Empty/single-window requests
use the sequential path. Errors propagate after active workers finish; no partial
result is returned. Concurrent callers must not mutate the supplied population
while counting. Worker count does not change numerical processing configuration.

## Count output and validity

`NeuralCounts` contains `eid`, copied `units` and `recordings`, `windows` (a tuple
of `CountWindow`), `configuration`, and `trial_sources` (empty for absolute
requests). Configuration identifies raw unsmoothed counts, source clock, bin
size, selection, effective unit coverage restrictions, and validity policy.
For trial processing, configuration also includes `trial_content_sha256`, which
identifies the consumed table, its original index, and its values. Generation
identity is supplied by the persistence boundary below.

Each `CountWindow` exposes:

| Field | Contract |
| --- | --- |
| `request_id` | Supplied request ID, or original trial ID for trial processing. |
| `interval` | Requested absolute `(start, end)`; may retain nonfinite missing timing. |
| `bin_edges` | Float64 `[T+1]` session seconds; bin `t` is `[edges[t], edges[t+1])`. Empty for missing timing. |
| `counts` | Int64 `[T, N]` counts of supplied events in each full interval, without clipping, smoothing, normalization, or padding. |
| `valid` | Boolean `[T, N]`; true only for fully observed, non-invalid bins. |
| `coverage_status` | Uint8 `[T, N]` codes described below. |
| `observed_duration` | Float64 `[T, N]` supported duration in seconds after subtracting known invalid intervals; NaN when support is unknown. For assumed support, duration is also qualified by its coverage provenance. |
| `status`, `reason` | Available, partial, unavailable, or empty-selection outcome and an explanatory reason when applicable. |

`T` varies by window; `N` is the complete selected population for every window,
including silent units. A missing-time window has shape `[0, N]`; an empty
population has shape `[T, 0]` and an explicit `empty_selection` outcome. A window
with missing timing remains unavailable even if the population is also empty.

`COVERAGE_STATES` maps codes to meanings:

| Code | State | Valid |
| --- | --- | --- |
| 0 | Unknown support | False |
| 1 | Fully observed support with no known invalid portion | True |
| 2 | Full support under a recorded assumption | False |
| 3 | Partially supported interval | False |
| 4 | Known invalid overlap without usable full support | False |
| 5 | Outside supplied observed support | False |

Numeric counts alone never establish validity. In particular, counts in partial,
invalid, unknown, or assumed cells are raw event tallies and **must not be used
as fully observed neural targets**. They are not scaled to compensate for missing
duration. A window is `partial` if it contains both valid and unusable cells;
if no cells are valid it is `unavailable`, with coverage states in its reason.
Per-cell statuses retain partial support even for an unavailable window.

Coverage is evaluated independently for each recording and propagated to its
units. Overlapping coverage intervals are unioned rather than double-counted,
and known invalid intervals are subtracted. Optional `unit_coverage` maps unit
axis positions to `Coverage` objects. These restrictions intersect known
recording support and combine invalid intervals; unit evidence can qualify
otherwise unknown support but cannot extend known recording coverage. Assumed
support remains qualified and is not promoted to observed validity. Original
recording evidence and effective unit restrictions remain in the output.

Processing errors raise rather than returning a silently truncated collection
of windows. No files are published by these APIs. Source-load failures continue
to propagate from the source preparation boundary.

## Standalone generation and loading

```python
from neural_data import generate_neural, load_generation

generation = generate_neural(
    access, eid, [RecordingRequest(pname="probe01", revision="")], "output/neural",
    event="stimOn_times", offsets=(-0.5, 1.5), bin_size=1/60,
)
loaded = load_generation(generation.path,
                         expected_generation_id=generation.generation_id)
counts = loaded.data
```

`generate_neural(access, eid, recordings, output_dir, *, intervals=None,
request_ids=None, event=None, offsets=None, bin_size=1/60, quality=None,
anatomy=None, unit_coverage=None, trial_collection="alf", trial_revision=None, workers=1)`
processes one session. Supply exactly one of absolute `intervals` or a trial
`event`; trial mode requires offsets and takes IDs from the original table.
Selection and coverage objects have the contracts above. Acquisition stays
within session-data. No visual artifact, behavioral eligibility filter, model,
or experiment split is required.

`workers` controls window counting and staged per-window NPZ compression in
separate shared-memory thread pools. Source loading and final verification and
publication remain serial. Each output file is written by one worker, and file
and outcome accounting retains requested order. A worker failure prevents
publication. Worker count is an execution setting, excluded from generation
configuration; identical numerical outputs retain the same content identity.

The returned `NeuralGeneration` has `generation_id`, absolute `path`, `data`
(`NeuralCounts`), and `manifest`. Publication uses a temporary sibling directory
and a final rename after the consumer loader verifies the artifact. Any source,
processing, serialization, or verification failure propagates without publishing
a completed generation. An interruption may leave an unselected temporary
directory after an unclean process termination; it is never used as a fallback.
Existing generations are not overwritten. An identical destination raises
`FileExistsError`, even after successful processing: reuse is an explicit load,
not an implicit success report for a new run.

`load_generation(path, *, expected_generation_id=None)` opens precisely the
selected generation directory. It checks schema, completion, manifest identity,
artifact hashes, shapes/dtypes, unit associations, and request accounting before
returning. Missing files raise filesystem errors; mismatched content, unsupported
schema, or inconsistent accounting raise `ValueError`. The optional expected ID
rejects an unintended selection. There is no latest-generation search, network
access, current-source revalidation, or replacement with another artifact. Older
explicitly selected generations remain loadable under the supported schema.

The manifest records decoded source-content identities, original dataset/source
identities, trial content identity where applicable, selected units, all request
IDs/outcomes, effective configuration, implementation source hashes, and relevant
package versions. A generation's ID is the SHA-256 fingerprint of its canonical
manifest excluding `generation_id`, including the artifact digests. Source or
configuration changes are processed anew and cannot silently reuse an existing
generation. Processing source changes during a run cause publication to fail.

`complete=true` means all requested recordings, selected units, and interval or
trial outcomes were accounted for, **not** that every count is a valid observed
target. Empty selection, missing timing, unknown/assumed coverage, and partial
observations retain their explicit meanings. Failures prevent publication rather
than masquerading as ordinary unavailability.

### Persistence schema 1

```text
<output_dir>/<eid>/<generation_id>/
    manifest.json
    units.parquet
    windows/000000.npz
    windows/000001.npz
    ...
```

`units.parquet` preserves the ordered unit table. Each numbered NPZ corresponds
to one request in manifest order and contains `interval`, `bin_edges`, `counts`,
`valid`, `coverage_status`, and `observed_duration`. Arrays use the in-memory
dtypes above, including int64 counts; loading disables pickle. Missing timing
and unknown duration remain nonfinite NumPy values, not synthetic zero values.
Variable lengths are stored separately without padding. Request IDs, status,
and reason are in the matching manifest `windows` entry.

Manifest fields include `schema_version`, `complete`, `eid`, `generation_id`,
`requested_recordings`, `requested_ids`, `recordings`, `trial_sources`,
`configuration`, `implementation`, `files`, `windows`, and `outcomes` (counts by
outcome). `files` maps relative artifact paths to SHA-256 digests. JSON metadata
uses strings for paths, lists for tuples, and null for unavailable scalar metadata;
unit coverage keys are restored to integers by the loader. Source arrays and the
trial table are fingerprinted in memory while processing consumes them, so later
changes to files at the same paths do not redefine the recorded input identity.

### CLI

From the checkout root using the project Python:

```text
python src/prepare_neural_data.py --eid EID --config request.json --access-policy local-only
```

The Bash entry point uses `script/environment.sh` to select the project Python
and checkout root, and forwards every argument and the Python exit status:

```bash
bash script/prepare_neural_data.sh --eid EID --config request.json --workers 4
```

Example request JSON:

```json
{
  "recordings": [{"pname": "probe01", "collection": "alf/probe01/pykilosort", "revision": ""}],
  "event": "stimOn_times",
  "offsets": [-0.5, 1.5],
  "bin_size": 0.016666666666666666
}
```

The configuration uses `generate_neural` keyword names, plus `recordings`.
Quality, anatomy, recording coverage, and unit coverage use JSON objects with the
corresponding dataclass fields. Unknown configuration keys fail. Absolute requests
use `intervals` and optional `request_ids` instead of `event`/`offsets`.

`--eids-file` and `--n-sessions` use the shared session-selection conventions;
the same request configuration applies to each selected session. `--workers N`
sets a positive number of counting/compression threads per session (default one).
This is a CLI option, not a request JSON field. Session source loading remains
serial and workers add no ONE/Alyx calls. `--cache-dir`
defaults to the project dataset cache and `--output-dir` to
`<VINED_OUTPUT_DIR>/neural`. Access defaults to `local-only`; acquisition requires
`--access-policy remote-allowed`. Sessions are processed individually. Output
reports each successful generation's ID/path/outcomes plus completed, failed,
and unprocessed sessions. Failures produce a nonzero exit and do not select an
older generation.

The combined alignment/training-dataset pipeline still requires its separately
scoped migration to this contract, including count-preserving serialization.
Its legacy eight-bit serializer is not used by standalone neural generations.

## Legacy adapter

`utils.ibl_data_utils.load_spiking_data` delegates to this boundary and retains
its `(spikes, clusters, sampling_freq)` return shape. `sampling_freq` is now
`None`. Collection/revision are explicit forwarded keyword arguments; unknown
keywords fail. `compute_metrics=True` fails explicitly. Legacy `qc` requests a
documented numeric lower bound on the supplied label. Returned cluster rows retain
original IDs; source records are available in DataFrame attributes.

`merge_probes` copies assignments before offsetting, validates associations, and
rejects inconsistent event-field sets instead of silently dropping fields. The
legacy preparation metadata retains raw labels in `good_clusters`, including
unknown values. Existing callers only store that field. Their binning and joint
selection semantics remain outside this source interface.
