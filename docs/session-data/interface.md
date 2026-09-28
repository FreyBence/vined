# Session data interface

## Purpose and entry points

`src/session_data` exposes original IBL session metadata and named datasets.
Import from `session_data` with the checkout's `src/` on the Python import path.
It does not filter trials, reconstruct stimuli, transform neural data, or assign
experiment roles.

```python
from session_data import SessionAccess

access = SessionAccess(policy="remote-allowed")
eid = "413a6825-2144-4a50-b3fc-cf38ddd6fd1a"
trials = access.load_dataset(eid, "_ibl_trials.table.pqt", collection="alf")
table = trials.data
revision = trials.source.revision
```

## Configuration and identity

`SessionAccess(*, policy, cache_dir=None,
base_url="https://openalyx.internationalbrainlab.org", force_reload=False)` requires an explicit policy:

- `AccessPolicy.LOCAL_ONLY` / `"local-only"`: cached identity and local files only;
  no remote fallback or Alyx client.
- `AccessPolicy.REMOTE_ALLOWED` / `"remote-allowed"`: remote metadata/catalog access
  and acquisition of missing files are permitted. ONE's cached HTTP responses may
  be used; this is not a guarantee of freshly queried remote availability.

`force_reload=True` requires `remote-allowed`. It bypasses cached HTTP metadata
responses and resolves session identity remotely. For `load_dataset` and
`load_datasets`, each requested dataset is re-downloaded once per access context,
even if already cached. Downloads are staged in the cache filesystem, checked
against available size/hash metadata, and decoded before replacing the cached
file. Failed downloads/validation do not replace an existing cached file and do
not fall back to it. This does not clear unrelated files or HTTP-cache entries,
and does not promise forced acquisition for separate spike/ephys adapter paths.

`policy`, `cache_dir`, `base_url`, and `force_reload` are read-only properties. The cache defaults
to `utils.paths.dataset_dir()` and its `VINED_DATA_DIR` / `VINED_OUTPUT_DIR`
conventions. Explicit relative cache paths resolve against the checkout root.
Existing ONE authentication configuration is used without interactive prompts.
Metadata and discovery persist ONE session/dataset tables for subsequent offline
use; source files remain in the same ONE-managed cache.

`SessionAccess.canonical_eid(eid)` accepts a UUID string and returns its canonical
lowercase form. `resolve_session(eid, *, require_local=False)` returns a frozen
`SessionSource(eid, path)`. `path` is the absolute expected session directory;
it need not exist. `require_local=True` requires the directory, not completeness
of any dataset. Identity resolution does not download recording data.

## Session metadata

`access.metadata(eid, *, include_probes=False)` returns `SessionMetadata`:

| Field | Contract |
| --- | --- |
| `eid` | Canonical session UUID. |
| `reported` | Dictionary of available ONE/Alyx session fields. Remote mode includes the full session record; offline mode exposes fields in the cached session table. Missing/null fields are unavailable, not inferred. |
| `derived` | `session_path`: resolved absolute cache path. |
| `probes` | `None` when not requested or unresolved offline; otherwise a tuple of insertion dictionaries. Remote records include `id`, `name`, and the factual Alyx fields. Available offline records contain `id`, `name`, and `session`. An empty remote tuple means no insertions were reported. |
| `origin` | `"remote"` or `"cache"`; remote may use ONE's HTTP cache. |

Task protocol and other version evidence are exposed where present in the source
record. Additional task settings/raw evidence are requested as named datasets;
this API does not infer software versions or choose historical behavior.
Remote probe records are not independently persisted as a project-owned catalog.
Offline probe resolution requires an existing ONE insertion table.

`access.probes(eid)` returns insertion dictionaries for consumer iteration. When
offline insertion metadata is unresolved, it derives probe names from cataloged
spike collections, with `id=None`, `session=eid`, and `origin="derived"`.
No matching offline sources raise `local_source_unavailable`; a reported empty
insertion list remains an empty tuple. Consumers decide whether that skips a session.

`access.load_trials(eid, *, collection="alf", revision=None)` returns
`LoadedTrials(data, sources)`: the original `_ibl_trials.table.pqt` DataFrame and
a tuple containing its `DatasetSource`. It follows named-dataset selection rules
and does not load unrelated auxiliary trial attributes or filter/renumber rows.

## Dataset discovery

`access.discover(eid, *, filename=None, collection=None, revision=None)` returns
`DatasetDiscovery(eid, datasets, status, origin)`. Filters use ONE wildcard
semantics; `None` includes all matching values, while `""` selects an empty
collection/revision. Discovery does not download datasets.

`datasets` is a tuple of `DatasetSource` records:

| Field | Contract |
| --- | --- |
| `eid`, `dataset_id` | Canonical session and dataset UUID strings. |
| `name` | Dataset filename. |
| `collection` | Effective collection; `""` means session root. |
| `revision` | Effective revision; `""` means unrevisioned. |
| `path` | Absolute expected ONE file path. |
| `availability` | `"local"` if the file exists, `"remote-known"` if only the remote catalog identifies it, or `"not-local"` if only a cached record identifies it. Presence alone does not verify integrity. |

`status` is `"available"` when records match (not necessarily locally),
`"unavailable"` when the remote catalog has no match, or `"unresolved"` when
the local catalog has no match. `origin` is `"remote"` or `"cache"`.
An unresolved session or failed lookup raises an error rather than returning a
misleading empty catalog. Discovery order carries no temporal or trial meaning.

## Named source loading

```text
access.load_dataset(eid, dataset, *, collection=None, revision=None, download_only=False)
access.load_datasets(eid, datasets, *, collection=None, revision=None, download_only=False)
```

Names are nonempty filenames or ONE filename patterns selecting exactly one
dataset each. Supply collection/revision separately, not as part of a path.
Loading collection/revision arguments are exact strings, without wildcards.
An explicit revision must exist exactly; an earlier revision is never substituted.
`revision=None` uses ONE's default/latest resolution, exposed in the returned
source identity. Ambiguous selection raises `source_conflict`.

`load_dataset` returns `LoadedDataset(data, source)`. `load_datasets` returns a
tuple of these results in request order, resolved from one catalog snapshot.
Every member must have the same EID, collection, and revision. Mixed selections
raise `source_conflict` before loading; distinct source groups may be requested
separately, leaving their cross-group compatibility explicit to the consumer.
Equal revision labels alone are not a scientific compatibility certification.

`data` is ONE's decoded representation (for example a DataFrame, NumPy array,
dictionary, or text). `download_only=True` instead returns the materialized
`Path`, including for raw files requiring consumer-specific decoding. `source`
is a `DatasetSource` with effective identity and `availability="local"`.
Recorded file size and hash are checked when available. Missing or inconsistent
files are not returned as valid empty observations. Valid empty decoded values
are preserved without a truthiness-based missing-data check.

Rows, array order, IDs, timestamps, units, and shapes are retained as stored;
there is no resampling, renumbering, normalization, or padding. For the current
replay inputs, trial-table rows remain original trial rows and wheel timestamps
and positions remain one-dimensional source arrays. These source loaders do not
establish downstream time alignment.

## Probe-aware spike-sorting sources

```python
result = access.load_spike_sorting(eid, pname="probe00", revision="2024-05-06")
spikes, clusters, channels = result.spikes, result.clusters, result.channels
```

`load_spike_sorting(eid, *, pid=None, pname=None, collection=None, revision=None)`
requires a PID UUID or an exact probe name. When both are supplied, they must
identify the same insertion in the requested session. No insertion is selected
implicitly. Offline, a probe name can resolve through the dataset catalog when
insertion metadata is unavailable; its PID remains `None`. A PID request requires
available insertion metadata.

An explicit collection must belong to that probe. Otherwise Brainbox selects
`iblsorter`, then `pykilosort`, then its shortest available probe collection.
Revision selection follows the named-source rules above, including exact explicit
revisions and rejection of mixed effective revisions. Missing local files never
trigger remote fallback under `local-only`.

The returned `LoadedSpikeSorting` contains unfiltered Brainbox-decoded ALF
`spikes`, `clusters`, and `channels` objects plus a frozen `SpikeSortingSource`:
`eid`, `pid`, `pname`, effective `collection`, effective `revision`, and `datasets`
(a tuple of materialized `DatasetSource` records). Standard Brainbox spike and
cluster attributes are loaded when cataloged; all unnamespaced channel attributes
are included. Manually curated namespaces are excluded. Spike `times` and
`clusters`, cluster `channels`, and a channel source are required.

Spike arrays retain their source order: `times` is seconds in the session clock,
and `clusters` contains insertion-specific cluster indices. Cluster and channel
arrays retain their respective source row identities and units. Zero-length
arrays remain valid. Channel fields retain ALF names (such as `mlapdv` in
micrometers and `brainLocationIds_ccf_2017`); this API does not replace them with
electrode-site or Alyx trajectory data, derive atlas labels, merge clusters,
compute metrics, filter neurons, merge probes, or bin spikes.

Absent insertions and missing source objects raise `source_unavailable` remotely
or `local_source_unavailable` offline. Other source errors use the reasons below.
Messages include the EID, PID/probe, and collection/revision request context;
consumers decide whether to skip or fail before downstream merging.

## Electrophysiology recordings and source time

```python
result = access.load_ephys(eid, pname="probe01", band="lf", stream=True)
recording = result.recording
session_times = result.samples_to_times([0, 2500])
sample_indices = result.times_to_samples(session_times)
traces = recording.get_traces(start_frame=0, end_frame=10)
```

`load_ephys(eid, *, pid=None, pname=None, band="lf", stream=False, revision=None)`
opens one insertion's compressed IBL recording. PID/probe selection follows the
spike-source identity rules: at least one is required; both must match when given.
It returns `LoadedEphys(recording, source, ...)`, exposing:

- `recording`: a SpikeInterface recording with one segment. Trace axes are
  `[stream samples, channels]`; the sync channel is excluded. The default traces
  retain raw ADC values, with gains, offsets, sampling frequency, and probe
  properties available through SpikeInterface. No signal preprocessing is applied.
- `source`: `EphysSource` with `eid`, `pid`, `pname`, `band`, effective `collection`
  and `revision`, `streaming`, and `datasets` (`DatasetSource` records for the
  recording, metadata, compression index, and source time mapping).
- `samples_to_times(samples)` and `times_to_samples(times)`: conversions between
  this stream's fractional sample indices and session-clock seconds. Scalars and
  arrays are accepted. Linear interpolation/extrapolation follows the source
  timestamp mapping; sample indices are not rounded or clipped. SpikeInterface's
  uniform recording times are not a substitute for this session-clock conversion.

`band` is `"ap"` or `"lf"`; the collection is `raw_ephys_data/<probe>`.
The time mapping contains AP sample indices and session seconds. Recorded AP and
stream sampling frequencies determine the conversion, including for LF; no fixed
AP/LF ratio is assumed. Missing or invalid timing evidence fails explicitly.

`stream=False` requires full `.cbin`, `.ch`, and `.meta` source files, loaded with
SpikeInterface's compressed IBL extractor. Under `remote-allowed`, missing files
may be downloaded, including the full recording. Under `local-only`, missing
files fail without network access. Cached stream chunks do not count as a complete
local recording.

`stream=True` requires `remote-allowed` and a resolved PID. It uses the IBL
recording extractor, acquiring metadata immediately and fetching chunks when
traces are read. The full recording may remain `remote-known`; partial chunks
remain in ONE's stream cache and are not a second source-data store. Lazy trace
reads can raise the underlying loader's network/read errors.

Explicit revisions are exact. Multiple recording/revision candidates raise
`source_conflict`; select an exact revision. Because the underlying streamer
cannot select revisions, streaming also requires an unambiguous recording and
metadata selection across its probe catalog. Otherwise use `stream=False` with
an exact revision. Joint recording and timing sources must share the effective
collection/revision. Missing insertions, recordings, or local files raise the
source availability errors below, with EID/PID/probe/band and selection context.

## Errors

`SessionAccessError` exposes `reason`, `eid` (canonical when known), and a message.
Underlying access exceptions are chained. Consumers decide whether to skip or
stop an operation. Invalid policy or malformed loading arguments raise `ValueError`.

| Reason | Meaning |
| --- | --- |
| `invalid_eid` | Invalid UUID string; no source lookup is attempted. |
| `session_unresolved` | Session identity is absent under the selected policy; a local miss does not establish remote nonexistence. |
| `local_source_unavailable` | Required local directory/file is missing, or the local catalog cannot resolve a required source. |
| `source_unavailable` | Requested dataset cannot be found or materialized through the configured remote source. |
| `revision_unavailable` | Remote catalog has no dataset matching the explicit revision and other request filters. |
| `source_conflict` | Ambiguous dataset selection, mismatched session identity, or different collections/revisions in a joint load. |
| `source_inconsistent` | Materialized file fails its recorded size/hash check. |
| `access_failed` | Configuration, service, filesystem, catalog, or decoding failure. |

## Current consumers and coverage

`visual_stim_gen.py` and `prepare_data.py` construct `SessionAccess` and accept
`--access-policy {local-only,remote-allowed}` (default `remote-allowed`). Replay
uses the configured source-cache root; preparation passes `--base_path` as its
cache root. Programmatic replay, neural preparation, trial masking, and LFP
preparation take a `SessionAccess` instead of a ONE client.

Replay loads trials and wheel as separately versioned objects, jointly resolving
wheel positions/timestamps. It retains the recovered-parameter ALF table
fingerprint check, original trial row IDs, source dataset identities, and both
revision groups in `replay_metadata.json`.

Neural preparation performs channel interpretation, cluster enrichment, filtering,
and probe merging after source loading. It retains per-cluster probe/PID associations
and original trial IDs. Aligned-data provenance includes source identities for
trials, spikes, and optional LFP, plus the retained neuron/probe order.

`prepare_lfp(access, eid, ..., trials=None, return_sources=False)` can consume the
already loaded original trial table. It uses each insertion's stream conversion
and concatenates channels in probe iteration order, returning
`[original trial rows, LF samples, concatenated channels]`. With `return_sources=True`
it returns `(data, source_records)`. Remote policy permits streaming; local-only
requires full cached recordings. Nonfinite event times retain unavailable rows.
The existing LFP frequency band and preprocessing remain consumer responsibilities.

Replay and preparation retain per-session completion, skip, and failure reports.
Source failures propagate with their context before downstream merging or publication.
