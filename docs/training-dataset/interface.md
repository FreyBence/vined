# Training-dataset interface

## Sample construction

With `src/` on the Python path, `training_dataset` exposes scientific sample construction in `src/training_dataset/samples.py`:

```python
from training_dataset import AlignmentSource, SampleConfig, build_samples, load_samples

samples = load_samples(
    [AlignmentSource(alignment_path, expected_generation_id=generation_id)],
    config=SampleConfig(max_time_length=100, max_neuron_count=800),
)
```

- `load_samples(sources: Iterable[AlignmentSource], *, config: Optional[SampleConfig] = None)` verifies each explicitly selected generation through `alignment.load_alignment`. Select exactly one generation per session; no automatic discovery or fallback occurs.
- `AlignmentSource(path: str | Path, expected_generation_id: str | None = None)` identifies one alignment generation and optionally requires its exact identity.
- `build_samples(trials: Iterable[alignment.AlignedTrial], *, config: Optional[SampleConfig] = None)` accepts already aligned in-memory trials. Source-generation provenance is null rather than invented.
- `SampleConfig(max_time_length: int | None = None, max_neuron_count: int | None = None)` independently enables right padding on each axis. Each configured maximum is a positive integer. `None` preserves the source length for that axis. Padding fills counts/features with zero, physical timestamps with NaN, and padded model positions with -1.

Both functions return a nonempty tuple of `TrainingSample` objects sorted by `(session_id, trial_id)`. Inputs require canonical session UUIDs, unique original trial IDs, matching neural/visual time dimensions, valid complete-bin timing, and source/resampling provenance. Counts are nonnegative int64; features are normalized finite float32. Feature width follows the supplied array rather than a constant in dataset construction. In one session, the ordered unit population and bin size must agree across trials.

## Sample contract

Let `T` and `N` denote the original temporal and neuron counts, `P` and `Q` the respective output sizes after optional padding, and `D` the visual feature width.

| Field | Contract |
| --- | --- |
| `sample_id` | SHA-256 fingerprint of session/trial identity, representation version, and padding configuration; independent of enumeration order and source path. |
| `session_id`, `trial_id` | Original source identities. |
| `neural` | Int64 `[P,Q]` raw spike counts; the leading `[T,N]` cells equal the source. |
| `visual` | Float32 `[P,D]`; the leading `[T,D]` cells equal the source without normalization or interpolation. |
| `physical_timestamps`, `bin_start_times`, `bin_end_times` | Float64 `[P]` session-clock seconds; original bin centers/edges followed by NaN padding. |
| `temporal_positions` | Int64 `[P]`; original positions `0..T-1`, then -1 padding. These are not physical timestamps. |
| `temporal_mask` | Bool `[P]`; true for the original `T` aligned bins, false for padding. |
| `neuron_identity` | Copied complete ordered unit DataFrame with `N` real rows; row `n` identifies neural column `n`. No padded unit identities are created. |
| `neuron_mask` | Bool `[Q]`; true for the original `N` channels, false for padding. |
| `sequence_length`, `neuron_count` | Original `T` and `N`, including when either is one. |
| `stim_on`, `stim_off`, `aligned_start`, `aligned_end`, `bin_size`, `discarded_tail_duration` | Unchanged source timing/duration values in seconds. |
| `metadata` | Copied alignment provenance, source alignment selection, and representation/padding configuration. |
| `split` | `None` before assignment; `train`, `val`, or `test` after `assign_splits`. |

Zero counts are valid observations when both masks are true. Numeric padding values never determine validity. Unit identity retains session, recording/probe, collection/revision, and source-unit scope; nullable insertion IDs and revisions are preserved. Optional unit metadata does not select, sort, or remove channels.

`metadata["alignment"]` preserves the complete source `alignment_metadata`. `metadata["source_alignment"]` is either null for in-memory construction or a dictionary containing the verified `generation_id`, absolute `path`, and alignment `schema_version`. `metadata["construction"]` records `representation="aligned-trial-v1"` and the padding configuration.

Samples are defensive copies of source arrays, unit tables, and alignment metadata. The dataclass fields are frozen, but contained arrays/tables/dictionaries remain mutable caller-owned values.

## Split assignment

`src/training_dataset/splits.py` exposes:

```python
from training_dataset import SplitConfig, assign_splits

dataset = assign_splits(
    samples,
    config=SplitConfig("within_session", ratios=(0.7, 0.1, 0.2), seed=42),
)
train_samples, validation_samples, test_samples = dataset.train, dataset.val, dataset.test
```

`assign_splits(samples: Iterable[TrainingSample], *, config: SplitConfig) -> DatasetSplits` accepts nonempty unassigned samples with unique sample IDs. Different sample representations of the same `(session_id, trial_id)` are grouped together; padding configuration and input enumeration do not influence assignment. Samples in each returned tuple are ordered by session ID, original trial ID, then sample ID.

`SplitConfig` accepts:

| Field | Contract |
| --- | --- |
| `strategy` | Required: `within_session` or `session_held_out`. |
| `ratios` | Optional train/validation/test ratios, in that order; finite nonnegative numbers summing to one. Required for ratio-based assignment. |
| `seed` | Explicit nonnegative int64-compatible integer, required for ratio-based assignment. |
| `session_assignments` | Optional mapping from canonical session UUIDs to `train`, `val`, or `test`. Supported only for `session_held_out`, instead of ratios and seed. Must name exactly the sessions retained after exclusions. |
| `exclusions` | Mapping from `(session_id, trial_id)` to a nonempty reason, default empty. All representations of each named source trial are excluded; absent source identities are errors. |

`within_session` apportions source trial groups separately for each session. `session_held_out` assigns whole retained sessions, either through explicit assignments or ratio-based allocation. Ratios apply to group counts, so held-out session allocation need not produce those same ratios of individual samples.

Ratio-based assignment ranks stable group identities using SHA-256 over strategy, seed, and identity. Counts use largest-remainder rounding with ties in train/validation/test order; when needed, a group is transferred from the largest split to populate a positive-ratio split. Each session in within-session mode, or the total retained sessions in held-out mode, must provide at least as many groups as there are positive ratios. Otherwise assignment fails rather than returning an unexpectedly empty requested split. Zero-ratio splits remain empty. Explicit assignments may intentionally leave splits empty.

`DatasetSplits` contains `train`, `val`, and `test` tuples plus JSON-serializable `metadata`. Metadata retains the complete split configuration, its fingerprint, algorithm version, grouping scope, ordered per-split membership records (`sample_id`, `session_id`, `trial_id`), and excluded sample records with reasons. Each returned sample carries its `split` and `metadata["split"] = {"name": ..., "configuration_id": ...}`.

Assignment copies sample metadata and sets membership without altering source samples. Scientific arrays and unit tables are shared with the supplied samples; splitting does not transform or copy their contents. The same seed/configuration/source identities produce identical memberships under reordered input. Changing the available groups can change ratio-based memberships.

Invalid split configuration, duplicate sample IDs, previously assigned samples, unknown exclusions, insufficient groups, incomplete session assignments, or exclusion of every sample raise `ValueError`. Unsupported sample/config types raise `TypeError`. No partial result is returned.

## Dataset publication and loading

`src/training_dataset/artifacts.py` exposes:

```python
from training_dataset import generate_dataset, load_dataset

generation = generate_dataset(
    [AlignmentSource(alignment_path, expected_generation_id=alignment_id)],
    output_directory,
    split_config=SplitConfig("within_session", ratios=(0.7, 0.1, 0.2), seed=42),
    sample_config=SampleConfig(max_time_length=100),
)
loaded = load_dataset(generation.path, expected_generation_id=generation.generation_id)
dataset = loaded.dataset
```

- `generate_dataset(sources, output_dir, *, split_config, sample_config=None)` verifies explicit alignment generations, constructs samples, assigns splits, and publishes a dataset.
- `publish_dataset(dataset: DatasetSplits, output_dir)` publishes an already constructed and assigned dataset after validating scientific arrays, masks, identities, provenance, population consistency, deterministic memberships, and exclusions.
- `load_dataset(path, *, expected_generation_id=None)` loads exactly one complete generation, checking its manifest fingerprint, exact payload accounting, file hashes, array schemas/dtypes/shapes, padding, unit associations, split assignments, and configuration. It does not reopen alignment artifacts or require them to remain available.

All three return `DatasetGeneration(generation_id, path, dataset, manifest)`, with an absolute `Path`, a `DatasetSplits` object, and JSON manifest data. Generation identity is a SHA-256 fingerprint of the manifest excluding its own identity, binding payload hashes, scientific metadata, construction/split configuration, and implementation source/package versions. Implementation versions are recorded as provenance; loading does not require the original source checkout or package versions to remain unchanged.

Publication writes a temporary sibling directory, performs verified consumer readback, checks that implementation sources remained unchanged during publication, then renames it to the final generation directory. Existing destinations raise `FileExistsError`; reuse requires explicit loading. Ordinary failures remove staging artifacts, and no loader discovers unfinished directories automatically.

### Persistence schema 1

```text
<output_dir>/generations/<generation_id>/
    manifest.json
    units/<session_id>.parquet
    samples/000000.npz
    samples/000001.npz
    ...
```

The manifest has `schema_version=1`, `kind="training_dataset"`, `complete=true`, `generation_id`, sorted `sessions`, ordered `samples`, `dataset_metadata`, `implementation`, and `files`. Sample entries store all scalar `TrainingSample` fields, split labels, and complete sample metadata. `dataset_metadata` preserves the split configuration, fingerprint, algorithm, grouping scope, memberships, and exclusions. Construction/padding configuration and alignment generation/configuration provenance remain in sample metadata.

Samples are stored in train/validation/test order and in canonical sample order within each split. Each NPZ contains exactly `neural`, `visual`, `physical_timestamps`, `bin_start_times`, `bin_end_times`, `temporal_positions`, `temporal_mask`, and `neuron_mask`, preserving their dtypes and values including NaN timestamp padding. Loading disables pickle. Each session's complete ordered unit table is stored once as Parquet; no padded neurons are serialized as real units.

JSON provenance stores arrays/tuples as lists, paths as strings, and unavailable/nonfinite metadata values as null. Loading restores resampling times/weights as float64 arrays and source indices as int64 arrays. The returned manifest remains JSON data.

Malformed, incomplete, hash-mismatched, or inconsistently accounted generations and unexpected generation identities raise `ValueError`. Missing files raise filesystem errors. No partial dataset is returned.

### Creation command

From the checkout root, with the project Python:

```text
python src/create_dataset.py --alignment-generation PATH --split-strategy within_session --split-ratios 0.7 0.1 0.2 --split-seed 42
```

`bash script/create_dataset.sh` forwards the same arguments using the project environment. Repeat `--alignment-generation` for multiple sessions. Optional repeated `--expected-alignment-generation-id` values must match the number and order of input paths. `--output-dir` defaults to `<VINED_DATA_DIR>/training-dataset`; optional `--max-time-length` and `--max-neuron-count` enable padding without truncation.

For explicit session-held-out membership, use `--split-strategy session_held_out --session-assignments FILE`, with a JSON object mapping every retained session to `train`, `val`, or `test`, instead of ratios/seed. `--exclusions FILE` accepts a JSON list of objects containing `session_id`, integer `trial_id`, and `reason`; duplicate or absent exclusion identities fail.

Success prints generation ID, absolute path, sessions, and split sizes as JSON and exits 0. Input/publication failures report the error and exit 1; CLI argument errors exit 2. The command has no model/trainer configuration dependency or acquisition/network behavior. The old positional `COUNT EID` launcher and objective/masking options are retired.

## Training and evaluation handoff

`load_dataset_splits(path, *, session_ids=None, expected_generation_id=None)` in `training_dataset.handoff` verifies one complete generation and returns four values:

```python
train, validation, test, metadata = load_dataset_splits(
    dataset_path, expected_generation_id=dataset_id,
)
```

Each split is a `PersistedSplit` with immutable membership/order and caller-owned `TrainingSample` values accessible by integer index. Empty persisted splits remain empty. Optional session selection must name unique sessions present in the generation; selection never recomputes memberships. Metadata provides `num_neurons`, `num_sessions`, sorted `eids`, `eid_list` mapping session IDs to real neuron counts, `dataset_generation_id`, `dataset_path`, `selected_session_ids`, and the full persisted `dataset_metadata`.

The existing dataset-owned `utils.dataset_utils.load_ibl_dataset` exposes the same four-value result when `cache_dir` or `aligned_data_dir` names a generation directory containing `manifest.json`, or when `dataset_generation=path` is supplied explicitly. Optional `expected_dataset_generation_id` verifies the selection. Use `split_method="predefined"`; runtime session-assignment lists and resplitting are rejected. `eid` selects one session; otherwise all persisted sessions are retained and `num_sessions` must accommodate them. `use_re`, `split_size`, and `seed` do not alter scientific dataset membership. Legacy paths without a generation manifest remain isolated in the older Hugging Face/cache branch; failed scientific loading never falls back to it.

Existing training/evaluation commands can supply the explicit dataset generation directory as their data path:

```python
from utils.dataset_utils import load_ibl_dataset
from loader.make_loader import make_loader

train, validation, test, info = load_ibl_dataset(
    dataset_path, dataset_path, num_sessions=2, split_method="predefined",
)
loader = make_loader(
    train, batch_size=8, target=["vision-clip"], mode="train",
    max_time_length=100, max_space_length=max(info["eid_list"].values()),
    eids=info["eids"], pad_value=-1.,
)
```

`BaseDataset` recognizes a verified `PersistedSplit` before consulting legacy `data_dir` caches. Derived `ibl_mm` cache paths supplied by existing callers are unused for this view. Its split and explicit EID selection must agree with the view. Source neuron order is authoritative: depth/region sorting, region filtering, and left padding are rejected. Optional metadata loading never removes channels, and both scientific modalities remain present regardless of prediction direction.

`model_sample(sample, *, max_time_length, max_space_length, pad_value=-1.)` adapts one sample to these field names. Runtime maxima can add padding or remove previously stored padding, but cannot shorten real observations or neurons. Physical grids and values are copied unchanged; no interpolation, normalization, or resplitting occurs. Source `sample_id` and construction metadata continue to identify the persisted scientific sample rather than the runtime tensor layout.

| Model-facing field | Contract |
| --- | --- |
| `spikes_data` | Float32 `[P,Q]` compatible model input; exact count conversion is required. Counts that cannot be represented exactly fail explicitly. |
| `vision-clip` | Float32 `[P,D]` unchanged real features with configured numeric padding. |
| `time_attn_mask`, `space_attn_mask` | Int64 validity masks; one for real temporal positions/channels, zero for padding. |
| `vision-clip_valid` | Bool `[P]`, matching real aligned temporal observations. |
| `spikes_timestamps`, `spikes_spacestamps` | Int64 model indices `0..P-1` and `0..Q-1`; physical time and biological identity are separate fields. |
| `neural`, `visual` | Exact scientific int64 counts and float32 features, with zero padding and explicit masks. |
| `temporal_positions` | Real indices followed by -1 for padded positions. |
| `physical_timestamps`, `bin_start_times`, `bin_end_times` | Float64 session timestamps followed by NaN padding. |
| `temporal_mask`, `neuron_mask` | Bool scientific validity masks. |
| `neuron_identity`, `metadata` | Full source unit table and sample provenance. |
| `neuron_depths`, `neuron_regions` | Optional source depths/acronyms; unavailable metadata is NaN/empty string, with no inferred anatomy. |
| `eid`, `session_id`, `trial_id`, `sample_id`, `split` | Explicit source and membership identities. |

True sequence/neuron counts, stimulus/aligned bounds, bin duration, discarded tail, and aligned `intervals` are also exposed. The legacy `bin_size` loader option does not override the persisted scientific bin duration.

`make_loader` retains standard PyTorch tensor collation and its existing sampler behavior. Scientific unit tables and provenance dictionaries travel as lists of per-sample sidecars rather than being coerced into tensors or collated across sessions. Tensor outputs have leading batch axis `B`. Empty selected splits retain the existing `make_loader` explicit empty-dataset error.

For direct evaluation metadata access, `PersistedSplit` supports `cluster_regions`, `cluster_uuids`, `eid`, `intervals`, `sample_id`, `trial_id`, `sequence_length`, `neuron_count`, and `split` column lookups. `cluster_uuids` uses recorded UUIDs when complete; otherwise it exposes stable `unit-<SHA-256>` labels derived from the full scoped unit identity, not invented biological UUIDs. Unit-level evaluation callers that assume one population should select an explicit EID. Arbitrary Hugging Face columns/methods and legacy behavioral modalities are not provided by this view.

## Errors and compatibility

Malformed or duplicate aligned trials, inconsistent session populations, invalid padding configuration, or maxima smaller than source dimensions raise `ValueError`. Unsupported trial/config types raise `TypeError`. Alignment loader verification and filesystem errors propagate. No partial sample tuple is returned and no failed source is skipped.

This API constructs, splits, persists, and exposes scientific datasets without applying prediction objectives. The verified split adapter supports existing training/evaluation loading calls, while legacy rows/caches remain a separate compatibility path. Existing orchestration that supplies the former positional creation arguments must migrate to explicit alignment selections; dataset roots and generation collections are not automatically searched for a generation.
