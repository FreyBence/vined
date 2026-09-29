# Visual features interface

## Observation input and selection

`visual_features.ObservationSelection(replay, *, sample_fps=None)` accepts a
fresh, pending public `ReplayStream` or `ReplayArtifactReader`. It requires
`image_space="mouse_view"`; display-only input raises `ValueError`. Saved NPZ
and legacy NPY decoding and content verification remain owned by the replay
reader. No video, renderer-internal access, CLIP model, or session acquisition
is required by this adapter.

`sample_fps=None` selects every source observation. A finite positive numeric
rate selects the first source observation at or after successive targets,
anchored at the first source observation in each trial. Targets passed by one
observation are coalesced: an observation is never duplicated, including when
the requested rate exceeds the supplied cadence. Selection is independent of
validity and batch boundaries and does not invent observations at target times.
Boolean/non-numeric rates raise `TypeError`; nonpositive/nonfinite rates raise
`ValueError`.

Iteration yields:

- `FeatureObservation(metadata, selected, rgb)` for **every** source observation.
  `metadata` is a defensive copy of the complete source observation metadata,
  including original identities, actual session timestamp, timing classification,
  status/reason, blank flag, image-space/format, and pixel hash. `selected` is an
  independent boolean selection decision. `rgb` is the source immutable
  `uint8[height, width, 3]` RGB array only for selected valid observations;
  otherwise it is `None`. An unselected valid observation remains valid in its
  metadata. A selected invalid observation has no pixels; it must not become a
  valid blank or zero-vector embedding.
- Public replay `TrialOutcome` records, with defensively copied metadata, for
  every requested trial, including zero-image trials. Requested domains,
  coverage partitions, reasons, timing, and upstream counts are unchanged.
  These counts describe replay observations, not selected feature counts.

Unselected metadata and trial outcomes must be consumed as well as selected
images: these preserve gaps and continuity boundaries between selected samples.
The adapter itself does not accumulate images or records. A consumer controls
batch memory and must retain the metadata needed by its feature publication.

## Identity and lifecycle

`definition` and `definition_id` expose the upstream reconstruction definition;
`selection` describes the configured sampling policy. `completion` is `None`
until successful iterator exhaustion, then returns a defensive copy of the
unchanged upstream completion, including generation and source-table identity.
It is not a feature-generation completion record. Accounted partial/failed
reconstructions remain partial/failed; they are not promoted to success.

`state` is `pending`, `running`, `completed`, or `interrupted`. The single-use
adapter owns and closes its source. Use it as a context manager or call `close()`
when stopping early. Interrupted iteration or source validation failure leaves
completion absent. Receiving the final trial outcome is insufficient: iterate
to exhaustion before finalizing derived output. Replay reader errors propagate.

```python
from visual_replay import ReplayArtifactReader, TrialOutcome
from visual_features import ObservationSelection, FeatureObservation

with ObservationSelection(ReplayArtifactReader(generation_directory),
                          sample_fps=5) as observations:
    for item in observations:
        if isinstance(item, FeatureObservation):
            # Retain metadata even when unselected or without pixels.
            if item.selected and item.rgb is not None:
                pass  # Supply pixels and their associations to the encoder.
        elif isinstance(item, TrialOutcome):
            pass  # Retain coverage and the original trial outcome.
    completion = observations.completion
```

## CLIP encoding

`ClipEncoder(model_name="openai/clip-vit-large-patch14", *, revision="main",
device=None)` resolves the model revision to an immutable commit and loads model
and image processor from that snapshot. The default device is CUDA when
available, otherwise CPU. The supported model has a square image input and
768 projected features. Weights are frozen; execution uses evaluation and
inference modes. Loading/download failures propagate.

`prepare_image(rgb, size, *, fill=(128,128,128))` accepts nonempty RGB8 pixels
and returns a square PIL RGB image. It fits the entire source image with one
common scale, rounds each dimension to the nearest integer (minimum one pixel),
resizes with Pillow bicubic interpolation, and centers it on neutral padding.
Odd padding places the extra pixel at the right/bottom. Aspect ratio is retained
up to raster rounding. No source content is cropped. This is feature input
preparation, not a change to replay geometry or a model of the environment.

`ClipEncoder.encode(frames)` accepts a nonempty list of RGB8 arrays and returns
`float32[B,768]` projected image features, L2-normalized per observation. The
processor's resizing and center cropping are disabled after full-view fitting;
RGB pixels are rescaled by `1/255` and normalized with the snapshot's image
mean/std. Nonfinite, incorrectly shaped, or zero/near-zero projected vectors
raise `FeatureExtractionError`; unusable vectors are not returned as features.
Preprocessing and inference failures use the same distinct extraction exception.

`encoder.provenance` returns a defensive copy containing immutable model
revision, snapshot file hashes, exact extracted output, width/dtype/normalization,
effective spatial preparation and processor settings, device, package versions,
and implementation hashes.

`iter_encoded_observations(observations, encoder, *, batch_size=32)` takes a fresh
`ObservationSelection` and yields records in source order, flushing before each
trial outcome. At most `batch_size` observation records are buffered. It yields:

- `EncodedObservation(metadata, selected, feature, status)`, retaining the full
  source metadata. Status is `encoded`, `not_selected`, or `source_unavailable`.
  The last category retains the exact upstream invalid/unavailable/failed status
  and reason in metadata. Only `encoded` records have an immutable float32
  `[768]` feature. Selected valid blanks are encoded normally.
- Unchanged `TrialOutcome` records, including trials without any features.

The generator owns its observation selector while iterating; close it explicitly
if stopping early (for example with `contextlib.closing`). Extractor failures
propagate rather than becoming upstream invalidity or valid placeholder vectors.
Consume the generator fully and inspect `observations.completion` before future
feature publication. No completed feature-generation identity or archive is
issued here.

## Feature publication

`write_features(observations, encoder, output, *, batch_size=32)` consumes a fresh
observation selector with the encoder and returns the completed artifact manifest.
It retains metadata for every source observation and every original trial outcome,
including zero-feature trials, alongside the selected valid feature vectors.
Vectors are spooled to disk; all session images or features need not fit in RAM.

Publication requires upstream iterator exhaustion and verified completion. A
temporary compressed archive is read back through the public reader before a
same-filesystem hard link atomically publishes the requested file. Existing
destinations raise `FileExistsError`; use a different output directory for a new
generation. There is no automatic reuse, overwrite, or fallback. Errors propagate
without publishing a partial destination. Temporary files are cleaned on ordinary
exceptions; forced process termination can leave temporary directories, which
are not published outputs. The output filesystem must support hard links.

An accounted partial/failed replay may produce a completed feature archive with
the unchanged upstream status and zero-feature outcomes. Feature completion does
not assert complete visual coverage. Encoder failures abort publication and are
distinct from recorded upstream invalidity.

### Persistence contract

`<eid>_visual_clip.npz` is a ZIP/DEFLATE archive with four members:

- `definition.json`: schema version 1, kind `visual_feature_definition`, complete
  replay definition and its ID, selection policy, encoder/preprocessing provenance,
  extraction batch size, and writer implementation hash.
- `records.jsonl`: ordered records for every replay observation and trial outcome.
  Observation records contain unchanged source `metadata`, boolean `selected`,
  extraction `status`, and `feature_index` (zero-based row or null). Trial records
  contain the complete unchanged source outcome, including domain and coverage.
- `features.npy`: little-endian float32 `[F,768]`, NPY format 2.0, in encoded-record
  order. `F` counts selected valid images; it can be zero. No invalid placeholders
  or model padding are included.
- `manifest.json`: schema version 1, kind `visual_feature_artifacts`, definition
  ID, complete upstream replay completion, total feature count, ordered per-trial
  feature/selection counts, SHA-256 of uncompressed record bytes and feature payload
  bytes (excluding NPY header), and `generation_id`.

Identities use SHA-256 of sorted compact ASCII-escaped JSON with nonfinite numbers
disallowed. `generation_id` hashes the manifest excluding itself. It binds source
completion, representation, selection, record membership, and actual feature bytes;
it is independent of the output filename. Trial/observation metadata retain source
session seconds, original IDs, source-table fingerprint, validity/blank distinctions,
timing classification, and support/coverage semantics. Feature indices never replace
source observation IDs.

## Verified loading and reuse

`FeatureArtifactReader(path, *, eid=None, replay_generation_id=None,
representation=None, selection=None)` is a single-use context-managed iterator.
Supplied compatibility arguments must exactly match the stored values; omitted
arguments mean explicit inspection of that archive, not approval for a different
requested extraction. For reuse, compare the requested source, representation,
and selection through these arguments and exhaust the reader before accepting it.

`definition` and `artifact_manifest` return defensive copies. The reader yields
the same `EncodedObservation` and `TrialOutcome` types as extraction, with immutable
feature vectors, and validates shape/type, unit norm, observation associations,
trial counts, member hashes, and the original replay record digest. Every trial
outcome and all unselected metadata remain accessible without renderer internals.
Iteration uses bounded reads rather than loading the complete feature array.

`completion` remains absent until successful exhaustion, then returns the verified
feature manifest. Reading a manifest header is not verification of its payload.
`state`, context-manager use, and early `close()` follow the input selector's
lifecycle. Missing files raise filesystem errors; malformed or incompatible data
raise validation/decoding errors. Legacy numeric feature archives are rejected.

```python
from visual_features import FeatureArtifactReader

with FeatureArtifactReader(path, eid=eid,
                           replay_generation_id=expected_replay_id) as features:
    for item in features:
        pass  # Consume both observations and trial outcomes, preserving metadata.
    completion = features.completion
```

## Extraction CLI

```bash
# Explicit run, one EID, explicitly subsample to 5 Hz:
bash script/prepare_visual_stim.sh --eid EID --replay-dir output/visual_replays/RUN --sample-fps 5

# All published replay sessions in a run, all source observations:
bash script/prepare_visual_stim.sh --replay-dir output/visual_replays/RUN
```

The equivalent entry point is `python src/prepare_visual_stim.py`. The shell wrapper
forwards named CLI options without overriding paths. Without `--eid`, sessions
are discovered from published replay manifests rather than `data/eids.txt`.
With `--eid EID`, only that session is processed. The old positional count/list
form and `--eids-file`/`--n-sessions` are no longer supported.
Defaults follow `VINED_REPLAY_DIR`
and `VINED_VISUAL_DIR` (normally `output/visual_replays` and
`output/datasets/vis_stim`). `--output-dir` selects a fresh output location.

`--replay-dir` accepts an explicit session generation, a run containing EID
directories, or a root with exactly one candidate per requested EID. Multiple
candidates are an error; select a run explicitly instead of relying on a newest-run
heuristic. Automatic discovery considers published manifests at the selected
directory, one level below it, or two levels below it. Unpublished directories
are excluded from automatic discovery; no published replays produces an error.
Explicit EID selection still rejects ambiguous or unfinished session directories.
The chosen source is printed and its EID is verified.

`--sample-fps` defaults to no subsampling; `--batch-size` defaults to 32.
`--clip-model`, `--clip-revision`, and `--device` configure the encoder. A revision
name resolves once when the encoder is loaded, and that encoder is shared across
the selected sessions. An immutable SHA can reproduce a chosen model revision.
The legacy `--video_dir` and `--frame-source` modes are no longer supported.

Exit codes are 0 for successful replay/extraction across all sessions, 2 when
published outputs retain partial/failed replay outcomes (also used by CLI argument
errors), and 1 for extraction, input, or publication errors. Ordinary session
failures are reported and later sessions continue. Outputs already successfully
published for other sessions are retained.

## Downstream compatibility boundary

The new archive intentionally does not masquerade as the old schema-2 arrays.
`utils.visual_data.load_archive` rejects it with an instruction to use
`FeatureArtifactReader`. The current `prepare_data.py` also rejects it because
its legacy top-level provenance field is absent. Those consumers do not yet handle
coverage partitions or generation identity. Alignment migration is separate work;
neural resampling and dataset construction are unchanged. Existing legacy files
are not converted or silently replaced.
