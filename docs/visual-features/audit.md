# Visual features audit

## Summary

The existing extractor provides batched, fixed-weight CLIP inference and compressed numeric feature archives, but remains attached to the legacy replay contract. It cannot consume the current replay artifacts or observation stream. Additional gaps concern spatial preprocessing, preservation of coverage and trial outcomes, and identities exposed by feature persistence/loading.

Comparison uses `spec.md`, `../dependencies.md`, and `../visual-replay/interface.md`. No visual-features architecture or interface document exists. Findings are based on source inspection, including installed processor source; no end-to-end CLIP inference was executed.

## Findings

### A01 — Current replay outputs cannot enter feature extraction

**Priority:** High

**Confidence:** Confirmed

**Category:** Pipeline

**Location:** `src/prepare_visual_stim.py:27` (`iter_frame_batches`), `:215` (`main`); `script/prepare_visual_stim.sh`.

**Finding:** The extractor opens `<video_dir>/<eid>/replay_metadata.json` and flat `trial_*.mp4` or legacy `trial_*.npz` sidecars. Current replay publication instead exposes a generation manifest, trial observation records, and compressed RGB images through `ReplayArtifactReader`, or observations through `ReplayStream`. The wrapper also points at the replay root rather than selecting a run. Neither supported frame-source option consumes the new boundary. The `render` option imports `ReplayConfig`, `create_grating_patch`, and `render_trial_frame` from the legacy renderer and reconstructs images inside feature extraction; it is not a mouse-view observation adapter.

**Expected:** Specification §§2–3 require public replay observations, intended mouse-perspective input, and extraction without video or renderer-internal access. Consume and verify replay completion before publishing derived features.

**Suggested disposition:** Replace the input adapter and renderer-internal path while retaining reusable encoder and bounded persistence code. The legacy paths are CLI-reachable, not proven dead code.

### A02 — Preprocessing does not enforce preservation of the supplied field of view

**Priority:** High

**Confidence:** Confirmed

**Category:** Spec

**Location:** `src/prepare_visual_stim.py:151` (`processor(...)`), `:197` (processor loading).

**Finding:** Images are passed directly to the downloaded CLIP processor without an explicit aspect-preserving spatial adaptation or rejection of cropping configurations. `padding=True` is not an implemented image letterboxing policy. The installed `CLIPImageProcessor` defaults to shortest-edge resizing and square center cropping. With that configuration, a 4:3 image loses horizontal peripheral content. The saved processor configuration makes this behavior inspectable but does not prevent it. The exact downloaded model snapshot configuration was not executed during this audit; actual stimulus removal depends on that configuration and image content.

**Expected:** Specification §4.2 requires retaining the supplied mouse-view spatial extent and proportions, with reproducible image preparation. A concrete adaptation policy remains an architecture/design choice; this audit does not prescribe one.

**Suggested disposition:** Refactor image preparation to enforce the accepted spatial policy before inference.

### A03 — Subsampling and archive structure discard coverage and unavailable-trial information

**Priority:** High

**Confidence:** Confirmed

**Category:** Data integrity

**Location:** `src/prepare_visual_stim.py:48–51`, `:109–115`, `:238–250`, `:295–306`; `src/utils/visual_data.py:25–46`.

**Finding:** Only selected timestamps and validity booleans survive sampling. Invalidity reasons, source frame indices, and stimulus-state distinctions present even in legacy sidecars are discarded. An invalid interval between two selected valid frames therefore disappears from the output. There are no persisted requested domains, continuity boundaries, blank-content flags, or trial outcomes. Trials marked invalid in the input manifest are filtered out; the loader requires strictly increasing offsets, so zero-feature trials cannot be represented in its current trial sequence structure. Alignment receives insufficient information to distinguish unavailable trials or gaps from absent selections.

**Expected:** Specification §§3.1 and 5 require preserving coverage limitations independently of temporal selection, valid blanks separately from unavailable/invalid content, and explicit outcomes for requested trials with no usable features.

**Suggested disposition:** Refactor selection and feature archive/reader contracts to retain source coverage and outcomes alongside sampled features.

### A04 — Feature persistence and loading do not expose sufficient generation identity

**Priority:** High

**Confidence:** Confirmed

**Category:** Interface

**Location:** `src/prepare_visual_stim.py:252–259`, `:295–317`; `src/utils/visual_data.py:16–46` (`load_archive`).

**Finding:** Persistence records a legacy manifest hash, contract fingerprint, encoder revision, and preprocessing provenance, but no current replay generation ID, source trial-table fingerprint, selected observation IDs, or explicit completed feature-generation identity. A legacy manifest/configuration hash is not verification of the actual consumed video pixels. More directly, `load_archive` neither requires nor returns the saved model, revision, or provenance: an archive stripped of all those fields can pass validation if its numeric arrays and EID match. Consumers receive only trial IDs, timestamps, values, masks, and a skip flag, preventing representation/source compatibility checks through this loader. Atomic replacement protects against partial publication, but the EID-only output filename is not a generation identity.

**Expected:** Specification §§4.1 and 6 require a complete representation definition, content-associated source identities, a completed feature generation, and enough information exposed on loading for downstream compatibility and reproducibility.

**Suggested disposition:** Refactor persistence and loading together; retain existing numeric validation and atomic publication.

### A05 — Implemented feature boundary has no component interface documentation

**Priority:** Medium

**Confidence:** Confirmed

**Category:** Interface

**Location:** `docs/visual-features/interface.md` (absent); `src/prepare_visual_stim.py`; `src/utils/visual_data.py`.

**Finding:** There is no component interface describing supported callers, source selection, representation, archive fields, load results, or failure/completion semantics. The current shared utility also contains alignment resampling, making source-file location alone insufficient to identify ownership. This is a documentation gap, not evidence that all functions in that file belong to visual-features or should be moved.

**Expected:** Repository interface rules require documentation of the completed public boundary, compatible with the component specification and directly connected consumers.

**Suggested disposition:** Wrap / centralize the public contract documentation after implementation resolves the required boundary; do not document legacy behavior as specification compliance.

## Conforming areas

- Encoder and processor load from the same immutable resolved model revision; model files, processor configuration, package versions, sampling configuration, and extraction sources are recorded during writing.
- `model.eval()` and `torch.no_grad()` provide inference behavior. Projected image features are L2-normalized, with width, finiteness, and valid-vector nonzero checks before publication.
- Extraction uses bounded image batches and disk spools; it does not retain all session images in memory. Compressed numeric archives load without pickle.
- Existing sampled timestamps retain float64 source-session values. Original trial IDs are retained for included trials, including across filename sorting and batches.
- Variable-length sequences use offsets without model padding. Matching the requested sample rate to native FPS can select the full supported regular sequence; a separate full-sequence mode is not inherently required for that legacy case.
- Archive writing uses a temporary location, verifies ZIP integrity, and replaces the destination only after successful extraction. A failed run does not publish its partial spool as a completed archive.
- Feature extraction does not call neural resampling. `resample_features` in the shared utility is alignment functionality and is outside this audit's implementation scope.
