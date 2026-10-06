# Training-dataset audit

## Summary

The NEDS-derived dataset helpers and immutable cache machinery are partially reusable, but the current builder cannot consume the supported alignment generation. Required variable-length samples, physical timing, neuron identity, and reproducible split construction are incomplete. No current training-dataset interface or architecture document exists; comparison uses `spec.md`, `docs/dependencies.md`, and `docs/alignment/interface.md`.

Findings are based on source inspection and direct execution of existing functions with small in-memory inputs. No end-to-end dataset generation was demonstrated.

## Findings

### A01 — Dataset creation does not consume the supported alignment output

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:** `src/create_dataset.py:91–128`; `src/utils/dataset_utils.py:228–394`; `docs/alignment/interface.md`, standalone publication and loading.

**Finding:** The builder loads already split Hugging Face datasets from `<eid>_aligned` and requires their `provenance.json`. Alignment instead exposes `load_alignment(...).trials` and verified schema-1 generations with `manifest.json`, `units.parquet`, and per-trial NPZ payloads. There is no dataset adapter for these artifacts, and splitting is assumed to have happened upstream. The sparse helper also requires legacy lab/subject/cluster metadata rather than the supplied aligned-trial contract.

**Expected:** Consume verified aligned trials, preserve their data and metadata, and own dataset splitting and packaging (spec sections 2–4, 17, 28–32).

**Suggested disposition:** Wrap / centralize

---

### A02 — Temporal padding rejects valid variable-length trials

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Runtime

**Location:** `src/loader/base.py:435–483, 516–546, 572–582`.

**Finding:** `_prepare_target_behavior` pads vision before `_preprocess_ibl_data` compares its length with unpadded spikes. A two-bin sample with a configured length of four fails with `AssertionError: Spike/vision mismatch: 2 vs 4`. Trials shorter than the default maximum therefore cannot be cached. The sample also lacks an explicit original sequence length. Oversized visual sequences fail explicitly, which should be retained; oversized neural axes can otherwise produce negative padding lengths and incorrectly sized masks.

**Expected:** Compare original aligned shapes, retain true sequence length, and expose masks matching the padded arrays; incompatible maxima must fail explicitly (spec sections 8–11, 14, 24).

**Suggested disposition:** Refactor

---

### A03 — Cache samples discard physical time and alignment metadata

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/loader/base.py:483–510`; `src/create_dataset.py:108–129, 131–152`.

**Finding:** Samples return positional `arange` values as `spikes_timestamps` and only retain legacy `intervals`. They omit physical bin centers/edges, stimulus bounds, true aligned bounds, bin size, and alignment metadata. A hardcoded `bin_size=0.05` cache option is not derived from the source. Cache manifests fingerprint legacy provenance and files but do not restore the missing scientific fields to samples.

**Expected:** Preserve physical bin-center timestamps independently of model positions, source timing and alignment provenance, and actual bin duration (spec sections 3, 7, 24, 26, 28, 30).

**Suggested disposition:** Refactor

---

### A04 — Neuron identity is lost and optional metadata can change channel content

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/loader/base.py:461–510, 553–570`.

**Finding:** Neural columns are represented by local positional indices; the aligned unit table and legacy cluster UUID/channel identities are not returned or cached. Sorting reorders counts without preserving the corresponding stable unit identities. With `load_meta=False`, placeholder metadata has length one and the default sorting path selects only column zero, silently dropping every other neuron. Unqualified `squeeze()` also collapses a one-bin or one-neuron matrix and can make later two-dimensional indexing fail.

**Expected:** Preserve `[T,N_session]` and the complete ordered, session-scoped neuron identity; metadata loading must not remove neurons, and permutations must apply equally to counts and identities (spec sections 6, 13–14, 24, 30).

**Suggested disposition:** Refactor

---

### A05 — Required split construction is incomplete and order-dependent

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:** `src/utils/dataset_utils.py:203–250, 255–394`; `src/create_dataset.py:91–102, 154–166`.

**Finding:** The active builder only accepts predefined upstream memberships. `random_split` and `session_based` do not initialize `val_dataset` or `meta_data` before the four-value return; the session-based branch also constructs paths from already constructed directory strings. The `aligned_data_dir` branch returns only three values. Discovery uses unsorted `os.listdir` and session slicing, and random assignment uses row positions, so an explicit seed alone does not make selection or assignment invariant to source order. Held-out exclusions compare unsuffixed paths against `_aligned` directory paths, so they do not reliably exclude the same session. No working identity-based within-session and session-held-out construction boundary persists its own strategy, ratios, seed, and memberships.

**Expected:** Deterministic sample/session grouping, explicit persisted split configuration, no held-out leakage, and a consistent usable return contract (spec sections 5, 17–22, 27–29).

**Suggested disposition:** Refactor

---

### A06 — Failed session loads can silently produce a partial dataset

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/utils/dataset_utils.py:324–358`; `src/create_dataset.py:118–129`.

**Finding:** The predefined branch catches every session-loading exception, prints it, and continues without a structured exclusion policy or record. Split datasets are appended before subsequent metadata checks complete, so a late failure can leave rows whose session is absent from `eid_list`. Earlier failures can instead omit requested sessions while the builder prepares manifests only for those retained in `eid_list`.

**Expected:** Invalid source data must fail clearly or be excluded through a deterministic, recorded policy; requested, retained, and excluded identities must remain accountable (spec sections 23, 27–28).

**Suggested disposition:** Refactor

---

### A07 — Sparse serialization narrows neural counts to eight bits

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/utils/dataset_utils.py:43–49, 57–69`.

**Finding:** `get_sparse_from_binned_spikes` converts counts to `np.ubyte`. Direct execution with counts `[[256,0],[0,1]]` yields sparse data `[0,1]`, changing 256 to zero. This cannot safely serialize the upstream raw int64 count contract even though lower-count examples may appear correct.

**Expected:** Preserve aligned neural values without silent narrowing or overflow (spec sections 3, 13, 30; alignment interface neural activity contract).

**Suggested disposition:** Refactor

---

### A08 — Dataset construction depends on downstream model configuration

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Dependency

**Location:** `src/create_dataset.py:30–34, 44–83, 108–113`; `src/loader/base.py:600–612`.

**Finding:** Dataset creation reads model and trainer configuration and accepts objective/masking options. Computed `modal_filter`, masking settings, and several CLI values do not affect cache construction. The target list nevertheless determines stored visual modalities, and cached loading hardcodes width 768. Scientific dataset packaging therefore depends on downstream settings and can omit vision when `--modality ap` is supplied, which also causes the unconditional vision lookup to fail. These options do not actually implement permanent training masks; their presence should not be interpreted as such.

**Expected:** Preserve both aligned modalities independently of prediction direction and training objectives; dataset configuration must not depend on model internals, and visual width must follow the source representation (spec sections 12, 15–16, 27, 31–32; dependency graph).

**Suggested disposition:** Refactor

---

### A09 — Obsolete dataset paths obscure the supported boundary

**Priority:** Medium  
**Confidence:** Likely  
**Category:** Dead code

**Location:** `src/utils/dataset_utils.py`, `get_data_from_h5`, `split_both_dataset`, upload/download helpers; `src/loader/base.py`, wrap-padding helpers and custom samplers; `src/loader/make_loader.py`, unused sampler imports and `weighted_sampler` option.

**Finding:** Repository caller searches find no active calls to the HDF5 acquisition helper, timestamp-proximity split helper, upload/download helpers, or wrap-padding helpers. Custom sampler classes remain imported but are never selected by `make_loader`, which always creates a plain DataLoader. The HDF5 helper references unimported `h5py` and undefined `self`; wrap padding duplicates observations, and proximity splitting groups by timestamps rather than trial identity. These paths are possibly obsolete, not confirmed safe to remove for external callers. `BaseDataset` still has a reachable non-IBL branch, so that entire branch is not classified as dead.

**Expected:** Keep the supported aligned-trial path unambiguous; intentionally retained compatibility behavior must not substitute synthesized observations or approximate grouping for the required sample contract (spec sections 4–5, 21, 33–34).

**Suggested disposition:** Refactor

## Conforming areas

- Legacy rows and cache samples retain explicit session and original trial IDs; construction and manifest loading reject duplicate trial identities within a session, including across splits.
- Cache generations use immutable payload locations, file hashes, configuration/source/package fingerprints, and atomically replaced session manifests. Loading selects declared payloads rather than enumerating loose cache files.
- For supported exact-length samples and compatible neural dimensions, temporal and channel masks are separate from numeric padding values, preserving the distinction between valid zero spikes and padding.
- Stored visual features are not re-extracted or renormalized by the dataset loader, and oversized visual sequences fail instead of being silently truncated.
