# Session data audit

## Summary

Session-data is partially implemented through shared EID/path helpers and source-loading code embedded in replay and neural preparation. The stable source-access boundary required by [spec.md](spec.md), [architecture.md](architecture.md), and [the dependency graph](../dependencies.md) is missing; [interface.md](interface.md) is empty. Existing ONE, Brainbox, and SpikeInterface integrations are useful migration foundations, not candidates for wholesale replacement.

The principal gaps are incorrect or unguarded probe selection, consumer-owned acquisition policy, missing collection/revision provenance, and incomplete metadata/discovery access. This is a source-inspection audit: remote availability and actual downloads were not exercised. Inspection of consumers was limited to their source-access calls and boundary behavior; neural processing and rendering algorithms were not audited.

## Findings

### A01 — Sessions without probes fail through empty-container operations

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Runtime

**Location:**  
`src/utils/ibl_data_utils.py:435`, `prepare_data()` and `merge_probes()`; `src/utils/preprocess_lfp.py:104`, `prepare_lfp()`.

**Finding:**  
When `eid2pid()` returns no insertions, neural preparation passes empty lists to `merge_probes()`, which reaches `pd.concat([])`. LFP acquisition instead immediately indexes `pids[0]`. Neither path reports that the requested source family is unavailable for this session. Separately, `load_spiking_data()` converts a missing merged-cluster result into a tuple of `None` values; preparation reduces this to the generic `SkipSession("Missing spike data")`, without retaining the affected insertion or source-resolution condition.

**Expected:**  
The specification's dataset-discovery and failure requirements distinguish unavailable source data from valid empty data and other access failures. A valid session need not contain every data family.

**Suggested disposition:**  
Refactor the source-access failure boundary while preserving the loaders and downstream decision about whether to skip a session.

### A02 — LFP acquisition repeatedly selects the first insertion

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/utils/preprocess_lfp.py:104–119`, `prepare_lfp()`; active caller in `src/prepare_data.py:132–134`.

**Finding:**  
The loop runs once per insertion but always assigns `pid, probe = pids[0], probes[0]`. Every `IblRecordingExtractor` therefore requests the first probe's stream; subsequent probes are never acquired. The time-conversion loader is also constructed only for the first insertion. This violates the claimed multi-probe source selection before neural processing begins.

**Expected:**  
Consistent session-source and probe-identity requirements apply to each requested stream. The architecture permits specialized ephys loaders but requires source identity to survive the boundary.

**Suggested disposition:**  
Refactor source selection to preserve the requested insertion/stream association, then wrap the existing access mechanism. LFP filtering and feature extraction remain neural-data responsibilities.

### A03 — Consumers still own the session-data boundary

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Architecture

**Location:**  
`src/visual_stim_gen.py:295–315,428–433`; `src/prepare_data.py:60–65`; `src/utils/ibl_data_utils.py`, `load_spiking_data()`, `load_trials_and_mask()`, `prepare_data()`; `src/utils/preprocess_lfp.py`, `prepare_lfp()`; `docs/session-data/interface.md`.

**Finding:**  
Replay and preparation construct their own ONE clients and pass them directly into source-loading functions. Replay calls `load_dataset()` itself; neural helpers combine source access with trial filtering, cluster filtering, or preparation; LFP processing constructs its source extractor internally. No shared consumer-facing source API or documented implemented contract covers these paths. Sharing utility files does not provide the accepted boundary when ordinary consumers still own client configuration and acquisition details.

**Expected:**  
The architecture's consumer-facing boundary and the dependency graph require replay and neural-data to obtain original IBL data through session-data. Source loading must remain separate from domain transformations. The interface must describe the actual public requests, results, identity, policy, and failures once implemented.

**Suggested disposition:**  
Wrap / centralize the existing acquisition portions. Keep working representations and consumer algorithms; extract source responsibility rather than moving entire mixed-purpose modules into session-data.

### A04 — Remote access and cache configuration lack a shared explicit policy

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/visual_stim_gen.py:428–429`; `src/prepare_data.py:34,60–65`; `src/utils/preprocess_lfp.py:119`; `src/utils/paths.py`, `dataset_dir()`; `script/prepare_data.sh`.

**Finding:**  
Normal entry points expose no session-data local-only/remote-allowed policy. Each underlying API call determines access behavior, including explicit streaming in the LFP path. Replay uses `dataset_dir()`, whereas direct preparation defaults its source cache to `EXAMPLE_PATH` through `--base_path`. The shell wrapper supplies `VINED_DATA_DIR`, so the wrapper mitigates this divergence but the Python entry point does not share that default. A caller cannot request a consistent offline operation across supported source families through a project boundary.

**Expected:**  
The acquisition-policy and access-context sections require explicit remote policy, coherent source-cache configuration, and an explicit unavailable condition when local-only access cannot satisfy a request. Existing ONE-managed storage should be reused.

**Suggested disposition:**  
Wrap / centralize configuration and policy using the existing path helpers and IBL access mechanisms; no second source-data store is needed.

### A05 — Effective dataset identity and revision coherence are not exposed

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/visual_stim_gen.py:297–315,338–344`; `src/utils/ibl_data_utils.py:31–55,435–485`; `src/utils/preprocess_lfp.py:104–119`.

**Finding:**  
Replay hard-codes collection `alf`, loads trials, offsets, and wheel arrays separately, and records `dataset_revision=None`. Spike/trial loading exposes neither a collection/revision request nor the effective source selections; `load_spiking_data()` consumes EID/probe keywords but does not forward additional revision requirements. Returned preparation metadata contains EID and some neural identifiers, but no source dataset IDs or effective revisions, and omits the PID column added before merging. There is no project-level check or exposed resolution result establishing coherence among jointly consumed sources. This confirms missing guarantees, not that a particular cached session is already mixed.

**Expected:**  
The specification's collections, revisions, consistent-source, and provenance requirements preserve explicit requests and make effective source identity or meaningful unresolved ambiguity visible. Supported IBL default resolution may remain in use.

**Suggested disposition:**  
Wrap / centralize source resolution and retain factual identity alongside existing data representations. Preserve replay's existing ALF-table fingerprint check as a useful, narrower safeguard.

### A06 — Metadata and dataset discovery are confined to partial consumer paths

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/utils/ibl_data_utils.py:436–437,470–480`; `src/inventory_visual_evidence.py`, `inventory()`; `src/recover_stimulus_parameters.py`.

**Finding:**  
Normal preparation fetches session details but exposes only selected fields such as subject and lab. There is no shared request boundary for task/software evidence, session metadata, raw task sources, synchronization datasets, or dataset availability. The inventory tool exposes some cached protocol/catalog information by inspecting `.rest` files and manually reconstructing session paths; it explicitly does not establish current remote availability. Parameter recovery consumes explicitly supplied local evidence. These useful tools do not supply ordinary consumers with the required metadata/discovery access or distinguish all local, remote-known, unavailable, and unresolved source states.

**Expected:**  
The metadata and dataset-discovery responsibilities require factual source information and meaningful availability states on request. Historical interpretation remains with consumers. The architecture explicitly excludes inventory cache traversal from the normal runtime boundary.

**Suggested disposition:**  
Wrap / centralize supported IBL metadata and discovery access. Keep inventory as specialized inspection tooling and parameter recovery as consumer-owned interpretation.

## Conforming areas

- `src/utils/sessions.py:21–39` canonicalizes UUIDs, preserves manifest order, supports comments/BOMs, and rejects duplicate or empty selections. Reuse this identity handling; it does not assign experiment roles.
- `src/utils/paths.py` supplies repository-relative defaults and environment overrides. Replay already uses the configured source root as its ONE cache, without making another IBL source-tree copy.
- ONE named-dataset loading, Brainbox `SessionLoader`/`SpikeSortingLoader`, and `IblRecordingExtractor` already cover useful source families. Their differing data representations are allowed by the architecture.
- Replay checks an available recovered ALF-table fingerprint, and `run_sessions()` distinguishes explicit skips from failures while retaining per-EID reporting. These safeguards should survive migration.

## Legacy and boundary assessment

Caller and CLI inspection found active uses of the source-loading helpers, optional LFP path, inventory CLI, and session-selection CLI. No source-access implementation was established to be dead, so removal is not recommended. The `utils.sessions --cache` branch invokes downstream dataset orchestration and is used by `script/run_create_dataset.sh`; reuse its generic identity helpers without treating that orchestration branch as session-data functionality.

The untracked `script/find_visual_eids.py` also constructs ONE independently for a specialized discovery workflow. It is additional consumer-side access evidence, not the basis for the findings above or a reason to expand this audit into experiment selection.

Source access is currently distributed across consumers rather than forming a distinct component; no separate implemented session-data dependency cycle was established. Migration should not import rendering, neural preprocessing, alignment, or model-dataset orchestration into the source boundary. Duplicate trial loading in `prepare_data()` is visible, but without a demonstrated bottleneck it does not warrant a separate optimization workstream.
