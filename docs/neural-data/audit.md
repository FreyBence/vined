# Neural-data audit

## Summary

The inherited spike-loading, probe-merging, and counting helpers are a reusable foundation, but the component only partially satisfies `spec.md`. Correctness and missing output-contract work outweigh dead-code cleanup: count columns can lose their metadata correspondence, unknown neural coverage becomes zero activity, and the only persisted result is a visually filtered, split dataset rather than an independently loadable neural generation.

This assessment uses the component specification, `../dependencies.md`, the `session-data` interface, active callers, and installed Brainbox/iblutil source. There is no neural-data architecture or interface document. Findings describe source-established behavior; their frequency in recorded sessions has not been measured. The preparation CLI imports and `--help` execute successfully. No full session generation or numerical runtime verification was performed, and no tests were written.

## Findings

### A01 — Count columns can be assigned to the wrong source units

**Priority:** High

**Confidence:** Confirmed

**Category:** Data integrity

**Location:**

`src/utils/ibl_data_utils.py:148–151, 206–227`; `src/prepare_data.py:79–102`.

**Finding:**

Both counting paths derive the unit axis from `np.unique` of spike assignments rather than the selected unit metadata. A selected unit with no events anywhere in the supplied source disappears. The caller discards `clusters_used_in_bins` and applies retained count-column positions directly to the original metadata lists. For source units `[0, 1, 2]` with events only for `[0, 2]`, count column 1 belongs to unit 2 but receives unit 1's UUID, anatomy, and potentially probe identity. Subset anatomical selection creates the same indexing risk. Units silent only within an individual trial are retained correctly if they appear elsewhere in the source.

**Expected:**

Specification §§4.2, 5.1, 9: use a declared selected-unit axis, preserve silent selected units, and keep every metadata field aligned through filtering.

**Suggested disposition:**

Refactor — retain the counting approach while explicitly mapping columns to selected source units.

### A02 — Unavailable and partially observed intervals become valid zero counts

**Priority:** High

**Confidence:** Confirmed

**Category:** Data integrity

**Location:**

`src/utils/ibl_data_utils.py:get_spike_data_per_interval`, `prepare_data`, `align_data`; `src/prepare_data.py:88–90, 143–151`.

**Finding:**

No recording coverage, invalid intervals, observed durations, or per-recording validity reach the counter. Every empty window is filled with zeros, including windows outside recording support and nonfinite trial windows. Alignment checks count finiteness, which cannot distinguish these cases from observed silence. The rate filter runs before the trial mask, so unavailable zero-filled windows also depress the population-selection statistic. Rejected trial IDs survive in aligned provenance, but neural-specific statuses and reasons do not. An empty population is not represented by an explicit selection outcome; the one-bin dependency additionally calls `cluster_ids.max()` on empty input.

**Expected:**

Specification §§6–7: distinguish observed silence, unknown/missing/partial coverage, missing timing, empty selection, and processing failure; retain original trial outcomes and reasons.

**Suggested disposition:**

Refactor — introduce neural validity and outcome handling around existing source loading and counting.

### A03 — Source associations are assumed rather than validated

**Priority:** High

**Confidence:** Confirmed

**Category:** Data integrity

**Location:**

`src/utils/ibl_data_utils.py:16–68, 206–209, 285–327`.

**Finding:**

Neural processing does not establish finite spike timestamps, matching event-array lengths, resolvable integer unit references, or unique scoped unit identities before enrichment and merging. Invalid cluster references can be removed silently by the later `np.isin` selection, and nonfinite spike times can disappear through interval comparisons. `merge_probes` assumes row-index cluster identities, offsets caller-owned arrays in place, and concatenates only the first probe's spike keys; repeating a merge on the same objects changes assignments again. Original cluster row IDs are not explicitly retained alongside PID and sorting identity, and the optional QC path resets those rows.

**Expected:**

Specification §§3, 4.2, 9: validate essential associations, preserve an explicit original-unit mapping, and keep processing deterministic for identical supplied sources. The session-data interface preserves source arrays; it does not establish these neural invariants.

**Suggested disposition:**

Refactor — validate at the neural boundary and make probe remapping explicit without mutating source assignments.

### A04 — Interval handling cannot represent the required variable windows

**Priority:** High

**Confidence:** Confirmed

**Category:** Spec

**Location:**

`src/utils/ibl_data_utils.py:135–227`; `src/prepare_data.py:40–47, 148–151`.

**Finding:**

The general interval path derives output length from the first interval and allocates that length for every request. Longer later intervals are truncated by `[:, :n_bins]`; shorter intervals can produce incompatible histogram shapes or misleading zero-filled tails. Invalid/reversed boundaries and nonpositive bin sizes lack explicit validation. The counter returns values and unit IDs but no actual bin edges. `ceil` permits shortened final bins without exposing their duration, while the persisted caller output stores only whole-trial boundaries and nominal bin size. The executable fixes the event, window, and resolution in source code.

**Expected:**

Specification §§5.2–5.3, 6, 8: configurable physical intervals, actual half-open bin boundaries including shortened final bins, variable trial lengths, and sufficient information for downstream regeneration/compatibility decisions.

**Suggested disposition:**

Refactor — preserve interval-local counting while making requested boundaries and returned temporal semantics explicit.

### A05 — Effective source and population selection are not fully exposed

**Priority:** High

**Confidence:** Confirmed

**Category:** Interface

**Location:**

`src/utils/ibl_data_utils.py:16–41, 119–132, 275–299, 323`; `src/prepare_data.py:40–47, 76–102, 174–178`.

**Finding:**

Preparation always enumerates all probes. `load_spiking_data` accepts arbitrary keyword arguments but forwards neither collection nor revision to `SessionAccess.load_spike_sorting`; an explicit sorting request supplied through those keywords is silently ignored. Quality filtering is unused by the active caller, and `good_clusters` is reduced to the unexplained numeric rule `label >= 1`. Anatomy is always mapped through Beryl and the caller selects all returned regions; exact-versus-descendant selection and mapping identity are not recorded. The active population filter is a hard-coded rate threshold of `1 / 0.2 = 5 Hz`; that threshold is recorded, but the remaining effective policies are incomplete.

**Expected:**

Specification §§3–4, 8: explicit recording/sorting, quality, and anatomical selection with source meanings and effective criteria preserved. The dependency already supports exact probe, collection, and revision requests.

**Suggested disposition:**

Wrap / centralize — expose and forward selection through the existing acquisition boundary, retaining useful legacy selection operations.

### A06 — Optional metadata are mandatory and sampling metadata are invented

**Priority:** High

**Confidence:** Confirmed

**Category:** Runtime

**Location:**

`src/utils/ibl_data_utils.py:19–30, 316–325`; installed `brainbox.io.one._channels_alf2bunch` and `SpikeSortingLoader.merge_clusters`.

**Finding:**

The neural wrapper unconditionally converts channel anatomy and indexes cluster depths, UUIDs, labels, subject, and lab. The channel converter requires coordinates and anatomical fields; absent optional metadata therefore fails even when no anatomical/quality selection was requested. Installed `merge_clusters` recomputes metrics when absent even with `compute_metrics=False`, adding implicit requirements for spike amplitudes/depths. The wrapper also returns a constant 30,000 Hz and publishes it as session sampling metadata, without recording evidence or an assumption. Spike counting itself uses session-clock seconds and does not require this invented frequency.

**Expected:**

Specification §§3–4, 6: optional information may remain unknown when processing does not depend on it; required metadata fail explicitly; units and timing provenance describe the supplied source.

**Suggested disposition:**

Refactor — retain enrichment where supported, make its prerequisites explicit, and preserve unknown optional metadata.

### A07 — The independent neural processing boundary is missing

**Priority:** High

**Confidence:** Confirmed

**Category:** Architecture

**Location:**

`src/prepare_data.py:56–65, 104–154, 181–226`; `src/utils/ibl_data_utils.py:1, 14, 275–336`; `docs/neural-data/`.

**Finding:**

Low-level spike helpers can count without visual values, but the only generation entry point requires a CLIP archive before loading spikes and persists only the result after joint visual filtering and experiment splitting. Neural loading also constructs a behavioral trial mask requiring choice, feedback, movement, and probability fields. The shared helper module imports visual utilities. There is no supported neural-only generation/load interface through which alignment can request and consume independently interpretable neural results. Shared files alone are not the defect; the missing usable boundary is.

**Expected:**

Specification §§2, 5.3, 6, 8 and the dependency graph: neural-data depends only on session-data; alignment owns joint selection and training-dataset owns splits. Neural preparation must be usable without visual artifacts and unrelated behavioral eligibility.

**Suggested disposition:**

Wrap / centralize — expose the existing neural operations independently while retaining orchestration in the downstream pipeline. No greenfield rewrite is justified.

### A08 — Neural generation identity and content provenance are incomplete

**Priority:** High

**Confidence:** Confirmed

**Category:** Spec

**Location:**

`src/utils/ibl_data_utils.py:31, 328–329`; `src/prepare_data.py:158–179, 228–233`; `docs/session-data/interface.md:DatasetSource` contract.

**Finding:**

Aligned provenance records dataset IDs, revisions, paths, processing source hashes, selected neuron order, and a visual content hash. The exposed spike/trial `DatasetSource` records do not contain content digests, and preparation does not compute identities for the neural/trial content actually consumed. Consequently, the fingerprint cannot distinguish changed source bytes under unchanged dataset metadata. There is no separately identifiable completed neural generation, loader, or accounting of all requested neural outcomes. Staged aligned publication and refusal to overwrite existing aligned output are useful protections, but do not supply this missing neural contract.

**Expected:**

Specification §8: content-identifiable sources and processing, completed-generation identity, complete request accounting, and loadable selected generations that support compatibility checks without owning downstream cache invalidation.

**Suggested disposition:**

Wrap / centralize — reuse existing provenance and staged-publication mechanisms for the required neural output boundary.

### A09 — The current persistence boundary silently narrows counts to eight bits

**Priority:** High

**Confidence:** Confirmed

**Category:** Data integrity

**Location:**

`src/utils/dataset_utils.py:43–48, 68–69`, reached from `src/prepare_data.py:create_dataset` calls.

**Finding:**

The only persisted consumer representation casts count matrices to `np.ubyte` without a range check. Counts above 255 cannot retain their numerical value, particularly when coarser neural intervals are requested. This is an observed outgoing contract conflict originating in training-dataset serialization, not a reason to expand this audit into dataset internals. No claim is made that overflow has occurred in the default 20 ms recordings.

**Expected:**

Specification §5.2: counts must not overflow silently or change meaning to fit a legacy dtype. Consumers must preserve the neural representation's numerical contract.

**Suggested disposition:**

Refactor — coordinate a count-preserving serializer at the declared downstream boundary; that external change needs an explicitly scoped implementation task.

### A10 — Obsolete return values remain, but major legacy paths are still reachable

**Priority:** Medium

**Confidence:** Confirmed

**Category:** Dead code

**Location:**

`src/utils/ibl_data_utils.py:305–308, 336`; `src/prepare_data.py:69`; `src/utils/ibl_data_utils.py:list_brain_regions` and its caller.

**Finding:**

`prepare_data` constructs an empty behavior dictionary and copies the trial mask into `good_trials_mask`; its sole repository caller discards both return values. These obsolete contract slots imply responsibilities that no longer produce useful outputs. The active region-list round trip selects every mapped region, and `single_region=True` still reaches the same union because the caller passes the whole list to `np.isin`, not separate region requests. Searches of source, launchers, and configuration found no alternate consumer implementing that advertised distinction.

The spike helpers themselves have active callers. The QC branch is unused by the current entry point but relates to required selection functionality; lack of a call is insufficient evidence to delete it. LFP is reachable through `--use_lfp` and has explicitly documented optional dependencies, so it is not dead code and its processing is outside the initial spike-count specification.

**Expected:**

An unambiguous current neural boundary without obsolete outputs or misleading inactive modes; retained compatibility behavior must be distinguished from the required pipeline.

**Suggested disposition:**

Remove — eliminate confirmed unused return slots and obsolete mode plumbing when updating their caller. Preserve useful selection primitives and reachable optional LFP behavior pending separate scope decisions.

### A11 — Binning repeatedly scans all spikes and copies the dense trial output

**Priority:** Low

**Confidence:** Confirmed

**Category:** Optimization

**Location:**

`src/utils/ibl_data_utils.py:66–67, 151–158, 226–227`; `src/prepare_data.py:91`.

**Finding:**

The merger already sorts spike times, but each trial allocates a full-source boolean time mask, giving work proportional to trials multiplied by session spike count. Binning allocates float64 `[trials, units, bins]`, then builds another dense array from transposed trial views; population filtering copies it again. These are concrete avoidable costs, although no runtime or peak-memory bottleneck has been measured.

**Expected:**

Specification §9: practical session processing without changing count, identity, overlap, or validity semantics. Optimization should remain proportional to demonstrated needs.

**Suggested disposition:**

Refactor — consider sorted-time interval slicing and avoid redundant full-output copies while repairing the counter; defer broader performance work.

## Conforming areas

- Spike acquisition goes through `SessionAccess`; neural helpers do not create an independent ONE client or project cache. Collection/revision identities returned by that dependency are retained.
- Probe merging offsets unit indices for ordinary valid ALF row-index inputs and stably sorts all corresponding event arrays together. PID/probe and available UUID metadata provide a useful basis for explicit population identity.
- Multi-bin counting explicitly filters `[start, end)`. The installed one-bin Brainbox implementation uses left-sided searches at both boundaries, also excluding the right endpoint. Equal-time events are not deduplicated.
- In-memory spike outputs are separate, unsmoothed, unnormalized unit counts; trial-silent units that occur elsewhere in the source receive zeros without changing position. No model-width truncation is performed by the neural helpers.
- Original trial row IDs are assigned before filtering and retained by the existing aligned handoff. Source-session timestamps and explicit event offsets are already used, rather than trial index as time.
- Sessions are processed individually; the counter allocates trial windows rather than a dense session-duration grid. Existing aligned output is staged before publication and is not silently overwritten.
