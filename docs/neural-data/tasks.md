# Neural-data implementation plan

## Scope

Implement the independent spike-count component defined by [spec.md](spec.md), using [audit.md](audit.md) and the existing [session-data interface](../session-data/interface.md). Preserve useful legacy loading, enrichment, and counting operations through incremental changes. The outcome is a neural generation that alignment can load or regenerate without visual inputs, model internals, or independent source acquisition.

Validation belongs within each task: use code inspection, existing checks, direct execution, and manual inspection of produced outputs. Do not write tests or validation infrastructure. Retain reachable optional LFP behavior; it is outside this plan.

## Tasks

### [x] T01 — Resolve and select a validated source population

**Summary:**

Provide neural source preparation through `SessionAccess` with explicit recording, sorting collection/revision, quality, and anatomical selection. Validate spike-to-unit associations and preserve deterministic source-unit identities and metadata through selection and multi-recording combination, without mutating supplied assignments. Keep optional metadata unknown when unused, make required metadata failures explicit, and retain source meanings, timing evidence, coverage evidence or limitations, and effective selection criteria. Remove invented sampling metadata and implicit metric-recomputation requirements. This addresses A03, A05, and A06 and establishes the population mapping needed for A01.

**Out of scope:**

Changing session-data acquisition/cache policy, computing a new spike sorting, reconstructing unavailable recording evidence, or introducing new quality metrics and fixed population thresholds.

### [x] T02 — Produce identity-preserving counts with explicit time and validity

**Summary:**

Expose visual-independent processing of configurable absolute intervals or original trial events and offsets, using the selected population from T01. Produce unsmoothed counts on a stable unit axis, including silent units, with actual half-open bin boundaries and variable window lengths. Preserve per-recording coverage distinctions, qualified unknown coverage, partial/invalid observations, empty selections, and traceable trial/request outcomes. Permit regeneration from source spikes for a requested grid without joint modality filtering or experiment splits. Directly verify boundary counting, silent units, multi-recording identity, missing timing, and coverage handling. This addresses A01, A02, A04, and the processing boundary in A07.

**Out of scope:**

Choosing the common multimodal grid, visual resampling, behavioral eligibility filters, model padding, or normalization. A11's sorted-time slicing and redundant-copy removal may be incorporated where straightforward; benchmarking and broader optimization are not separate deliverables.

### [x] T03 — Publish and load complete standalone neural generations

**Summary:**

Provide a configurable neural-only generation entry point and loader for T02 outputs. Preserve counts without narrowing, identities, interval boundaries, validity, all requested outcomes, and effective processing configuration. Identify consumed source content and processing implementation, publish only fully accounted generations, and preserve explicitly selected generation identity on reload without silently substituting older results. Demonstrate generation and reload without visual artifacts and document the implemented public contract in `interface.md`, including regeneration and downstream compatibility information. Reuse existing provenance/publication utilities where appropriate. Remove superseded neural internals and obsolete plumbing where callers remain compatible; retain narrow legacy adapters where downstream migration is separate. This addresses A07, A08, and applicable cleanup from A10.

**Out of scope:**

Migration of historical artifacts, recursive downstream cache invalidation, redesign of the combined preparation pipeline, or changes to alignment and training-dataset behavior.

### [x] T04 — Parallelize independent neural windows

**Summary:**

Expose configurable shared-memory trial/interval counting and staged window compression through the standalone neural API and CLI, preserving source loading, unit and request ordering, count/coverage semantics, and verified publication.

**Out of scope:**

Concurrent source acquisition, spike-sorting changes, downstream alignment changes, and performance benchmarking infrastructure.

### [x] T05 — Prepare neural data without a separate request file

**Summary:**

Discover session probes and count recorded full-trial intervals at the specified default resolution when no config is supplied. Preserve explicit custom requests, original trial identity, coherent revision selection, source coverage qualifications, and documented defaults.

**Out of scope:**

Inventing recording coverage, changing source acquisition policy, selecting a common multimodal grid, and implicit quality/anatomical filtering.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.
- T04 depends on T03.
- T05 depends on T03.

## Downstream boundary

The current combined preparation caller must eventually consume the neural contract while alignment retains joint coverage decisions and training-dataset retains split application. That consumer migration is separate cross-component work; this plan completes the producer and verifies its exposed contract without silently expanding into downstream implementation.

A09 originates in training-dataset's `np.ubyte` serializer and remains an external correctness issue. The new neural generation must preserve counts, but that alone does not fix the existing aligned-dataset serialization path. Address the serializer and preservation of neural identities, intervals, and validity in the explicitly scoped downstream integration. Do not narrow the neural contract to accommodate that legacy format.
