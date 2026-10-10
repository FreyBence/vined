# Alignment implementation plan

Produce unsplit, stimulus-bounded aligned trials according to [spec.md](spec.md),
using the prepared [visual-features](../visual-features/interface.md) and
[neural-data](../neural-data/interface.md) public contracts. The plan addresses
[audit.md](audit.md) without changing the accepted component dependencies.

## Tasks

### [x] T01 — Consume and pair prepared modality inputs

**Summary:**  
Expose an alignment input boundary for verified visual feature generations and
prepared neural counts. Pair records by original session/trial identity, retain
visual observation and coverage metadata, preserve the full ordered neural unit
axis and scoped identities, and obtain exact stimulus bounds from supplied
upstream metadata. Validate input timing, source compatibility, and neural
coverage explicitly; unknown, assumed, partial, or invalid neural support must
not silently become fully observed targets. Reject missing required metadata or
incompatible neural grids with actionable errors. Covers A01, A04–A06, and the
input side of A07.

**Out of scope:**  
Raw recording acquisition, upstream generation changes, neural selection or
rebinning inside alignment, and conversion of obsolete feature archives.

### [x] T02 — Implement complete-bin stimulus alignment

**Summary:**  
Produce deterministic aligned trials satisfying the interval, grid, identity,
and provenance requirements in the specification. Retain only complete supplied
neural bins anchored at stimulus onset, exclude the trailing partial bin, and
derive visual queries from their physical centers. Resample visual features
with normalized linear interpolation only within valid consecutive coverage;
reject missing observations, coverage gaps, extrapolation, and incompatible
timing rather than publishing placeholder observations. Preserve raw counts,
silent units, exact stimulus bounds, discarded duration, and source/resampling
provenance in the unsplit output. Covers A02–A06 and A08.

**Out of scope:**  
Synthetic gap repair, interpolation of neural counts, padding, model masking,
behavioral eligibility filters, and LFP processing.

### [x] T03 — Publish standalone alignment output and expose its interface

**Summary:**  
Provide a supported alignment launcher that selects explicit prepared inputs
and produces readable, variable-length unsplit aligned output for
training-dataset. Preserve count precision, identities, physical timestamps,
configuration, and source generation provenance across save/load, with clear
failure behavior and no publication of incomplete output as successful data.
Retire or isolate the legacy combined alignment route so the supported alignment
entry point performs neither acquisition nor dataset splitting/packaging.
Demonstrate generation and loading through direct execution and inspection,
verify the downstream handoff at the declared boundary, and document the actual
public API/artifact contract in `interface.md`. Covers A07–A08 and completes
integration of A01–A06.

**Out of scope:**  
Training-dataset implementation changes, training/evaluation migration, rewriting
working upstream preparation, and removal of legacy helpers still used outside
the replaced alignment route.

### [x] T04 ? Parallelize alignment trial work

**Summary:**
Expose a configurable worker count for trial alignment and compression while
preserving canonical ordering, scientific values, and atomic publication.

**Out of scope:**
Multi-session orchestration, upstream preparation, and dataset construction.

### [x] T05 ? Default alignment inputs to upstream output locations

**Summary:**
Resolve a selected session's prepared inputs from the neural and visual output
roots, retaining explicit overrides and rejecting ambiguous neural generations.

**Out of scope:**
Acquisition, automatic generation replacement, and multi-session orchestration.

### [x] T06 - Select alignment sessions from the configured EID list

**Summary:**
Use shared EID selection and session reporting when no explicit session or input
pair is supplied, retaining per-session input resolution and trial workers.

**Out of scope:**
Acquisition and parallel execution across sessions.

### [x] T07 - Publish usable trials with complete exclusion accounting

**Summary:**
Retain valid trials from damaged sessions, preserve requested/retained/excluded
identities and reasons across publication/loading and dataset handoff, and avoid
repeating full session provenance in every trial. Keep integrity errors fatal.

**Out of scope:**
Repairing missing observations, inventing coverage, and changing split policy.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.

- T04 depends on T03.

- T05 depends on T03.

- T06 depends on T05.

- T07 depends on T06.
