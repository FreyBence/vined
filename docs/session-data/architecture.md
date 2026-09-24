# Session Data Architecture

## Architectural goal

The `session-data` component provides one stable ViNED boundary for access to original IBL session data.

It organizes the repository's existing ONE, Brainbox, SpikeInterface, and local-cache access paths behind a coherent component boundary without unnecessarily rewriting working research code.

The component is an integration layer around supported IBL data-access mechanisms, not a replacement for them.

For this existing codebase, the target architecture is the practical intersection between:

* the accepted `session-data` specification;
* the useful structure already present in the repository;
* the cost and risk of changing working research code.

The preferred migration strategy is:

```text
keep
→ expose through the component boundary
→ wrap / adapt
→ extract shared responsibility
→ refactor where necessary
→ replace only when necessary
```

Existing code quality alone is not sufficient reason for replacement.

---

## Architectural boundary

The component sits between external IBL infrastructure and ViNED processing components.

```text
                       External
                     ONE / Alyx
                         │
                         ▼
               ┌──────────────────┐
               │   session-data   │
               │                  │
               │ session identity │
               │ access context   │
               │ metadata access  │
               │ source access    │
               │ provenance       │
               └────────┬─────────┘
                        │
               ┌────────┴────────┐
               ▼                 ▼
         visual-replay       neural-data
```

Downstream components specify what source information they need.

`session-data` determines how supported source information is resolved and obtained.

The boundary hides normal ONE/cache/acquisition mechanics from consumers while preserving source information required for correct downstream interpretation.

---

# Internal responsibilities

The component has five logical responsibilities:

```text
Session identity
       │
       ▼
Access context
       │
       ├──────────────┐
       ▼              ▼
Metadata access    Source access
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
        ALF       spike data    ephys streams
          │           │            │
         ONE       Brainbox    SpikeInterface
          │           │            │
          └───────────┴────────────┘
                      │
                      ▼
             source/provenance
```

These are architectural responsibilities.

They do not require one class or module per box.

The implementation should remain as small as practical.

---

## 1. Session identity

Session identity provides canonical handling of IBL EIDs.

Responsibilities include:

* canonicalizing EIDs;
* validating identifier syntax;
* preserving original session identity;
* loading generic session manifests where useful;
* avoiding competing EID parsing behavior in active data-access paths.

Existing behavior in `src/utils/sessions.py` should be reused where it satisfies this responsibility.

### Experiment roles

Session identity does not assign experiment meaning.

The following are not properties of `session-data`:

```text
training
validation
testing
held-out
fine-tuning
evaluation-only
```

A file containing EIDs may use shared session parsing, but the semantic role assigned to those sessions belongs to the consumer or experiment configuration.

---

## 2. Access context

Normal session-data acquisition should use a shared project-level access context.

The access context owns concerns such as:

* ONE endpoint configuration;
* ONE client construction;
* source-data cache root;
* remote acquisition policy;
* local-only behavior where requested;
* shared source-access configuration.

This removes the need for downstream components to independently decide how ONE should be configured.

### Existing source-data location

The existing configured source-data location should be reused.

Current `VINED_DATA_DIR` / `dataset_dir()` behavior may remain shared infrastructure where appropriate.

Do not introduce a second project-managed copy of the IBL source-data hierarchy solely to provide the new component abstraction.

The intended source-storage relationship is:

```text
ONE-managed source cache
        │
        ▼
   session-data
```

not:

```text
ONE cache
   +
separate ViNED copy of ALF/ephys source data
```

Derived ViNED artifacts remain outside this source-data responsibility.

---

## 3. Metadata access

Metadata access exposes factual session and dataset metadata required by downstream consumers.

Typical information may include:

* EID;
* subject;
* laboratory;
* session date/time;
* session number;
* task protocol;
* available task/software version information;
* probe/insertion identity;
* dataset collection;
* dataset revision;
* remote dataset identity;
* availability information.

ONE/Alyx remains the authoritative external source where applicable.

Metadata access must not interpret historical behavior for downstream components.

For example:

```text
session-data
    → task/software version evidence

visual-replay
    → renderer/source-profile selection
```

The first is source information.

The second is domain interpretation.

---

## 4. Source access

Source access is organized by source-data family rather than forced through one universal loader.

Different supported IBL libraries may continue to provide the data families they already handle appropriately.

The architecture standardizes the ViNED boundary and acquisition policy, not every internal loading implementation.

---

### ALF and named dataset access

Used for source information such as:

* trial tables;
* wheel timestamps;
* wheel positions;
* stimulus timing arrays;
* synchronization arrays where required;
* other explicitly requested ALF datasets.

ONE dataset/object loading may remain the underlying implementation.

Consumers should not need to construct their own ONE client for ordinary supported ALF access.

---

### Trial access

Existing `SessionLoader` behavior may remain useful where it provides an appropriate standardized representation of trial information.

The architecture does not require every trial consumer to receive data from the same internal loader.

For example, a direct ALF table and a `SessionLoader` representation may both remain valid internally when each serves an existing supported path.

The requirement is that the consumer reaches source data through the `session-data` boundary rather than owning independent acquisition policy.

---

### Spike-sorting access

Spike-sorting data has richer loading and metadata semantics than simple named ALF datasets.

Existing Brainbox mechanisms such as `SpikeSortingLoader` may remain the underlying implementation.

This path may expose source representations including:

* spike times;
* spike cluster identities;
* cluster metadata;
* channel metadata;
* probe/insertion information.

Model-oriented transformations do not belong here.

For example:

```text
session-data
    → spike/source representation

neural-data
    → filtering
    → binning
    → model-ready neural representation
```

---

### Electrophysiological stream access

Raw or streamed electrophysiological data such as LFP may continue to use specialized mechanisms such as `IblRecordingExtractor`.

`session-data` owns access to the source stream.

The following remain downstream neural-processing responsibilities:

* filtering;
* bad-channel processing;
* trial-window extraction;
* model-oriented transformation;
* model-specific temporal representation.

The boundary is:

```text
session-data
    ↓
source electrophysiology access

neural-data
    ↓
processed neural representation
```

---

### Raw task and auxiliary evidence

Where available, downstream components may require additional session evidence such as:

* raw task records;
* encoder/task parameters;
* display/synchronization information;
* historical task metadata.

`session-data` may expose these as source evidence.

It must not interpret them into visual replay behavior or neural preprocessing rules.

---

## 5. Source identity and provenance

Data leaving `session-data` must retain enough source identity for downstream interpretation when that identity can materially affect the result.

Relevant information may include:

* EID;
* source-data kind;
* collection;
* effective revision;
* remote dataset identifier;
* probe/insertion identity where relevant;
* local/cache/remote resolution state;
* factual task/software metadata.

The exact representation belongs to `interface.md`.

The architectural requirement is that source identity must not be lost merely because the underlying loader returns a convenient array or table.

---

# Acquisition policy

Remote acquisition is a `session-data` concern.

Downstream components must not accidentally determine download behavior merely by selecting a particular ONE function.

The component must support the conceptual distinction between:

```text
local/cache access only
```

and:

```text
remote acquisition allowed
```

The concrete API belongs to `interface.md`.

When remote acquisition is allowed:

```text
request
   ↓
resolve through supported IBL mechanism
   ↓
use cached/local data when available
   ↓
acquire remotely when required
   ↓
return source representation
```

When remote acquisition is disabled:

```text
request
   ↓
resolve locally
   ↓
available? ─ yes → return
   │
   no
   ↓
explicit unavailable result/error
```

Acquisition mode must not change silently.

---

# Collection and revision handling

Collection and revision handling belong to source resolution.

Prefer supported ONE/IBL mechanisms over manually reconstructing cache paths.

Manual traversal of the ONE cache tree may remain useful for specialized inspection tools, but it should not define normal runtime source access.

---

## Explicit collection/revision requests

When a consumer explicitly requires a particular collection or revision, that requirement must be preserved.

---

## Unspecified revision

When the caller does not request a revision, supported IBL resolution behavior may be used.

The effective revision should remain discoverable where it materially affects reproducibility or downstream historical compatibility.

---

## Multiple revisions

The component should not build a large independent revision-management framework.

It must, however, avoid silently hiding meaningful ambiguity.

If multiple revisions could materially change the requested source data, the effective choice or unresolved ambiguity must remain visible.

The concrete resolution representation belongs to `interface.md`.

---

# Consumer-facing boundary

Consumers should conceptually operate as:

```text
consumer
   │
   │ EID + source requirement
   ▼
session-data
   │
   │ source representation
   │ + relevant source identity
   ▼
consumer-owned transformation
```

Consumers should normally not need to know:

* ONE cache directory layout;
* `.rest` cache structure;
* how session paths are constructed;
* which IBL library performs acquisition;
* how ONE clients are configured;
* where revision directories appear physically;
* whether the source was already cached or downloaded during the request.

These remain internal source-access concerns unless explicitly needed for provenance.

---

# No universal data container

The architecture does not require every source-data type to be converted into one common representation.

Appropriate representations may remain different.

Examples:

```text
trial data       → DataFrame
wheel data       → NumPy arrays
spike data       → Bunch/dictionary + metadata
streamed ephys   → extractor/array-oriented access
metadata         → structured metadata
```

The component should standardize:

* how source data is requested;
* how source identity is preserved;
* how acquisition policy is applied;
* how failures are exposed.

It should not erase useful distinctions between data families merely for architectural uniformity.

---

# Error boundary

`session-data` should provide coherent component-level failure semantics.

Internally, ONE, Brainbox, SpikeInterface, filesystem, and network errors may differ.

Consumers should be able to distinguish meaningful conditions such as:

```text
invalid session identity
session unresolved
required source data unavailable
remote acquisition disabled/unavailable
collection/revision resolution failure
external service/access failure
unsupported source request
```

A complex custom exception hierarchy is not required unless implementation demonstrates a need for one.

Existing exceptions may be preserved, wrapped, or normalized only where doing so improves the component boundary.

Valid empty data must not be confused with unavailable data.

---

# Legacy architecture preservation

`session-data` is being introduced over an existing research codebase rather than designed in a greenfield repository.

The goal is therefore not to replace the current implementation with an idealized clean architecture.

The target architecture should represent the best practical intersection between:

* the accepted component specification;
* the useful hidden architecture already present in the repository;
* the cost of migration;
* the risk of modifying working research code.

Follow an Open/Closed-style migration principle:

> Prefer extending, wrapping, adapting, or centralizing existing behavior over modifying stable implementation internals.

Use this preference in the following order:

```text
keep
→ expose through the component boundary
→ wrap / adapt
→ extract shared responsibility
→ refactor existing internals
→ replace only when necessary
```

Working code may remain internally imperfect when it still satisfies the accepted component behavior and boundary.

---

## When existing code should be changed

Existing implementation should be modified when there is a concrete reason, such as:

* it violates the accepted specification;
* it prevents the required component interface;
* it produces incorrect data;
* it causes supported execution to fail;
* it breaks the component boundary;
* it introduces an invalid or circular dependency;
* duplicated implementations materially disagree;
* it silently corrupts identity, timing, revision, collection, or source semantics;
* wrapping it would create greater complexity or risk than targeted refactoring.

Do not refactor solely because:

* another abstraction is cleaner;
* naming is inconsistent;
* multiple supported IBL libraries are used;
* the code is procedural;
* the code is old;
* the code would be structured differently in a greenfield project.

---

## Greenfield components

The preservation bias applies to functionality with meaningful existing implementation.

When a genuinely new component or capability has no useful existing implementation to preserve:

* architecture may be designed directly from the accepted specification;
* unnecessary legacy patterns should not be reproduced merely for consistency;
* project dependency constraints and component conventions still apply.

This distinction allows legacy components to evolve incrementally without constraining future greenfield design.

---

# Reuse of current implementation

The existing repository already contains several useful partial abstractions.

These should be reused where practical rather than rewritten automatically.

---

## `src/utils/sessions.py`

Provides useful behavior for:

* EID canonicalization;
* manifest parsing;
* basic session selection;
* batch session execution/reporting.

Canonical identity and generic manifest behavior are strong candidates for reuse.

Experiment-specific train/test meaning remains outside `session-data`.

---

## `src/utils/paths.py`

Provides configured project storage roots.

The existing path configuration may remain shared project infrastructure.

It should not be duplicated solely to make `session-data` appear self-contained.

`session-data` should consume the relevant configured source-data/cache location.

---

## `src/utils/ibl_data_utils.py`

Contains existing:

* trial loading;
* spike loading;
* metadata access;
* preprocessing.

The source-access portions are candidates for reuse or extraction behind the new boundary.

Neural transformations remain under `neural-data`.

A complete rewrite is not required.

---

## `src/visual_stim_gen.py`

Currently combines:

```text
source acquisition
+
visual replay generation
```

The visual behavior remains in `visual-replay`.

Only source access should move behind or be redirected through the `session-data` boundary.

The renderer should not be rewritten merely because its source-loading logic is extracted.

---

## `src/utils/preprocess_lfp.py`

Currently combines specialized stream acquisition with neural processing.

The existing acquisition mechanism may remain useful.

Source access belongs to `session-data`.

Processing belongs to `neural-data`.

---

## `src/inventory_visual_evidence.py`

This tool directly inspects cached ONE/Alyx responses and local files.

It should not define normal runtime session-data architecture.

Useful inspection behavior may:

* remain specialized tooling;
* support audits;
* be reused selectively if an active runtime requirement genuinely needs it.

Do not redesign normal data access around the inventory implementation.

---

## `src/recover_stimulus_parameters.py`

This currently accepts explicitly supplied local source files.

Parameter interpretation and recovery belong primarily to `visual-replay`.

If common source resolution is required, source-file acquisition may use `session-data`, while parameter interpretation remains outside it.

---

# Migration architecture

Migration should be incremental.

## Current state

Conceptually:

```text
visual replay ─────────→ ONE
prepare data ──────────→ ONE / Brainbox
LFP preprocessing ─────→ ONE / SpikeInterface
inventory ─────────────→ cache filesystem
parameter recovery ────→ explicit local source paths
```

Several consumers independently own source-access details.

---

## Migration state

```text
visual-replay ──┐
                │
neural-data ────┼────→ session-data boundary
                │             │
LFP access ─────┘             ├── ONE
                              ├── Brainbox
                              └── SpikeInterface
```

Existing underlying loaders may continue to operate.

Their ownership changes before their implementation necessarily changes.

---

## Target state

```text
                 ONE / Alyx
                     │
                     ▼
                session-data
                 /         \
                ▼           ▼
        visual-replay    neural-data
```

Normal downstream components no longer own independent project-level IBL acquisition policy.

Specialized internal source-access strategies may remain different where appropriate.

---

# Package organization

The target implementation should expose a recognizable source boundary for `session-data`.

A possible initial organization is:

```text
src/session_data/
    __init__.py
    access.py
    metadata.py
    sessions.py
```

An `errors.py` module or specialized access modules should be added only if actual implementation complexity justifies them.

Possible later splits include:

```text
alf.py
spikes.py
ephys.py
```

but these are not required architecture.

Do not mirror every conceptual responsibility with a separate Python module merely for symmetry.

The implementation may also reuse existing modules outside `src/session_data/` where moving them would add cost without improving the boundary.

The component boundary is architectural, not dependent on perfect physical file relocation.

---

# Dependency direction

The conceptual dependency direction is:

```text
consumer
   ↓
session-data public boundary
   ↓
source access / metadata
   ↓
ONE / Brainbox / SpikeInterface
```

`session-data` must not depend on:

* `visual-replay`;
* `visual-features`;
* neural preprocessing behavior;
* alignment;
* training datasets;
* model implementation;
* training;
* evaluation.

Consumer-specific interpretation must not flow back into source access.

---

# Relationship to documentation

This architecture defines the accepted internal organization and migration direction.

The following remain separate responsibilities.

## `interface.md`

Defines the concrete current technical contract, including where appropriate:

* public callable boundaries;
* request arguments;
* return representations;
* acquisition-policy representation;
* source/provenance representation;
* failure representation.

## `audit.md`

Compares current source code against:

* `spec.md`;
* this architecture;
* `interface.md`;
* project dependencies.

It determines which existing behavior should be:

```text
Keep
Wrap / centralize
Refactor
Replace
Remove
```

The audit must not assume that architectural cleanliness requires rewriting working legacy code.

## `tasks.md`

Contains the minimal work required to resolve the accepted audit findings.

There is no required one-to-one mapping between architecture sections, audit findings, and tasks.

---

# Architectural principles

1. **Provide one ViNED boundary for original IBL session-data access.**
2. **Reuse supported IBL loaders rather than replacing them.**
3. **Centralize acquisition policy, not every internal data representation.**
4. **Keep source access separate from visual and neural transformations.**
5. **Use the ONE cache instead of creating a duplicate IBL source-data store.**
6. **Keep collection and revision identity visible where relevant.**
7. **Hide routine cache/filesystem mechanics from downstream consumers.**
8. **Preserve EID and source identity across the component boundary.**
9. **Allow specialized source loaders when they provide useful existing behavior.**
10. **Introduce abstractions only when they enforce a real component boundary or requirement.**
11. **Preserve useful existing structure when it already satisfies required behavior.**
12. **Prefer extension, wrapping, and adaptation over invasive modification of working legacy code.**
13. **Refactor because of a concrete specification, correctness, interface, boundary, or dependency problem—not merely because a cleaner redesign is possible.**
14. **For greenfield functionality, design directly for the accepted specification instead of inheriting unnecessary legacy constraints.**
15. **Optimize for the smallest architectural change that produces a coherent and maintainable component boundary.**
16. **Do not force unrelated IBL data families through one artificial loader or container.**
17. **Do not allow downstream consumers to redefine source acquisition semantics.**
18. **Do not create architectural work whose only benefit is aesthetic uniformity.**
