# Session Data Specification

## Purpose

The `session-data` component provides the project-wide boundary between ViNED and original IBL session data.

Its responsibility is to make session data and session metadata available to downstream components without requiring those components to independently implement:

* ONE/Alyx access;
* dataset discovery;
* remote acquisition;
* ONE cache handling;
* session-path resolution;
* collection handling;
* dataset revision handling;
* session metadata lookup.

The component represents **source research data access**, not transformed model-ready data.

Downstream components should request the session information they require and remain independent of the mechanics used to locate or obtain it.

---

## External source

The primary external source is the International Brain Laboratory data ecosystem accessed through ONE/Alyx.

ONE/Alyx is external to ViNED and is not itself a project component.

The `session-data` component is the ViNED boundary to that external system.

---

## Responsibilities

### Session identity

The component must support canonical IBL session identification by EID.

It must:

* accept valid session identities;
* preserve canonical EIDs;
* resolve session metadata when required;
* distinguish an invalid identifier from a valid identifier that cannot be resolved.

The component may support reading configured lists of EIDs, but experiment-specific roles such as training, validation, or testing are not part of session-data semantics.

---

### Session metadata

The component must make relevant session-level metadata available when requested.

Relevant metadata may include:

* EID;
* subject;
* laboratory;
* session date/time;
* session number;
* task protocol;
* task/software version information when available;
* probe/insertion identity where required;
* dataset collection and revision information.

The component must preserve the distinction between:

* metadata directly reported by IBL/Alyx;
* information derived locally;
* information that is unavailable.

It must not silently invent unavailable metadata.

---

### Dataset discovery

The component must support determining whether required IBL datasets are available for a session.

This includes dataset families required by current or future downstream components, such as:

* ALF trial data;
* wheel position and timestamps;
* stimulus timing data;
* spike-sorting data;
* neural channel/cluster metadata;
* probe/insertion information;
* raw task information when available;
* synchronization data when required;
* LFP/raw electrophysiology streams when explicitly requested.

The component must not assume that every session contains every dataset.

---

### Data acquisition

The component must provide a consistent path for obtaining required session datasets.

When remote acquisition is allowed, missing local data may be obtained through the supported IBL data-access mechanisms.

When remote acquisition is not allowed, unavailable local data must be reported explicitly rather than silently triggering an alternative source or fabricated fallback.

Remote acquisition policy must therefore be explicit rather than an accidental consequence of whichever ONE function a downstream component happened to call.

The component must not require downstream consumers to construct or configure their own ONE client for ordinary session-data access.

---

### Local and cached data

The component must support locally available IBL data and ONE-managed cached data.

It must distinguish relevant states such as:

* available locally;
* available through the configured ONE cache;
* known remotely but not currently local;
* unavailable;
* unresolved because remote access was not performed or failed.

The exact technical representation of these states belongs to the component interface and architecture rather than this specification.

The component must not create a second independent copy of the original IBL session tree merely to provide its abstraction.

Derived ViNED artifacts are outside this responsibility.

---

### Collections

IBL datasets may exist in different collections.

The component must preserve collection identity where it affects which data is used.

A downstream component must not need to know the filesystem layout used to represent a collection in the ONE cache.

Where a caller requires a specific collection, that requirement must be respected.

Where no collection is specified, the effective collection used must remain determinable.

---

### Dataset revisions

Dataset revisions must be treated as meaningful source identity.

The component must:

* preserve the effective revision when it can be determined;
* allow explicitly required revisions to be distinguished;
* avoid silently combining datasets from incompatible revisions;
* expose enough source identity for downstream provenance when revision matters.

The component is not required to reconstruct unavailable historical revisions.

When multiple applicable revisions exist and selecting between them could change the consumed data, the choice must not be hidden.

The concrete revision-resolution policy belongs to the architecture and interface.

---

### Consistent session source

Data returned together for one operation must represent a coherent session source.

The component must avoid silently combining:

* different sessions;
* incompatible revisions;
* mismatched collections;
* unrelated local files discovered only because they have similar names.

Where source consistency cannot be established sufficiently for the requested operation, the condition must be visible to the consumer.

---

### Data availability and failure

Failure modes must remain distinguishable where they have different meanings.

Examples include:

* invalid EID;
* unknown session;
* missing required dataset;
* remote dataset not downloaded;
* requested revision unavailable;
* local cached file missing or inconsistent;
* network or remote-service failure;
* unsupported data request.

The component must not convert all such conditions into an indistinguishable empty result.

Whether a particular condition should stop a downstream operation remains the responsibility of that operation.

---

## Consumers

The primary current consumers are expected to include:

### Visual replay

Requires source information such as:

* trial data;
* wheel observations;
* stimulus timing;
* task/session metadata;
* raw visual/task evidence when available.

`visual-replay` owns stimulus reconstruction.

`session-data` only provides the underlying session evidence.

### Neural data

Requires source information such as:

* spike-sorting results;
* clusters/channels;
* probe identities;
* session metadata;
* trial events;
* optional electrophysiological streams such as LFP.

`neural-data` owns neural preprocessing and representation.

`session-data` only provides access to the underlying source data.

Other components should normally consume transformed outputs from these components rather than bypass them to access IBL source data directly.

---

## Source-data boundary

The component ends at access to original or externally supplied IBL session data.

It does not own ViNED-generated derivatives such as:

* reconstructed visual stimulus frames;
* replay videos;
* replay sidecars;
* recovered normalized stimulus parameter manifests;
* CLIP feature archives;
* aligned visual/neural representations;
* Hugging Face training datasets;
* model sample caches;
* checkpoints;
* evaluation results.

Those belong to downstream components.

---

## Session lists and experiment splits

Canonical EID parsing and session identity handling may be shared through this component.

However, semantic experiment membership is outside its responsibility.

The component does not decide that a session is:

* training;
* validation;
* testing;
* held-out;
* fine-tuning;
* evaluation-only.

Those roles belong to experiment/data-pipeline configuration outside the source-data access layer.

The same IBL session may therefore be consumed in different experiment roles without changing its session-data representation.

---

## Transformation boundary

The component may perform only transformations necessary to access and consistently expose source data.

It must not perform domain-specific transformations owned by consumers.

Examples outside this component include:

* Gabor/stimulus reconstruction;
* wheel-to-stimulus trajectory reconstruction;
* CLIP preprocessing;
* neural spike binning for model input;
* neuron filtering for a specific experiment;
* visual-neural temporal alignment;
* train/validation/test splitting;
* model-specific padding;
* training-time masking.

Library-level decoding required simply to load an IBL dataset does not count as a downstream research transformation.

---

## Provenance

Downstream processing must be able to determine the source identity of data when that identity can materially affect reproducibility.

Relevant provenance may include:

* EID;
* dataset name/type;
* collection;
* effective revision;
* remote dataset identifier where available;
* local/cache origin where useful;
* source metadata relevant to historical compatibility.

The component should expose factual provenance.

It is not responsible for maintaining full downstream artifact provenance after source data leaves the component.

---

## Version-sensitive downstream use

Historical software or task versions may affect how downstream components interpret session data.

The `session-data` component must expose available version-related session metadata but must not decide how another component should reproduce historical behavior.

For example:

```text
session-data
    → exposes task/software version evidence

visual-replay
    → selects the appropriate rendering/source profile
```

This preserves the component boundary between source evidence and domain interpretation.

---

## Independence from consumers

Source-data access must not contain behavior that exists only to implement a specific downstream algorithm.

In particular:

* visual rendering rules must not enter `session-data`;
* neural preprocessing rules must not enter `session-data`;
* training configuration must not control source acquisition semantics;
* evaluation behavior must not alter how the underlying IBL session is identified.

Consumer-specific requirements may determine **which data is requested**, but not redefine the source-data component itself.

---

## Out of scope

The following are outside the `session-data` component:

* visual stimulus reconstruction;
* historical renderer selection;
* CLIP feature extraction;
* spike sorting;
* model-oriented spike binning;
* visual-neural alignment;
* train/validation/test experiment assignment;
* model dataset construction;
* training-time masking;
* model training;
* evaluation;
* storage of derived model artifacts;
* reconstruction of unavailable historical IBL infrastructure;
* creation of a new replacement for ONE/Alyx.

---

## Required behavioral properties

The component must provide the following project-level guarantees:

1. Downstream components do not need independent ONE/Alyx acquisition logic for supported session-data access.
2. A requested session remains identifiable by its original EID.
3. Dataset collection and revision identity are preserved where relevant.
4. Local availability and remote acquisition are not silently conflated.
5. Missing data is distinguishable from valid empty data.
6. Session metadata required for version-sensitive downstream behavior can be obtained when available.
7. Source-data access does not perform visual, neural, alignment, training, or evaluation transformations.
8. Multiple downstream components can consume the same session source without implementing competing cache/download logic.
9. Important source provenance remains available to downstream consumers.
10. The component does not silently substitute unrelated data when the requested source cannot be resolved.

---

## Relationship to other components

At project level:

```text
                 external
                ONE / Alyx
                    │
                    ▼
               session-data
                /         \
               ▼           ▼
       visual-replay    neural-data
```

The authoritative full component dependency graph is maintained in `docs/dependencies.md`.

This specification defines only the responsibilities of `session-data`.
