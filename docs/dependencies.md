# ViNED component dependencies

This document defines the project-level dependency graph between ViNED components.

It describes architectural dependency direction, not execution status or task order.

The graph must remain acyclic.

Codex must treat this document as authoritative architecture and must not modify it unless the user explicitly requests a project-level architecture change.

## Components

The current top-level components are:

* `session-data`
* `visual-replay`
* `visual-features`
* `neural-data`
* `alignment`
* `training-dataset`
* `model`
* `training`
* `evaluation`

## External data source

IBL ONE/Alyx is an external data source, not a ViNED component.

Access to ONE, dataset discovery, downloading, local materialization, revision handling, and caching belong to the `session-data` component.

Other components should consume the `session-data` interface instead of independently implementing ONE/cache handling.

## Dependency graph

```text
                         ONE / Alyx
                             │
                             ▼
                        session-data
                         /         \
                        ▼           ▼
                visual-replay    neural-data
                        │           │
                        ▼           │
                visual-features     │
                        \           /
                         ▼         ▼
                          alignment
                              │
                              ▼
                      training-dataset
                         /         \
                        ▼           ▼
                    training     evaluation
                       │            ▲
                       ▼            │
                     model ─────────┘
```

Explicit component dependencies:

```text
session-data → visual-replay
session-data → neural-data

visual-replay → visual-features

visual-features → alignment
neural-data → alignment

alignment → training-dataset

training-dataset → training
model → training

training-dataset → evaluation
model → evaluation
training → evaluation
```

Arrows indicate that a downstream component may consume the declared interface of the upstream component.

They do not require direct source-code imports between every connected component.

---

## Component responsibilities

### `session-data`

Provides the project-wide boundary to IBL session data.

Responsibilities include:

* resolving session identities and configured EIDs;
* discovering required ONE datasets;
* retrieving datasets when required;
* using and maintaining the project's local data/cache representation;
* resolving collections and revisions where relevant;
* exposing locally usable session data and metadata to downstream components;
* preserving the distinction between unavailable, not downloaded, cached, and locally materialized data where that distinction affects behavior.

Typical consumers are:

```text
visual-replay
neural-data
```

The component should hide ONE/download/cache mechanics from those consumers wherever practical.

It does not perform visual stimulus reconstruction or neural preprocessing.

---

### `visual-replay`

Produces trial-aligned visual stimulus representations from session evidence and accepted reconstruction assumptions.

Depends on:

```text
session-data
```

It may consume session data such as:

* trial tables;
* wheel observations;
* task/version metadata;
* raw task information when available;
* stimulus timing information.

It should not independently own general ONE download or cache management.

---

### `visual-features`

Consumes visual replay frames or equivalent replay outputs and produces model-ready visual feature representations.

Depends on:

```text
visual-replay
```

---

### `neural-data`

Produces neural activity representations and relevant neural metadata from session electrophysiological data.

Depends on:

```text
session-data
```

It should use the shared session-data boundary rather than implement an independent project-wide acquisition/cache path.

It has no dependency on the visual pipeline.

---

### `alignment`

Combines compatible visual and neural representations on the intended trial and physical-time structure.

Depends on:

```text
visual-features
neural-data
```

---

### `training-dataset`

Packages aligned multimodal representations into the model-ready dataset structures consumed by training and evaluation.

Depends on:

```text
alignment
```

This component is distinct from `session-data`.

`session-data` represents acquired/cached research data.

`training-dataset` represents processed model-ready samples.

---

### `model`

Defines the learnable multimodal architecture and model-level input/output behavior.

The model is conceptually separate from data acquisition and preparation.

---

### `training`

Runs optimization and adaptation using model-ready datasets and the model implementation.

Depends on:

```text
training-dataset
model
```

---

### `evaluation`

Evaluates trained or loaded model behavior using prepared data and the model implementation.

Depends on:

```text
training-dataset
model
training
```

The `training` dependency represents trained checkpoint/artifact production when evaluation consumes training outputs. Evaluation may also consume an already available compatible checkpoint.

---

## Dependency rules

Dependencies may only point in the declared direction.

A downstream component may read the `interface.md` of an upstream dependency.

It must not require knowledge of the upstream component's internal implementation unless the current task explicitly spans both components.

For example:

```text
visual-replay
```

may rely on:

```text
session-data/interface.md
```

but should not normally need to inspect the internal ONE download/cache implementation.

Do not introduce reverse dependencies or cycles to simplify implementation.

Examples of forbidden architectural drift include:

```text
session-data ← visual-replay-specific rendering behavior
session-data ← neural preprocessing rules
visual-replay ← visual-features internals
alignment ← training-dataset control logic
training-dataset ← model internals
```

If implementation appears to require a new component relationship, treat that as a project architecture question rather than silently changing this graph.

---

## Context routing

The dependency graph also controls documentation access.

For work on `visual-replay`, normal external component context is:

```text
session-data/interface.md
```

For work on `alignment`, normal external component context is:

```text
visual-features/interface.md
neural-data/interface.md
```

Do not recursively load the specifications, architectures, audits, task files, or archives of upstream components merely because they are dependencies.

---

## Status independence

This graph is independent of implementation progress.

It must not contain:

* blocked states;
* completion states;
* active task selection;
* temporary execution order;
* development history.

Those concerns do not alter project architecture.
