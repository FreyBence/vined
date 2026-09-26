# Interface Documentation Rule

This rule defines how the agent must create and maintain module-level
`interface.md` files after implementation is complete.

## Purpose

Each module may contain an `interface.md` describing the public interface exposed by the implemented module to the rest of the system.

The interface document is created or finalized **after implementation**.

Its purpose is to answer:

> How should another module use this module?

It documents the implemented public contract without describing unnecessary internal details.

---

# Documentation lifecycle

The project documentation flow is:

```text
spec
  ↓
optional architecture
  ↓
audit
  ↓
tasks
  ↓
implementation
  ↓
interface
```

Each document has a different responsibility:

### `spec.md`

Defines:

> What must the module do?

### `architecture.md`

When present, defines:

> What important internal structural relationships must be preserved?

### `audit.md`

Defines:

> How does the current implementation differ from the intended design?

### `tasks.md`

Defines:

> What implementation work is required?

### `interface.md`

Defines:

> How does the completed module expose its functionality to other modules?

---

# Source of truth

`interface.md` must be based on the **completed implementation**, but it must remain compatible with:

1. `spec.md`;
2. `architecture.md`, when present;
3. the completed implementation tasks.

The interface documentation must not be used to legitimize an implementation that violates the specification.

If the implementation and specification disagree, the mismatch must be resolved in the implementation rather than silently documented as the intended interface.

---

# Public interface only

The interface document must describe only elements that another module or external caller needs to use.

This may include:

- public functions;
- public classes;
- public methods;
- public data structures;
- public configuration objects;
- serialized input/output schemas;
- supported command-line entry points;
- stable file or artifact formats;
- errors that callers are expected to handle.

It must not document every internal helper function.

---

# Concrete implementation details are allowed

Unlike `spec.md` and `architecture.md`, `interface.md` may reference concrete implementation symbols.

For example:

```text
src/alignment/alignment.py
align_trial(...)
```

or:

```python
def align_trial(
    neural: NeuralTrial,
    visual: VisualTrial,
    config: AlignmentConfig,
) -> AlignedTrial:
```

Concrete signatures are useful here because the implementation already exists.

However, internal implementation structure should still be omitted unless callers depend on it.

---

# Required sections

Each `interface.md` should contain only the sections relevant to that module.

A typical interface document should include:

## 1. Purpose

A short description of what the module exposes.

Do not repeat the full specification.

---

## 2. Public entry points

List the supported public functions, classes, commands, or services.

For each entry point include:

- symbol name;
- source location;
- purpose;
- parameters;
- return value;
- important exceptions or failure conditions.

Example:

```text
align_trial(...)
```

Purpose:

> Align one neural trial and one visual-feature trial onto the common neural temporal grid.

---

## 3. Input contracts

Document the concrete data accepted by the implementation.

Include:

- type;
- shape;
- required fields;
- units;
- identity fields;
- relevant validity semantics.

Example:

```text
visual_features: float32[T_visual, D_visual]
timestamps: float64[T_visual]
session_id: str
trial_id: int
```

Do not repeat scientific explanations already fully covered by the spec unless needed to use the API correctly.

---

## 4. Output contracts

Document the concrete output representation.

Include:

- type;
- shape;
- important fields;
- units;
- ordering guarantees;
- masks;
- identity information.

Example:

```text
AlignedTrial
    neural_activity: [T, N]
    visual_features: [T, D]
    bin_center_times: [T]
```

---

## 5. Shape semantics

Where tensors are exposed, define every axis.

For example:

```text
[B, T, N]
```

where:

- `B` = batch;
- `T` = temporal positions;
- `N` = neural channels.

Never assume tensor axes are obvious.

---

## 6. Time and unit semantics

Explicitly document relevant units.

Examples:

```text
timestamps       → seconds in session clock
bin_size         → seconds
spike values     → spike counts per bin
model output     → log expected spike count
```

Do not rely on variable names alone to communicate units.

---

## 7. Identity semantics

Where applicable, explain how identifiers are represented.

Examples:

- `session_id`;
- `trial_id`;
- neuron/cluster identity;
- sample identity;
- checkpoint identity.

Document whether ordering has semantic meaning.

---

## 8. Validity and padding

Where the interface exposes masks, document them explicitly.

For example:

```text
temporal_mask[t] = True
```

means:

> the temporal position contains a real scientific observation.

and:

```text
temporal_mask[t] = False
```

means:

> the position is padding.

A numeric zero must never implicitly define validity.

---

## 9. Configuration

Document only configuration that callers need to provide or understand.

Include:

- field names;
- allowed values;
- defaults where stable;
- effect on the interface.

Do not duplicate the entire configuration file if most fields are internal.

---

## 10. Error behavior

Document errors that callers can reasonably encounter.

Examples:

- session mismatch;
- unsupported feature dimension;
- malformed timestamps;
- incompatible checkpoint;
- missing visual coverage.

Describe whether the implementation:

- raises an exception;
- rejects the item;
- returns an explicit invalid result.

Do not document unexpected implementation bugs as supported behavior.

---

## 11. Persistence format

If the module exposes persisted artifacts, document their external schema.

Examples:

- `.npz`;
- `.pt`;
- `.npy`;
- JSON metadata;
- directory layouts.

Document fields and semantic meaning, not internal serialization code.

---

## 12. Minimal usage example

Where useful, include one short example showing the intended public usage.

Example:

```python
aligned = align_trial(
    neural=neural_trial,
    visual=visual_trial,
    config=config,
)
```

Examples should use public interfaces only.

Do not use private helper functions.

---

# Interface stability

The interface document describes the supported boundary between modules.

Internal refactoring should not require changes to `interface.md` unless the public contract changes.

For example:

```text
StitchEncoder implementation changed
```

does not necessarily change the model interface.

But:

```text
model output changed
[B,T,N] → [B,N,T]
```

does change the interface and must update `interface.md`.

---

# Do not expose unnecessary internals

Avoid documenting:

- private helper functions;
- internal class inheritance;
- internal caches;
- temporary implementation structures;
- exact call chains;
- optimizer internals;
- debugging utilities;
- implementation-specific intermediate tensors;

unless another module directly depends on them.

If another module currently depends on a private implementation detail, this should be treated as a possible audit/design concern rather than automatically declaring it public.

---

# No new design decisions

Creating `interface.md` must not introduce new behavior.

Codex must not decide during interface generation that:

- a parameter should become optional;
- a new default should exist;
- a new public function should be added;
- a different serialization format should be used;
- a new fallback should exist.

If the implementation does not provide an interface required by the spec, this is an implementation issue.

The interface file documents the finished contract; it does not redesign it.

---

# Public versus internal classification

When inspecting the completed implementation, classify symbols as:

### Public

Used or intended to be used across module boundaries.

Document these.

### Internal

Used only inside the module.

Do not document these unless their behavior is necessary to understand a public contract.

### Legacy

Present for compatibility but not part of the required current pipeline.

Document these only in an optional compatibility section if they remain intentionally supported.

---

# Cross-module verification

Before completing `interface.md`, verify calls from direct downstream modules.

For example:

```text
visual-features → alignment
alignment → training-dataset
training-dataset → training
training → model
model → evaluation
```

Check that:

- argument names/types are compatible;
- tensor shapes agree;
- identity fields agree;
- masks have the same semantics;
- units agree;
- returned data actually contains what the consumer expects.

The interface file should describe the resulting confirmed boundary.

---

# No duplication of specification

Do not copy large parts of `spec.md` into `interface.md`.

For example, the spec may explain why:

> zero spikes represent valid neural observations.

The interface only needs the operational contract:

```text
neural_activity[t, n] == 0
```

is valid when:

```text
temporal_mask[t] == True
neuron_mask[n] == True
```

The interface should be precise and practical.

---

# No duplication of architecture

Do not reproduce internal architecture diagrams unless they are required to understand public usage.

For example:

```text
visual projection
→ transformer
→ neural stitcher
```

belongs in `model/architecture.md`.

The public model interface instead documents:

```text
inputs
→ model(...)
→ outputs
```

---

# Legacy behavior

Inherited NEDS functionality may be documented separately when intentionally retained.

Example:

```text
## Optional legacy interfaces
```

Such interfaces must be clearly distinguished from the required current pipeline.

The existence of legacy code alone does not make it part of the public interface.

---

# Documentation accuracy

Codex must inspect the final implementation before writing `interface.md`.

Do not infer signatures from:

- old documentation;
- audit findings;
- task descriptions;
- previous NEDS behavior.

The final implementation is authoritative for concrete symbol names and signatures, subject to compatibility with the spec and architecture.

---

# Interface completion criteria

An `interface.md` is complete when a developer working on the directly connected module can determine:

1. what they should call;
2. what they must provide;
3. what they receive;
4. what shapes and units are used;
5. how identity is represented;
6. how validity and padding are represented;
7. what errors they must handle;
8. which interface is stable/public;
9. which legacy interfaces are optional;
10. how to use the interface without reading module internals.

---

# Recommended generation procedure

After implementation is complete, Codex should:

```text
1. Read spec.md
2. Read architecture.md if present
3. Read completed audit.md and tasks.md for context
4. Inspect the final implementation
5. Identify actual cross-module callers
6. Identify public entry points
7. Verify input/output contracts against callers
8. Verify shapes, units, masks and identities
9. Write interface.md
10. Re-check direct producer/consumer compatibility
```

The final document must describe the implemented and supported module boundary, not the historical development process.

---

# Final principle

`interface.md` answers:

> How do I correctly use this completed module without needing to understand its internal implementation?

If information is not required to answer that question, it normally does not belong in the interface document.