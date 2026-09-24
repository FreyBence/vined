# Context selection rules

These rules define which repository documentation should be read for a task.

The goal is to provide the minimum authoritative context required to perform the current work correctly.

Do not recursively inspect documentation merely because additional files exist or may contain potentially useful information.

## Core principle

Start from the target component.

Read:

1. repository-wide agent rules;
2. the project documentation map;
3. the project dependency graph;
4. the authoritative documents of the target component;
5. only the interfaces of other components that are relevant through declared dependencies.

Do not read the internal documentation of another component unless the current task explicitly requires work inside that component.

The existence of a document does not make it relevant to the current task.

---

## Component boundaries

A component is a meaningful functional part of the research pipeline with an independently understandable responsibility.

Examples include:

* visual replay;
* visual feature extraction;
* neural data preparation;
* visual–neural alignment;
* dataset construction;
* model definition;
* training;
* evaluation.

Source files, helper modules, classes, and utilities are not automatically separate components.

Work should remain inside the target component unless the accepted task explicitly includes a cross-component change.

---

## Documentation roles

Each component may contain:

```text
spec.md
architecture.md
interface.md
audit.md
tasks.md
archive/
```

Their roles are different.

### `spec.md`

Defines required component behavior.

Authoritative.

Read when working on the component.

Codex must not modify it unless the user explicitly requests a specification change.

### `architecture.md`

Defines accepted internal structure or design where a separate architecture document is necessary.

Authoritative.

Read when present and when working on the component.

Codex must not modify it unless the user explicitly requests an architectural change.

### `interface.md`

Defines the current implemented boundary of the component:

* inputs;
* outputs;
* artifacts;
* identifiers;
* timing/unit conventions;
* compatibility information relevant across component boundaries.

Codex may update it when implementation changes the actual interface while remaining compatible with the accepted specification, architecture, and project dependency graph.

It must not be used to silently redefine requirements or architecture.

### `audit.md`

A snapshot comparing repository state with the accepted specification and architecture.

Read when performing an audit or deriving a new implementation plan.

Do not normally read it while executing a sufficiently clear existing task.

Do not maintain it as a progress report during implementation.

### `tasks.md`

Contains the accepted implementation plan and completion state.

Read when implementing planned work.

Follow `planning.md` when creating or modifying it.

### `archive/`

Contains inactive historical documentation.

Treat archived documentation as outside normal agent context.

Do not inspect `archive/` during normal planning, auditing, or implementation.

Read a specific archived artifact only when:

* the user explicitly requests historical information; or
* an active authoritative document explicitly identifies that artifact as necessary evidence.

Do not search archives merely because current information is incomplete.

---

## Project dependency graph

`docs/dependencies.md` defines allowed component-level dependencies.

Use it to determine which external component interfaces may be relevant.

Do not infer arbitrary cross-component relationships from repository proximity, imports, filenames, or historical documents when an explicit component dependency already exists.

Do not introduce:

* reverse dependencies;
* dependency cycles;
* new cross-component dependencies;

unless the user explicitly approves an architecture change.

---

## Cross-component context

When component `B` depends on component `A`, work inside `B` may read:

```text
A/interface.md
```

to understand the contract supplied by `A`.

It should not normally read:

```text
A/spec.md
A/architecture.md
A/audit.md
A/tasks.md
A/archive/*
```

The implementation details of `A` are hidden behind its interface for work performed in `B`.

Read another component's internal documentation only if the current task explicitly includes changes to that component.

---

## Context by work type

### Implementation

Read:

* target `tasks.md`;
* target `spec.md`;
* target `architecture.md` if present;
* target `interface.md`;
* `docs/dependencies.md`;
* interfaces of directly relevant dependencies;
* relevant source code.

Do not read the target audit by default when the task already provides sufficient direction.

Do not read unrelated component documentation.

### Audit

Read:

* target `spec.md`;
* target `architecture.md` if present;
* target `interface.md`;
* `docs/dependencies.md`;
* directly relevant dependency interfaces;
* relevant source code.

The audit may inspect the implementation broadly inside the target component.

It must not broaden into unrelated documentation or redesign the component.

### Planning

Read:

* target `spec.md`;
* target `architecture.md` if present;
* target `interface.md`;
* target `audit.md`;
* `docs/dependencies.md`;
* directly relevant dependency interfaces.

Create or revise `tasks.md` according to `planning.md`.

### Interface maintenance

Read:

* target specification and architecture;
* current target interface;
* affected dependency interfaces;
* the implemented boundary being changed.

Update only factual current contract information.

If the required interface change conflicts with the specification, architecture, or dependency graph, do not silently change those authoritative documents.

---

## Progressive disclosure

Do not preload all project documentation.

Expand context only when the current task requires it.

Use this order:

```text
repository rules
→ project documentation map
→ dependency graph
→ target component
→ required dependency interfaces
→ relevant source code
```

Stop expanding context once enough information is available to perform the task correctly.

Do not perform repository-wide documentation discovery as a precaution.

---

## Cross-component conflicts

If implementation inside one component appears to require changing another component:

1. verify the declared dependency and interface;
2. determine whether the accepted task already includes that change;
3. if it does, work only within the explicitly included components;
4. otherwise do not silently expand the task.

Do not redesign the dependency graph or another component to make the current implementation easier.

---

## Default context bias

When uncertain:

* read less documentation;
* prefer authoritative active documents;
* prefer interfaces over another component's internals;
* prefer the target component over repository-wide exploration;
* ignore historical material unless specifically required;
* expand context only in response to a concrete missing dependency or requirement.

The goal is not to know everything about the repository.

The goal is to know exactly what is necessary to perform the current task correctly.
