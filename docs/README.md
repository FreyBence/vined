# ViNED documentation structure

This directory contains the active technical documentation for ViNED.

Documentation is organized by functional component rather than by source-code module.

The documentation system separates:

* project structure;
* component requirements;
* component architecture;
* implemented interfaces;
* repository audits;
* implementation plans;
* historical material.

Repository-wide agent behavior is defined under `.agent/rules/`.

## Documentation hierarchy

The project-level research specification is primarily represented by the MSc research work and is not duplicated as a single repository specification.

The repository represents the project architecture through:

1. the component structure under `docs/`;
2. the project-level dependency graph in `dependencies.md`;
3. component specifications and optional component architectures.

A component is a meaningful functional part of the research pipeline, not necessarily a Python package or module.

---

## Project-level documents

### `README.md`

Defines how repository documentation is organized and interpreted.

### `dependencies.md`

Defines the acyclic dependency relationships between project components.

This is an architectural document.

It is authoritative and must not be modified by Codex unless the user explicitly requests an architecture change.

---

## Component directories

Each major component receives a directory:

```text
docs/<component>/
```

A component directory may contain:

```text
spec.md
architecture.md
interface.md
audit.md
tasks.md
archive/
```

Not every component requires every document.

Small components normally do not require a separate `architecture.md` when their structure is sufficiently clear from the specification and implementation.

---

## `spec.md`

Defines what the component is required to do.

The specification is created and maintained through human/ChatGPT design work.

It is authoritative input to Codex.

Codex must not modify a specification merely because the current implementation differs from it or because another design appears easier.

A difference between implementation and specification is an audit finding, not permission to rewrite the specification.

---

## `architecture.md`

Defines how a sufficiently complex component is intentionally structured.

It is optional.

Use it only when internal relationships, stages, boundaries, or design decisions are too complex to be represented clearly by the specification alone.

Architecture documents are created and maintained through human/ChatGPT design work.

They are authoritative input to Codex and must not be modified without an explicit architecture change request.

---

## `interface.md`

Defines the component's current technical boundary.

It should contain only information required by the component itself or its consumers, such as:

* inputs;
* outputs;
* produced artifacts;
* required identifiers;
* data shapes where important;
* time and unit conventions;
* version/compatibility fields relevant across boundaries.

`interface.md` describes the actual implemented contract.

Codex may maintain it as implementation evolves, provided the changes remain compatible with:

* the component specification;
* the component architecture;
* the project dependency graph.

It must remain concise.

It is not:

* an implementation guide;
* an architecture document;
* a progress report;
* an audit;
* a task list.

---

## `audit.md`

An audit compares the current repository implementation against the accepted component specification and architecture.

It identifies only meaningful gaps such as:

* implemented;
* partially implemented;
* missing;
* incompatible;
* obsolete behavior.

An audit is a snapshot created for a development cycle.

It is not maintained continuously during implementation.

It must not redesign the component or expand its scope.

Audit findings are converted into implementation tasks.

---

## `tasks.md`

Contains the implementation work derived from the accepted specification, architecture, and audit.

Follow `.agent/rules/planning.md`.

Task files are stable plans.

Normal implementation-time changes should preferably be limited to:

```text
[ ] → [x]
```

Task files are not progress reports.

---

## `archive/`

Each component maintains its own archive when historical documentation must be preserved.

Use:

```text
docs/<component>/archive/
```

rather than one repository-wide archive.

This preserves component context while keeping inactive material outside the active documentation set.

Typical archived material includes:

* superseded specifications;
* old architecture drafts;
* previous task plans;
* investigation notes;
* feasibility reports;
* old audits;
* temporary design explorations.

Archived files are not part of normal agent context.

Do not use them as active requirements.

Version control preserves history; archive only documents whose historical content remains worth retaining.

---

## Document ownership

| Document          | Role                           | Codex modification              |
| ----------------- | ------------------------------ | ------------------------------- |
| `docs/README.md`  | Documentation structure        | No, unless explicitly requested |
| `dependencies.md` | Project component architecture | No                              |
| `spec.md`         | Required component behavior    | No                              |
| `architecture.md` | Accepted component design      | No                              |
| `interface.md`    | Current implemented contract   | Yes                             |
| `audit.md`        | Implementation gap snapshot    | Yes, during audit               |
| `tasks.md`        | Implementation plan/status     | Yes, under planning rules       |
| `archive/*`       | Historical material            | Normally no                     |

When a conflict exists, authoritative design documents take precedence over implementation-support documents.

---

## Component documentation lifecycle

The intended lifecycle is:

```text
research/project intent
        ↓
component spec
        ↓
optional component architecture
        ↓
repository audit
        ↓
task plan
        ↓
implementation
        ↓
interface maintained as needed
```

The first two design stages are not Codex responsibilities unless the user explicitly requests otherwise.

Codex primarily operates from an already accepted design.

---

## Documentation scope

Keep active documentation small.

Do not create a new document merely because information could be documented separately.

Prefer:

* one canonical source for each fact;
* references instead of duplication;
* active current-state documentation instead of historical narratives;
* component-local documentation instead of repository-wide mixed documents.

If a document no longer contributes to understanding or operating the current system, either archive it or remove it.

---

## Context selection

Follow `.agent/rules/context.md`.

Do not recursively read the entire `docs/` directory.

For cross-component work, use declared dependencies and `interface.md` files rather than loading the internal documentation of every connected component.
