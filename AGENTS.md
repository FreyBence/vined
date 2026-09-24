# Repository instructions

These instructions apply to the entire repository.

## Required repository rules

Before planning, auditing, or implementing work, read and follow:

* [.agent/rules/instructions.md](.agent/rules/instructions.md) for stable project context and development conventions.
* [.agent/rules/scope.md](.agent/rules/scope.md) for project scope, proportional validation, completion criteria, and rules against unnecessary scope expansion.
* [.agent/rules/planning.md](.agent/rules/planning.md) for task decomposition, dependency planning, task-file structure, and progress tracking.
* [.agent/rules/context.md](.agent/rules/context.md) for selecting the minimum authoritative documentation required for the current component and task.
* [.agent/rules/audit.md](.agent/rules/audit.md) for component audits, finding priorities, repository-gap classification, and audit output structure.

These rules apply throughout the task, not only during initial planning.

## Documentation system

The active project documentation is organized under `docs/`.

Use:

* [docs/README.md](docs/README.md) to understand documentation roles, ownership, and component structure.
* [docs/dependencies.md](docs/dependencies.md) as the authoritative component-level dependency graph.

Do not recursively read the `docs/` directory.

Use `.agent/rules/context.md` to determine which component documents and dependency interfaces are relevant to the current work.

The existence of a document does not make it part of the current task context.

## Documentation authority

Treat the following as authoritative design inputs:

* `docs/dependencies.md`
* component `spec.md`
* component `architecture.md`, when present

Do not modify these files unless the user explicitly requests a specification or architecture change.

A mismatch between implementation and an authoritative document is not permission to rewrite the authoritative document.

Codex-maintained implementation-support documents are:

* component `interface.md`
* component `audit.md`
* component `tasks.md`

Modify them only according to their roles defined in `docs/README.md`, `.agent/rules/context.md`, and `.agent/rules/planning.md`.

Component `archive/` directories are outside normal working context. Do not inspect them unless explicitly required by the user or by a specific active authoritative document.

## Component boundaries

Keep work inside the target component unless the accepted task explicitly spans multiple components.

Cross component boundaries only through dependencies declared in `docs/dependencies.md`.

When another component is relevant only as a dependency, prefer its `interface.md` rather than reading its internal specification, architecture, audit, task list, or archive.

Do not introduce new component dependencies, reverse existing dependencies, or create dependency cycles without explicit user approval of an architecture change.

## Scope discipline

Keep all work proportional to the concrete goal of the current task and the needs of the research pipeline.

Follow `.agent/rules/scope.md` when deciding:

* what needs to be implemented;
* what level of validation is justified;
* whether an uncertainty needs to be resolved now;
* whether additional infrastructure or investigation is necessary;
* and when a task is complete.

Do not expand a task into stronger validation, benchmarking, historical reconstruction, infrastructure work, optimization, or unrelated research unless it is required by the requested functionality, a concrete correctness risk, a downstream dependency, or an explicit user requirement.

When functional completion and additional confidence-building work compete for effort, prioritize completing the required functional pipeline according to `.agent/rules/scope.md`.

## Planning and task tracking

Follow `.agent/rules/planning.md` whenever creating, revising, or updating implementation plans or `tasks.md` files.

Keep plans minimal, stable, and focused on meaningful functional outcomes.

In particular:

* create only tasks required for the intended project outcome;
* avoid turning implementation details, investigation, validation, or discovered possibilities into separate tasks unless they represent a real completion boundary;
* maintain one fixed dependency list rather than dynamic blocked/ready/in-progress states;
* give each task a concise summary and out-of-scope boundary;
* avoid progress reports, implementation journals, evidence logs, completion dates, runtime summaries, and similar historical detail in task files;
* after a plan is accepted, prefer `[ ] → [x]` as the only routine modification;
* modify the plan structure only when the existing plan is materially incorrect, incomplete, or changed by an explicit requirement.

Use `.agent/rules/scope.md` to decide whether work belongs in the project and `.agent/rules/planning.md` to decide how that work should be represented and tracked.

## Audit behavior

Follow `.agent/rules/audit.md` when inspecting a component against its accepted specification, architecture, interfaces, and dependency boundaries.

Audits identify the current implementation delta. They do not redesign components, modify authoritative design documents, or act as progress reports.

## No test writing

* Do not create, add, or modify test code in this repository, including unit, integration, regression, end-to-end, snapshot, or temporary test scripts.
* Validate changes through code inspection, existing checks, execution of the implemented functionality, or manual verification without writing tests.
* Apply the validation principles in `.agent/rules/scope.md`: validation should be proportional to the concrete risk and project claim.
* Do not create automated validation infrastructure as a substitute for directly demonstrating that the requested functionality works.
* Do not delete existing tests unless the user explicitly requests their removal.

## No CI/CD automation

* This repository does not use continuous integration, continuous delivery, or continuous deployment.
* Do not add CI/CD workflows, pipeline configuration, automated dependency-update bots, or scripts and documentation intended to support those systems.
* Use code inspection, existing local checks, direct execution, and manual verification as appropriate.
* Review dependency updates manually and keep them consistent with the repository's constraints files.

## Task completion

Unless the user or a task specification explicitly requires something stronger, stop when the requested functionality satisfies the completion principles in `.agent/rules/scope.md`.

When task tracking is involved, update the plan according to `.agent/rules/planning.md`.

Normally this means:

```text
[ ] → [x]
```

without adding a completion report or rewriting the plan.

Do not create additional work merely because a more exhaustive implementation, validation strategy, research investigation, abstraction, or future-proofing effort would be possible.
