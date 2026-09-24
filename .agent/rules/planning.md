# Planning and task tracking rules

These rules define how implementation plans and `tasks.md` files should be created and maintained.

Read and apply [scope.md](scope.md) before creating, expanding, or revising a plan. Scope determines **whether work is necessary**; this file determines **how necessary work should be divided and tracked**.

## Purpose of task files

Task files are stable implementation plans and lightweight completion trackers.

They exist to answer:

1. What concrete work is part of the planned implementation?
2. What does each task mean?
3. What is explicitly outside each task's responsibility?
4. What are the fixed dependencies between tasks?
5. Which planned tasks are complete?

Task files are **not**:

* progress reports;
* implementation journals;
* execution logs;
* validation reports;
* research notebooks;
* changelogs;
* dynamic project-management boards.

The task file should change as little as possible after the initial plan is accepted.

The preferred implementation-time modification is:

```text
[ ] → [x]
```

Any other modification should be made only when it is genuinely necessary.

---

## Plan for the required outcome

Start planning from the concrete functional outcome requested by the user or required by the project pipeline.

Decompose only the work necessary to reach that outcome.

Before creating a task, ask:

> Is this work necessary to produce, integrate, or use a required project output?

If not, do not add it unless explicitly requested.

Apply the proportional validation and scope rules from `scope.md`.

Do not turn optional confidence-building, research, benchmarking, historical reconstruction, optimization, or future-proofing into required tasks.

---

## Prefer the smallest sufficient plan

Create the smallest task set that clearly represents the required implementation path.

Do not create tasks merely because implementation can be divided further.

Prefer:

```markdown
- [ ] Implement normalized stimulus parameter loading
```

over:

```markdown
- [ ] Add parameter dataclass
- [ ] Add loader function
- [ ] Add manifest parser
- [ ] Add validation helper
- [ ] Update imports
```

Those smaller actions are implementation details of the same functional task.

A separate task is justified when it represents a meaningful completion boundary, such as:

* a distinct functional capability;
* an independently consumed artifact;
* a required integration between major components;
* a genuine dependency boundary;
* a significant user-visible or research-pipeline result.

---

## Task structure

Each task should contain enough context to prevent implementation ambiguity without becoming a mini-specification or progress report.

Use this structure:

```markdown
### [ ] Task title

**Summary:**  
Short description of the concrete result this task must produce.

**Out of scope:**  
Short description of nearby work that must not be included in this task.
```

The summary should explain the intended result, not list low-level implementation steps.

The out-of-scope section should prevent predictable scope expansion.

Example:

```markdown
### [ ] Implement source-derived stimulus renderer

**Summary:**  
Generate stimulus frames from the normalized replay state using rendering behavior derived from the historically appropriate source profile.

**Out of scope:**  
Pixel-level comparison against the historical runtime, recreation of unavailable hardware, and unrelated renderer generalization.
```

A task title alone is usually insufficient when it could reasonably be interpreted in multiple ways.

Keep both sections concise.

---

## Out-of-scope guidance

The `Out of scope` section exists to constrain implementation, not to enumerate every imaginable excluded activity.

Include only exclusions that are likely to prevent real misunderstanding or scope creep.

Good:

```markdown
**Out of scope:**  
Historical framebuffer fidelity testing and reconstruction of unavailable display hardware.
```

Avoid:

```markdown
**Out of scope:**  
Do not rewrite module A, do not modify helper B, do not change CLI C, do not add logging D, do not optimize E...
```

unless those restrictions are genuinely required.

Global exclusions already defined in `scope.md`, `AGENT.md`, or repository rules do not need to be repeated in every task unless they are especially relevant to that task.

---

## Do not turn investigation into tasks by default

Investigation is usually part of implementation.

Do not automatically create separate tasks such as:

```markdown
- [ ] Investigate possible approaches
- [ ] Research validation strategies
- [ ] Analyze implementation differences
- [ ] Document findings
```

when those activities are simply needed to complete another task.

Research or investigation should become its own task only when:

* its result is itself a required project deliverable; or
* implementation cannot meaningfully proceed until a specific decision is resolved.

Otherwise, perform the necessary investigation inside the corresponding functional task.

---

## Do not create tasks from every discovered possibility

During implementation, new possibilities, unknowns, improvements, abstractions, edge cases, and validation ideas may appear.

Discovery alone does not justify changing the plan.

Before adding a task, ask:

> Does the current project outcome depend on resolving this?

If not, do not add it to the task file.

Do not grow the task list with:

* hypothetical future requirements;
* optional architectural improvements;
* stronger validation than the current project claim requires;
* speculative abstractions;
* unsupported edge cases with no current impact;
* optimization without a demonstrated bottleneck;
* additional documentation that is not required to use or maintain the implemented functionality.

Follow `scope.md` when uncertain.

---

## Task granularity

Each task should describe a concrete result rather than a list of coding operations.

Good:

```markdown
### [ ] Implement source-derived stimulus rendering

**Summary:**  
Produce stimulus frames from resolved replay parameters using the selected historical source profile.

**Out of scope:**  
Reference-image fidelity certification and unrelated backend abstraction.
```

Too granular:

```markdown
- [ ] Create renderer class
- [ ] Add shader helper
- [ ] Add coordinate conversion function
- [ ] Add CLI option
- [ ] Update imports
```

Too broad:

```markdown
- [ ] Finish visual pipeline
```

A task should normally be large enough to produce a meaningful result but small enough that completion is unambiguous.

---

## Validation belongs inside the task

Validation should normally be part of the functionality being implemented, not a separate workstream.

For example:

```markdown
### [ ] Implement source-derived stimulus renderer

**Summary:**  
Render representative replay inputs and produce usable stimulus output.

**Out of scope:**  
Historical pixel-fidelity certification.
```

may naturally include running representative inputs and inspecting the generated output.

It does not normally require separate tasks such as:

```markdown
- [ ] Establish renderer fidelity
- [ ] Build validation dataset
- [ ] Define comparison metrics
- [ ] Analyze discrepancies
```

unless those outputs are explicitly required by the project.

Use the validation level defined by `scope.md`.

---

## Fixed dependency list

Define task dependencies once during planning.

Keep them in a single dedicated section.

Example:

```markdown
## Dependencies

- Parameter recovery → Replay timeline
- Replay timeline → Stimulus renderer
- Stimulus renderer → Mouse-perspective renderer
- Mouse-perspective renderer → CLIP integration
```

Or, when task identifiers are useful:

```markdown
## Dependencies

- T02 depends on T01
- T03 depends on T01, T02
- T04 depends on T03
```

The dependency list should describe the planned structural relationships between tasks.

It is not a live execution-state tracker.

### Do not maintain dynamic blocker state

Do not update the task file to track:

* which task is currently blocked;
* which task is currently executable;
* temporary environment limitations;
* transient missing files;
* current agent progress;
* which dependency has just become available.

Do not annotate tasks with changing states such as:

```text
blocked
ready
in progress
waiting
not applicable
selected
```

unless the user explicitly requests such project-management information.

The dependency list should normally remain unchanged after planning.

Whether a task can be executed at a particular moment should be determined from the dependency graph and current repository state during implementation, not written back into the plan.

---

## Keep dependencies minimal

Add a dependency only when one task genuinely requires an output produced by another task.

Do not create dependencies simply because two tasks are conceptually related.

Do not place optional research, validation, or documentation work in the critical path unless it is genuinely required.

Prefer:

```text
recover parameters
→ reconstruct replay
→ render stimulus
→ render perspective
→ extract CLIP features
```

over dependency chains inflated by optional confidence-building work.

---

## Stable plan principle

After the plan is accepted, treat its structure as stable.

During normal implementation:

* do not reorder tasks;
* do not rewrite task descriptions;
* do not rewrite out-of-scope sections;
* do not rewrite dependency relationships;
* do not add completion dates;
* do not add evidence summaries;
* do not add implementation notes;
* do not append verification results;
* do not continuously refine already accepted wording.

The preferred update is only:

```text
[ ] → [x]
```

This keeps the task file compact, predictable, and cheap to re-read.

---

## When modifying the plan is allowed

A plan may be changed when the existing plan is materially wrong or insufficient.

Examples:

* a required functional stage was omitted;
* two tasks are discovered to represent the same required outcome and must be merged;
* a planned task is technically impossible and the project requires a different implementation path;
* an upstream requirement changes;
* the user explicitly changes the scope;
* an incorrect dependency prevents the plan from representing the real implementation order.

Plan modifications should be exceptional, not routine.

When a change is required, make the smallest necessary edit.

Do not rewrite unrelated tasks.

---

## Completing tasks

The normal completion operation is exactly:

```text
[ ] → [x]
```

For example:

```markdown
### [x] Implement normalized parameter recovery
```

Do not add:

* completion dates;
* `Status: Complete`;
* `Implemented:` sections;
* `Evidence:` sections;
* `Verification:` sections;
* runtime reports;
* environment reports;
* hashes;
* handoff reports;
* lessons learned;
* completion summaries.

The implementation itself is the primary record of what was done.

If durable technical information must be documented, place it in the relevant specification, design documentation, code comments, or user documentation — not in the task tracker.

---

## Do not track historical task states

The task file represents the current accepted plan and its completion state.

It does not need to preserve the full history of how the plan evolved.

Avoid retaining entries such as:

```text
Not applicable under previous decision
Previously blocked
Superseded by task X
Originally planned for backend Y
```

unless that history is necessary to understand the current implementation.

Prefer a clean current plan over a historical project-management record.

Version control already preserves previous versions of the file.

---

## Acceptance criteria

Do not automatically generate large acceptance-criteria sections.

The task summary should normally define the intended outcome sufficiently.

Add a small number of acceptance points only if completion would otherwise remain ambiguous.

For example:

```markdown
### [ ] Integrate perspective frames with CLIP

**Summary:**  
Use the generated perspective frames as the visual input for the existing CLIP extraction path while preserving trial identity and timing.

**Out of scope:**  
Changing the CLIP model or redesigning the neural alignment pipeline.

**Required behavior:**
- Preserve original trial IDs.
- Preserve session timestamps.
- Produce the expected feature representation.
```

Use such extra criteria sparingly.

Do not turn them into exhaustive internal implementation checklists.

---

## Do not duplicate specifications

Task files should reference existing specifications rather than copy them.

Prefer:

```markdown
### [ ] Implement source-derived stimulus renderer

**Summary:**  
Implement the renderer behavior defined in `visual-replay-spec.md`.

**Out of scope:**  
Historical pixel-fidelity certification.
```

Do not copy several paragraphs of source semantics, implementation details, or research findings into the task file.

This reduces duplication and prevents the task list from diverging from the real specification.

---

## Planning procedure

When creating a new plan:

1. Identify the concrete final project outcome.
2. Read `scope.md`.
3. Remove work that is not necessary for that outcome.
4. Identify the major functional stages.
5. Create one task per meaningful completion boundary.
6. Give every task a concise `Summary`.
7. Add a concise `Out of scope` section where it prevents likely misunderstanding.
8. Merge low-level implementation steps into their parent task.
9. Merge validation into the functionality it validates unless validation is itself a required deliverable.
10. Remove speculative future work.
11. Create one fixed dependency list.
12. Review the dependency list for unnecessary blockers.
13. Stop decomposing once implementation can proceed without ambiguity.
14. Treat the accepted plan as stable.

Before finalizing each task, ask:

> If this task were removed, would the requested functional outcome become incomplete or incorrect?

If not, remove it or merge it into another task.

Then ask:

> Is the task description clear enough that another implementation session would understand what must be produced and what nearby work should not be included?

If not, improve the `Summary` or `Out of scope` text rather than creating more subtasks.

---

## Examples

### Good task

```markdown
### [ ] Render mouse-perspective scene

**Summary:**  
Place the generated stimulus output on the configured physical screen geometry and render it from the project's fixed schematic mouse viewpoint.

**Out of scope:**  
Eye tracking, retinal optics, ambient illumination reconstruction, and physical display calibration.
```

### Avoid implementation decomposition

Prefer the task above over:

```markdown
- [ ] Create camera class
- [ ] Add screen plane
- [ ] Add projection function
- [ ] Add image resize
- [ ] Add video writer
```

### Avoid progress-report growth

Prefer:

```markdown
### [x] Recover normalized parameters

**Summary:**  
Recover available session parameters and resolve required fallback values.

**Out of scope:**  
Reconstruction of unavailable raw task logs.
```

Do not transform it after completion into:

```text
Status: Complete (2026-09-22)

Evidence: ...

Implemented: ...

Verification: ...

Runtime: ...

Remaining gaps: ...

Handoff: ...
```

### Avoid dynamic blocker tracking

Do not change:

```markdown
### [ ] Implement renderer
```

into:

```markdown
### [ ] Implement renderer — blocked by missing reference capture
```

and later:

```markdown
### [ ] Implement renderer — ready
```

and later:

```markdown
### [ ] Implement renderer — selected backend
```

The dependency graph and implementation context determine executability.

The task file should remain stable.

---

## Default planning bias

When uncertain:

* prefer fewer tasks;
* prefer stable tasks;
* prefer functional outcomes over research activities;
* prefer implementation over additional validation;
* prefer one fixed dependency list over dynamic status tracking;
* prefer `Summary` and `Out of scope` over many subtasks;
* prefer `[ ] → [x]` as the only routine file modification;
* prefer specifications for detail and task files for direction;
* prefer existing abstractions over speculative future ones;
* prefer finishing the current functional pipeline over preparing hypothetical future work.

Examples clarify these rules; they do not create additional requirements.

The plan is complete when it is sufficient to guide implementation to the requested outcome.

It does not need to record every action taken while getting there.
