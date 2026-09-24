# Audit rules

These rules define how repository audits should be performed and how component `audit.md` files should be written.

Read and apply:

* `scope.md` for deciding which findings materially matter to the project;
* `context.md` for selecting the documentation and source code relevant to the audited component;
* `planning.md` only after the audit, when accepted findings are converted into implementation tasks.

An audit describes the current repository state relative to the accepted component design.

It does not redesign the component and does not implement fixes.

---

## Purpose

A component audit answers:

> What in the current implementation prevents, contradicts, weakens, duplicates, or unnecessarily complicates the behavior required by the accepted specification, architecture, interfaces, and project dependency graph?

The audit should identify meaningful implementation gaps and risks.

It should not become a general code-quality review.

---

## Authoritative comparison

Audit the implementation against:

1. the target component `spec.md`;
2. the target component `architecture.md`, when present;
3. the target component `interface.md`;
4. `docs/dependencies.md`;
5. directly relevant dependency interfaces;
6. actual source behavior.

Authority is not equal between these documents.

The intended design is defined by:

```text
specification
architecture
project dependency graph
```

The interface describes the current technical component boundary.

If implementation and interface disagree, report the inconsistency.

Do not silently change the specification, architecture, dependency graph, or interface while performing the audit.

---

## Audit scope

Stay inside the target component.

Inspect external components only through the context allowed by `context.md`.

Do not recursively audit upstream or downstream components merely because a problem crosses a boundary.

When a problem originates outside the target component, identify the external dependency and the observed contract conflict without expanding the audit into that component.

---

# Finding priorities

Use three priorities.

Priority reflects **project impact**, not code aesthetics or implementation difficulty.

## High priority

A finding is high priority when it can make the pipeline incorrect, unusable, misleading, or structurally inconsistent.

High-priority findings include:

### Runtime failures

Examples:

* exceptions on supported inputs;
* broken imports;
* incorrect function signatures;
* missing required artifacts;
* code paths that cannot complete their intended operation;
* invalid configuration handling that prevents normal execution.

### Incorrect pipeline behavior

Examples:

* wrong input reaches a component;
* output does not represent the expected data;
* trial/session identities shift;
* timestamps become misaligned;
* masks apply to the wrong samples;
* data is silently dropped or substituted;
* a downstream component receives an incompatible representation.

### Specification violations

Behavior contradicts a requirement in the accepted component specification.

### Architecture violations

Examples:

* responsibility implemented in the wrong component;
* accepted component boundary is bypassed;
* consumer owns logic that belongs to its dependency;
* implementation requires knowledge intentionally hidden behind another component's interface.

### Interface violations

Examples:

* implementation output differs from the declared interface;
* consumer expects fields the producer does not provide;
* units, shapes, identifiers, or timing semantics disagree;
* interface information is stale in a way that can cause incorrect use.

### Dependency violations

Examples:

* undeclared cross-component dependency;
* reverse dependency;
* component cycle;
* direct access to another component's internal implementation when only its interface should be consumed.

Circular dependencies are always high priority.

### Data integrity risks

Examples:

* mixing sessions;
* mixing incompatible dataset revisions;
* silent trial renumbering;
* mismatched IDs;
* incorrect temporal alignment;
* wrong units;
* stale cache accepted as current;
* invalid data treated as valid.

### Silent incorrect fallback

A failure or missing input is replaced by another value or behavior without the project explicitly allowing that fallback.

### Destructive behavior

Anything that can unexpectedly overwrite, delete, corrupt, or mix source or derived research data.

---

## Medium priority

A finding is medium priority when the current system may work, but the implementation creates meaningful maintainability, duplication, or consistency risk.

Examples include:

### Duplicate responsibility

The same important behavior is independently implemented in several places.

Examples:

* multiple EID parsers;
* multiple incompatible ONE configurations;
* repeated source-resolution logic;
* competing implementations of the same conversion.

Duplication becomes high priority if the copies already produce different or incorrect behavior.

### Boundary leakage

A component performs work that belongs elsewhere, but the current behavior is still functionally correct.

### Inconsistent error semantics

Equivalent failures are represented differently across code paths in a way that complicates consumers but does not currently corrupt results.

### Excessive coupling

A component depends on internal details that make future changes unnecessarily risky.

### Fragile assumptions

The implementation depends on undocumented ordering, filenames, directory layouts, implicit defaults, or similar assumptions that are likely to break supported behavior.

### Legacy compatibility burden

Obsolete paths materially complicate active code even though they are still reachable.

---

## Low priority

A finding is low priority when it is primarily an improvement opportunity and does not currently threaten functional correctness or component boundaries.

Examples include:

### Optimization opportunities

* unnecessary repeated loading;
* avoidable allocation;
* repeated metadata lookup;
* inefficient iteration;
* cache opportunities;
* unnecessary recomputation.

Optimization remains low priority unless there is evidence that it causes a real project bottleneck.

Do not promote theoretical performance improvements without measurements or an observed problem.

### Local code simplification

* unnecessarily complex helper;
* avoidable branching;
* small readability improvement;
* harmless duplication.

### Non-blocking consistency improvements

Naming or internal organization that could be cleaner but does not create ambiguity at a component boundary.

Low-priority findings should be included only when they are concrete and useful.

Do not fill the audit with style observations.

---

# Dead code

Dead code is a major audit concern.

Give dead-code detection **high audit attention** because inherited and experimental code can:

* obscure the active pipeline;
* mislead future implementation;
* create false dependencies;
* preserve obsolete behavior;
* increase agent context and repository exploration;
* make it unclear which implementation is authoritative.

Dead code includes:

* unreachable functions or branches;
* modules with no active caller;
* obsolete implementations replaced by another path;
* configuration that is no longer consumed;
* abandoned experimental paths;
* duplicated implementations where one is no longer reachable;
* compatibility code for functionality no longer supported by the project;
* stale entry points;
* generated or copied source artifacts incorrectly retained as active code.

Do not classify code as dead merely because no direct call was found.

Check, where relevant:

* imports;
* CLI entry points;
* configuration-driven invocation;
* dynamic loading;
* shell scripts;
* training/evaluation entry points;
* indirect callbacks;
* serialization/checkpoint compatibility.

When deadness cannot be established confidently, mark it as **possibly obsolete** rather than recommending removal.

Confirmed dead code should normally be treated as high priority for cleanup when it materially increases ambiguity or maintenance burden.

Trivial unreachable code with no architectural impact may be medium priority.

---

# Existing code quality is not itself a defect

This repository contains inherited and rapidly developed research code.

Do not recommend replacement simply because code is:

* old;
* inelegant;
* procedural;
* inconsistent in style;
* not structured as it would be in a new implementation;
* implemented using several external IBL APIs;
* missing an abstraction that would look cleaner.

Ask instead:

> Does this code prevent the accepted component behavior or boundary from working correctly?

If no, it may remain.

The preferred audit disposition is incremental:

```text
Keep
Wrap / centralize
Refactor
Replace
Remove
```

## Keep

Use when the implementation works and does not materially violate the accepted design.

Code quality alone is not sufficient reason to change it.

## Wrap / centralize

Use when existing behavior is useful but should be exposed through the accepted component boundary rather than independently owned by consumers.

Prefer this over rewriting working functionality.

## Refactor

Use when behavior should remain but internal structure prevents the component from satisfying the accepted design cleanly.

## Replace

Use only when existing behavior cannot reasonably satisfy required functionality or component boundaries.

Replacement requires a concrete reason.

## Remove

Use for confirmed obsolete/dead behavior that no longer contributes to the supported pipeline.

Do not convert these dispositions directly into tasks during the audit.

They are guidance for later planning.

---

# Optimization

Inspect for optimization opportunities, but keep them secondary.

Optimization should not distract from:

1. correctness;
2. pipeline integrity;
3. specification compliance;
4. component boundaries;
5. dependency correctness;
6. dead-code cleanup.

Report optimization when:

* repeated expensive work is clearly visible;
* data is unnecessarily loaded multiple times;
* an existing cache is bypassed;
* operations scale poorly with the expected dataset size;
* a known project bottleneck exists.

Do not propose:

* speculative micro-optimization;
* performance-oriented rewrites without evidence;
* new caching layers merely because caching is possible.

Optimization findings are low priority by default.

Promote them only when measured or clearly dominant runtime/memory behavior threatens the intended workflow.

---

# Error handling

Audit error behavior where it affects component correctness.

Look for:

* swallowed exceptions;
* ambiguous `None`/empty return values;
* missing-vs-empty confusion;
* partial output treated as complete;
* fallback after an error without explicit policy;
* inconsistent handling of equivalent failure states;
* errors that appear far downstream from their cause.

Do not require elaborate error frameworks.

The goal is that meaningful failure states remain distinguishable and do not silently corrupt the pipeline.

---

# Data and scientific correctness

For research-data components, explicitly inspect:

* identity preservation;
* time alignment;
* units;
* array/tensor shape assumptions;
* ordering;
* missing-data semantics;
* masking semantics;
* revision/source identity where relevant;
* deterministic vs stochastic processing where it affects reproducibility;
* provenance required by the specification.

These have higher importance than ordinary style or abstraction concerns.

---

# Dependency and boundary audit

For every component audit, check:

1. Does the component use only declared upstream dependencies?
2. Does it bypass another component's interface?
3. Does another component bypass this component's intended boundary?
4. Has consumer-specific behavior leaked into this component?
5. Does this component depend on downstream behavior?
6. Does any relationship create a cycle?
7. Are responsibilities duplicated across component boundaries?

Report architectural dependency problems even when the current code happens to run successfully.

---

# Finding evidence

Every finding must be tied to concrete repository evidence.

Prefer:

```text
file path
function / class / configuration
observed behavior
```

Do not report speculative problems as confirmed findings.

Distinguish:

* **Confirmed** — directly established from source/runtime evidence.
* **Likely** — strong source evidence exists but execution or dynamic behavior prevents complete confirmation.
* **Uncertain** — potentially relevant but insufficient evidence.

Uncertain observations should be rare and should not automatically become tasks.

---

# Audit output format

A component `audit.md` should remain concise and structured.

Use:

```markdown
# <Component> audit

## Summary

Short description of the current implementation state and the most important gaps.

## Findings

### A01 — <short finding title>

**Priority:** High | Medium | Low  
**Confidence:** Confirmed | Likely | Uncertain  
**Category:** Runtime | Pipeline | Spec | Architecture | Interface | Dependency | Data integrity | Dead code | Maintainability | Optimization

**Location:**  
Relevant file(s), function(s), or configuration.

**Finding:**  
What the current implementation does and why it matters.

**Expected:**  
The relevant required behavior or boundary.

**Suggested disposition:**  
Keep | Wrap / centralize | Refactor | Replace | Remove

---

### A02 — ...

## Conforming areas

Optional short list of important areas that already satisfy the accepted design and should preferably be preserved.
```

Do not add:

* completion dates;
* implementation progress;
* task checkboxes;
* execution diaries;
* long runtime logs;
* every file inspected;
* speculative future improvements;
* detailed fix plans;
* acceptance criteria for future tasks.

The audit identifies the delta.

The task plan decides how to resolve it.

---

# Finding granularity

A finding should represent one meaningful problem.

Do not create separate findings for every line or helper affected by the same root cause.

Prefer:

```text
ONE client configuration duplicated across three acquisition paths
```

over three independent findings that describe the same architectural issue.

Likewise, do not combine unrelated high-impact problems into one generic finding.

A later task may resolve several related audit findings together.

There is no required one-to-one relationship between findings and tasks.

---

# Audit snapshot behavior

`audit.md` is a snapshot of repository state when planning begins.

After the audit is accepted:

* do not update findings as implementation progresses;
* do not mark findings fixed;
* do not append implementation notes;
* do not convert it into a progress report.

Progress is represented by `tasks.md`.

If a future major development cycle requires another audit, perform a new audit against the then-current specification and implementation rather than maintaining the old audit as a live status document.

---

# Audit-to-planning boundary

Audit findings are inputs to planning, not automatic tasks.

Before converting a finding into a task, `planning.md` and `scope.md` must determine:

* whether resolving it is required;
* whether multiple findings belong in one task;
* whether the issue can be handled as an implementation detail;
* whether a low-priority optimization should be deferred;
* whether existing code can be wrapped rather than rewritten.

In particular:

> A finding does not create a task merely because it exists.

---

# Default audit priority

When inspecting a component, focus in this order:

1. runtime failures;
2. data/scientific correctness;
3. broken pipeline behavior;
4. specification violations;
5. interface incompatibilities;
6. dependency cycles or invalid dependencies;
7. responsibility/component-boundary violations;
8. dead or obsolete code;
9. duplicated or fragile behavior;
10. maintainability concerns;
11. optimization opportunities;
12. cosmetic/style improvements.

Dead-code detection should receive deliberate attention even though confirmed runtime/correctness failures take precedence.

Optimization and style must never dominate the audit unless explicitly requested.

---

# Audit completion

An audit is complete when it provides enough reliable information to derive a minimal implementation plan.

It does not need to enumerate every imperfection in the codebase.

Stop when:

* required behavior has been compared with implementation;
* meaningful pipeline and boundary violations are identified;
* relevant dead code has been examined;
* important runtime/data risks are identified;
* major duplication or fragility affecting the component is recorded;
* sufficient information exists for planning.

Do not continue searching merely to make the audit exhaustive.

---

## Legacy architecture assessment

When auditing existing functionality, do not compare it against an imagined greenfield implementation.

Evaluate it against:

* the accepted specification;
* the accepted architecture;
* the component interface;
* the project dependency graph;
* actual functional correctness.

Existing structure should be preserved when it can satisfy these requirements without introducing material correctness or maintenance risk.

For legacy code, prefer findings that support incremental migration:

```text
Keep
Wrap / centralize
Refactor
Replace
Remove
```

Use the least invasive disposition that resolves the actual problem.

In particular:

* do not recommend refactoring solely because a different abstraction would be cleaner;
* do not recommend replacement solely because the implementation is old or inconsistent;
* do not require all existing access paths to share identical internals when they can safely share a stable external boundary;
* prefer adapters and boundary extraction when they allow working implementation to remain closed to unnecessary modification.

A greenfield component is different: when no meaningful existing implementation exists, the accepted specification and architecture may define a new structure without preserving historical implementation patterns.
