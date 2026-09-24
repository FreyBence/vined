# Project scope and validation principles

## Core principle

Prioritize implementing the intended research pipeline over proving an idealized level of correctness that is not required by the current project goals.

The default objective of a task is to make the required functionality work correctly within the available data, source material, interfaces, and project assumptions.

Do not expand implementation tasks into broader validation, benchmarking, infrastructure, or research work unless that additional work is necessary to establish that the requested functionality operates as intended.

## Functional completion comes first

For every task, first identify the concrete functional outcome that the task exists to produce.

Examples include:

* generating a visual stimulus from resolved parameters;
* transforming an input representation;
* extracting model features;
* aligning visual and neural data;
* producing training samples;
* running model training;
* computing an evaluation metric;
* publishing an artifact consumed by the next pipeline stage.

A task is primarily complete when that intended output exists, can be consumed by the next stage where applicable, and behaves consistently with the inputs that are supposed to control it.

Do not substitute increasingly strict validation work for completion of the actual functionality.

## Validation should be proportional to the decision it supports

Validation is required only to the level necessary to answer the practical question relevant to the task.

Prefer the least expensive validation that provides enough confidence to continue development.

Typical validation levels are:

### 1. Functional validation

Confirms that the implementation actually performs its intended operation.

Examples:

* code executes successfully;
* expected artifacts are produced;
* outputs are non-empty and readable;
* relevant input parameters affect the result;
* downstream code can consume the result.

This is the default required level.

### 2. Integration validation

Confirms that the implementation is connected correctly to the rest of the project.

Examples:

* IDs remain aligned;
* timestamps remain associated with the correct samples;
* compatible source/model/configuration versions are selected;
* expected schemas and interfaces are respected;
* the next pipeline stage receives the intended representation.

Use this when pipeline correctness depends on these relationships.

### 3. Fidelity or reference validation

Compares the implementation against an external ground truth, historical implementation, reference output, physical measurement, or independently validated system.

Examples:

* pixel-level comparison with a historical renderer;
* numerical equivalence with another implementation;
* reproduction of a published benchmark;
* validation against physical measurements.

Do not require this level unless:

* the project explicitly needs such a claim;
* a downstream decision depends on that level of accuracy;
* the reference is readily available and comparison is inexpensive;
* or the user explicitly requests it.

Absence of an ideal reference must not automatically block functional implementation.

## Do not create validation requirements that exceed the project claim

Validation should support the claim the project actually intends to make.

Do not introduce requirements intended to prove a stronger claim than the project needs.

For example:

* if the project claims that a source-derived renderer is used, source-consistent implementation and functional behavior may be sufficient;
* exact historical framebuffer equivalence would only be necessary if the project claimed pixel-identical reproduction;
* if a feature representation is used as a model input, it must be produced consistently and aligned correctly; proving that it is the uniquely optimal representation is not required;
* if an evaluation metric is used to compare models, its implementation must be correct; reproducing every published benchmark using that metric is not automatically required.

Always distinguish between:

* what the pipeline needs to function,
* what is useful to verify,
* and what would only support a stronger scientific claim.

## Evidence and assumptions

Use the strongest evidence that is reasonably available, but do not turn unavailable evidence into a blocking research task unless the functionality depends on it.

Preferred order:

1. session-specific or directly observed data;
2. version-compatible source code or configuration;
3. authoritative documentation;
4. historically appropriate defaults;
5. explicit project assumptions or fallbacks.

Record important assumptions and fallbacks when they materially affect the generated data.

Do not silently replace missing evidence with the newest available implementation or unrelated defaults.

However, documenting an uncertainty is often sufficient. Resolving every uncertainty is not automatically part of the task.

## Version compatibility

Version handling matters when behavior may have changed across historical data, software, models, preprocessing, or configuration.

When processing older data:

* prefer the implementation or configuration appropriate for that data;
* do not silently apply the newest behavior;
* use available metadata to select the appropriate profile or compatibility path;
* record unresolved compatibility assumptions when exact reconstruction is impossible.

Version routing should be validated when it affects the output.

Do not reconstruct entire historical environments merely to increase confidence unless that environment is necessary to make the required functionality work.

## Avoid scope expansion

Do not add work solely because it would make the implementation more theoretically complete, reproducible, elegant, benchmarked, or scientifically exhaustive.

Before adding a new validation or infrastructure task, ask:

> What concrete failure in the current project would this detect or prevent?

If there is no important current failure that it would detect or prevent, it is probably outside the present scope.

Examples of common unnecessary scope expansion:

* creating reference datasets solely for validation;
* rebuilding historical environments that are not needed for production;
* implementing extensive benchmark suites;
* introducing new CI/CD or automated testing infrastructure;
* testing every theoretically possible edge case;
* optimizing performance before performance is a demonstrated limitation;
* creating abstractions for hypothetical future backends;
* investigating unknowns that do not materially affect the current output;
* proving equivalence to an external implementation when approximate/source-derived behavior is sufficient.

## Prefer direct evidence of functionality

When possible, validate functionality using the artifact the task is intended to produce.

Examples:

* renderer task → inspect/generated video exists and reflects parameters;
* CLIP extraction → expected feature vectors are produced for the intended frames;
* alignment → aligned samples retain correct IDs and timestamps;
* training pipeline → model receives valid batches and completes a training/evaluation step;
* metric implementation → controlled inputs produce interpretable expected values.

Do not replace these direct checks with large supporting validation systems unless necessary.

## Testing philosophy

Automated tests are useful when they protect stable interfaces, transformations, identifiers, version routing, or calculations that are likely to regress.

They are not mandatory merely because a component can theoretically be tested.

Prefer tests for:

* deterministic transformations;
* schema handling;
* ID and timestamp preservation;
* source/profile selection;
* unit conversions;
* important mathematical calculations;
* known failure modes.

Prefer simple execution or manual inspection for:

* visual appearance;
* exploratory research components;
* generated videos;
* approximate scene construction;
* tasks without reliable ground-truth outputs.

Do not invent automated tests whose oracle would itself require assumptions or unavailable reference data.

## Definition of done

Unless a task specifies a stronger requirement, consider it complete when:

1. its required functionality has been implemented;
2. the intended output is successfully produced;
3. important inputs demonstrably affect that output where applicable;
4. required pipeline interfaces and identities are preserved;
5. important unsupported cases fail clearly or use documented fallbacks;
6. the next dependent stage can use the result.

Additional validation may be recorded as future work without preventing completion.

## Decision priority

When there is a tradeoff between:

* finishing the functional research pipeline, and
* increasing confidence beyond what is necessary for the current project claim,

prefer finishing the functional pipeline.

Only increase the validation burden when a concrete correctness risk, downstream dependency, scientific claim, or explicit user requirement justifies it.
