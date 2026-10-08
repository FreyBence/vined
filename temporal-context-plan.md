# Temporal context implementation plan

## Objective and scope

Support independently trainable encoding and decoding runs with `strict` or
`full_trial` temporal context, preserving session-specific stitching, existing
masking capabilities, scientific trial identities, and persisted split membership.

`src/configs/scenarios.json` supplies the terminology only. Entry-script arguments
select context for this implementation; neither scenario loading nor context
selection from model/training JSON is included. Resolved arguments must still be
recorded in run artifacts and checkpoints for reproducibility and compatibility.

This plan incorporates the accepted shared held-state alignment policy. It does
not authorize changes to upstream CLIP/neural extraction, the dependency graph,
or unrelated authoritative documents. Existing alignment documentation describes
interpolation; the accepted held-state requirement supersedes that behavior for
this work. Keep that discrepancy explicit until separately authorized document
updates are made.

## Runtime contract

### Entry-script arguments

Expose these arguments through `src/train.py` and `src/finetune.py`, using their
shared `trainer.setup.add_setup_arguments` boundary:

```text
--context-mode strict|full_trial     default: full_trial
--context-bins 1|3|6|9|12            required for strict; rejected for full_trial
```

Expose the same arguments through `src/eval.py`, with `--context-mode` required.
Evaluation arguments declare the intended context and must match the selected
checkpoint; they cannot override its learned context semantics. Resume and
pretrained adaptation must also verify context compatibility. Do not infer context
from `scenarios.json`, a scenario ID, or a model/training JSON override. Checkpoint
metadata is retained for restoration and compatibility, not as an alternative
experiment-selection interface. Existing shell launchers should forward arguments.

Examples after implementation:

```text
python src/train.py --dataset-generation PATH --model_mode encoding --context-mode strict --context-bins 3
python src/train.py --dataset-generation PATH --model_mode decoding --context-mode full_trial
python src/eval.py --checkpoint CHECKPOINT --dataset-generation PATH --context-mode strict --context-bins 3
```

### Shared alignment

- A visual feature represents the stimulus state active from its source update
  timestamp until the next display update, bounded by the stimulus interval.
  Use timestamp-aware held-state mapping; do not interpolate CLIP embeddings.
- Neural counts retain their supplied physical intervals and complete-bin exposure.
  Do not shift counts for physiological latency or silently rebin prepared counts.
- Exact 1:1 correspondence requires verified coincidence of display-update times
  and neural bin boundaries. Reconstructed nominal cadence is not measured update
  evidence. Preserve timing classification and supporting provenance.
- Validate that a mapped state covers the intended neural interval. If an update
  falls inside a bin or timestamp support is insufficient, fail explicitly rather
  than silently claiming one state describes that entire bin. Any different
  treatment requires an explicit physical-interval policy.
- Holding a known active state until its next update or known stimulus offset is
  supported temporal representation. Repeating observations to invent missing
  history, bridge unknown updates, or repair incomplete source coverage is invalid.
- Alignment remains shared by both context modes and prediction directions.
  A changed alignment policy requires new alignment and dependent dataset artifacts,
  but changing context mode or length reuses those artifacts.

### Context and target support

For a trial with `T` real bins and target position `k`:

| Mode / direction | Available source context | Target |
| --- | --- | --- |
| strict encoding | `V[k-L+1:k+1]` | `N[k]` |
| strict decoding | `N[k:k+L]` | `V[k]` |
| full_trial encoding | All valid visual positions in the same trial | `N[k]` |
| full_trial decoding | All valid neural positions in the same trial | `V[k]` |

`L` includes the current bin and limits raw source information across the entire
model, not the distance of a single attention layer. Strict encoding excludes
future visual observations; strict decoding intentionally permits later neural
observations. Targets retain their original coordinates.

`full_trial` preserves the current full-sequence, unrestricted bidirectional
temporal computation and configured corruption behavior. It may use future visual
information in encoding and is an explicitly noncausal control. The new common
target-support rule applies to this mode as well. Legacy finite forward/backward
attention settings must not silently redefine either mode; reject conflicting
settings and keep existing unrestricted defaults compatible.

Use fixed `H=12` during optimization, validation, and final evaluation:

- Encoding selects `11 <= k < T`, requiring eleven preceding real bins.
- Decoding selects `0 <= k <= T-12`, requiring eleven following real bins.
- Apply the same selection to every strict length and full-trial control.
- Keep scientific observation validity, input corruption, context eligibility,
  and objective target selection separate. Excluded targets can remain valid context.
- Trials shorter than twelve bins have no eligible targets; retain source identity
  and report excluded coverage. Empty active target selections must fail clearly.
- Context cannot cross missing support, trial boundaries, or session boundaries.
  Do not extend the stimulus-bounded aligned interval for post-offset responses.

## Implementation tasks

### [x] T1 - Produce held-state aligned trials

**Summary:**
Update `src/alignment/temporal.py` and the prepared-input/publication boundary as
needed to publish timestamp-aware held-state features alongside unchanged neural
counts and physical intervals. Validate update support and interval correspondence;
record the mapping policy and source associations so interpolated and held-state
generations cannot be confused. Verify boundary behavior directly against supplied
timestamped states, including the final supported state before stimulus offset.

**Out of scope:**
Upstream extraction redesign, inventing actual display timestamps from cadence,
physiological offsets, and post-stimulus interval extension.

### [x] T2 - Expose trial-local context views and common support

**Summary:**
Extend the `training_dataset` runtime handoff with a reusable, direction-aware
context-index and eligibility boundary. Preserve complete persisted trial samples,
source positions, unit identities, and splits. Expose H=12 selection for both modes
without removing valid source observations. Keep loader/collation changes minimal;
verify all five lengths, short trials, right padding, and unequal session populations.

**Out of scope:**
Context-specific persisted datasets, resplitting, permanent corruption, and new
sampler infrastructure.

### [x] T3 - Implement exact strict context alongside full-trial inference

**Summary:**
Extend `MultiModal` in `src/multi_modal/mm.py` to consume the runtime context
contract while reusing registered input projections, Transformer layers, embeddings,
and `StitchEncoder`/`StitchDecoder`. Isolate target-anchored windows before any
temporal mixing, read the final encoding or first decoding position, and restore
predictions to original trial coordinates. Preserve full-trial execution and
external prediction shapes. Include mode/length/semantics in model identity and
provide prediction eligibility independently of scientific validity. Process
windows in bounded chunks when necessary; do not gather from already mixed
full-trial latents. Verify exclusion of out-of-window information with multiple
Transformer layers and correct dispatch for singleton/mixed-session batches.

**Out of scope:**
New model families, per-length model implementations, stitching redesign, sparse
attention kernels, and changing inherited multimodal objectives. Preserve the
legacy multimodal path; reject strict mixed-direction requests unless their
direction-specific context is explicitly supported.

### [x] T4 - Train and validate context-selected runs through CLI arguments

**Summary:**
Integrate the entry-script arguments through `src/trainer/setup.py`, model
construction, `objective.py`, and `base.py`. Combine H=12 eligibility with existing
target selectors in losses, accumulation denominators, validation metrics, and
plots. Record resolved CLI settings and support policy in checkpoints/run metadata;
verify resume/adaptation compatibility and reject conflicting configuration inputs.
Preserve masking ratios/modes independently of context restriction. Demonstrate
an optimization and validation step for each direction and context mode using
compatible prepared data, covering all five strict lengths proportionally.

**Out of scope:**
Scenario/configuration-controller integration, region selection, optimizer changes,
and source-data generation. Existing time maxima must fit real trials or fail
clearly; do not silently truncate trials to the current 100-bin default.

### [x] T5 - Evaluate and persist predictions on common directional support

**Summary:**
Extend `src/eval.py` and `src/evaluation/{setup,predictions,metrics,artifacts}.py`
to verify the explicit CLI context against checkpoint identity, infer using its
model semantics, and persist separate encoding/decoding scoring eligibility.
Apply H=12 to neural trial R2, PSTH contributors, BPS observations and baseline
means, visual cosine, and plots. Preserve full source coordinates/observations and
coverage reporting; version changed artifact schemas explicitly. Demonstrate
checkpoint restoration, matched target positions across modes/lengths, and
prediction artifact publication/readback. Finalize affected component interfaces
from the implementation, preserving the existing dependency boundaries.

**Out of scope:**
Cross-scenario comparison/statistics, checkpoint reselection using test results,
and post-offset decoding.

## Dependencies

- T2 depends on T1's accepted aligned-data contract.
- T3 depends on T2's runtime context contract.
- T4 depends on T2 and T3.
- T5 depends on T2, T3, and T4's checkpoint contract.

## Validation discipline

Validation belongs within each task. Use source inspection, existing checks,
direct execution, and manual artifact verification. Do not write or modify test
code, temporary test scripts, CI/CD configuration, or validation infrastructure.
Confirm exact source windows, original target alignment, identical H=12 support,
padding/corruption separation, session correctness, and checkpoint/artifact
compatibility. Full-trial equivalence concerns temporal forward behavior with the
same inputs and corruption; held-state preprocessing and common target selection
intentionally change the overall experiment relative to older runs.

## Completion

Complete when compatible held-state aligned trials can be reused by CLI-selected
strict and full-trial encoding/decoding runs, strict receptive fields are exact,
training/validation/evaluation share H=12 support, and predictions remain traceable
through session-correct checkpoint restoration and evaluation artifact readback.
Mark tasks complete using only `[ ]` to `[x]`; keep the dependency list stable.
