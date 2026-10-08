# Evaluation implementation tasks

Implement final visual-to-neural evaluation according to [spec.md](spec.md), informed by [audit.md](audit.md). Consume only the declared [training-dataset](../training-dataset/interface.md), [model](../model/interface.md), and [training](../training/interface.md) boundaries. Preserve usable inherited metric and plotting functionality.

## Tasks

### [x] T01 — Load an explicitly selected compatible checkpoint and test dataset

**Summary:**  
Update evaluation setup and its launcher to accept a selected checkpoint and verified dataset generation, optional session selection, and effective evaluation configuration. Restore the complete saved model through the training restoration interface, retaining its configuration, session mapping, training run, and checkpoint-selection record. Verify ordered populations, feature/output semantics, temporal representation, and test membership against checkpoint provenance before inference. Support persisted within-session and held-out-session strategies where compatible mappings already exist; reject empty selections, unavailable mappings, and overlap with optimization or checkpoint-selection observations. Replace guessed paths and filtered/randomly initialized checkpoint restoration.

**Out of scope:**  
Checkpoint search or selection using test scores, historical checkpoint migration, upstream dataset changes, resplitting, and session adaptation during evaluation.

### [x] T02 — Collect complete deterministic predictions with scientific identity

**Summary:**  
Evaluate every selected test sample exactly once across configurable batches using the checkpoint's actual prediction direction, inference mode, and explicit temporal/neuron validity. Collect neural log expected counts and original targets with sample/trial/session identities, ordered units, physical temporal coordinates, bin durations, and masks. Preserve session-specific populations and verify collection coverage. Remove ineffective held-out processing and obsolete controls from the active path. If optional decoding or multimodal evaluation remains exposed, apply the same coverage/identity guarantees and retain full visual embeddings.

**Out of scope:**  
New decoding experiments, stochastic corruption experiments, parameter updates, reconstruction of missing observations, rebinning, and distributed evaluation infrastructure.

### [x] T03 — Compute validity-aware neural metrics and explicit aggregation

**Summary:**  
Produce per-neuron trial-level R², PSTH R², and Poisson BPS independently within each session, excluding invalid cells while retaining valid zero counts and negative scores. Define trial/time axes, PSTH grouping and temporal comparability, baseline statistics and their permitted source, and neuron/session/global weighting in evaluation configuration and results. Use identical valid observations for predicted and observed PSTHs and persisted bin durations for any rate conversion. Expose constant-target, zero-spike, insufficient-observation, and nonfinite conditions with metric-specific reasons and explicit aggregation coverage instead of silent omission. Retain compatible inherited calculations without the implicit firing-rate gate. Correct metric naming and validity handling for optional visual scoring if retained.

**Out of scope:**  
Additional scientific metrics, benchmark reproduction, new baseline research, inferred behavioral conditions, and changes to training losses or upstream representations.

### [x] T04 — Publish traceable evaluation artifacts and finalize the public interface

**Summary:**  
Connect the entry point to structured machine-readable neuron/session/global results and optional persisted predictions containing complete identity, masks, coordinates, and provenance. Record checkpoint identity and selection policy, training/adaptation lineage, dataset generation and split strategy, alignment provenance, effective evaluation configuration, metric definitions, baseline/grouping/aggregation policies, and available software versions. Prevent unrelated artifact reuse or overwriting and reject incomplete/incompatible results. Derive configured diagnostic plots from the same valid predictions and metric definitions. Remove confirmed obsolete evaluation helpers as part of completing this path, directly demonstrate representative multi-batch execution and artifact readback using available compatible inputs, and write `interface.md` from the completed implementation.

**Out of scope:**  
Model-comparison tooling, new analysis dashboards, upstream documentation redesign, automated validation infrastructure, and exhaustive performance or fidelity studies.

## Dependencies

- T02 depends on T01.
- T03 depends on T02.
- T04 depends on T01, T02, and T03.

Validation belongs inside each task: use code inspection, existing checks, direct execution, and manual artifact verification proportional to its concrete risks. Do not write or modify tests. Optional visual evaluation is compatibility work only if retained; it must not delay the required neural encoding pipeline.
