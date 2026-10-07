# Training audit

## Summary

The inherited training path provides explicit visual-to-neural modality selection, AdamW optimization, validation without gradients, and model/optimizer/scheduler checkpoint serialization. It does not yet satisfy the training specification for validity-aware neural monitoring, reproducible effective configuration, checkpoint provenance and continuation, or session adaptation. Training currently has no architecture or interface document; the model dependency has no interface document. Model-facing compatibility was checked narrowly against the actual constructor/forward boundary rather than inferred from an absent contract.

The current training-dataset interface supplies verified persisted splits, explicit temporal/neuron masks, identities, and generation provenance. This audit includes that working-tree interface. Training imports succeed in the project Python environment; findings below are established by source inspection, without a full optimization run. No test code or implementation changes were made.

## Findings

### A01 — Neural validation metrics include padded time positions

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/trainer/base.py:330`, `_collect_eval_results`; `:384`, `_collect_enc_results`; `eval_epoch`, `eval_enc_epoch`.

**Finding:**  
Neural collection trims the neuron axis using `space_attn_mask` but retains every time position. Predictions and targets are passed to BPS without applying `time_attn_mask`; padded targets therefore affect the metric used to retain checkpoints. The visual collection already excludes invalid temporal positions, but the neural path does not.

**Expected:**  
Specification §§8, 17–18, 25: monitoring and checkpoint selection must reflect valid neural observations, preserving the distinction between zero spikes and padding.

**Suggested disposition:**  
Refactor

---

### A02 — Singleton session groups disappear from neural validation

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:**  
`src/trainer/base.py:330`, `_collect_eval_results`; `:384`, `_collect_enc_results`; `eval_epoch`, `eval_enc_epoch`, `_plot_log_epoch`.

**Finding:**  
`np.argwhere(...).squeeze()` produces a scalar when a batch contains one trial for a session. Both neural collectors interpret that scalar as zero neurons and discard the trial. The dataset handoff retains partial batches and permits mixed-session batches, so this affects ordinary supported data. Main validation catches empty concatenation and continues with missing metrics; encoding-specific validation concatenates without that guard. Plotting assumes session index zero has results even when collection skipped it or the session has no validation members.

**Expected:**  
Specification §§7, 9, 17, 30: every valid validation sample must participate, and empty per-session membership must be handled explicitly rather than silently losing observations.

**Suggested disposition:**  
Refactor

---

### A03 — Scheduler duration does not match optimizer updates

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Runtime

**Location:**  
`src/train.py:122–135`, `:299–313`; `src/finetune.py`, scheduler construction; `src/trainer/base.py:290`, `train_epoch`.

**Finding:**  
Pretraining computes scheduler steps using `num_train // global_batch_size`, although the loader keeps partial batches. In non-encoding runs this undercounts steps whenever the sample count is not divisible by batch size; OneCycleLR can exhaust its schedule before training finishes. A dataset smaller than one batch gives zero configured steps. Encoding computes the schedule for 4,000 epochs while the trainer executes the independently forced 130 epochs. Both entry points divide the scheduler duration by `gradient_accumulation_steps`, but the trainer zeroes gradients and steps the optimizer/scheduler on every batch without accumulation. Distributed epoch and batch-size adjustments introduce further disagreement with the trainer's actual loop.

**Expected:**  
Specification §§14–16, 28: scheduling and accumulation must use the actual configured optimization trajectory.

**Suggested disposition:**  
Refactor

---

### A04 — Effective training settings differ from requested configuration

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/train.py`, configuration setup, search handling, CLI defaults; `src/trainer/base.py:22`, `set_seed`, and `:294`; `src/configs/multi_modal/trainer_mm.yaml`.

**Finding:**  
Encoding overwrites configured epochs with 130 and optimizer learning rate with `5e-4`. Pretraining reads CLI/search masking ratio and CLI masking mode for logging/path names but never applies them to `config.model.masker`; stochastic multimodal masking retains YAML values. `--modality` is declared but ignored in favor of a local fixed list. Each training epoch reseeds with `42 + epoch`, ignoring `config.seed` for that phase. The direct pretraining CLI defaults to nonexistent `configs/`, while active configurations are under `src/configs/` (the shell launcher supplies the correct path).

**Expected:**  
Specification §§11–12, 16, 22, 28–29: effective numerical/scientific settings must be explicitly controlled and accurately recorded.

**Suggested disposition:**  
Refactor

---

### A05 — Checkpoints lack provenance and complete continuation state

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Interface

**Location:**  
`src/trainer/base.py:577`, `save_model`; `:201`, `train`; `src/train.py:316–346`, continuation branch.

**Finding:**  
Checkpoints contain only epoch, model weights, optimizer state, and scheduler state. They omit effective configuration, dataset generation/split identities, ordered unit identity, session mapping, random-generator/sampler state, validation history, and best-selection state. W&B configuration is optional and does not bind those missing dataset identities to the checkpoint. Continuation loads weights and optimizer/scheduler state but resets best metrics and lacks explicit compatibility checks beyond state-dict loading. Its fixed `pretrained/model_epoch.pt` lookup is not the location written by this trainer. Matching session names and tensor shapes cannot establish matching ordered neural populations or dataset generations.

**Expected:**  
Specification §§19–21, 27, 29, 33: artifacts must be traceable and continuation must restore a verified compatible run, including selection and stochastic state where applicable.

**Suggested disposition:**  
Refactor

---

### A06 — Canonical best-checkpoint branch is unreachable

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Dead code

**Location:**  
`src/trainer/base.py:228–250`, `train`.

**Finding:**  
The per-metric loop includes `eval_avg_metric` and updates its best value before the following comparison against that same value. The subsequent strict improvement condition can never succeed. Consequently `model_best.pt` is never written by this branch and `best_eval_loss` remains infinity. `model_best_avg.pt` and per-modality checkpoints are still written by the preceding loop. Selection is hard-coded to maximize metrics rather than exposed as a configured rule.

**Expected:**  
Specification §§18, 26, 28, 33: the selected artifact and reported selection information must implement a coherent, explicit rule.

**Suggested disposition:**  
Refactor

---

### A07 — Fine-tuning depends on evaluation and fails encoding adaptation

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Dependency

**Location:**  
`src/finetune.py:23`, import of `utils.eval_utils.load_model_data_local`; model-loading call and new-session embedding loop around `:277`.

**Finding:**  
Fine-tuning delegates model loading to an evaluation-owned helper, reversing the declared training → evaluation dependency and creating a component-level cycle. That call also loads a test dataset through YAML cache paths rather than using the already selected training generation; the supplied `data_path` is not used by the helper's dataset load. For multi-session encoding adaptation, fine-tuning then iterates all modalities and accesses `encoder_embeddings["spike"]`, although encoding constructs input embeddings only for `vision-clip`. This raises a missing-key error before adaptation completes.

**Expected:**  
Dependency graph and specification §§6, 9, 21, 32, 34–36: adaptation must use the declared dataset/model boundaries and select compatible session-specific components for the configured prediction direction.

**Suggested disposition:**  
Wrap / centralize

---

### A08 — Non-finite outputs and losses do not stop optimization

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Runtime

**Location:**  
`src/trainer/base.py:290`, `train_epoch`; `eval_epoch`, `train`.

**Finding:**  
The loop calls backward and optimizer/scheduler steps without checking model outputs or loss for NaN/Inf. Non-finite validation metrics can also flow into `nanmean` and failed checkpoint comparisons without an explicit run failure. Printing the eventual scalar does not prevent corrupted updates or an unusable selected result.

**Expected:**  
Specification §§30–31, 37: unsupported numerical failures must be visible and stop the optimization trajectory before invalid updates continue.

**Suggested disposition:**  
Refactor

---

### A09 — Run paths do not distinguish scientifically different runs

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/train.py:223–247`; corresponding log-name and overwrite handling in `src/finetune.py`.

**Finding:**  
Normal run directories use session count, a five-character single-session prefix or `multi`, modalities, and masking labels. They omit the dataset generation, full selected-session identities, split configuration, seed, learning rate, and architecture. Different runs therefore share a directory. The guard checks only `model_last.pt`, so an interrupted run that already wrote best/epoch checkpoints can be overwritten without `--overwrite`. Explicit overwrite also reuses the directory, allowing older best artifacts to remain when a new run does not replace them.

**Expected:**  
Specification §§27, 29, 33: configurations, datasets, logs, and checkpoints from different runs must remain distinguishable and correctly associated.

**Suggested disposition:**  
Refactor

---

### A10 — Neuron validity is replaced by a numeric padding convention

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Interface

**Location:**  
`src/trainer/base.py:129`, `_forward_model_inputs`; current `MultiModal.forward`/loss call boundary in `src/multi_modal/mm.py`.

**Finding:**  
Training receives explicit neuron validity but does not forward it to the model/loss boundary. Instead, the current model excludes neural cells by `targets != -1`. Entry points currently request `pad_value=-1`, so this convention excludes their padded channels; this is a contract gap rather than evidence that those channels currently enter the loss. Correctness nevertheless depends on the numeric sentinel instead of the dataset's authoritative neuron mask, and sample/unit provenance is not carried into checkpoint compatibility checks.

**Expected:**  
Specification §§7–10, 13 and the training-dataset handoff: explicit scientific validity must control which neural cells contribute, independently of numeric padding. The model boundary needs a compatible validity contract; this audit does not redesign that dependency.

**Suggested disposition:**  
Wrap / centralize

---

### A11 — Training unnecessarily requires a populated test split

**Priority:** Medium  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:**  
`src/train.py:197`, test loader construction; corresponding construction in `src/finetune.py`; trainer constructor.

**Finding:**  
Both entry points build a test loader and pass it through kwargs, but `MultiModalTrainer` never stores or consumes it. The dataset interface permits intentionally empty test splits; `make_loader` rejects them, blocking otherwise usable train/validation datasets. This is unused runtime setup, not evidence of test-driven parameter updates or checkpoint selection.

**Expected:**  
Specification §§4, 32: test observations belong to downstream evaluation and should not be a runtime prerequisite for optimization.

**Suggested disposition:**  
Remove

## Conforming areas

- Encoding explicitly selects visual inputs and neural outputs; unimodal masks hide the neural target while leaving visual inputs available.
- Training consumes predefined dataset splits; no training-owned resplitting or temporal realignment was found.
- Gradient updates iterate only over the training loader. Validation uses evaluation mode and `torch.no_grad()`; no test-driven checkpoint selection was found in the trainer.
- Explicit EIDs reach the model-facing modality dictionaries and session-specific construction.
- Temporal validity and visual availability are intersected at the model boundary; visual validation masks unavailable/padded positions.
- AdamW, periodic validation/saving, and optimizer/scheduler state serialization provide useful infrastructure to retain.
