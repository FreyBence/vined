# Evaluation Module Specification

## 1. Purpose

The `evaluation` module is responsible for the final scientific assessment of a trained neural encoding model.

It evaluates a selected model checkpoint on data that was not used for optimization or checkpoint selection.

The primary evaluated task is:

```text
visual stimulus representation
        ↓
trained model
        ↓
predicted neural activity
        ↓
comparison with observed neural activity
```

The module converts model predictions and ground-truth neural observations into reproducible scientific metrics and evaluation artifacts.

---

# 2. Position in the pipeline

Conceptually:

```text
visual-replay
    ↓
visual-features ─┐
                 ├── alignment
neural-data ─────┘
                      ↓
               training-dataset
                      ↓
                   training
                      ↓
             trained checkpoint
                      ↓
                  evaluation
                      ↓
        scientific results / metrics
```

Evaluation is downstream from all scientific preprocessing and training.

---

# 3. Responsibilities

The module is responsible for:

- loading a selected trained checkpoint;
- loading the evaluation dataset;
- verifying model/dataset compatibility;
- performing deterministic inference;
- collecting neural predictions;
- preserving trial, session, neuron, and temporal identity;
- calculating configured scientific metrics;
- aggregating metrics at explicitly defined levels;
- comparing predictions against appropriate baselines where required;
- producing machine-readable evaluation results;
- producing diagnostic plots or summaries where configured;
- preserving complete evaluation provenance.

The module must not:

- update model parameters;
- modify dataset splits;
- select training hyperparameters;
- select the model checkpoint using test results;
- perform neural/visual alignment;
- reconstruct missing observations;
- re-bin neural activity;
- change visual feature representations;
- fine-tune the model.

---

# 4. Evaluation input

The module consumes:

```text
selected model checkpoint
+
model configuration
+
evaluation dataset
+
evaluation configuration
```

The evaluation dataset is produced by `training-dataset`.

The scientific representation provided by upstream modules is authoritative.

---

# 5. Test-set isolation

Final scientific evaluation uses the test split.

The test split must not previously have influenced:

- gradient updates;
- optimizer state;
- hyperparameter selection;
- early stopping;
- checkpoint selection;
- model architecture selection;
- threshold selection.

Conceptually:

```text
training:
    train
    validation

evaluation:
    test
```

Evaluation results must not retroactively influence the evaluated training run.

---

# 6. Validation versus evaluation

Training-time validation and final evaluation serve different purposes.

## Validation

Used during training for:

- convergence monitoring;
- checkpoint selection;
- training diagnostics.

## Evaluation

Used after model selection for:

- final model performance assessment;
- scientific interpretation;
- cross-session or cross-trial generalization analysis;
- result reporting.

Metrics may overlap between the two stages, but their roles remain distinct.

---

# 7. Checkpoint selection boundary

The checkpoint evaluated by this module must already be selected before the final test evaluation begins.

Evaluation must record:

- checkpoint identity;
- checkpoint-selection policy;
- training run identity.

The evaluation module must not inspect multiple checkpoints on the test set and select the best-performing one.

---

# 8. Model compatibility

Before inference, the module must verify compatibility between the checkpoint and evaluation dataset.

Relevant compatibility includes:

- model architecture;
- visual feature dimension;
- temporal representation;
- session mappings;
- neural output dimensionality;
- neuron ordering;
- positional representation;
- output semantics.

Matching tensor shapes alone are not sufficient evidence of semantic compatibility.

---

# 9. Session compatibility

For session-specific models, every evaluated sample must have a valid neural output mapping.

The evaluation module must not silently reuse another session's output mapping.

If an evaluation session requires explicit adaptation or fine-tuning, that adaptation must occur before final evaluation and produce a distinct checkpoint or model state.

Evaluation itself remains parameter-free.

---

# 10. Evaluation modes

The module must support evaluation strategies corresponding to dataset split design.

## Within-session evaluation

Evaluates unseen trials from sessions also represented during training.

This measures generalization across trials within known recording sessions.

## Held-out-session evaluation

Evaluates sessions excluded from the original training split.

This measures cross-session generalization where a valid evaluation-session mapping exists.

The evaluation result must identify which strategy was used.

---

# 11. Deterministic inference

Evaluation must use inference behavior.

At minimum:

```text
model.eval()
no gradient calculation
no optimizer update
```

Training-only stochastic behavior such as dropout must be disabled according to model inference semantics.

Training-time random corruption or masking must not be applied unless the evaluated experiment explicitly requires it.

---

# 12. Scientific validity

Evaluation must respect all dataset validity information.

Only real scientific observations may contribute to metrics.

The evaluation module must distinguish:

```text
valid zero spike count
```

from:

```text
padding / invalid observation
```

A neural target of zero at a valid location is a real observation.

It must not be filtered out merely because its numeric value is zero.

---

# 13. Temporal validity

Padded temporal positions must not contribute to evaluation metrics.

Metrics must operate only over valid aligned temporal positions.

The evaluation module must not infer validity from prediction or target tensor values.

Validity masks supplied by the dataset are authoritative.

---

# 14. Neural validity

Where neural dimensions are padded across sessions, padded neural channels must not contribute to metrics.

The evaluation must preserve:

- session identity;
- real neuron count;
- neuron identity;
- neuron validity.

Metrics must be calculated only for valid biological neural units.

---

# 15. Prediction semantics

The current primary model output represents:

```text
log expected spike count
```

for each:

```text
(time bin, neuron)
```

Evaluation must explicitly convert this representation when a metric requires expected spike counts:

```text
predicted_count = exp(model_output)
```

The evaluation module must not interpret log expected counts directly as:

- spike counts;
- firing rate in Hz;
- probabilities.

---

# 16. Time-bin semantics

Predictions correspond to the same uniform neural bins produced by `alignment`.

For the current default configuration:

```text
bin size = 20 ms
```

but evaluation must use dataset metadata rather than assume a permanent hard-coded duration.

If a metric requires firing rate rather than expected count, conversion must use the configured bin duration explicitly.

---

# 17. Prediction collection

Inference output must remain traceable to its source observations.

Conceptually:

```text
EvaluationPrediction
    session_id
    trial_id

    neuron_identity
    timestamps

    predicted_neural
    observed_neural

    temporal_mask
    neuron_mask
```

Predictions must not be stored without sufficient identity metadata to reconstruct their scientific meaning.

---

# 18. Prediction persistence

Evaluation should support persistence of raw predictions and corresponding ground truth.

This allows:

- recalculation of metrics;
- plotting;
- error analysis;
- comparison between models;

without repeating model inference.

Persisted predictions must remain associated with:

- checkpoint;
- dataset;
- sample identity;
- session;
- neuron ordering;
- temporal coordinates.

---

# 19. Metric levels

Evaluation metrics may describe performance at different aggregation levels.

The aggregation level must always be explicit.

Relevant levels include:

```text
time bin
trial
neuron
session
dataset
```

A single scalar metric must not obscure its aggregation semantics.

---

# 20. Per-neuron evaluation

Where meaningful, evaluation should first calculate performance for individual neurons.

Conceptually:

```text
metric(session, neuron)
```

Session-level or global results may then be derived through an explicit aggregation rule.

This avoids allowing sessions with more neurons to dominate an average unintentionally.

---

# 21. Session-level evaluation

Metrics must support session-level reporting.

Conceptually:

```text
SessionResult
    session_id

    neuron_metrics
    aggregate_metrics

    number_of_trials
    number_of_neurons
```

Cross-session averages must define how individual sessions contribute to the final value.

---

# 22. Trial-level neural prediction

Evaluation must support measuring how well the model predicts trial-specific neural activity.

Trial-level evaluation compares predicted and observed neural activity without first averaging responses across repeated stimulus conditions.

This measurement reflects the model's ability to predict individual trial responses.

---

# 23. Trial-level R²

The evaluation module may calculate trial-level coefficient of determination.

Conceptually:

```text
R² = 1 - prediction_error / baseline_error
```

The implementation must explicitly define:

- evaluated axes;
- averaging order;
- handling of constant targets;
- validity filtering;
- session/neuron aggregation.

A negative result is valid and means prediction error exceeds the corresponding baseline error under the defined metric.

Trial-level R² must not be silently clipped to zero.

---

# 24. PSTH evaluation

Evaluation must support comparison of predicted and observed peri-stimulus time histograms.

Observed and predicted neural activity are aggregated across comparable trials while preserving temporal position.

Conceptually:

```text
trials
  ↓ average
time × neuron
  ↓
PSTH
```

Predicted and observed PSTHs must use identical:

- trial selection;
- temporal bins;
- neuron ordering;
- aggregation rules.

---

# 25. PSTH R²

PSTH R² measures how accurately the model predicts the average temporal response profile.

It is distinct from trial-level R².

Conceptually:

```text
trial-level R²
    → individual-trial prediction quality

PSTH R²
    → average temporal response prediction quality
```

A model may perform differently under these two metrics, and both results must therefore remain separately identifiable.

Negative PSTH R² values are valid and must not be clipped.

---

# 26. Bits per spike

The module must support Bits per Spike (BPS) for neural encoding evaluation where the probabilistic output semantics are compatible.

BPS compares model likelihood with an explicitly defined baseline model and normalizes the improvement by the number of observed spikes.

Conceptually:

```text
model log-likelihood
        -
baseline log-likelihood
        ↓
normalize by observed spikes
```

The exact baseline and likelihood implementation must be explicit and reproducible.

BPS interpretation:

```text
BPS > 0
```

means the model assigns greater likelihood to the observed spike activity than the configured baseline.

```text
BPS = 0
```

means no improvement over the baseline.

```text
BPS < 0
```

means the model performs worse than the baseline under the defined likelihood measure.

Negative BPS values are valid results.

---

# 27. BPS baseline

The baseline used for BPS must be scientifically meaningful and explicitly defined.

Where the inherited NEDS implementation uses a mean-rate or equivalent null model, that behavior may be retained if it is compatible with the current evaluation objective.

The evaluation result must record sufficient information to reproduce the baseline.

Changing the baseline changes the interpretation of BPS and therefore constitutes an evaluation-configuration change.

---

# 28. Likelihood consistency

Likelihood-based metrics must use prediction semantics compatible with training.

For the current Poisson encoding model:

```text
model output = log expected count
```

and the evaluation likelihood must interpret this consistently.

The evaluator must not apply a different statistical interpretation to the same output tensor.

---

# 29. Correlation metrics

Correlation-based metrics may be calculated as diagnostic measures.

If used, their aggregation and input representation must be explicit.

Correlation does not replace likelihood-based metrics or R² because it captures different aspects of prediction quality.

---

# 30. Optional additional metrics

Additional metrics may be supported where scientifically justified.

Examples include:

- explained variance;
- Pearson correlation;
- prediction loss;
- firing-rate error.

Such metrics are supplementary unless explicitly promoted to primary evaluation metrics.

Their definitions and aggregation policies must be recorded.

---

# 31. Visual metrics

The primary project objective is visual-to-neural encoding.

Therefore neural prediction metrics are the required evaluation outputs.

Legacy NEDS visual-decoding metrics such as CLIP cosine similarity may remain available for optional decoding or multimodal experiments.

They are not required for encoding-only evaluation.

---

# 32. Metric independence

Evaluation metrics must be calculated independently where possible.

Failure or inapplicability of one metric must not silently invalidate unrelated metrics.

For example, an undefined R² for a constant target does not necessarily make likelihood-based evaluation invalid.

Metric-specific invalidity must be represented explicitly.

---

# 33. Metric aggregation

Aggregation rules must be explicit.

Possible levels include:

```text
per trial
    ↓
per neuron
    ↓
per session
    ↓
global
```

or another documented order.

Changing aggregation order may change the result and therefore must not occur implicitly.

---

# 34. Avoiding population-size bias

When combining results from multiple sessions, the evaluator must explicitly decide whether aggregation is:

```text
neuron-weighted
```

or:

```text
session-weighted
```

A session with more recorded neurons must not automatically dominate the global result unless that behavior is explicitly intended.

The chosen aggregation policy must be recorded.

---

# 35. Invalid metric values

Scientific metric values may legitimately be:

- negative;
- zero;
- positive.

The evaluator must not clip valid negative values merely to improve presentation.

Undefined results caused by mathematical degeneracy must be distinguished from valid poor performance.

For example:

```text
R² = -0.5
```

is a valid result.

```text
R² = undefined
```

due to zero target variance is a different condition.

---

# 36. Numerical stability

Evaluation must detect non-finite values.

Examples include:

```text
NaN
+Inf
-Inf
```

Such values must not silently participate in aggregated metrics.

The module must:

- identify the affected metric/sample;
- expose the condition;
- apply only explicitly defined handling rules.

---

# 37. Baseline comparisons

Where meaningful, evaluation may compare the model against explicit baselines.

Examples may include:

- mean neural response;
- mean firing-rate model;
- constant-rate Poisson model.

Baseline computation must use only information permitted by the evaluation protocol.

A test target must not be used to fit a baseline in a way that introduces evaluation leakage unless the baseline definition explicitly requires test-distribution statistics and is reported as such.

---

# 38. Temporal diagnostic analysis

Evaluation may expose performance over stimulus-relative time.

Conceptually:

```text
performance(t)
```

This can reveal whether prediction quality changes during:

- early stimulus processing;
- decision period;
- feedback-related period;
- late stimulus presentation.

Such temporal analyses must use the aligned temporal coordinates preserved upstream.

---

# 39. Event metadata

Where behavioral/task event timestamps are retained as metadata, evaluation may use them for grouping or interpretation.

Examples include:

- stimulus onset;
- choice;
- feedback;
- stimulus offset.

These events must not be used to alter the already established neural/visual alignment.

They are analysis metadata only.

---

# 40. Trial grouping

PSTH or condition-specific analyses may require grouping trials.

Grouping criteria must be explicit and scientifically meaningful.

Examples may include grouping by:

- stimulus condition;
- response condition;
- feedback condition;

when such metadata is available.

The evaluator must not infer undocumented trial categories.

---

# 41. Evaluation output

The evaluation module must produce structured results.

Conceptually:

```text
EvaluationResult
    evaluation_id

    checkpoint_id
    dataset_id

    evaluation_strategy

    session_results
    neuron_results

    trial_r2
    psth_r2
    bits_per_spike

    optional_metrics

    prediction_artifact
    plots

    configuration
    provenance
```

The exact serialization format is an implementation detail.

---

# 42. Machine-readable results

Scientific metrics must be available in machine-readable form.

Plots or console output alone are insufficient.

Machine-readable output enables:

- experiment comparison;
- automated reporting;
- later statistical analysis;
- reproducibility.

---

# 43. Diagnostic visualization

The evaluator may generate plots such as:

- predicted versus observed PSTH;
- neuron-level performance distribution;
- session-level performance;
- prediction traces;
- metric distributions.

Plots are derived artifacts.

They must not be the sole source of metric values.

---

# 44. Reproducibility

Given identical:

- model checkpoint;
- evaluation dataset;
- evaluation configuration;
- numerical environment;

evaluation should produce equivalent predictions and metrics within the deterministic limitations of the framework.

Any randomized evaluation operation must use an explicit seed.

Default final evaluation should avoid unnecessary randomness.

---

# 45. Evaluation configuration

Evaluation behavior must be explicitly configurable.

Relevant configuration may include:

- evaluated checkpoint;
- evaluated split;
- evaluation strategy;
- enabled metrics;
- metric aggregation policy;
- BPS baseline;
- trial-grouping policy;
- output location;
- plot generation;
- inference batch size.

Values that change scientific interpretation must be preserved in provenance.

---

# 46. Evaluation provenance

Every evaluation result must remain traceable to:

- model checkpoint;
- training run;
- model configuration;
- evaluation dataset;
- dataset split;
- source sessions;
- neuron identities;
- alignment configuration;
- evaluation configuration;
- metric definitions;
- software version where available.

---

# 47. Multiple model comparison

The module may support comparing evaluation results from multiple trained models.

Each model must first be evaluated independently under compatible conditions.

Comparisons must ensure that models are evaluated using compatible:

- test samples;
- sessions;
- neurons;
- temporal intervals;
- metric definitions;
- aggregation policies.

Incompatible evaluations must not be directly ranked using nominally similar metric names.

---

# 48. Failure handling

Evaluation must fail explicitly when required invariants are violated.

Examples include:

- incompatible checkpoint;
- unknown session mapping;
- mismatched neuron ordering;
- malformed masks;
- missing prediction target;
- incompatible output semantics;
- invalid dataset split;
- corrupted evaluation artifact.

Scientific incompatibilities must not be silently repaired.

---

# 49. Compatibility principle

The existing NEDS-derived evaluation code should be preserved where it satisfies this specification.

Existing implementations of:

- Bits per Spike;
- PSTH evaluation;
- trial-level prediction evaluation;
- session-wise aggregation;
- diagnostic plotting;

may be retained if their semantics are compatible with the current encoding task.

The module must not be rewritten solely to create a cleaner design.

The later implementation audit must determine:

- which existing metrics satisfy the required definitions;
- whether aggregation is scientifically correct;
- whether masks and padded channels are handled correctly;
- whether cross-session evaluation preserves session identity;
- whether evaluation artifacts contain sufficient provenance.

---

# 50. Core invariants

The evaluation module must guarantee the following.

## Test isolation

Final test data does not influence model training or checkpoint selection.

## No parameter updates

Evaluation is inference-only.

## Scientific validity is respected

Padding and invalid positions do not contribute as real observations.

## Zero spikes are valid targets

Valid zero neural counts are not treated as missing data.

## Session identity is preserved

Predictions are evaluated only against the correct neural population.

## Neuron identity is preserved

Prediction and target channel ordering must correspond.

## Temporal identity is preserved

Predictions and targets refer to the same aligned temporal positions.

## Output semantics are explicit

Log expected counts are interpreted consistently with the neural encoding model.

## Metric definitions are reproducible

A metric name alone is insufficient; aggregation and baseline semantics are part of its definition.

## Negative performance remains visible

Valid negative R² or BPS values are not clipped.

## Raw predictions remain traceable

Scientific metrics can be linked back to predictions, trials, sessions, and neurons.

## Evaluation does not repair upstream data

Alignment, padding semantics, and neural/visual representations remain upstream responsibilities.

---

# 51. Primary evaluation set

For the current visual-to-neural encoding task, the primary scientific evaluation should include at least:

```text
Trial-level R²
PSTH R²
Bits per Spike
```

These metrics describe complementary aspects of neural prediction performance.

### Trial-level R²

Measures prediction quality for trial-specific neural activity.

### PSTH R²

Measures prediction quality for the average stimulus-related temporal neural response.

### Bits per Spike

Measures probabilistic predictive improvement over an explicit neural baseline.

No single one of these metrics should be treated as a complete summary of model quality.

---

# 52. Summary

The `evaluation` module performs the final scientific assessment of trained visual-to-neural encoding models.

It operates only after model selection.

A fixed checkpoint is applied to a fixed evaluation dataset without parameter updates.

Predictions remain associated with their:

- session;
- trial;
- neuron;
- temporal position.

The primary evaluation metrics are:

- Trial-level R²;
- PSTH R²;
- Bits per Spike.

Metric definitions, aggregation rules, likelihood assumptions, and baselines must be explicit and reproducible.

Negative results remain scientifically valid.

Evaluation results must be machine-readable and traceable to the checkpoint, dataset, configuration, and source observations that produced them.

The existing NEDS-derived evaluation infrastructure should be preserved wherever its metric semantics and validity handling satisfy this contract.