# Evaluation interface

## Checkpoint and test setup

From the checkout root, use the project Python:

```text
python src/eval.py --checkpoint OUTPUT/runs/ATTEMPT/model_best.pt --dataset-generation DATASET/generations/ID --output-dir OUTPUT/evaluation --save-plot
```

`script/eval.sh` forwards these same arguments through the project environment. The former positional launcher and inferred checkpoint paths are retired. `--data_path` is an alias for `--dataset-generation`. Optional arguments are `--expected-dataset-generation-id`, repeated `--eid` session selection, positive `--batch-size` (default 32), `--device` (default CPU), and nonnegative `--seed` (default 42, used by the test loader generator).

`evaluation.setup.resolve_setup(*, checkpoint_path, dataset_generation, expected_generation_id=None, session_ids=None, batch_size=32, device="cpu", seed=42)` returns `EvaluationSetup` with:

- `model`: completely restored model on the requested device in evaluation mode, with the checkpoint's configuration and session mapping.
- `dataset`: verified, selected `PersistedSplit` for test observations only.
- `dataloader`: ordered, non-shuffled test loader retaining every selected sample, including incomplete final batches.
- `configuration`: effective inference settings, prediction direction, modalities, split, and selected sessions.
- `provenance`: checkpoint path/hash/epoch and selection record, training run and adaptation lineage, resolved `checkpoint_lineage` with each checkpoint's path/hash/run/epoch/selection, model identity/configuration, dataset generation/path, and persisted split metadata/strategy.

`setup.summary()` returns JSON-compatible provenance and configuration with test sample/batch counts. `--setup-only` prints this summary after restoring the model and constructing only the test loader; it does not infer, score, or publish evaluation artifacts.

## Compatibility and errors

Inputs must be a trusted versioned training checkpoint supported by `trainer.pretrained.model_from_checkpoint` and the exact scientific dataset generation recorded in its configuration. Generation identity binds alignment, temporal/bin representation, visual features, and split membership. Different generations are rejected because their semantic compatibility is not established by the checkpoint contract.

Each selected test session must have a checkpoint population with identical ordered unit columns, dtypes, rows, and neuron count. Feature width and real temporal/neural dimensions must fit the restored model. Unknown sessions require prior adaptation; setup never initializes replacement heads or changes observations.

Validation-selected checkpoints must contain weights from the recorded selected epoch. Final selection requires the configured final epoch. Test observations are compared by original `(session_id, trial_id)` against optimization and validation membership in the checkpoint and all recorded resume/pretrained ancestors. Referenced ancestor checkpoints must remain available; pretrained hashes must agree. No training or validation loaders are constructed.

Empty test selections, unsupported selection/split policies, incompatible identities/populations/dimensions, nonfinite model parameters, overlap with previously consumed observations, and unavailable ancestry raise explicit errors. Artifact verification and filesystem/device errors propagate without fallback. Supported split strategies are `within_session` and `session_held_out`, provided a compatible session mapping already exists.

Model inputs/outputs and scientific batch metadata follow the [model](../model/interface.md) and [training-dataset](../training-dataset/interface.md) contracts: neural outputs are log expected counts per persisted bin; temporal positions are indices, physical timestamps are session seconds, and temporal/neuron masks identify real observations.

## Prediction collection

`evaluation.predictions.collect_predictions(setup: EvaluationSetup) -> PredictionCollection` visits every selected test sample once, in persisted order, under `torch.inference_mode()` and `model.eval()`. It verifies sample/trial/session membership, ordered units, input/target values, validity, positions, timestamps, and bin duration against the scientific source; duplicate, missing, unexpected, or reordered samples fail explicitly. Returned arrays are detached CPU NumPy copies; unit tables and metadata are also copied.

`PredictionCollection.predictions` is an ordered tuple of `EvaluationPrediction` records. Each record exposes `sample_id`, `session_id`, original `trial_id`, the complete ordered `neuron_identity` table, `metadata`, `bin_size` in seconds, source `stim_on`, `stim_off`, `aligned_start`, `aligned_end` in session seconds, and:

| Field | Shape and semantics |
| --- | --- |
| `temporal_positions` | Int64 `[T]` model indices; padded positions are -1. |
| `physical_timestamps`, `bin_start_times`, `bin_end_times` | Float64 `[T]` session-clock seconds, with NaN padding. |
| `temporal_mask`, `neuron_mask` | Boolean `[T]` and `[N]` scientific validity. |
| `observed_neural` | Exact int64 `[T,N]` counts, retaining valid zeros. |
| `observed_visual` | Float32 `[T,D]` source features, retaining all coordinates. |
| `predicted_neural` | `[T,N]` log expected counts, or `None` for decoding-only checkpoints. |
| `predicted_visual` | `[T,D]` visual feature prediction, or `None` for encoding-only checkpoints. |

`T` and `N` are the checkpoint runtime padding sizes; `D` is its visual width (currently 768). Real unit rows identify the leading valid neural columns. Neural cells are interpretable only where both masks are true; padded model fill is never a valid prediction. No exponentiation, rate conversion, normalization of observations, or metrics are applied during collection.

Encoding uses visual-only keyword inference without neural targets or model loss. Decoding uses the neural encoder without targets. Multimodal checkpoints receive two deterministic passes: hide all real neural tokens for encoding and all real visual tokens for decoding while preserving scientific validity. No random training masks, input from the predicted target modality, stochastic sampler, optimizer, or singleton duplication is used.

The collection carries detached copies of setup `configuration` and `provenance`. `collection.summary()` exposes total samples, session trial/neuron counts, direction, and checkpoint/dataset identities. The former inferred-path/held-out evaluation flow, `--base_path`, and unused legacy helpers in `utils.eval_utils` are retired; callers use the evaluation entry points documented here.

## Scientific metrics

`evaluation.metrics.compute_metrics(collection, *, config=None)` returns a JSON-compatible dictionary containing `configuration` (definitions/policies), `session_results`, `global_metrics`, and checkpoint/dataset `provenance`. `MetricConfig(aggregation="session_weighted", bps_baseline="test_mean_count", psth_grouping="all_trials")` supplies the supported policies. The CLI exposes corresponding `--aggregation`, `--bps-baseline`, and `--psth-grouping` options, publishes the result, and prints JSON including its `artifact_path`. Diagnostics go to stderr.

Encoding scores all real neurons without a firing-rate threshold. Each neuron result carries its session-scoped source identity, column, metric records, and available PSTH profile. Decoding scores full visual vectors; multimodal checkpoints report both. Metrics operate in float64 and retain finite negative results:

- `trial_r2`: per session/neuron, pool all valid trial/time cells, then calculate `1 - sum((y-p)^2)/sum((y-mean(y))^2)` using expected counts `p=exp(log_count)`. No trial averaging occurs first.
- `psth_r2`: within each session/neuron, average all trials at each valid model position using identical observations for target and prediction. Corresponding physical bin edges relative to `stim_on` must agree to absolute tolerance `1e-8` seconds. Unequal lengths contribute only at their real positions. Incompatible grids make only PSTH R² unavailable; no interpolation or regrouping occurs. The stored profile records relative bin edges, positions, per-position trial counts, and observed/predicted means. R² is calculated over those means across positions.
- `bits_per_spike`: Poisson log-likelihood improvement over a constant expected count equal to that neuron's mean over valid test observations, divided by its total observed spikes and `log(2)`. The baseline explicitly uses test-distribution statistics, matching the inherited null model; its fitted count, observations, and spike total are recorded. Factorial terms cancel and log predictions are used directly for the log-likelihood term. Bins remain counts rather than Hz; zero counts remain observations.
- `visual_cosine`: cosine similarity across all feature coordinates, averaging valid time vectors within a session and then equally across sessions. Zero-norm/nonfinite vectors make the session metric unavailable and identify affected samples.

Session neural scores are equal means over available neuron metrics. Global `session_weighted` scores equally average available session means; `neuron_weighted` pools available neuron metrics across sessions. Visual global scores always weight sessions equally. Each aggregate records available/total/unavailable contributor counts and `partial`; global scores also identify sessions without available values. Every unavailable neuron remains present in the structured output.

Individual metric records contain `value`, `status`, `reason`, and coverage. Undefined constant-target R², fewer than two R² observations, empty observations, zero-spike BPS, incompatible PSTH bins, and nonfinite input/calculation conditions produce `value=None` with a reason. Neural metric records identify contributing samples and samples with nonfinite predictions or overflowed expected counts. A failure in one metric does not suppress unrelated metrics. Invalid masks, dimensions, session populations, bin duration, or missing required predictions raise `ValueError`; they are not silently repaired.

`trial_r2(observed, predicted)` and `poisson_bps(observed, log_expected_count)` expose the same one-dimensional valid-observation calculations for direct callers. Their inputs must already be filtered by scientific validity; they do not infer padding from numeric values.

## Publication and readback

`evaluation.artifacts.publish_evaluation(collection, output_dir, *, metric_config=None, persist_predictions=True, save_plots=False) -> EvaluationArtifact` computes the configured scores, writes and verifies a complete staged artifact, then renames it into a new UUID-named directory. Existing evaluation directories are never replaced or automatically reused. The return exposes `path`, `manifest`, `result`, and `collection` (or `None` when predictions are omitted).

CLI `--output-dir` defaults to `<VINED_OUTPUT_DIR>/evaluation`. Raw prediction persistence is enabled by default; `--no-predictions` produces metrics/provenance only. `--save-plot` (also `--save_plot`) adds a session metric plot and the first available neuron PSTH profile per session, explicitly labelled by session and column. Plots retain negative scores, identify unavailable values, and use stimulus-relative seconds and counts per recorded bin. Neither plots nor display values define the metrics. `--setup-only` continues to avoid inference and publication.

Schema version 1:

```text
<output-dir>/<evaluation_id>/
    manifest.json
    result.json
    units/000000.parquet
    predictions/000000.npz     # optional
    plots/*.png               # optional
```

`manifest.json` records `schema_version=1`, `kind="evaluation"`, `complete=true`, `evaluation_id`, its `manifest_id` fingerprint, exact payload paths/SHA-256 hashes, inference/metric configuration, checkpoint/dataset provenance, software versions/source hashes, sample index, and session-to-unit-table mapping. Each indexed sample records original identities, scalar timing and full alignment/construction metadata, plus its NPZ path and array shapes/dtypes when persisted. JSON metadata converts arrays to lists and unavailable numeric metadata to null. Parquet retains complete ordered unit columns and dtypes. NPZ contains the array fields in the prediction contract above, including masks, exact targets, physical coordinates and full predictions; absent prediction directions omit their array. NumPy loading disables pickle.

`result.json` carries the configured neuron/session/global metric records and PSTH profiles, evaluation ID, inference settings, checkpoint selection/run/adaptation lineage, dataset generation/split strategy, software provenance, and relative plot/prediction locations. Source alignment provenance remains associated with every indexed sample even when raw predictions are omitted.

`load_evaluation(path, *, expected_evaluation_id=None, expected_checkpoint_sha256=None, expected_dataset_generation_id=None) -> EvaluationArtifact` reads one explicitly selected directory. It checks manifest fingerprint, schema/completeness, exact file accounting and payload hashes, result/manifest agreement, sample membership, and prediction array schemas. With persisted predictions, it rebuilds a detached collection and verifies the associated scores using their recorded policy; the checkpoint and source dataset need not remain available. A metrics-only artifact returns `collection=None` and cannot regenerate raw predictions.

Missing files propagate filesystem errors; unsupported, incomplete, corrupt, or mismatched artifacts raise errors without reuse or repair. Optional expected identities reject unrelated results before consumption. Prediction metadata is a JSON-compatible readback representation; scientific arrays and unit tables retain their stored dtypes.

```python
from evaluation.artifacts import load_evaluation
from evaluation.metrics import compute_metrics

artifact = load_evaluation(evaluation_directory, expected_checkpoint_sha256=checkpoint_hash)
if artifact.collection is not None:
    recalculated = compute_metrics(artifact.collection)
```
