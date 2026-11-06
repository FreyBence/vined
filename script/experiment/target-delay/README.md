# Controlled target-delay experiment

This experiment compares how much visual history should precede a neural target.
It uses the production ViNED model, configuration resolver, trainer, dataset
loader, checkpoint lifecycle and complete test-split inference. It does not
regenerate stimuli, CLIP features or neural data. The shared `training.target_granularity` selects `region` (the current setting)
or `neuron`. Region mode reuses production anatomical count aggregation: each
target channel is the sum of recorded neuron counts in that area. No new
subregion grouping or neuron filtering is introduced.

## Hypothesis and limits

A positive stimulus-to-response delay may improve single-bin likelihood and
temporal response prediction relative to D=0 and a training-derived constant
predictor. Larger response windows may expose slower response structure, but
smoothing alone can improve correlation while suppressing amplitude. Inspect
likelihood, error, variance, target-channel distributions and trial consistency together.

This is exploratory selection with one controlled initialization seed. Selecting
delays on the test-split comparisons means that split is no longer an independent
confirmation set for the chosen delay. There are no significance tests, confidence
claims, causal latency estimates or claims of held-out-session generalization.
Follow-up confirmation needs untouched data and a separately specified protocol.

## H, D and W

- **H=12:** number of visual bins in each independently encoded context.
- **D=0,2,4,6,8,12:** delay in 60 Hz bins; positive D puts stimuli before targets.
- **W=1,3,6,12:** trailing offline response-window length. Both predicted and
  observed counts are averaged over W bins and multiplied by 60 to give
  spikes/second per target channel (regional sum in region mode, neuron in neuron
  mode). W never changes the training targets.

The target mapping is exactly:

```python
neural_target[t] <- visual_context[t-D-H+1 : t-D+1]
```

The right endpoint is exclusive. At t=23, D=0 uses `[12:24]`, D=2 uses
`[10:22]`, and D=12 uses `[0:12]`. Bin indices are onset-aligned; D=12 is
200 ms. D=0 includes visual bin t, not a positive physiological delay.

Each context is gathered before the transformer. The model executes isolated
H-bin windows through all layers and reads the last visual token into target t.
It does not rely on per-layer causal attention to prevent multi-layer leakage.
No neural observation enters the encoding input. Context indices are checked
again by the model against D and H. No learned architecture or head is changed.

All runs train and score original single-bin counts from t=23 onward, because
H+max(D)=24 bins must exist. W comparisons use identical endpoints t=34 onward:
the W=12 trailing window then begins at 23. Single-bin NLL uses the larger
original-bin support t>=23, independent of W. Curve metrics, including W=1,
use t>=34. All contexts and aggregation windows stay inside the same trial.

## Dataset and configuration

Run from the checkout root with the project `.venv` Python. Commands below use
Windows syntax; on Linux substitute `.venv/bin/python`.

`configs/experiment.json` explicitly pins the published dataset generation and
expected generation ID, session selection, H/D grid, subordinate configuration
paths and experiment output root. It is independent of future edits to
`src/configs/run.json`. The initial dataset/session values were copied from that
production controller. Change them explicitly for another experiment before
creating outputs. `session_id=null` selects all persisted sessions.

The loader verifies generation hashes, original counts/features, memberships and
ordered units. Preflight additionally requires onset-aligned consecutive 60 Hz
bins ending no later than stimulus offset. Every retained trial must support
every W; short trials fail explicitly rather than silently changing membership.
All evaluated sessions must have training observations. Source alignment and CLIP
generation/representation provenance remain in the dataset metadata, resolved
configuration and prediction sidecars. No source artifact is written.

`configs/training-defaults.json` contains the production controller's current
entry arguments and inline overrides: seed 42, configured epochs, batch 1, accumulation
4, AdamW 0.0002/0.01, no scheduler, validation every epoch, validation-loss
selection, and the 2-layer/128-hidden/4-head model with dropout 0.1. The current
target granularity is explicitly `region`; the epoch count is
`entry_arguments.epochs` (currently 300). Unspecified settings are
resolved from the existing production training/model JSON profiles; the entire
effective configuration is saved. All six run configurations contain only their
ID, H and D, so optimization and initialization cannot accidentally vary by run.
Models start from scratch using the same initialization policy and seed. Resume
only accepts a checkpoint from the same run directory and checks the production
compatibility record, including dataset, model, D, support and optimization state.

`configs/evaluation.json` selects the whole persisted test split, W grid, inference
batch/device/seed, deterministic representative-plot count and minimum contrast
trial count. `configs/comparison.json` selects completed run IDs and D=0 reference.

## Independent commands

First inspect resolved parameters and actual common support without optimization:

```powershell
.venv\Scripts\python.exe script/experiment/target-delay/run.py --stages preflight
```

Every selected configuration is resolved before the orchestrator starts training.
Preflight saves the mapping, target/curve counts, resolved architecture,
optimizer, seed and compatibility identity. Review `preflight.json` and
`resolved.json`, including dataset generation and per-trial index lists.

Train one model independently:

```powershell
.venv\Scripts\python.exe script/experiment/target-delay/train.py --run-config script/experiment/target-delay/configs/delay_04.json
```

Evaluate a completed run independently:

```powershell
.venv\Scripts\python.exe script/experiment/target-delay/evaluate.py --run-config script/experiment/target-delay/configs/delay_04.json
```

Optional `--device cpu` changes inference hardware. All other scientific settings
remain pinned to the run snapshot. To compare previously generated evaluations:

```powershell
.venv\Scripts\python.exe script/experiment/target-delay/compare.py
```

For a subset, include the configured reference:

```powershell
.venv\Scripts\python.exe script/experiment/target-delay/compare.py --run-ids delay_00 delay_04
```

Run the complete workload only when ready for six independent training jobs:

```powershell
.venv\Scripts\python.exe script/experiment/target-delay/run.py
```

Select stages or runs with `--stages train evaluate` and
`--runs delay_00 delay_04`. All scripts accept `--experiment-config PATH`.
Run IDs in `run.py` select the corresponding experiment-local JSON files.
The orchestrator invokes subordinate scripts sequentially and stops at the first
failure with its nonzero exit code. It never implements optimization or inference.

## Repeat, retry and resume

The orchestrator skips completed compatible stages after verifying all retained
artifact hashes. Standalone scripts run a new attempt by default; pass
`--skip-completed` to reuse a verified compatible stage. `run.py --repeat` repeats
selected stages into new attempt directories. A repeated train starts from scratch
unless an explicit resume checkpoint is provided. For another independent run,
use `train.py ... --run-id delay_04_repeat`, then pass that ID to `evaluate.py` and
`compare.py --run-ids ...` (update comparison config if the reference changes).

Resume a failed/interrupted run from a retained complete epoch checkpoint:

```powershell
.venv\Scripts\python.exe script/experiment/target-delay/train.py --run-config script/experiment/target-delay/configs/delay_04.json --resume-checkpoint output/experiment/target-delay/region/runs/delay_04/train-attempts/ATTEMPT/training/model_epoch_0.pt
```

Use the latest usable epoch checkpoint; partial files and the final checkpoint
with no remaining epochs cannot resume. Production resume restores optimizer,
scheduler, RNG and loader state and carries selection ancestry forward. Saving
frequency follows the shared production configuration, so interruption can lose
work since the last saved epoch. The orchestrator does not guess a resume source:
resume with the standalone command, then restart orchestration to skip completion.

Partial/failed attempts remain available. Ordinary exceptions and Ctrl+C record
failed/interrupted states. A hard process kill may leave `*.lock` and a running
state; verify the process is gone before manually removing that lock. A directory
alone never proves completion. Changing scientific settings or source code after
preflight requires a new `output_root` under the experiment directory. Existing
incompatible runs are rejected. Re-evaluating a compatible newly trained attempt
or recomparing a new selection creates fresh outputs and retains older attempts.

## Output contracts

```text
output/experiment/target-delay/region/
  manifest.json                    # experiment, sources, common support, baseline
  runs/<run_id>/
    resolved.json                  # complete effective configuration and provenance
    preflight.json                 # mapping, counts and resolved controls
    train.json                     # execution state and attempt/artifact hashes
    train-attempts/<id>/
      train.log
      execution.json
      result.json                  # selected/final checkpoint references and hash
      training/                    # original production manifests/history/checkpoints
    evaluate.json
    evaluate-attempts/<id>/
      execution.json
      metrics.json
      support.json
      provenance.json
      predictions/index.json       # sample/session/trial identities and source metadata
      predictions/000000.npz
      plots/trial_000000.png
      report.md
  comparison/
    compare.json
    compare-attempts/<id>/
      execution.json
      comparison.json
      matrix.csv
      plots/
      report.md
```

Execution manifests use schema 1, state `running`, `failed`, `interrupted` or
`completed`, an input identity, unique attempt path and completed artifact SHA-256
hashes. Stage readers verify every listed payload before reuse. `train.json`
is completed only after final-epoch and selected checkpoint validation plus
production manifest readback. Evaluation requires complete trial and target-cell
coverage and readable prediction artifacts before publishing completion.

Prediction NPZ arrays preserve `observed` int64 `[T,N]`, `log_expected` and
`expected` float64 `[T,N]`, training `baseline` `[N]`, `target_indices` `[K]`,
`curve_endpoints` `[C]`, onset-relative `time` and session-clock timestamps/bin
edges `[T]`. T is the full original trial without padding, N the ordered real
target channels (regions or neurons according to the recorded granularity). Predictions outside `target_indices` must not be scored. Full-trial plots
show all curve-supported endpoints; the initial unsupported interval is absent.
The support manifest retains exact ordered unit tables and original target counts
fingerprints. NPZ loading uses `allow_pickle=False`.

## Metrics and comparison

Single-bin Poisson NLL matches production training (`log_input=True`, `full=False`,
omitting the target-only factorial term). Reports include cell-weighted NLL,
training-baseline NLL, gain and bits/spike, per trial, target channel, region and population.
The constant predictor is the per-target-channel mean over common-support **training** bins;
zero means are floored at 1e-8 only for likelihood. It is not fit on evaluation data.

Every W reports within-trial Pearson correlation, R², MSE/MAE/RMSE, error relative
to the training mean, predicted/observed means, standard deviations, variance ratio
and peak-to-peak amplitude. Constant or short series have null correlation and an
explicit reason. Zero-spike BPS and zero-variance ratios are unavailable. Curves
and channel/trial summaries use equal contributing trial-channel weighting, while
likelihood pools valid count cells. Population/region curve summaries first average
the same target channels, then summarize trials. Correlations are not pooled
across arbitrary trial boundaries. No W>1 Poisson objective is calculated.

Original persisted contrast values are retained with their side labels where
available. Contrast groups need the configured minimum trial count; missing or
small groups are explicitly unavailable. Contrast information is never inferred
from embeddings or modified. Representative plots use evenly spaced trial indices
in deterministic persisted order, independent of prediction performance.

Comparison verifies run/evaluation manifests, original generation and split
provenance, units, counts, physical target bins, common W support and controlled
training settings, allowing only D to vary. It reads stored metrics and predictions
without loading weights or repeating inference. Outputs include D-by-W matrices,
likelihood curves, channel/trial gain distributions, regional curves, matched temporal
plots, paired per-trial differences against D=0, contrast summaries and a cautious
interpretation. Lowest pooled single-bin NLL is a descriptive candidate, not a
ranking by selected-trial maximum correlation. The complete JSON retains the
regional, target-channel, contrast and trial summaries needed to judge consistency.

## Interface adaptations and limitations

`temporal_context_view` adds optional `target_delay_bins=0` and
`target_support_bins=12`. Nondefault values require strict encoding with sufficient
complete support. These values travel through production setup, objective,
checkpoint restoration and full-test inference; the model checks the exact delayed
indices. Default D=0/H-support=12 calls and checkpoint identities are unchanged.
Production training CLI accepts `--target-delay-bins` and `--target-support-bins`.
The experiment supplies its JSON settings through these runtime contracts without
changing global production profiles. The model architecture and learned tensors
remain the same.

The ordinary general evaluation metric/report pipeline retains its existing H=12
and test-mean-baseline protocol. Use this experiment's `evaluate.py` and
`compare.py` for delayed runs and the training-derived baseline; its custom metric
layer is needed for these different scientific definitions. Production full-split
inference and R² primitives are reused. Comparison does not use example-pair or
ground-truth visualizations as a substitute for complete evaluation.

This workflow supports one process per invocation, ordinary production device
placement, known training sessions and the configured test split. Very long trials
and retaining all full-trial predictions can consume substantial memory/storage.
No full six-run training workload is needed to verify implementation; preflight
and bounded real-data forward/backward/inference diagnostics suffice for setup.


In region mode, preflight records the complete original-neuron-to-region mapping
and fingerprints aggregated counts. Bias initialization, validation BPS and offline
baselines use the same common-support training means. Regional predictions cannot
provide per-neuron prediction metrics. Existing JSON fields named `neurons`,
`neuron_summary`, and matrix `mean_neuron_correlation` retain their schema names
but index target channels; `target_granularity` identifies their meaning. Plot
units use spikes/s/region, with no division by contributing neuron count. The
new `region/` output root keeps earlier neuron runs separate.
