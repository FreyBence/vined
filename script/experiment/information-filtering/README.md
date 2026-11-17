# ViNED Experiment 3: information-based neural filtering

This controlled experiment compares six configurations at H=12 and D=0/6.
`unfiltered_00` and `unfiltered_06` reference the existing Experiment 2 models;
the other four models train independently from scratch for 50 epochs. CLIP,
stimulus generation, Transformer architecture, optimization, seed 42,
save-every 5, eval-every 1 and validation-loss checkpoint selection remain fixed.
Resolved settings are compared against both existing unfiltered runs before
training. Original Experiment 1/2 S=24 common support is retained: first target
index 23, not 11 or 17. No cross-trial context or targets outside the stimulus.

## Commands

Run from the checkout root with the project environment (`.venv/Scripts/python.exe`
on Windows or `.venv/bin/python` on Linux). The examples use `python` for brevity.

```text
python script/experiment/information-filtering/run.py
```

The default runs CPU scoring and all four preflights; it never trains. Each stage
also runs independently:

```text
python script/experiment/information-filtering/score.py --skip-completed
python script/experiment/information-filtering/train.py --run-config script/experiment/information-filtering/configs/training/neuron_filtered_00.json --preflight
python script/experiment/information-filtering/train.py --run-config script/experiment/information-filtering/configs/training/neuron_filtered_00.json --skip-completed
python script/experiment/information-filtering/evaluate.py --run unfiltered_00 --device cpu --skip-completed
python script/experiment/information-filtering/evaluate.py --run neuron_filtered_00 --skip-completed
python script/experiment/information-filtering/compare.py --skip-completed
```

Explicitly launching all four full training runs is a separate action:

```text
python script/experiment/information-filtering/run.py --stages train
python script/experiment/information-filtering/run.py --stages evaluate compare
```

Preflight must have been reviewed before GPU training. `--stages` executes in
the supplied order; it does not silently train missing dependencies. Evaluation
always includes both reused reference models plus the chosen filtered runs.
`--runs` restricts filtered configurations; `--device cpu` overrides inference
device. Standalone training accepts `--resume-checkpoint PATH` for an epoch-boundary
checkpoint under the same run and `--run-id ID` for independent training repeats.
Resume restores production latest state, optimizer, scheduler, history and random
states. Best validation weights and latest resumable state remain separate.
The fixed six-model comparison consumes canonical configured run IDs.

All entrypoints accept `--experiment-config PATH`. JSON files under `configs/`
declare experiment inputs, screening, inherited training settings, native run
identities, evaluation and comparison. No original datasets are republished.
Use a new output subdirectory and run IDs for a changed controlled experiment;
existing incompatible resolved runs are rejected.

## Scoring method and metadata audit

The verified published dataset exposes original int64 neuron counts, complete
ordered unit tables (including source/probe/unit identity and recorded Allen IDs),
train/val/test trial IDs, 60 Hz physical bins and 768-dimensional visual CLIP
features. Recorded cortical layers refine parent areas; area-only and
noncortical labels remain as recorded. Optional contrast metadata is retained
only for descriptive stratification and is never inferred from embeddings.

The default readout is deliberately small:

1. A seed-42 fixed Gaussian projection maps the unchanged 768 CLIP coordinates
   to 12 coordinates. This data-independent compression has no fitted population
   statistics. For target t and delay D, concatenate the projected last token
   at t-D and the projected mean of the 12-token causal window t-D-11..t-D.
2. Seeded original `(session, trial)` groups define three folds entirely inside
   the persisted training split. Every bin from a trial stays in the same fold.
   Use only original common-support target bins 23..T-1.
3. In each fold, standardize each visual coordinate using fit-fold moments only.
   Fit an intercept and ridge count readout with fixed alpha=10:
   `B=(X'X+alpha I)^-1 X'(Y-mean_fit)`. The intercept is unpenalized.
4. Predict expected counts as `max(mean_fit + X_held B, 1e-8)` and compare against
   `max(mean_fit, 1e-8)`. Score original held-out counts with Poisson NLL
   `lambda-y log(lambda)` (factorial omitted from both). Pool held-out NLL sums
   over all training bins, divide by the held-out bin count, and take baseline
   NLL minus readout NLL. Compute neuron and complete-subregion scores separately.
5. Average D=0 and D=6 gains with equal delay weight. This one score defines the
   population used at both delays. There is no regularization search or
   validation/test selection refinement.

Ridge count fitting minimizes squared error rather than Poisson deviance. It
provides a deterministic, inexpensive regularized conditional-mean approximation
that is evaluated with the requested predictive Poisson criterion. Negative
linear predictions require a fixed floor; nonlinear tuning may be missed.
Absolute NLL gain depends on count scale; the score is neither mutual information
nor proof of purely visual tuning. It can capture correlated task variables.
It is not a claim that the selected readout class is optimal.

Neuron screening keeps `ceil(0.5 * group_neurons)`, at least one, within every
original nonempty subregion. Complete-subregion screening keeps
`ceil(0.5 * subregions)` and all neurons inside those groups. Fractions are
configurable predeclared screening policies, not optimized fractions. Undefined
scores (no training spikes or nonfinite score) rank last; stable original source
column/group order breaks ties. Negative but defined scores remain eligible.
Selected and excluded identities, group rankings, cutoff values, actual fractions,
mean counts/rates and zero fractions are persisted, including undefined scores.
Combined neuron-and-subregion filtering is outside this experiment.

## Evaluation contracts

Every model is evaluated on the entire persisted **validation** split for
model-development decisions. Checkpoint selection has also used validation;
these results are not an independent final generalization estimate. Existing
test results remain exploratory and never enter scoring or policy optimization.

Primary observations are the same full original regional int64 count sums as
Experiments 1/2. For each region, sum expected counts from learned included
subregions and training common-support means of omitted source neurons. For
complete-subregion screening, omitted groups contribute their full neuron sums.
For neuron screening, omitted neurons contribute only their own training means.
These constants use the original training split and original bin support and
are identical at both delays. A parent with no learned output uses its full
training mean. Omitted validation or test observations never construct the
prediction. This is a constant prediction for omitted activity, not temporal
recovery of that activity. Reconstruction records learned/constant contributions
per parent and trial, including constant-only regions.

Native diagnostics use actual selected counts and their own training baselines.
They include original-bin NLL, BPS over training constant, activity summaries,
per-target and trial metrics, temporal correlation, variance ratio, MSE and
baseline improvements. Native NLL cannot rank models with different populations.

NLL and paired gains use original bins and exclude factorial terms. Comparison
reports global cell-weighted NLL and paired gain, equal-trial mean gain, median
trial gain, fraction improved and eligible region-bin elements. Positive gain is
`reference NLL - candidate NLL`. Comparisons to training constants remain
separate from comparisons to trained models. Experiment BPS uses training
constants, independently of general evaluation's test-mean baseline BPS.

W=[1,3,6,12] curves use trailing mean counts times 60, identical endpoints from
bin 34, with no trial crossings. W=1 likelihood uses every bin starting at 23.
Curve summaries average trial-target metrics; they retain unavailable correlation
counts and trial spread. Better smoothed correlation does not prove better
original-bin prediction. All recorded regions enter primary metrics; VIS*
subregion diagnostics receive a separate explicit report without filtering
other anatomy out of the global result. Example trials are evenly spaced by
saved trial order, never selected by favorable scores.

## Artifacts and reproducibility

Outputs live under `output/experiment/information-filtering/`:

- `selection/score.json` points to a completed, hashed `score-attempts/<id>/`.
  `selection.json` contains exact method/configuration, train sample/trial IDs,
  fit/held-out fold identities, fixed projection, fitted fold moments/readout
  coefficients and constants, per-delay NLLs, scores, original anatomical unit
  mappings, ranks, cutoffs, included/excluded identities, training neuron means,
  dataset generation/provenance and seeds. Strategy JSON files are exact
  session/subregion-to-source-column maps consumed by production setup.
  Selection plots show complete-subregion score ranks and before/after counts.
- `runs/<filtered_id>/resolved.json`, `preflight.json`, `selection.json` and
  `manual-verification.json` expose fixed settings, mapping, output widths,
  original support, exact full-region ground-truth reconstruction and excluded
  constants before training. `train.json` points to production logs and best/latest
  checkpoints; `train-attempts/` retains every attempt.
- `runs/<any_id>/evaluate.json` points to a completed inference attempt. It holds
  `metrics.json` (full region), `native-metrics.json`, `subregion-diagnostics.json`,
  `decomposition.json`, common/native support, checkpoint provenance, separately
  indexed parent/native NPZ predictions, temporal plots and a Markdown report.
  Reused reference evaluations are written here, leaving Experiment 2 intact.
- `comparison/compare.json` points to all-six-model metrics/region CSV matrices,
  all 15 pairwise comparisons and paired-trial distributions, constant-baseline
  audits, native diagnostics, decomposition, VIS* diagnostics, matched-trial
  regional overlays, smoothing/variance curves, weighted-gain plots and `report.md`.

Stage locks reject concurrent writes. Attempts and failed attempts remain
inspectable. `--skip-completed` verifies recorded hashes before reuse;
`run.py --repeat` reruns requested stages without discarding old attempts.
Scoring, evaluation and comparison identities include relevant sources; changed
selection/configuration cannot silently enter an existing trained run.

## Manual verification and interpretation

See [verification.md](verification.md) for the executed local checks, actual
selection counts, reference-validation diagnostics and pending empirical results.

Before training, inspect selection provenance/folds, selected/excluded counts and
identities, anatomical partitions, full-region reconstruction, excluded constants,
common support and all four resolved settings/output dimensions. Preflight builds
actual heads and checks widths without optimization. Source checks bind reused
checkpoints to Experiment 2 and verify original population/trial/bin hashes.

The implementation verification does not execute the four full training runs.
Consequently, final filtered-model metrics and an empirical Experiment 4 choice
require subsequent explicit training and evaluation. Until those artifacts exist,
retain both delays and the unfiltered reference; treat either fixed 50% screen as
an unproven candidate. The completed comparison report assesses whether global
and equal-trial gains agree at both delays, exposes difficult regions and keeps
inconsistent results explicit. It does not force a winner, infer visual causality
or claim robustness from a favorable smoothed trace.
