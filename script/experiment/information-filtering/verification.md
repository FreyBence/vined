# Experiment 3 manual verification

## Selection and provenance

The local generation is
`01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c`,
session `0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63`.
It contains 218 training, 31 validation and 63 test trials, 280 recorded neurons,
13 recorded subregions and 8 parent regions. Scoring consumed 121,229 eligible
training bins and no validation/test responses. Three grouped folds contain
complete original trial identities; normalization, ridge coefficients and
constants are fitted only to their fit folds. The fixed projection is independent
of all observations.

Two CPU scoring executions reproduced all per-delay scores and selected
populations exactly. No neuron score was undefined. Both delays use the same
selection for each strategy.

| Strategy | Selected neurons | Excluded neurons | Learned subregions |
| --- | ---: | ---: | ---: |
| Neuron filtering | 144 | 136 | 13 |
| Complete-subregion filtering | 73 | 207 | 7 |

Complete-subregion filtering retains MB, VISpm1, VISpm2/3, VISpm4, VISpm6a, bsc
and or. It retains every neuron in those groups. Neuron filtering retains at
least one neuron in every original group, including the singleton fp group.
Exact identities, source columns, rank thresholds and counts are in the active
`output/experiment/information-filtering/selection/score.json` attempt's
`selection.json`. The active pointer and file hashes are checked by consumers.

## Count and temporal contracts

Preflight compares current full-subregion population, trial/bin support,
timestamps, count hashes and dataset metadata against both Experiment 2 runs.
It verifies full-region ground truth as original int64 source-column sums on
all 312 trials. The first target remains original index 23 (H=12/S=24); smoothing
uses common endpoints beginning at 34. The published CLIP representation,
stimulus generation and original datasets are unchanged.

All four standalone preflights completed successfully. Resolved settings match
the actual Experiment 2 controls, with 50 epochs, seed 42, save-every 5,
eval-every 1 and validation-loss checkpoint selection. Actual output heads have
13 channels for both neuron-filtered delays and 7 channels for both
complete-subregion-filtered delays.

Fresh-model CPU forwards were executed at D=0 and D=6 for both strategies on
the first validation trial (original trial 3, 462 eligible bins). They produced
13 or 7 native channels and reconstructed all 8 full-region channels. No
optimization step or checkpoint training was performed. Native observations
were checked against declared original selected source-column sums; full-region
observations matched recorded common-support hashes.

Excluded constants were independently recalculated as pooled original training
counts of excluded neurons. Maximum disagreement was 4.45e-16 counts for neuron
filtering and 1.34e-15 for complete-subregion filtering, consistent with floating
point summation order. Complete-subregion filtering correctly produces
constant-only parent predictions for APN, CA1, DG and fp. No excluded validation
or test response contributes to prediction construction.

`output/experiment/information-filtering/manual-verification/filtered-forward.json`
records output widths, temporal coverage, exact target checks and zero
optimization steps. Each filtered run's `manual-verification.json` records its
resolved dimensions, full-population reconstruction and excluded constants.

## Reused checkpoints and evaluation

Both existing Experiment 2 selected checkpoints were verified against their
training manifests and hashes, then evaluated on all 31 validation trials on
CPU. Full-region metrics cover the same 160,608 eligible region-bin elements.
Parent observations and saved temporal coordinates match exactly across delays.
Native and common predictions/metrics are stored separately. Selection/ranking
never uses these validation results.

The existing comparison helpers and new paired-weighting calculation were
executed on those two real reused-model evaluations. With unfiltered D=0 as
reference and D=6 as candidate, global paired NLL gain was approximately
`-0.00035824`, equal-trial mean gain `-0.00030593`, median trial gain
`+0.00008276`, and 16/31 trials improved. These are validation reference
diagnostics, not filtered-model results or a decision to eliminate either delay.
The saved reference-only pair and plots are under `manual-verification/`.

Selection-ranking and saved validation-curve plots were visually inspected.
Python source and JSON configurations were parsed, and `git diff --check` was
run. No test code, test files or CI/CD infrastructure was created or modified.

## Limits and Experiment 4 recommendation

The four new full training runs were deliberately not launched. Their selected
checkpoints, native filtered metrics, complete six-model comparison and empirical
filtering decision therefore remain unavailable until explicit training and
evaluation. Fresh untrained forwards above establish pipeline compatibility,
not predictive performance.

Keep both delays and the unfiltered reference as Experiment 4 candidates until
the controlled comparison exists. VISpm2/3 and VISpm6a have the strongest positive
complete-subregion readout gains and are provisional population candidates.
Only those two complete groups have positive scores under this particular
readout; the fixed top-50% policy also retains negative-scoring groups, as
predeclared. This is not evidence that another fraction would improve the main
model. Do not optimize screening from the inspected validation/test curves.

The score is a projected linear count-readout diagnostic, fitted by squared
error and assessed by predictive Poisson NLL. It can miss nonlinear visual
structure and reflect correlated task variables. Native NLL is target-dependent;
excluded constants do not recover omitted temporal activity. Neither smoothing
nor a favorable example establishes a full-population improvement or visual
causality. The final comparison must consider effect magnitude, both weighting
schemes, trial consistency, delay dependence, variance and correlation together;
inconsistent or negligible differences warrant retaining multiple candidates.
