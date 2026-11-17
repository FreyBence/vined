# ViNED Experiment 2: subregion aggregation

Controlled exploratory 2×2 comparison: existing `region_00`/`region_06` and fresh
50-epoch `subregion_00`/`subregion_06`. Checkpoints from Experiment 1 remain in
their original locations. No model is initialized from their weights.

Run commands from the checkout root with the project Python environment
(`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux).

```powershell
# Default: inspect both configurations, construct CPU heads, publish diagnostics;
# no optimizer steps or GPU inference.
.venv/Scripts/python.exe script/experiment/subregion-aggregation/run.py

# Independent preflight / training / evaluation / four-model comparison
.venv/Scripts/python.exe script/experiment/subregion-aggregation/train.py --run-config script/experiment/subregion-aggregation/configs/training/subregion_00.json --preflight
.venv/Scripts/python.exe script/experiment/subregion-aggregation/train.py --run-config script/experiment/subregion-aggregation/configs/training/subregion_00.json
.venv/Scripts/python.exe script/experiment/subregion-aggregation/evaluate.py --run-config script/experiment/subregion-aggregation/configs/training/subregion_00.json
.venv/Scripts/python.exe script/experiment/subregion-aggregation/compare.py

# Explicit full workload (two new 50-epoch runs); launch separately when desired.
.venv/Scripts/python.exe script/experiment/subregion-aggregation/run.py --stages train evaluate compare
```

Use `subregion_06.json` for the other independent run. Evaluation accepts
`--device cpu` or `--device cuda`. Entrypoints accept `--experiment-config`;
comparison accepts `--comparison-config`. Default orchestration skips completed
stages; `--repeat` creates new attempts. Standalone stages accept
`--skip-completed`. Training accepts `--run-id` for isolated repeats and
`--resume-checkpoint` for an epoch-boundary checkpoint under the same run.
Comparison intentionally requires the four canonical IDs; use a separately
configured output root for a complete repeated experiment.

## Representation and audit

The source neural pipeline validates finite session-clock spike timestamps,
integer unit assignments and stable recording-scoped unit identities. Its
explicit quality/anatomical policy is preserved in the published generation;
it retains original Allen IDs/acronyms, source-unit IDs, recording/probe,
sorting collection/revision and available quality metadata. It does not infer
quality from firing rate. Experiment 2 consumes that exact published population,
applying no additional quality, activity, information or minimum-size filtering.
The consumed generation explicitly records `quality=null` and `anatomy=null`;
all recorded source units in that generation are eligible.

`trainer.targets.transform_neural_targets(..., "region")` sums original int64
spike counts within recorded parent areas, merging recorded layer labels into
their parent. The `subregion` extension refines this exact partition by each
recorded Allen anatomical ID/acronym. Valid noncortical labels and recorded
area-only labels remain intact. Unknown anatomy fails explicitly, as in the
parent experiment; it is not silently excluded or assigned an invented layer.
Hemisphere handling follows the existing absolute-Allen-ID parent contract.

The discovered population is 280 neurons: APN, CA1, DG-mo, DG-sg, MB,
VISpm1, VISpm2/3, VISpm4, VISpm5, VISpm6a, bsc, fp and or. These are 13 targets
within 8 parent regions. The mapping is discovered at runtime, not hardcoded.
`mapping.json` records every contributing original neuron identity, source
column, parent, count, units, aggregation and mapping limitations. `reconstruction.json`
records every original trial and bin index. Source generation metadata retains
the upstream selection policy and source identities.

Both regional and subregion heads use the existing `RegionDecoder`: a
session-specific linear projection with no output LayerNorm. The unchanged
transformer produces log expected **summed counts per original 1/60-second bin**.
Poisson loss uses `log_input=True, full=False`, reduced over valid cells.
The new models have 13 output channels instead of 8; native losses across these
spaces are not comparable. Biases, validation BPS null rates and offline
baselines use training-only mean counts on the common support.

## Controlled settings and provenance

`configs/training/base.json` was derived from the actual regional resolved
configuration. Preflight compares resolved scientific settings to **both**
regional runs and fails if anything besides delay, granularity, target mapping
or output width changes. Settings include seed 42, 50 epochs, batch sizes 1,
gradient accumulation 4, AdamW LR 0.0002, weight decay 0.01, scheduler `none`,
two transformer layers, hidden size 128, 4 heads, intermediate size 256 and
dropout 0.1. Save frequency is 5, validation frequency 1, selection is minimum
validation loss. Full resolved values are retained per run.

The generation is
`01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c`, session
`0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63`. There are 218 training, 31 validation
and 63 test trials. H=12 and S=24 preserve Experiment 1's D=12-wide sweep support:
single-bin targets start at original bin 23 for both delays. Neural timestamps
never shift. The visual slice for target t is `[t-D-11:t-D+1]` within its own
trial. CLIP features and contrast remain unchanged. Common W-curve endpoints
start at bin 34, for every W=1,3,6,12. Longer W is offline averaging only.

Preflight verifies source stage hashes, checkpoint hashes, exact generation,
split/sample/trial identities, ordered neurons, physical-time hashes and
target-count hashes against Experiment 1. It checks an exhaustive neuron
partition and exact integer reconstruction across all 312 trials, and matches
the saved regional test observations, timestamps and target indices directly.
It fails with the incompatible field/relation instead of substituting data or
retraining a regional model. Existing region/neuron contracts remain unchanged.

## Evaluation and reporting

`evaluate.py` performs complete persisted-test inference with production setup
and collection. It scores native subregions and reconstructs parent predictions
by `logsumexp` over subregion log intensities (equivalent to summing intensities).
It sums observed counts exactly and checks them against the original saved
regional ground truth. No averaging or neuron-count reweighting is appropriate
for this **sum-count** contract.

The metric implementation is reused directly from target-delay `analysis.py`:
single-bin Poisson NLL, training-mean NLL gain and bits/spike; trailing W-bin
rates and within-trial Pearson correlation, variance ratio, MSE gain, RMSE,
MAE, bias, standard deviations and peak-to-peak ranges. Constant-series
correlations are explicitly unavailable. Cell-weighted likelihood and macro
trial/channel curve summaries remain distinct. Native metrics additionally
retain per-subregion activity ranges, mean rate, zero fraction, neuron counts
and trial consistency. Native regional-named fields denote subregion channels
in `native-metrics.json`; they are never used in the common comparison matrix.

`compare.py` reads only saved evaluations, requires all four models, checks
identical parent observations, means, identities, timestamps and bin support,
and emits A (granularity D=0), B (granularity D=6), C (subregion delay) and D
(all four; all six paired trial contrasts). NLL columns repeated across W
always describe **single-bin** NLL. It does not compute a Poisson likelihood
for averaged signals. Region-specific metrics cover every recorded parent,
including VISpm. Native activity and temporal metrics remain separate.

Plots include matched full-trial curves per configuration, per-region overlays
of all four predictions, temporal metrics against W, NLL/constant-baseline
comparisons, variance ratios, native subregion curves, neuron counts/activity,
region improvements and paired trial differences. Representative trials are
uniformly spaced in the same persisted order; W=1 raw and averaged traces are
explicitly labeled. W=1 overlays include the full eligible interval, while
metric summaries use the common endpoints required by Experiment 1.

## Artifacts and manual verification

Under `output/experiment/subregion-aggregation/`:

- `experiment-manifest.json`: resolved identities and exact regional references.
- `manifest.json`: immutable experiment settings/common support.
- `runs/<id>/resolved.json`, `preflight.json`, `mapping.json`,
  `reconstruction.json`, `manual-verification.json`, `mapping-diagnostics.png`.
- `runs/<id>/train-attempts/<uuid>/`: logs, effective source/config snapshots,
  original production checkpoints, resumable state, history and selection.
- `runs/<id>/evaluate-attempts/<uuid>/`: common `metrics.json`, `support.json`,
  `predictions/`, `native-metrics.json`, `native-support.json`,
  `native-predictions/`, `subregion-diagnostics.json`, provenance and plots.
- `comparison/compare-attempts/<uuid>/`: `comparison.json`, `matrix.csv`,
  `regions.csv`, `report.md`, plots and execution manifest.

Stage pointer JSON files and exclusive locks follow the target-delay lifecycle.
Every attempt preserves failures and outputs; completed payloads are hashed.
The final report lists the original checkpoint, resolved configuration,
training/evaluation attempt and provenance for each model. Large regional
checkpoints are referenced, never copied.

Before launching training, manually inspect the entire `mapping.json`, exact
reconstruction flags, common trial/bin support, resolved configuration and
`manual-verification.json` head dimensions. CPU preflight initializes heads but
does not optimize. No new test scripts or test infrastructure are used.

The implementation does not launch training by default. A scientific finding
about subregion prediction requires the two explicit 50-epoch runs and their
evaluations; preflight alone establishes representation compatibility.

Small groups VISpm1 (2 neurons), bsc (2) and fp (1), and sparse targets, remain
present as diagnostics. These observations motivate examining training-only
reliability/information in Experiment 3, not automatic exclusions here. Use
validation to guide exploratory configuration choice. Previously inspected
test observations cannot provide independent unbiased confirmation or derive
future filtering rules. Both delays remain candidates; one seed cannot establish
a reliable interaction. Better predictions do not establish purely visual
responses under IBL closed-loop visual/motor confounding.
