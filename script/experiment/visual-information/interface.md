# E1 experiment interface

## Entry point and inputs

`python script/experiment/visual-information/run.py [--output PATH] [--threads N]`
runs the complete diagnostic experiment. Run from the checkout root with the
project environment. `PATH` must be a fresh directory; `N` is positive, default
four. Failures propagate and leave an incomplete output directory available for
inspection; an existing directory is never overwritten or reused as a cache.

Frozen inputs are the single session
`0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63` and generations:

| Stage | Generation |
| --- | --- |
| Dataset | `01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c` |
| Alignment | `7e635484272fdd3e209ab442db97a391b6fc9352f31101fa9e08dc2a3145b229` |
| Features | `8605a05d5fbf67eb01d4155fb3d552e3cbae666509dfd9148747869718714cb7` |
| Replay | `4f1b31578f8e28ec1b926b1cab7d36e9f0124cd6eb31728274cb7b16c3ca8184` |

Their standard repository output paths are recorded in `data.py` and the output
manifest. The split is fixed at 218 train, 31 validation, 63 test trials.
With the user's explicit authorization, whole-generation public readers access
test payloads only to verify integrity. No test values return to experimental
views or enter labels, diagnostics, preprocessing, fitting, plots or decisions.

`FeatureArtifactReader` and `ReplayArtifactReader` are exhausted;
`load_alignment` and `load_dataset` complete against the expected identities.
`integrity.json` records completion without retaining test values. Selected
train/validation metadata and payload hashes are additionally checked directly,
including pixel hashes for representative categories and exact reconstruction
of every selected aligned feature from source associations. No production
reader contract is changed.

## Probe contract

Source and aligned stages are separate. A/B/C have widths 768/768/1536 and
share eligible anchors within each stage and target. Every difference uses
adjacent valid observations from the same original trial and a regular physical
1/60-second interval. Unknown values, gaps, event/visibility/rendering changes
and raw angular differences of at least 180 degrees are excluded. First anchors
have no difference and are omitted for all three variants.

Labels describe reconstructed horizontal position and displacement in degrees;
direction 0/1 means left/right; visible motion 0/1 means visible stationary/
changing translation. Contrast classes 0–4 correspond to
`[0, 0.0625, 0.125, 0.25, 1]`. Motion and spatial targets require a nonblank
stimulus and an on-screen center. Unchanged pixels with commanded movement and
true RGB128 blanks remain separate controls. Aligned labels use saved source
indices and weights only within qualified smooth intervals; direction reversals
and invisible constituent movements cannot become positive aligned motion.

Ridge uses alpha=1. Logistic regression uses C=1, lbfgs, tolerance=1e-4 and at
most 1000 iterations; multiclass fitting uses multinomial softmax. Train-only
weighted StandardScaler parameters are independently fitted per target/variant.
There is no L2 normalization of difference vectors, model selection or tuning.
Every contributing trial has equal total training weight. Classification
balances observed classes within each trial; constant-per-trial contrast cannot
also be globally balanced without breaking equal trial totals. All weights are
rescaled together to mean sample weight one.

Primary metrics use weights summing to one per validation trial. Pooled and
individual-trial metrics are supplementary. R² uses the weighted held-out target
mean as its denominator; baseline constants and majority classes use train only.
Null metrics and their causes are retained. Direction overlap diagnostics use
fixed 10-degree position/contrast strata with at least three trials per direction;
they qualify interpretation rather than selecting fitted examples.

## Outputs

| Artifact | Contract |
| --- | --- |
| `manifest.json` | Frozen generation IDs, paths, membership, representation provenance, source hashes, versions, fixed protocol and verification boundary. |
| `integrity.json` | Completed public-reader identities; test access is integrity-only. |
| `implementation/*.py` | Exact executed source snapshot. Restore these files at their original checkout paths to reproduce a historical run. |
| `anchors-{stage}.jsonl` | One qualified adjacent pair per row; original trial, split, anchor, session seconds, interval seconds, reconstructed targets, category, visibility and spatial qualification. |
| `support.json` | Per-target/stage/split counts, class/trial coverage including per-trial class counts, shared A/B/C anchor fingerprints, target exclusions and continuity exclusions. |
| `metrics.json` | Baselines and A/B/C metrics, weighted/pooled/per-trial results, strata, convergence warnings and unchanged-pixel false positives. |
| `paired-comparisons.json` | Descriptive A/B/C score differences with a common anchor fingerprint; no frame-IID significance claim. |
| `diagnostics.json` | Position/side/direction and trial-level associations, overlap support, stationary sequences, spatial limitations and interpolation dependence. |
| `predictions-{stage}-{target}.npz` | Validation arrays `trial`, `anchor`, `time`, `target`, `contrast`, `magnitude`, `A`, `B`, `C`, each `[observations]`. |
| `models/{stage}-{target}-{variant}.npz` | StandardScaler `mean`, `scale` `[width]`, model `coefficients`, `intercept`, and classification `classes` (empty for regression). |
| `plots/*.png`, `control-*.png` | Prediction/error diagnostics, row-normalized confusion matrices and representative saved RGB pair previews. Plot display thinning does not affect metrics. |
| `report.md` | Protocol, measured comparison, scientific interpretation and limits. |

Files are standalone research artifacts. Consumers can read JSON and NPZ with
pickle disabled; coefficient/scaler arrays reproduce linear inference without
loading serialized Python estimators. Unsupported/mismatched identities,
nonregular support, invalid payloads or absent sources fail explicitly.

This is experiment orchestration across existing public artifact schemas. It
adds no production component dependency and changes no production interface.
