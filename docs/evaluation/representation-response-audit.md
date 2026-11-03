# Scenario 11 visual representation, response timing, and regional aggregation audit

## Executive summary

The current CLIP input contains measurable stimulus-motion information. Small temporal changes and low effective dimensionality are confirmed, but missing neural-relevant visual information has not been demonstrated. All completely constant-feature trials in this analysis are zero-contrast trials; adjacent similarity alone is not an extraction defect.

A broad short-delay association is reproduced on validation: earlier CLIP change magnitude correlates with later regional counts most strongly around 1–3 bins. This is a descriptive association inside the existing H=12 context, not an identified physiological latency. Its interpretation is limited by autocorrelation, movement, reconstructed display timing, and the choice of scalar visual signal.

The regional sum is implemented correctly. Neurons differ substantially in firing rate and response profiles. Positive and negative **baseline deviations** can offset, but the available comparisons do not establish that summation destroys reproducible stimulus information or causes the model's weak performance. Aggregate temporal variance is actually greater than the sum of individual temporal variances because of positive covariance.

Some coarse condition averages are repeatable in training, especially right/high-contrast trials and longer diagnostic smoothing windows. Validation is sparse and imbalanced, and matching side/contrast does not match the complete visual trajectory. The predictable fraction of neural variance and a noise ceiling remain unidentified.

**Best-supported interpretation:** the model fits little within-trial structure in a slowly varying, strongly condition-dependent representation, while response reliability and movement/position confounds remain insufficiently resolved. H1 has direct support as a conditioning concern; H2 has support as a short, broad association; H3 has support for heterogeneity but not demonstrated harmful information loss. No single primary cause is established.

**Recommended next action:** perform a condition-and-movement-matched response-reliability diagnostic, including zero-contrast movement controls and separate onset support, before changing the model. This is a proposed follow-up, not an experiment run in this audit.

## Scope, identities, and evidence

This is a research audit spanning visual features, alignment, neural data, training-dataset, model, training, and evaluation. It is not a replacement for any component's implementation-gap `audit.md`. Authoritative specifications, architectures, dependency graph, existing task plans, runtime source, datasets, configurations, and checkpoints were not changed. Three focused scientific diagnostic scripts and this report are the only additions outside generated outputs. No training, predictive-model fitting, HPO, new inference, or new test-set analysis was performed.

| Item | Verified identity |
| --- | --- |
| Session | `0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63` |
| Region / recording | VISpm; `probe01`, PID `1f3d3fcb-f188-47a2-87e5-ac1db6cf393a`, `alf/probe01/pykilosort`, unrevisioned sorting |
| Scenario / checkpoint | `scenario-11/002`, `model_best.pt`, zero-based epoch 12; selected by validation loss |
| Checkpoint SHA-256 | `fc933d97758b169c44c366f0dc78bd754cc9091a67d8b12554a8125a55c472d7` |
| Training dataset | `01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c` |
| Alignment generation | `7e635484272fdd3e209ab442db97a391b6fc9352f31101fa9e08dc2a3145b229` |
| Neural generation | `2e491dc413d9785fa8d687c504a41e4093b286ae0e8a86ad4c88e4a96e1c5ae7` |
| Feature generation | `8605a05d5fbf67eb01d4155fb3d552e3cbae666509dfd9148747869718714cb7` |
| Replay generation | `4f1b31578f8e28ec1b926b1cab7d36e9f0124cd6eb31728274cb7b16c3ca8184` |
| Frozen prediction source | `output/audit/scenario-11/002/temporal-001/` |
| New audit outputs | `output/audit/scenario-11/002/representation-response-001/` |
| Historical descriptive comparison | Scenario 15/006 dataset `99e803760e5f3769a8a87755bbb9ea00005b2fcf2cb46d92c662225d8f61cc05` |

Exact train/validation memberships, sample IDs, ordered unit identities, original-to-selected column mapping, diagnostic parameters, and input hashes are recorded in the new JSON/CSV outputs. The 218 training and 31 validation trials match the frozen diagnostic records. Validation already selected this checkpoint; it is a separate confirmation split for these descriptive analyses, not an untouched final test.

The dataset stores 280 source neurons. Runtime anatomical selection retains 99 VISpm neurons, then sums them into one channel without altering persistent data. Source unit IDs, in order:

```text
147, 149–234 inclusive, 261–270 inclusive, 276, 279
```

This shorthand denotes exactly 99 identities, scoped by the session, insertion, sorting collection, and revision above. Full identity and quality rows are in `results.json:ordered_units`. Layer membership is VISpm5: 50, VISpm6a: 19, VISpm2/3: 17, VISpm4: 11, VISpm1: 2. No quality filter was applied. Supplied labels are 0 (20 units), 1/3 (27), 2/3 (36), and 1 (16). These are retained source metadata; this audit does not reinterpret them as biological relevance or claim a universal label definition.

### Verified implementation path

| Component | Inspected implementation and effective behavior |
| --- | --- |
| Visual feature extraction | `src/visual_features/encoder.py`, `observations.py`, `artifacts.py`: fixed CLIP projected pooled image features; ordered observations and timestamp provenance retained |
| Alignment | `src/alignment/inputs.py`, `temporal.py`, `artifacts.py`: complete observed neural bins; timestamp bracketing; float64 interpolation, L2 normalization, float32 storage; no extrapolation |
| Dataset preparation | `src/training_dataset/samples.py`, `artifacts.py`, `context.py`: counts, original identities, masks, physical timestamps, and trial boundaries retained; split assignment stored explicitly |
| Source counts / aggregation | `src/neural_data/counting.py`, `sources.py`; `src/trainer/selection.py`, `targets.py`: source `[start,end)` counts; anatomical descendants; int64 per-bin summation in source order |
| Model projection / context | `src/models/stitcher.py`, `src/multi_modal/encoder_embeddings.py`, `mm.py`: session-specific LayerNorm(768) then linear projection to 128; contextual embeddings; two transformer layers, four heads; isolated 12-token windows with final-token readout |
| Encoding / scoring | `src/evaluation/predictions.py`, `metrics.py`; `src/trainer/temporal_diagnostics.py`: visual-only encoding, log expected counts, explicit scientific/context/H=12 masks; Poisson objective and count-scale metrics |

Strict windows are isolated **before** the transformer stack. Thus two layers do not expand the receptive field to earlier aligned tokens. Learned absolute positions and RoPE remain present. Coordinate-wise input LayerNorm does not standardize each feature's temporal variance or add temporal information. The regional head is linear without output normalization. Small predicted amplitude is not mechanically imposed by a singleton LayerNorm in that head.

## Methods and support

| Analysis | Training support | Validation support |
| --- | ---: | ---: |
| Valid aligned bins, including onset context | 126,243 | 20,789 |
| Adjacent within-trial feature pairs | 126,025 | 20,758 |
| H=12 eligible neural / frozen prediction bins | 123,845 | 20,448 |
| Trials | 218 | 31 |
| Median duration | 2.433 s | 2.767 s |
| Duration range | 1.150–62.900 s | 1.283–60.633 s |

Feature statistics use all real aligned bins, including the first eleven context-only bins; they are explicitly source-representation analyses. Neural distribution, covariance, and prediction statistics use H=12 support. No padding is analyzed. Count sums were checked against every frozen train/validation observed target and matched exactly. Loaded current sample and frozen prediction payloads were checked against their recorded hashes; checkpoint hash also matched. Source feature generation identity and reconstructed interpolated feature values were checked directly.

For lag comparisons, response positions are fixed at `t=12..T-1` for every candidate `k=0..11`. The visual signal is `||V[t-k]-V[t-k-1]||`; the response is the regional count at `t`, or its first difference. Positive `k` means earlier visual signal, later neural activity. Each correlation centers its own trial-local pair. Detrended results additionally remove each sequence's linear time trend. No trials are concatenated to form a correlation.

`lag-comparison.json` is the canonical lag comparison: it keeps the **same finite-correlation trial cohort across all twelve lags**, 186 train / 25 validation. Raw exploratory rows in `results.json` retain variable per-lag contributor counts (186–191 train / 25 validation); they must not replace the common-cohort comparison. Five additional training trials have a constant scalar signal on at least one lag's support. Completely constant-feature trials are excluded from correlation denominators, not assigned zero correlation.

Confidence intervals for equal-trial effects resample whole trials, not bins: 1,000 bootstrap resamples in primary/supplemental summaries, 2,000 paired resamples in the common-cohort comparison; seed 20261027. They are pointwise descriptive intervals, without correction for inspecting multiple lags or conditions. Random split-half summaries use 200 whole-trial partitions; their distributions are **not** confidence intervals or independent biological repetitions. Undefined correlations are JSON null, with contributor counts retained.

Effective dimensionality uses every eighth bin of each trial-centered feature sequence, preserving trial-local centering, and the covariance participation ratio `(sum eigenvalues)^2/sum(eigenvalues^2)`. This is a descriptive subsample estimate, not a full-data rank or an estimate of neural-relevant dimension. Within/between variance decomposition and per-dimension variances use all valid feature bins, weighted by bin count.

## Quantitative results

### Visual representation and interpolation

| Measurement | Training | Validation |
| --- | ---: | ---: |
| Mean adjacent cosine, unit-vector distance identity | 0.9999354 | 0.9999484 |
| Mean adjacent Euclidean distance | 0.004222 | 0.002762 |
| Median adjacent distance | 0.001364 | 0 |
| 95th percentile adjacent distance | 0.015073 | 0.009885 |
| Maximum adjacent distance | 0.219106 | 0.192448 |
| Exactly unchanged adjacent vectors | 34.48% | 60.80% |
| Approximately unchanged, distance ≤ 1e-6 | 34.48% | 60.80% |
| Within-trial variance trace | 0.005681 | 0.005772 |
| Between-trial variance trace | 0.040816 | 0.041370 |
| Within / total feature variance | 12.22% | 12.24% |
| Within-trial participation ratio | 4.38 | 3.12 |
| Dimensions explaining 95% of sampled within-trial variance | 23 | 12 |
| Whole trials with constant features | 27 | 6 |
| Mean per-trial repeated-source-pixel fraction | 55.25% | 58.58% |
| Equal-trial source motion / CLIP-change correlation | 0.9013 (191 contributors) | 0.8807 (25 contributors) |

The cosine summary uses `1-||delta||²/2` for normalized vectors; float32 storage can introduce negligible norm-rounding error. Euclidean difference statistics are calculated directly in float64 from the current float32 prepared features. Per-dimension variances are exported separately, not inferred from the participation ratio.

The 27/6 completely constant trials coincide with the 27/6 zero-contrast trials. Source metadata identifies sustained stationary, closed-loop, and frozen stimulus states. For all analyzed consecutive source pairs with identical RGB hashes, none had a feature difference above 1e-6. Actual decoded representative source frames were inspected, and their RGB hashes matched observation metadata. They show a mostly gray field with a localized Gabor moving across the reconstructed display. This supports source persistence and blank trials as real contributors to high similarity. It does not certify historical display fidelity or quantify all information discarded by spatial downsampling/CLIP.

CLIP is `openai/clip-vit-large-patch14`, immutable revision `32bd64288804d66eefd0ccbe215aa642df71cc41`, projected pooled vision output, 768 coordinates. Extraction used CUDA float16 autocast, float32 weights/normalization/storage, SDPA, and exact-prepared-image reuse with capacity 512. The entire mouse-view image is fitted to 224×224 with aspect-preserving bicubic resizing and centered RGB(128,128,128) padding; processor resizing/cropping are disabled and CLIP channel normalization retained. All source observations were selected. This can reduce fine spatial detail, but this audit does not establish which neural-relevant details were lost.

All 249 analyzed trials use a later right-hand source frame, averaging **8.333 ms after the bin-center query**. With source cadence and neural grid both 60 Hz, these are approximately midpoint mixtures. Interpolation creates intermediate feature values and spreads a source transition across adjacent estimates; it does not add observed frames. An abrupt source change is already partly represented at the preceding bin-center query. L2 renormalization increases the equal-trial mean step size relative to unnormalized mixtures by only about 0.11% train / 0.12% validation; maximum pre-normalization norm deficit is 1.90% / 1.52%.

For all 218/31 matched historical train/validation trials, current and historical count arrays and physical timestamps match exactly. The complete ordered 280-unit tables also match, preserving the same selected 99-unit population and ordering. Historical held-state versus current interpolation equal-trial mean adjacent distances are 0.010706 → 0.006672 train and 0.009550 → 0.005915 validation. Historical equal-trial unchanged fractions are 56.95% / 61.86%. These values use equal trial weighting, unlike the bin-weighted distance table. Source feature-generation IDs in the compared manifests identify the same feature generation. This is a descriptive alignment comparison, not a controlled performance attribution: Scenario 15 also used full-trial attention.

| Lag (bins) | Trial-centered feature autocorrelation train / val | Neural count autocorrelation train / val |
| --- | --- | --- |
| 1 | 0.978 / 0.983 | 0.325 / 0.338 |
| 3 | 0.909 / 0.932 | 0.081 / 0.120 |
| 6 | 0.804 / 0.847 | 0.050 / 0.034 |
| 11 | 0.610 / 0.664 | 0.031 / 0.039 |
| 30 | 0.218 / 0.289 | 0.001 / 0.012 |
| 60 | -0.208 / -0.124 | 0.003 / 0.004 |

Feature autocorrelation is a normalized inner product over all coordinates after subtracting each trial's feature mean; scalar neural autocorrelation is a within-trial Pearson correlation on eligible support. Both are equal-trial summaries with available counts in JSON. Finite-trial centering can induce negative long-lag values; these are not evidence of oscillatory physiology. Longer trials tend to have smaller mean visual steps (duration correlation -0.398 / -0.540), making pooled and equal-trial statistics materially different.

### Response timing

| Delay bins / ms | Common-cohort equal-trial r, train | Common-cohort equal-trial r, val |
| --- | ---: | ---: |
| 0 / 0 | 0.1468 | 0.1755 |
| 1 / 16.7 | 0.1578 | 0.1911 |
| 2 / 33.3 | 0.1587 | 0.1930 |
| 3 / 50.0 | 0.1570 | 0.1901 |
| 4 / 66.7 | 0.1530 | 0.1691 |
| 6 / 100.0 | 0.1130 | 0.1304 |
| 9 / 150.0 | 0.0603 | 0.0721 |
| 11 / 183.3 | 0.0283 | 0.0391 |

At lag 2, train r=0.1587, 95% trial-bootstrap interval [0.1337, 0.1849]; validation r=0.1930 [0.1142, 0.2820]. The paired lag-2-minus-lag-0 changes are 0.0119 [0.0034, 0.0217] and 0.0175 [0.0023, 0.0342]. This modest short-delay preference reproduces, but does not isolate a unique lag: bootstrap maxima occupy lags 1–4 in training and mostly 1–3 in validation. Linear detrending retains lag-2 r=0.1405 / 0.1834. Count-difference relationships are much weaker: lag-2 r=-0.0012 / 0.0026, with maxima near zero delay.

A half-trial circular phase shift of the scalar visual sequence gives raw equal-trial count correlations near zero at these lags (training roughly -0.009 to 0.004; validation -0.029 to -0.001). It preserves trial identity and each shifted sequence's circular autocorrelation, but introduces a circular join and is only a reference for privileged temporal phase, not a calibrated permutation p-value or a causal null.

Using training duration-tertile cutoffs, lag-2 correlations are 0.254, 0.131, 0.072 in training and 0.376, 0.049, 0.143 in validation. The corresponding available contributor counts are 69/60/58 and 9/9/7. These exploratory strata retain per-lag availability rather than the common cohort. They show duration dependence and limited validation precision; they do not establish an invariant delay.

Per-neuron exploratory maxima span the whole 0–11 range; only 24 of 95 units with available maxima in both splits have the same maximizing lag. This is evidence of unstable/heterogeneous **descriptive maxima**, not proof of heterogeneous physiological delays. There are no per-unit latency confidence claims or neuron selections based on validation maxima.

The 12-bin token window is `[V[t-11],...,V[t]]`: 200 ms of bin exposure, 183.3 ms between first and last centers. H=12 model scoring starts at bin 11, center 191.7 ms after onset. The audit cannot identify early onset latency from model scores because the first eleven bins are input context, not scored targets. Source-only onset profiles retain them separately. Neural bins are `[stimOn+k/60, stimOn+(k+1)/60)`; centers are `stimOn+(k+0.5)/60`. Source observations are scheduled at reconstructed `stimOn+j/60`, not measured display-frame timestamps. Positive source-frame availability, target-bin exposure, and aligned-token causality must remain distinct.

### Neuron heterogeneity and aggregation

| H=12 measurement | Training | Validation |
| --- | ---: | ---: |
| Median source-unit rate | 1.307 Hz | 1.185 Hz |
| Minimum / maximum rate | 0.00048 / 13.901 Hz | 0 / 13.844 Hz |
| Units below 1 Hz | 43/99 | 46/99 |
| Units with no spikes on analyzed support | 0 | 4 |
| Highest-rate unit spike share | 6.01% | 5.75% |
| Ten highest-rate units' spike share | 42.71% | 43.46% |
| Mean pairwise trial-centered correlation | 0.0293 (4,851 pairs) | 0.0253 (4,465 pairs) |
| Variance of trial-centered regional counts | 15.319 | 14.471 |
| Sum of individual trial-centered count variances | 3.757 | 3.900 |
| Ten largest covariance contributions / regional variance | 37.14% | 40.46% |

Pairwise correlation excludes constant units; contributor counts refer to unit pairs, not biological repetitions. Per-neuron rate, zero fraction, variance, spike share, and covariance contribution are in `neurons.csv`. Covariance contribution is `mean((Y_i-trial_mean_i)*(Y_region-trial_mean_region))`; these contributions sum to regional centered variance. The ten highest-rate units account for 35.37% / 38.95% of this variance, rather than monopolizing temporal structure. Ranking each split separately is descriptive only; no target population was changed.

All nonnegative counts add. Cancellation concerns deviations from a baseline, never disappearance of spikes. For coarse training condition-average profiles, the norm of the summed temporal deviations divided by the sum of individual deviation norms ranges from 0.169 to 0.416. Mixed response directions are therefore present in these noisy profiles, but this ratio is **not** a fraction of predictive information lost: noise alone can produce opposing deviations.

For an event diagnostic, the first reconstructed azimuth change >1e-6 degrees was used, with six complete preceding eligible bins and twelve following bins. Most first movements occur too early for that support; only 8 train / 3 validation trials contribute. Relative to each unit's six-bin baseline, 46/21 training units and 24/21 validation units have positive/negative post-event mean deviations; remaining units have zero estimated change. Regional post-event change is +0.438 counts/bin [−0.615, 1.594] in training and −0.417 [−1.333, 0.833] in validation. Signed-to-absolute mean-change ratios are 0.228 / 0.152. Sparse, mixed-condition, movement-associated events cannot establish a reproducible aggregation loss or a preferred response latency.

Region-level prediction remains a justified measurement of total selected VISpm activity. It should not be interpreted as a complete population code, nor abandoned solely because individual profiles differ.

### Response consistency and temporal resolution

Available condition metadata are recorded side and contrast, recorded onset/offset, go cue, response and feedback, plus reconstructed position/stage. Closed-loop and freeze times use documented event proxies. Stimulus phase is synthetic and trial-specific; several geometry/rendering parameters are nominal assumptions. Wheel trajectories, outcomes, and trial lengths differ. Side/contrast matching therefore provides comparable coarse conditions, not repeated identical visual sequences.

The table below uses identical first-second H=12 support, positions 11–59, and complete trailing means. Counts remain expressed per original bin. Each entry is mean random whole-trial split-half correlation within training; negative values are retained. These are descriptive reliability checks of condition averages, without a noise-ceiling interpretation.

| Training condition | Trials | 16.67 ms | 50 ms | 100 ms | 200 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| left, 0 | 10 | -0.115 | -0.196 | -0.304 | -0.449 |
| left, 0.0625 | 22 | 0.021 | 0.134 | 0.134 | 0.131 |
| left, 0.125 | 23 | -0.081 | -0.153 | -0.199 | 0.008 |
| left, 0.25 | 31 | 0.176 | 0.318 | 0.422 | 0.552 |
| left, 1 | 28 | -0.098 | -0.166 | -0.215 | -0.060 |
| right, 0 | 17 | 0.050 | 0.107 | 0.164 | 0.179 |
| right, 0.0625 | 18 | -0.105 | -0.211 | -0.417 | -0.549 |
| right, 0.125 | 24 | 0.346 | 0.546 | 0.687 | 0.845 |
| right, 0.25 | 23 | 0.385 | 0.549 | 0.705 | 0.813 |
| right, 1 | 22 | 0.751 | 0.893 | 0.938 | 0.963 |

Validation condition counts, in the same order, are 3, 1, 4, 4, 2, 3, 4, 6, 3, 1. Conditions with fewer than two trials have unavailable split-half statistics. With two or three trials, 200 random partitions merely revisit a handful of divisions. The strong right/1 training profile has only one validation trial, preventing independent within-validation reliability estimation.

Train-versus-validation condition-average correlations at 16.67 → 200 ms include right/0.125: 0.141 → 0.766 (24/6 trials), right/0.25: 0.465 → 0.817 (23/3), and left/0.25: −0.043 → 0.323 (31/4). Other conditions remain weak or negative. Reduced temporal resolution and repeated overlapping windows increase apparent smoothness and reduce independent information; these correlations do not prove single-trial predictability. Individual-neuron split-half and cross-split profile correlations, complete profiles, and contributor counts are exported in `response-profiles.json`.

### Frozen model interpretation and position effects

| Frozen H=12 metric | Train | Validation |
| --- | ---: | ---: |
| Equal-trial within-trial Pearson | 0.11533 | 0.09134 |
| Trial-mean Pearson | 0.42313 | 0.35465 |
| Trial-centered predicted / observed variance | 0.001225 | 0.001157 |
| Trial-centered bin-weighted Pearson | 0.04720 | 0.00397 |
| Training-null BPS | 0.002986 | 0.001939 |
| Mean observed / predicted counts | 3.8547 / 4.0154 | 4.0147 / 4.0046 |
| Equal-trial Pearson after 200 ms trailing mean | 0.22019 | 0.20329 |
| Trial-centered bin-weighted Pearson after 200 ms | 0.06438 | -0.02454 |

These are existing frozen diagnostics, not new test results. The validation mean calibration is close to one, but this does not imply good temporal calibration. Compression occurs in training as well as validation; a generalization-only explanation is incomplete. Equal-trial smoothing gains coexist with near-zero/negative bin-weighted validation tracking, reflecting the influence of long trials and weighting.

New descriptive correlations on the same frozen arrays show duration versus predicted trial mean r=0.520 / 0.503 and duration versus observed trial mean r=0.340 / 0.473. Equal-trial prediction-versus-elapsed-time r averages −0.204 / −0.196; observed count-versus-time averages −0.023 / +0.018, with intervals spanning zero. Predictions therefore have a stronger consistent position trend than observations. Learned global positions can support such trends, but duration is not supplied as a model input, and averaging over longer support itself changes trial means. These observations do not demonstrate a positional shortcut or establish that visual information is ignored. A controlled position/content ablation is required.

Expected counts need not reproduce every stochastic spike-count peak or match raw observed variance. Nevertheless, weak training within-trial correlations and large variance compression justify investigating conditioning and repeatable response structure. They do not justify artificial amplitude amplification or changing the likelihood solely to make traces look more variable.

## Verified findings

### A01 — Small, low-dimensional visual dynamics coexist with represented movement

**Priority:** High  
**Confidence:** Confirmed for measurements; uncertain as primary performance cause  
**Category:** Data integrity  
**Affected components:** visual-features, alignment, model

**Location:** `src/visual_features/encoder.py`, `src/alignment/temporal.py`; current dataset; new `results.json`, `trials.csv`, feature-variance CSVs.

**Finding / evidence:** Within-trial variation is ~12.2% of total feature variance; participation ratio ~3–4; adjacent vectors are very similar. Movement/change correlations ~0.88–0.90, distinct source frames, and nonzero transitions show information is present. Constant whole-trial features coincide with blank trials.

**Expected:** Preserve source representations and alignment without asserting that cosine similarity alone diagnoses missing information.

**Scientific interpretation:** Weakly expressed temporal information is a plausible conditioning limitation. Delta-CLIP may expose existing changes; it adds no source information.

**Limitations:** No neural-relevant information comparison against pixels or an alternative encoder; rank and motion correlations are not sufficiency tests.

**Suggested disposition:** Keep. Consider a future controlled temporal-feature experiment after reliability characterization.

### A02 — Reconstructed timing and short-delay association do not identify physiological latency

**Priority:** High  
**Confidence:** Confirmed association and timing limitations; uncertain physiology  
**Category:** Data integrity  
**Affected components:** visual-replay, alignment, training-dataset, evaluation

**Location:** aligned resampling metadata, replay timing/event evidence; `lag-comparison.json`, `response-profiles.json`.

**Finding / evidence:** Common-cohort associations favor a broad 1–3-bin neighborhood, with modest improvements over zero lag. Right source frames are 8.333 ms later than queries. Count-difference effects are weak, first-event support is sparse, and source times are reconstructed.

**Expected:** Explicit physical-time interpretation, fixed lag support, trial boundaries, and original H=12 context.

**Scientific interpretation:** A short delayed association exists inside the allowed context; neither a single physiological lag nor insufficient context length is established.

**Limitations:** Visual change norm discards direction/content; stimulus motion and behavior covary; no measured per-frame display timing or reliable event-specific latency estimate.

**Suggested disposition:** Keep accepted alignment/context; do not shift the window or select a precise delay from the maximum.

### A03 — Heterogeneous units and baseline offsets do not establish harmful aggregation

**Priority:** Medium  
**Confidence:** Confirmed heterogeneity and exact sum; uncertain information loss  
**Category:** Data integrity  
**Affected components:** neural-data, training, evaluation

**Location:** `src/trainer/selection.py`, `targets.py`; stored units/counts; `neurons.csv`, condition/movement profiles.

**Finding / evidence:** Sum matches frozen targets on every authorized bin. Ten units contribute ~43% of spikes, maximum unit ~6%; positive covariance increases aggregate centered variance. Mixed baseline-relative response directions occur, but sparse or unstable cross-split profiles do not establish reproducible cancellation.

**Expected:** Count-preserving aggregation over the exact selected population; distinguish spike sums from baseline-deviation cancellation.

**Scientific interpretation:** The sum omits population identity by design; demonstrated loss of useful repeatable structure remains unresolved.

**Limitations:** Unequal firing rates, sparse counts, small condition groups, and unstable per-neuron maxima preclude ranking neurons by reliable selectivity here.

**Suggested disposition:** Keep regional target and preserved per-neuron source data.

### A04 — Coarse response repeatability is condition-dependent; predictable variance is unidentified

**Priority:** High  
**Confidence:** Confirmed evidence limitation  
**Category:** Data integrity  
**Affected components:** visual-replay, neural-data, evaluation

**Location:** replay parameter/event provenance; coarse-condition profiles and split-half distributions.

**Finding / evidence:** Some right-condition training averages repeat, others do not. Validation has 1–6 trials/condition, exact trajectories are unmatched, and reconstruction contains synthetic phase and nominal geometry/task assumptions.

**Expected:** Repetition claims must match actual recorded/reconstructed conditions; no invented noise ceiling.

**Scientific interpretation:** Response reliability is a material confound when interpreting amplitude compression and representation adequacy.

**Limitations:** No exact repeated movie, no complete condition/movement matching, no pre-stimulus baseline in the prepared aligned trials. This is not proof that the neural response is intrinsically unpredictable.

**Suggested disposition:** Keep existing observations; prioritize a focused matched-state reliability follow-up.

### A05 — Weak temporal fit and position-related predictions coexist

**Priority:** Medium  
**Confidence:** Confirmed descriptive behavior; uncertain shortcut mechanism  
**Category:** Runtime  
**Affected components:** model, training, evaluation

**Location:** frozen `temporal-001/results.json` and prediction payloads; new `trials.csv`.

**Finding / evidence:** Training temporal fit is weak and compressed; trial-mean association exceeds within-trial association. Predicted elapsed-time trends and duration/trial-mean relationships reproduce across splits.

**Expected:** Interpret expected-count metrics on fixed eligible support and distinguish within-trial, between-trial, pooled, and equal-trial effects.

**Scientific interpretation:** Position/slow-statistic dependence is plausible, but visual dynamics and elapsed position are confounded.

**Limitations:** No controlled content/position intervention was run; future trial duration is not directly available to this strict model.

**Suggested disposition:** Keep architecture; isolate content and position in a future controlled comparison.

### Conforming areas

- Correct count summation, unit identity, physical grid, trial membership, and valid-zero handling on examined support.
- Current interpolated values reproduce from the recorded bracketing observations.
- Fixed H=12 target support and isolated strict contexts are explicit; no two-layer receptive-field expansion.
- Historical source artifacts are preserved and Scenario 15/11 performance is not treated as a single-variable ablation.
- No new implementation defect requiring a model, alignment, loss, or aggregation change was confirmed by this audit.

## Hypothesis assessment

| Hypothesis | Supporting evidence | Contradicting / limiting evidence | Confidence and justified follow-up |
| --- | --- | --- | --- |
| H1: representation limitation | Small temporal variation, low effective dimension, strong persistence, spatial reduction to 224 square | Actual movement is represented; blank/sustained states explain many repeats; missing neural-relevant information unproven | Moderate for conditioning concern; low for information absence. Reliability first, then delta-feature control if justified |
| H2: temporal mismatch | Reproduced modest short-delay preference; survives linear detrending | Broad maxima, weak count-difference relation, timing/behavior confounds; all candidate preferred lags fit inside H=12 | Moderate for delayed association; low for physiological delay or causal model limitation. Preserve H=12 |
| H3: regional aggregation limitation | Heterogeneous rates, directions and exploratory maxima; averaged baseline deviations can offset | Exact sum, positive shared covariance, no demonstrated reproducible information loss; regional profiles can repeat | High for heterogeneity; low for harmful aggregation. Compare reliability on matched states before changing target |
| Reliability / position confounds | Strong condition imbalance, sparse validation, differing coarse-profile consistency; prediction position trends | Some condition profiles reproduce, and visual changes associate with counts | Material unresolved confounds; no universal noise ceiling or shortcut conclusion |

## Ranked future controlled experiments

These are proposals only. All model experiments retain the current alignment, 60 Hz grid, original split/unit population, Poisson objective, checkpoint-selection policy, and exact H=12 input boundary unless a future explicit scientific decision changes them. Validation cannot become a search over many unreported alternatives.

| Rank | Hypothesis and single primary change | Components / scientific benefit | Required train/validation measurements | Confounds and decision criteria |
| --- | --- | --- | --- | --- |
| 1 | Response reliability differs after **matching reconstructed movement/state within recorded side/contrast**; change only diagnostic grouping | visual-replay metadata, source neural counts, evaluation diagnostics; determine which regional and unit response profiles actually repeat | Contributor counts; whole-trial split-half distributions; train/val profiles; 1/3/6/12-bin windows fixed in advance; individual and summed baseline deviations; zero-contrast movement controls; source-only onset separately | Exact pixels/phase and behavior still vary; sparse groups must remain unavailable. Accept useful support only if a predefined profile/direction reproduces across independent trial groups with uncertainty excluding a zero effect; otherwise report insufficient evidence, not a noise ceiling |
| 2 | Explicit differences improve conditioning; add **within-window visual differences** as the sole input change | model input preparation/projection and training; test accessibility of existing temporal information | Same-support BPS, equal-trial and trial-centered pooled Pearson, calibration, variance ratios, condition/duration strata; predeclared seeds and fixed training budget | Extra parameters and scaling can confound. Derive differences only between tokens already inside `[t-11,t]`; the first delta must be masked or zero, never use `V[t-12]`. Accept only reproducible validation improvement in BPS plus within-trial tracking, without reliance on variance amplification; reject or defer if gains are unstable |
| 3 | Prediction uses position beyond visual content; **remove learned additive absolute position** while retaining configured rotary ordering | model/training; isolate one positional mechanism | Same budget/support, BPS, within-trial metrics, elapsed-time trends, constant-feature/condition strata | Changes optimization and positional capacity; RoPE remains. Accept shortcut concern only if reduced position dependence preserves or improves validation visual-response tracking. Loss of performance alone does not prove a shortcut |
| 4 | Summation obscures reliable unit substructure; **change the descriptive target summary to a fixed training-defined anatomical layer partition** | neural metadata, training target transformation, evaluation; only after rank-1 evidence supports reproducible opposing profiles | Identical unit set/bin support; reliability by layer and total; summed expected counts vs original total; condition-specific and pooled BPS with explicit denominators | More heads/targets and loss weighting confound neural-model comparison. No neuron-level retraining is implied. Accept only if cross-split reliable substructure and total-count prediction improve under a predeclared fair comparison; reject if merely raw variance or training fit increases |

No precise-delay reweighting experiment is prioritized: the current descriptive lag preference is broad and already inside the accepted window. No alternative likelihood, larger context, or capacity increase is justified as the immediate next action by these measurements.

## Missing evidence and recommended next action

Missing evidence includes measured per-frame display availability, exact repeated stimulus trajectories with known phase, historically validated renderer parameters, adequate validation repetitions for strong training conditions, pre-stimulus counts in the current aligned sample, condition-and-movement-matched unit response consistency, and a controlled position/content intervention. Available source spikes/counts could support some further diagnostics without rebuilding targets; missing historical display evidence may remain irrecoverable.

Run **one condition-and-movement-matched reliability diagnostic** next, using training to define grouping and validation to check the same contrasts/support. Compare original units and their exact regional sum, including blank movement controls. Keep source-only early-onset bins separate from model-scored bins. This directly addresses whether a reproducible target exists and whether summation obscures it, while controlling the strongest observed visual-motion confound.

This has greater immediate diagnostic value than increasing capacity (weak reliable signal has not been established), changing Poisson NLL (no likelihood implementation defect found), amplifying predictions (expected counts need not match stochastic variance), expanding context (short associations already lie inside H=12), or restoring neuron-level training (heterogeneity is not evidence of useful recoverable predictive information).

## Diagnostic figures and reproducibility

Figures were inspected as rendered images. Duration-quartile examples are training trials 55, 213, and 170, chosen by duration rather than correlation. Representative-trial curves use bin-start display coordinates `i/60` seconds; stored physical bin centers are 8.333 ms later. The movement plot uses same-index source-event offsets; its heatmap x-axis is offset-array index 0–17, corresponding to bins −6..11. The image montage uses source frames at the beginning, midpoint, and near the end; original RGB hashes were checked. Overview lag curves show raw available-trial summaries; use `lag-comparison.json` for fixed-cohort inference. Heatmaps show descriptive deviations, not significant/selective neurons.

![Aggregate CLIP dynamics, lag relationships, neuron contributions, duration and position effects](../../output/audit/scenario-11/002/representation-response-001/overview.png)

![Representative trials selected by duration quartiles](../../output/audit/scenario-11/002/representation-response-001/representative-trials.png)

![Decoded original reconstructed source frames](../../output/audit/scenario-11/002/representation-response-001/source-frames.png)

![First reconstructed movement response; sparse contributor support](../../output/audit/scenario-11/002/representation-response-001/movement-responses.png)

![A condition with at least six repetitions in each split; per-neuron deviations on the same support](../../output/audit/scenario-11/002/representation-response-001/condition-profiles.png)

From the checkout, run the three scripts in order using the project Python:

```powershell
.venv/Scripts/python.exe script/audit/audit_representation_response.py
.venv/Scripts/python.exe script/audit/audit_response_profiles.py
.venv/Scripts/python.exe script/audit/audit_lag_comparison.py
```

They refuse existing destination outputs. To reproduce, select a new output directory by changing the primary script's `OUT` constant in a copied script set; the supplements import that constant. No acquisition/network call, model restoration, optimizer, predictive fit, target smoothing in the pipeline, or historical overwrite occurs. Decoding the compressed source feature array necessarily loads its shared storage; statistics index only the authorized train/validation observations. Historical sample payloads are opened only for matched train/validation trials. No test prediction or test-target statistic is generated.

Artifacts: `results.json` (provenance, representation/neural statistics, raw lag curves); `lag-comparison.json` (canonical fixed-cohort lag estimates); `response-profiles.json` (condition, movement, per-unit timing and reliability); `trials.csv`; `neurons.csv`; `lags-by-trial.csv`; `historical-alignment.csv`; train/validation feature-variance CSVs; five PNG figures. Primary and supplemental per-condition random partitions use separately initialized deterministic streams and can differ slightly; the reliability table here consistently uses `response-profiles.json`.
