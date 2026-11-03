# Neural activity peak prediction and temporal dynamics audit

## A. Executive summary

The saved `scenario-15-006` evaluation confirms that this model captures trial means much better than within-trial temporal variation. Its expected-count variance is only **0.158%** of observed-count variance. This is a measured symptom, not proof that all missing observed variance should be reproduced: a conditional expectation should exclude unpredictable counting variability.

Two confirmed contract discrepancies precede interpretation of latency or loss experiments:

1. **The reference checkpoint is noncausal.** Scenario 15 uses `full_trial`, with unrestricted attention to past and future visual bins. H=12 is its scoring-support rule, not its receptive field. The implementation supports genuinely isolated strict windows, but this run did not use them.
2. **The reference dataset does not contain the accepted bin-center interpolated visual features.** It records timestamp-aware held states over neural intervals. Alignment implements that policy, and training/evaluation explicitly reject other policies. This conflicts with the user's accepted contract and `docs/alignment/spec.md` sections 11–14 and 27. No policy was changed in this audit.

The strongest candidate explanations for weak temporal learning are the weakly varying global visual representation, a limited reproducible visual signal relative to observed variability, and generalization limits. The history supports **later overfitting**, rather than a claim that optimization never learns: training loss improves while validation deteriorates after the selected epoch. Missing training prediction trajectories prevent deciding whether variance compression already occurs on training observations. Neither a physiological latency defect nor Poisson loss being the root cause is established.

**Recommended next action:** a frozen-checkpoint train/validation trajectory diagnostic, with no optimization, no test-based tuning, and explicit reporting of the historical full-trial/held-state semantics. This resolves the principal missing evidence before changing features, model capacity, or likelihood. Contract discrepancies must be resolved before a new experiment is claimed to satisfy the accepted strict/interpolated protocol.

## Evidence scope and provenance

This is a cross-component audit requested by the user, covering visual-features, alignment, training-dataset, model, training, and evaluation. It follows repository priority/confidence/category conventions; the additional hypotheses and experiments are requested research-report sections, not additions to component task lists. Existing component audit snapshots, specifications, architecture, interfaces, source, configuration, checkpoints, datasets, and evaluation artifacts are preserved.

Primary artifacts:

- `output/runs/scenario-15/006/{config.json,history.json,selection.json,run.json,provenance.json,model_best.pt}`.
- `output/evaluation/scenario-15/006/{manifest.json,result.json,psth.json,coverage.json,predictions/*.npz}`.
- Dataset generation `99e803760e5f3769a8a87755bbb9ea00005b2fcf2cb46d92c662225d8f61cc05` under `output/datasets/training-dataset/generations/`.

The selected checkpoint is epoch **10**, SHA-256 `1a076caa44157ca13d4056dd405ef84f287aa267c830b62092e956bc25044ddd`, selected by validation loss. The evaluation checkpoint hash and all **67** declared payload hashes match. All five evaluation source hashes recorded in the manifest match the inspected current files. The artifact records a dirty source checkout; this audit does not infer exact historical training source identity from a Git revision alone. Current code behavior is identified separately from behavior directly demonstrated by saved arrays/configuration.

This run contains one session, `0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63`, and one regional channel, **VISpm**, summing **99** recorded units. Its split is within-session: 218 train, 31 validation, 63 test trials. This is not evidence of held-out-session generalization or performance across multiple regions.

Read-only analysis used existing arrays and inline Python, without creating test scripts or running training/inference. Test-array summaries below describe the already selected reference result; they were not used to select lags, feature scales, smoothing widths, checkpoints, or model hyperparameters. The elevated-activity threshold was calculated from training counts only.

## B. Verified findings

### A01 — Reference run violates the requested strict temporal restriction

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity  
**Affected:** Training, validation, evaluation  
**Location:** `src/configs/scenarios.json`, scenario 15; `src/training_dataset/context.py:30`; `src/multi_modal/mm.py:285,341`; saved run `config.json` and evaluation `manifest.json`.

**Finding:** The run records `mode=full_trial`, `bins=null`, `target_support_bins=12`; model context forward/backward limits are both `-1`. In full-trial mode `_context_forward` calls `forward_encoder` on the whole valid trial. Its attention mask excludes padding keys but imposes no causal restriction. Future visual information can influence a prediction. This is explicit historical configuration, not a train/eval mode mismatch. It prevents interpreting this reference as strict 200 ms causal encoding; it does **not** establish that future access causes its low variance.

**Expected:** Under the requested strict encoding contract, output at t can use only `[t-11, ..., t]`. The fixed H=12 target mask excludes the first eleven bins; it does not restrict attention by itself.

**Suggested disposition:** Keep the historical result labeled noncausal; use the existing strict execution path for any subsequent strict experiment. Do not reinterpret or overwrite this checkpoint.

The strict path gathers isolated per-target windows before all transformer layers and reads their final encoding token. Multilayer attention therefore cannot expand beyond the gathered window. Bidirectional mixing *inside* that past-only window does not expose anything later than the target. Training and evaluation consume the same dataset-owned context semantics. Twelve bins cover 200 ms of exposure; the oldest and newest bin centers are separated by eleven bins, approximately 183.3 ms. Delayed effects within that history are representable without shifting the dataset. Longer delays or excluded pre-onset history are not available under that restriction.

### A02 — Held-state alignment replaces the accepted interpolation contract

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec  
**Affected:** Prepared inputs consumed by training and evaluation  
**Location:** `src/alignment/temporal.py:14,65,134,148`; `src/trainer/setup.py:301`; `src/evaluation/setup.py:151`; `docs/alignment/spec.md` sections 11–14, 27; `docs/alignment/interface.md`, input requirements.

**Finding:** `_visual` selects the visual update active at the neural **interval start**, requires its support to cover the whole interval, and copies its feature. Left/right source indices are identical and interpolation weights are zero. All 63 saved test samples record `timestamp-aware held-state over neural intervals v1`. Training and evaluation require this policy string and reject interpolated generations. The interface describes held states while the authoritative specification requires normalized linear interpolation at bin centers. Equal nominal 60 Hz cadence does not resolve that discrepancy.

**Expected:** Preserve the accepted bin-center interpolated estimate and its physical timestamp semantics. The audit must not silently replace that policy with held states or legitimize it through interface text.

**Impact:** Different visual values and transition timing reach the model; this confounds interpretation of temporal errors and comparison with the accepted design. An exact performance penalty or physiological offset cannot be inferred from the discrepancy alone. No generic one-bin spike-count indexing error was found.

**Suggested disposition:** Refactor the policy enforcement and alignment together in a separately accepted implementation task; retain existing generations for provenance. No implementation or regeneration is performed here.

Interpolation between adjacent frame observations can itself involve a source timestamp later than the query center. The user explicitly accepts that reconstructed estimate. Strict *token-index* causality and strict availability of every underlying source timestamp are different claims: report interpolation provenance and do not claim measured real-time availability. The discrepancy must not be “resolved” by silently changing either the accepted representation or context horizon.

### A03 — Severe temporal variance compression is measured on the selected test result

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline  
**Affected:** Evaluation demonstrates the prediction-quality limitation; its training origin remains unverified  
**Location:** Saved evaluation `predictions/*.npz`, `result.json`, `psth.json`; `src/evaluation/metrics.py:57,73,105`.

**Finding:** Recalculation on the 33,773 H=12 eligible bins exactly reproduces Trial R² **0.002402048592321515** and BPS **0.001677173581419456** (roundoff in the final digits). Saved PSTH R² is **-0.006871869255576701**.

| Diagnostic, expected counts per 1/60 s bin | Observed | Predicted |
| --- | ---: | ---: |
| Mean | 3.926421 | 3.998224 |
| Pooled variance, population denominator | 14.569335 | 0.023048 |
| Minimum–maximum | 0–58 | 3.406031–4.580558 |
| Median | 3 | 3.999514 |
| 95th percentile | 11 | 4.242345 |
| Bin-weighted variance after centering each trial | 14.215950 | 0.011430 |

Pooled Pearson correlation is **0.05453**; equal-trial mean correlation is **0.06729**, median **0.02368**. Correlation between the 63 observed and predicted trial means is **0.59763**. The pooled variance ratio is **0.00158193**; the ratio of bin-weighted within-trial variances is approximately **0.000804**. Thus the reference captures between-trial variation substantially better than rapid fluctuations. Pooled “Trial R²” is not an average of within-trial temporal scores.

The training 95th-percentile count threshold is **11**. On test bins, 1,705 observations meet `y >= 11`; no prediction reaches 11, giving **0 elevated-bin recall** and mean signed amplitude error **-10.96683 counts** on those bins. This is a training-threshold elevated-activity diagnostic, not top-k ranking recall or proof that every such count is visually predictable. Integer ties mean the selected fraction need not be exactly 5%.

**Expected:** Better stimulus-dependent temporal prediction, without forcing the variance of an expected count to match noisy realized counts.

**Suggested disposition:** Keep the official scores; add separately labeled diagnostics in future work. Diagnose train/validation behavior before changing amplitudes or objectives.

### A04 — Existing histories cannot resolve training-time variance; later overfitting is supported

**Priority:** Medium  
**Confidence:** Confirmed for missing trajectories and loss divergence; Likely for overfitting interpretation  
**Category:** Pipeline  
**Affected:** Training/validation diagnosis and checkpoint interpretation  
**Location:** `src/trainer/base.py:225,308`; `src/trainer/artifacts.py:236`; saved `history.json`, `selection.json`; `script/audit/region_diagnostics.py:36`.

| Zero-based epoch | Training Poisson objective | Validation Poisson objective | Validation BPS |
| --- | ---: | ---: | ---: |
| 0 | -1.338558 | -1.555439 | -0.002454 |
| 10, selected | -1.350586 | -1.565851 | 0.001288 |
| 20 | -1.358897 | -1.559471 | -0.001005 |
| 60 | -1.415305 | -1.535802 | -0.009511 |
| 99 | -1.480274 | -1.490269 | -0.025873 |

**Finding:** All 100 epochs report 55 optimizer updates and a constant learning rate of 0.0002, consistent with recorded `scheduler=none`; no nonfinite abort is evidenced. Training improves while validation worsens later. Comparing absolute train/validation loss directly is misleading because target distributions differ, factorial terms are omitted, and training includes dropout plus changing weights within an epoch. The selected early checkpoint prevents attributing final-epoch degradation to the reported test checkpoint.

Training persists scalar losses/counts, not train trajectories, moments, dynamic ranges, or correlations. Validation predictions are assembled transiently but are absent from this attempt's saved files; plotting was disabled. Existing `region_diagnostics.py` predicts **validation only** and requires a compatible neuron source checkpoint matching regional initialization lineage. Attempt 006's `run.json` records an ordinary fresh training run without that source initialization; the diagnostic cannot be assumed to apply unchanged.

**Expected:** Sufficient split-specific evidence to distinguish a near-constant learned mapping from a generalization failure.

**Suggested disposition:** Wrap/centralize a frozen split diagnostic around existing checkpoint restoration and keyword encoding inference. Required collection procedure is in section E. Do not infer train variance from an epoch-average scalar loss. Missing gradient norms prevent asserting that all optimization was stable; finite progress only excludes obvious recorded numerical collapse.

### A05 — Official evaluation does not isolate temporal peaks or reproducible responses

**Priority:** Medium  
**Confidence:** Confirmed  
**Category:** Pipeline  
**Affected:** Evaluation interpretation  
**Location:** `src/evaluation/metrics.py:14,20,105`; saved `psth.json`; `src/trainer/base.py:368`.

**Finding:** Official metrics are pooled Trial R², all-trials PSTH R² and BPS. None directly measures within-trial correlation, peak recall/amplitude, variance compression, or lag. The PSTH combines different trials without stimulus-condition grouping. Its 3,809 scored positions have between 1 and 63 contributors: **1,820 positions have fewer than five**, and 179 have one. Late positions are a duration-selected subset, not a repeat-averaged response with uniform reliability. This is a limitation of interpretation, not an arithmetic error.

Regional validation BPS uses the training constant (3.854673 counts/bin); official test BPS uses the test mean (3.926421). These explicitly different baselines prevent treating the two BPS values as identical measures of generalization. Preserve both official policies and label auxiliary training-baseline comparisons.

**Expected:** Preserve official scoring while supplying separate temporal diagnostics with clear support, aggregation and baseline definitions.

**Suggested disposition:** Wrap/centralize supplemental diagnostics; retain the official metric contract.

## Conforming areas and objective/sampling assessment

**Physical timing and counts.** For every saved test sample, real bin widths match 1/60 s within 1e-10 s, physical timestamps are bin centers, the first bin begins at stimulus onset, the final edge is within stimulus offset, and the encoding target mask equals real positions >=11. `alignment.inputs._window` validates the onset-anchored source edges and selects complete observed bins, dropping partial terminal exposure. Trial-local context construction does not cross trials. These checks support stored-coordinate consistency; reconstructed replay timing is not measured display timing (`verified_display_bin_alignment=false` throughout).

**Poisson semantics.** `trainer.objective.forward_objective` uses `F.poisson_nll_loss(z, y, log_input=True, reduction="none")`, with default `full=False`, then sums selected terms. Its parameter-dependent term is `exp(z) - y*z`; gradients with respect to z are `exp(z)-y`. Omitting the target-only factorial allows negative objectives and does not suppress peaks or change their gradients. The [PyTorch PoissonNLLLoss definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.PoissonNLLLoss.html) confirms this parameterization; repository code and recorded runtime settings determine the active behavior.

`trainer.targets.transform_neural_targets` sums original int64 counts over disjoint recorded-unit groups, without averaging by population size, dividing by bin duration, clipping, smoothing or z-scoring. The regional `RegionDecoder` is a linear log-count head without output LayerNorm. Conversion to Hz or per-source-neuron rates belongs only to labeled presentation. In particular, the legacy single-output LayerNorm concern does not explain this regional run. Sums preserve count semantics; an independent-Poisson interpretation would additionally require independence, which is not established for these units.

**Feature and latent scaling.** Fixed-weight CLIP extraction uses image preprocessing, inference mode, projected pooled image features, float32 L2 normalization and float32 storage; this artifact records `openai/clip-vit-large-patch14`, width 768, CUDA float16 autocast before float32 normalization. Identical prepared RGB images may reuse exact cached embeddings. No temporal encoder operates inside frame-level CLIP extraction. Session-specific input projection uses feature LayerNorm then Linear; learned position/session/modality context is added. Both model forward paths normalize the temporal latent with `F.normalize` before the neural head. This removes latent radius, but head weights/bias remain trainable and outputs are not clipped. Whether that constraint suppresses useful temporal amplitude is a hypothesis, not an established defect.

**Weighting.** Loaders shuffle whole trials in training, retain all samples (`drop_last=False`), and do not oversample peaks. The run has batch size one with four-trial accumulation. Each update divides its sum by the number of eligible cells in that accumulation window; within a window, longer trials contribute more cells. Update windows themselves receive one optimizer step each, so this is not exactly a single global bin-weighted objective under variable window lengths. Epoch reporting pools all valid-cell numerators and denominators. Rare high counts are not removed, but peak-rich short trials have no special sampling advantage. High-count residuals can produce larger gradients; Poisson does not explicitly downweight peaks to zero.

For this single-channel run there is no inter-region weighting imbalance. More generally, valid cells across regions are equally counted, not normalized by each region's source-unit count; larger/high-rate regional residuals can have larger absolute influence. Session weighting is also determined by trial sampling and eligible-cell counts, not an equal-session objective.

Training eligible trial lengths range from 58 to 3,763 bins (median 135); the longest contributes 3.04% of training bins. Validation lengths range 66–3,627 (median 155); one trial supplies **17.74%** of validation bins, materially affecting loss-based selection. These are weighting properties and sensitivity risks, not reasons to change the fixed split.

**Inference paths.** Training and validation use the modality-dictionary forward and explicit valid-cell objective. The trainer duplicates singleton-session samples without supervision to retain compatibility, then slices original predictions. Current session dispatch explicitly handles singletons, so the comment explaining the historical workaround is stale; no resulting train/eval scientific mismatch is demonstrated. Test inference calls keyword visual-only encoding with no neural targets, under `eval()` and `inference_mode()`. Both paths use the same projections, context execution, latent normalization and head. Encoding leaves visual input uncorrupted (`enc_task_var=all`), despite legacy masker parameters remaining in the saved configuration. The configured 30% masker ratio therefore is not evidence that this run randomly discards 30% of visual input. Dropout is active only during training, as expected. No unrelated legacy cleanup is required by this audit.

## C. Unverified hypotheses

### H1 — Visual input changes are weak or omit neural-relevant transitions

On real adjacent **training** bins, mean cosine similarity is **0.99983156**, median **0.99999810**; median difference norm is **0.00194900**, 95th percentile **0.02304125**, and **38.53%** have difference norm <1e-7. Norms remain within approximately 1.4e-7 of one. Validation has **64.13%** effectively identical pairs. These include genuine sustained scenes and exact image-cache reuse; they do not establish erroneous duplicate extraction. Test adjacency statistics are similar (mean cosine 0.99981518; 43.15% effectively identical).

The [CLIP paper](https://proceedings.mlr.press/v139/radford21a) describes image/text representation learning, not supervision for millisecond neural dynamics. It is therefore plausible, but not demonstrated, that pooled CLIP features miss subtle visual transitions. Needed evidence: train/validation transition-stratified response reliability, corresponding replay/feature trajectories, and controlled frozen-feature temporal baselines. Similarity alone is insufficient, particularly because the model's feature LayerNorm changes sensitivity.

### H2 — A consistent neural response delay is insufficiently learned

No reliable physiological lag estimate follows from the saved reference. Full-trial predictions can use future tokens; pooled neural peaks mix stimulus response and counting/state variability. Norm(delta_CLIP) is not a validated neural response regressor. Physical grid checks do not measure biological latency or actual display synchronization.

Needed evidence: train/validation-only trial-local lag curves, stratified by reproducible visual transitions or matched conditions, with trial-centered counts/predictions and common support. Positive lag must explicitly mean earlier visual information predicting later neural counts. Report trial-level uncertainty and stimulus persistence/autocorrelation; a broad correlation maximum is not proof of an indexing defect.

### H3 — Compressed conditional means versus a generalization failure

Near-mean training predictions remain unmeasured. Later loss divergence supports overfitting, but does not settle whether epoch-10 training has useful within-trial variance. Required evidence is frozen train/validation prediction moments, correlations and losses from the same checkpoint and deterministic inference path. Optional final-checkpoint train/validation comparison may explain overfitting, without reselecting a checkpoint on test performance.

### H4 — Observed peaks contain substantial unpredictable variability

Training marginal mean/variance are **3.854673/15.594464** (Fano **4.04560**); validation **4.014671/14.814045** (Fano **3.68998**); test Fano is **3.71059**. Marginal overdispersion can result from varying stimulus-dependent means, correlated units, unobserved state and true conditional overdispersion. It does not identify their proportions or justify Negative Binomial loss by itself. Required evidence: matched-condition/repeated-response reliability where available and conditional residual dispersion assessed on validation after training-only fitting. No universal noise ceiling can be estimated from these summaries.

### H5 — Regional aggregation cancels selective temporal responses

Summing 99 VISpm units preserves counts but could cancel opposing stimulus-dependent deviations or mix neurons with different delays. It can also improve signal-to-noise. Required evidence: retained source-unit responses grouped by the same region on train/validation, comparing reproducible components and delays, while keeping region-level model targets. Do not restore neuron-level targets merely to answer this hypothesis.

### H6 — Latent normalization, position reliance or representation capacity limits dynamics

Unit-normalized 128-wide latents, learned absolute position embeddings plus RoPE, two transformer layers and 0.1 dropout are confirmed configuration. They may favor position/mean shortcuts or limit useful magnitude coding; none is confirmed causal. Required evidence: the split diagnostic first, then one-variable validation ablations if necessary. No evidence currently justifies a larger model or amplitude scaling.

## Diagnostic definitions and interpretation

Use expected counts `p=exp(z)`, original counts y and official H=12 eligible masks intersected with real-region validity. Never calculate across padding or join trial boundaries. Produce per `(session, region, trial)` records first, then equal-trial means per session/region, equal-region session summaries, and equal-session summaries. Record unavailable/constant cases and contributor counts. Bin-weighted pooled versions are useful secondary summaries, explicitly named; do not silently replace the official aggregation. Bootstrap whole trials within sessions for uncertainty, retaining the limitation of one available session.

| Supplemental metric | Definition and interpretation |
| --- | --- |
| Within-trial Pearson | Correlate y and p across eligible bins separately within each trial/region. Constant or fewer-than-two observations are unavailable. Equal-trial summary isolates temporal tracking from between-trial means; report median/distribution as well. It does not assess calibration. |
| Elevated-bin Recall@5% | Fit each session/region's q95 from eligible **training** counts (declare bin-weighted distribution); freeze it. Define true elevated bins by `y>=q95` and detections by `p>=q95`; recall is intersection / observed elevated bins, unavailable when denominator is zero. Report integer ties, predicted positive fraction, precision and false-positive rate. This is amplitude-sensitive and can penalize a correct conditional mean for noisy extremes. |
| Ranking Recall@5% | A separate optional definition: intersect predicted top-k and observed top-k eligible bins within each trial, k=ceil(0.05*n), with fixed deterministic tie handling. Divide by k; chance overlap is roughly 5%. Rank-based recall measures localization, not amplitude. Never present it as the threshold metric above. |
| Peak amplitude error | On true elevated bins from the frozen training threshold, report mean signed `p-y`, MAE and counts, per trial/region. For local-maximum event diagnostics, predeclare minimum separation and matching tolerance on validation; report matched and missed events separately. Avoid taking an unconstrained predicted maximum elsewhere in a trial as a match. |
| Variance ratio | `Var(p)/Var(y)` on eligible bins per trial/region; zero-target variance is unavailable. Also report pooled and trial-centered bin-weighted variance ratios. A ratio of one is not a scientific optimization target because y includes irreducible variability. |
| Lag-dependent correlation | Within each trial correlate `p[t-lag]` with `y[t]`; positive lag means predictions lead observations. For visual-response analysis use a training-fitted visual predictor or declared transition signal, not an arbitrary coordinate. Use the intersection valid for **every** candidate lag for comparisons, center separately per trial, and aggregate as above. Select any offset only on train/validation; never shift official test predictions to maximize correlation. |

**Smoothing:** Report diagnostic-only temporal sums/means at predeclared widths such as 3, 6 and 12 bins (50, 100 and 200 ms), applying identical filters to y and p within each trial and retaining only complete eligible windows. Sums preserve count units; means must be labeled. Do not pad across onset/offset or score windows partially supported by excluded targets. Centered filters use future data and belong only to offline plots/reliability diagnostics, never model input or a causal claim. Prefer clearly labeled trailing windows when illustrating causal availability. Correlation after smoothing can rise simply by removing noise/autocorrelation effects; it is not proof that individual raw peaks are predictable. Condition-specific repeat averages, when enough matching trials exist, give stronger evidence than an all-trials PSTH. Preserve original official targets and scores.

## D. Recommended experiments, in order

These are proposed controlled experiments, not an implementation plan or authorization to train during this audit. Keep the existing session selection and trial memberships. Existing reference artifacts remain immutable. Contract conformity and historical diagnosis must not be mixed into an unlabeled comparison.

### 1. Frozen train/validation trajectories and temporal diagnostics

- **Hypothesis:** The early checkpoint learns trial means while failing to learn temporal changes even on training data, or temporal learning fails mainly to generalize.
- **Conceptual change:** None to model behavior. Restore the selected checkpoint and collect deterministic train/validation predictions separately; attach identities and existing support.
- **Components:** Training-owned diagnostic orchestration, existing model and dataset interfaces; evaluation-style supplemental calculations.
- **Required artifacts:** Selected checkpoint/configuration, exact dataset generation/mapping, split manifests; original-bin y, log p, masks, times and trial IDs. No source neuron checkpoint is needed.
- **Metrics:** Same-path Poisson loss, training-null BPS, means/ranges, trial-centered variance, within-trial Pearson, calibration, fixed-training-threshold recall/amplitude, and labeled smoothing diagnostics.
- **Expected improvement:** Diagnostic discrimination, not an immediate score gain.
- **Risks/trade-offs:** Full-trial inference cost; historical predictions remain noncausal; selected validation is not an unbiased final test.
- **Accept/reject:** Similar tiny train/validation temporal variance supports representation/objective/learnability investigation; substantially better train temporal metrics supports generalization/overfitting investigation. Define practical magnitude thresholds before inspecting the new output; do not use noisy count variance as a required target.

### 2. Establish a contract-conforming temporal baseline

- **Hypothesis:** Current conclusions are confounded by held-state inputs and unrestricted context.
- **Conceptual change:** Restore the specified bin-center representation and existing strict H=12 execution in a separately scoped implementation cycle. Preserve exact source trials/splits; fail explicitly if accepted interpolation lacks source support. Do not silently select a different eligible population.
- **Components:** Alignment, training/evaluation policy compatibility and scenario configuration; model strict implementation can be retained.
- **Required artifacts:** Existing source CLIP observations/timestamps, neural counts, fresh explicitly identified alignment/dataset generation and new run; old artifacts untouched. Reuse the frozen diagnostic from experiment 1.
- **Metrics:** Official metrics and supplemental train/validation temporal metrics on matched source identities; inspect context support before fitting.
- **Expected improvement:** Valid interpretation under the requested protocol; no guaranteed numerical gain. A causal score can be lower than a noncausal one.
- **Risks/trade-offs:** Changed visual representation and context are two factors. Where informative and compatible, use controlled train/validation comparisons separating them; do not infer which factor caused a score difference from a combined change. Interpolation support changes may prevent a strictly matched comparison.
- **Accept/reject:** Accept protocol conformity only when the accepted physical resampling, unchanged trial memberships and isolated `[t-11,t]` receptive field are demonstrated. This is a prerequisite for causal claims, independent of score direction.

### 3. Train/validation lag and transition reliability diagnostic

- **Hypothesis:** Reproducible delayed responses exist but are inadequately exploited.
- **Conceptual change:** First analyze frozen outputs/transition-aligned counts; predeclare lag range 0–11 bins with signed negative lags as descriptive controls. Fit any compact visual readout on training only and validate its lag preference. No data realignment.
- **Components:** Training diagnostic, model input policy only if a later ablation is justified.
- **Required artifacts:** Train/validation trajectories/features, physical timing, meaningful transition/condition metadata where available; trial-level uncertainty and common lag support.
- **Metrics:** Lag curves, within-trial correlation, Poisson loss/BPS, raw and diagnostic-smoothed metrics, matched-condition reliability.
- **Expected improvement:** Identify whether emphasizing earlier visual bins could help; effect size is unknown.
- **Risks/trade-offs:** Stimulus autocorrelation, behavior-linked image changes, noisy transitions and sparse repeats can create apparent lags. Do not report test-selected maxima.
- **Accept/reject:** Support a delay intervention only if validation agrees with a stable training preference across trials/conditions and uncertainty excludes a practically flat curve. Reject a single universal delay if preferences vary or replicate poorly.

If a later input experiment uses a positive delay d, choose information **within the original allowed window**, e.g. `[t-11,...,t-d]`. Simply shifting a full H=12 window to `[t-11-d,...,t-d]` expands the allowed lookback and violates the preserved horizon. A delay beyond eleven bins cannot be introduced under the current strict contract. No response-delay correction is applied in this audit.

### 4. Window-local temporal differences from cached CLIP features

- **Hypothesis:** Explicit differences make weak existing transition information easier to learn.
- **Conceptual change:** Add a learned difference branch alongside the unchanged visual-content branch; compare against the same strict baseline. `delta_V[t]=V[t]-V[t-1]` is a deterministic function of existing vectors and adds **no information** when both are already visible. It changes inductive bias/conditioning; it cannot recover detail discarded by CLIP.
- **Components:** Model input projection/configuration/checkpoint identity, with a training-owned runtime adapter if needed. No scientific alignment or persisted dataset values need change.
- **Required artifacts:** Existing cached aligned CLIP vectors are sufficient. Use training-only scale statistics if needed; do not L2-normalize each tiny delta, which would amplify numerical noise. An additive 768-to-hidden difference projection can retain the existing content width; concatenation would instead require changing the 1536-wide input projection and checkpoint contracts.
- **Metrics:** Official plus temporal, transition-stratified, peak and calibration diagnostics on validation, using the same initialization/optimization budget.
- **Expected improvement:** Easier learning of transitions, not automatic variance inflation.
- **Risks/trade-offs:** **Strict-window boundary leakage:** globally precomputed delta at the oldest allowed token reads one bin outside H. Calculate differences within each gathered window, setting its first delta to zero (optionally with a validity indicator). Reset at every independent trial; no previous-trial features. H=1 then supplies no valid delta. Adding a delta only at the target is another restricted option when H>=2.
- **Accept/reject:** Retain only for reproducible validation temporal gains without degraded calibration/official scores or receptive-field expansion; reject a mere increase in prediction variance.

### 5. Multi-scale Poisson auxiliary objective

- **Hypothesis:** Reproducible slower rate modulation is obscured by fine-bin noise and weak training signals.
- **Conceptual change:** Retain fine-bin Poisson NLL and add a fixed-weight auxiliary term on complete trial-local blocks: `Y_b=sum(y_t)`, `Lambda_b=sum(exp(z_t))`, loss `Lambda_b-Y_b*log(Lambda_b)`. Compute log-sums stably. Predeclare block sizes/weights on training/validation, with no test tuning.
- **Components:** Training objective/configuration and checkpoint objective identity; model output remains log expected fine-bin regional counts.
- **Required artifacts:** Existing counts/masks, matched baseline run and reliable smoothed/condition response evidence from earlier diagnostics.
- **Metrics:** Unchanged official fine-bin metrics, calibration, within-trial correlation, raw peaks and labeled 50–200 ms diagnostics.
- **Expected improvement:** Better integrated activity if the model currently misses reproducible coarse changes. Such evidence is not yet established here.
- **Risks/trade-offs:** Overlapping scales reuse observations and form a composite objective, not independent additional data likelihood. Coarse loss can reward correct totals with wrong timing and further smooth peaks. Normalize per scale so exposure/number of bins does not accidentally dominate. Require full support inside each trial; pooling targets is training supervision, not permission for future input. Complexity is low to moderate.
- **Accept/reject:** Retain only if validation improves reproducible temporal structure and integrated count accuracy without sacrificing fine-bin timing/calibration; reject coarse-only gains coupled to worse raw temporal metrics.

### 6. Negative Binomial conditional-count model

- **Hypothesis:** Residual counts remain conditionally overdispersed after predictable means are modeled.
- **Conceptual change:** Replace the optimized likelihood in a separate scenario with NB mean `mu=exp(z)` and positive dispersion k, `Var(Y|V)=mu+mu^2/k`; begin with a training-fitted per-session/region scalar k. Preserve mean output semantics and region-level counts.
- **Components:** Training loss/configuration, dispersion parameter persistence and possibly a separate dispersion head. Official Poisson BPS remains available as its defined score; label NB likelihood separately.
- **Required artifacts:** Training-fitted conditional residual assessment, validation calibration/likelihood and controlled baseline. Marginal Fano >1 alone is insufficient.
- **Metrics:** Official metrics, validation NB likelihood, conditional dispersion and temporal/peak diagnostics; compare the same means and support.
- **Expected improvement:** Better uncertainty/likelihood calibration if conditional overdispersion is real, not guaranteed higher mean peaks. For fixed k the log-mean gradient is `(mu-y)/(1+mu/k)`, so large residual influence can be reduced rather than boosted.
- **Risks/trade-offs:** Extra dispersion can absorb missing visual structure; unstable dispersion estimates or excessive downweighting can worsen mean learning. Scalar dispersion is moderate complexity; time-varying dispersion needs an architectural/output-contract extension and is deferred.
- **Accept/reject:** Retain only if train-fitted dispersion generalizes and validation conditional likelihood/calibration improve with acceptable temporal means. Reject when improvement comes solely from broad uncertainty while mean timing deteriorates.

## E. Recommended next action

Create **one frozen-checkpoint train/validation temporal diagnostic**, rather than start another training run. Restore `output/runs/scenario-15/006/model_best.pt` through `trainer.pretrained.model_from_checkpoint`; load its exact dataset through the existing handoff, then apply the checkpoint's recorded visual-region selection and regional aggregation. Verify ordered source-unit mapping, split/sample identities, physical bins and H=12 masks against the checkpoint.

Collect every training and validation trial once in ordered, non-shuffled loaders, using `model.eval()` and `torch.inference_mode()` with keyword visual-only encoding and the checkpoint's recorded context view. Never call `train()`, construct an optimizer, update weights, change the checkpoint context, or invoke the test-only evaluation setup for a training split. Use the same deterministic path on both splits; it deliberately removes dropout so the comparison concerns the learned mapping. Preserve masks and original timestamps and save only to a new diagnostic destination. Compute training thresholds/statistics first and freeze them before validation scoring.

The required result is a split-labeled report and original-bin trajectories showing mean/range, Poisson objective, calibration, within-trial correlations and variance ratios, plus the supplemental diagnostics above. This extends the useful inference pattern in `region_diagnostics._predict` without inheriting its unrelated source-checkpoint comparison requirement. It is a future diagnostic task, not code added by this audit.

This has the strongest evidence-to-effort ratio because it distinguishes two materially different next steps: a model that cannot fit temporal changes on training data calls for representation/learnability investigation; a model that fits them but loses them on validation calls for generalization/regularization investigation. Either conclusion is more actionable than increasing amplitudes, applying a guessed lag, or replacing Poisson loss. Label this historical diagnostic **full-trial, held-state** throughout; resolving A01/A02 is necessary before claiming a new contract-conforming experiment.
