# E1b experiment interface

`python script/experiment/visual-motion-controlled/run.py [--e1 PATH]
[--output PATH] [--threads N]` consumes a completed E1 experiment from the
checkout root. Output must be a fresh directory. Failure leaves incomplete
outputs for inspection; only `completion.json` signals a successful run.

The frozen E1 manifest selects dataset, alignment, feature and replay
generations and the original 218 training / 31 validation trials. Frozen E1
JSONL anchors supply qualified labels. The reader checks generation identities,
the E1 feature archive hash, selected dataset/alignment payload hashes, trial
identity, source times, aligned physical times and pair feature finiteness.
Unselected feature records are skipped without JSON decoding and their numeric
payloads are skipped by seeking. Compressed bytes may be traversed; test
observations are never decoded. E1 integrity evidence is inherited, not rerun.

Feature variants are A=V[t], B=V[t]−V[t−1], C=[A,B], D=abs(B), E=L2(B),
with widths 768/768/1536/768/1. D/E serve presence and absolute displacement;
signed direction/displacement use A/B/C. All variants in a comparison share
anchors. Regression units are degrees per regular adjacent interval; speeds
are degrees/second. Source features retain float32 subtraction; scaling and
fixed linear fitting use float64. Source and aligned times remain distinct.

Primary aligned anchors satisfy `11 <= t < min(T,146)`. Corresponding source
anchor times use the physical bin-edge interval
`[aligned_start+11/60, aligned_start+min(T,146)/60)`; they are not aligned
centers. Preceding context may lie before the scored anchor interval. Full
frozen E1 support is a separately fitted supplementary analysis.

Matched strata use floor(position/10) and E1 contrast class. Both splits need
three independent trials per class and at least six union trials. Trials that
contribute both classes are deterministically allocated to a single class per
stratum/split, balancing contributor counts after allocating exclusive trials.
Only strata supported in both splits are used. Signed regression matches
opposite visible motion directions; absolute regression matches moving versus
stationary states. No missing or ambiguous labels are promoted to motion.

Training totals are equal per trial; matched fits balance stratum/class cells
within trial. The additional `matched_balanced` metric equalizes all
stratum/class totals, giving contributing trials equal weight inside each cell.
Global trial totals can differ for this supplemental standardized metric.
Primary `trial_balanced` and secondary `pooled`/`per_trial` scores retain E1
definitions. Per-trial balanced accuracy uses classes present in that trial,
so its average paired difference can differ from aggregate balanced accuracy.

Training-only weighted StandardScaler, ridge alpha=1 and logistic C=1 are fixed.
Logistic uses lbfgs, tol=1e-4, at most 20000 iterations; a training convergence
warning alone triggers a warm-start continuation with 80000 iterations.
Unconverged fits are excluded. No validation threshold or model selection is
performed. Classification decisions use the estimator's fixed rule.

| Output | Contract |
|---|---|
| `manifest.json` | Frozen inputs, hashes, selected memberships/payloads, E1 provenance/integrity, versions and protocol. |
| `support-{stage}.json` | Coverage/exclusions before fitting, independent trial IDs per stratum/class, allocations, refined support, common anchor hash and training speed tertiles. |
| `confounds.json` | Inherited E1 associations/repeated-image diagnostics plus interval-specific side, phase, position, speed, categories and stationary runs. |
| `results-{stage}.json` | Separate interval/target/mode scores, baselines, confusion matrices, optimization evidence, refined-support scores and unchanged-pixel false positives. |
| `paired-{stage}.json` | Common-anchor differences and individual-trial differences; descriptive reused-validation comparisons. |
| `predictions-*.npz` | Validation trial, anchor, time, target, evaluation_weight and variant predictions, all `[N]`. |
| `models/*.npz` | Scaler mean/scale, coefficients/intercept and classes; pickle-free linear inference. |
| `implementation/*.py` | Exact executed runner and imported E1 helper snapshots. |
| `report.md`, `verification.md` | Scientific comparison and manual verification evidence. |
| `summary.md` | Compact, manually reviewed interpretation of the completed scores and limitations. |
| `matched-confounds.json` | Post-run descriptive class-conditional position, displacement, speed, side and phase summaries joined to matched validation prediction anchors. |
| `manual-checks.json` | Direct artifact inspection: completion hashes, validation membership, variant lengths and disjoint contributor counts. |
| `completion.json` | Successful-run flag and top-level output hashes at runner completion. |

Refined side/speed/phase evaluations reuse the matched fitted model, with
independent contributor support checked again on its existing matched anchors.
They report trial-balanced scores and do not refit or choose parameters.
Direction speed bins are training tertiles; phase bins are fixed quadrants.
Raw phase and trial appearance are not fully randomized by these controls.

The compact summary, verification and post-run inspection files are authored
after successful execution; they do not change predictions, matching or model
parameters and are not included in the runner's completion-time hash list.

These standalone research artifacts have no production downstream consumer;
no production interface or component dependency changes.
