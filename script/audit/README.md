# Audit scripts

Run these commands with the project Python. Python entry points resolve checkout
imports and default output paths independently of the current working directory.
Audit reports and diagnostics default to `output/audit/` (or
`<VINED_OUTPUT_DIR>/audit/` when configured).

| Command | Default result location |
| --- | --- |
| `analyze_motion_neural.py` | `output/audit/motion-neural/` |
| `analyze_clip_similarity.py` | `output/audit/clip-similarity.json` |
| `audit_clip_motion.py` | `output/audit/clip-motion/` |
| `audit_replay.py` | `output/audit/audit-replay.txt` |
| `find_visual_eids.py` | `output/audit/eid-relevanc.txt` |
| `temporal_diagnostics.py --checkpoint CHECKPOINT` | `output/audit/<scenario>/<attempt>/temporal-NNN/` |
| `region_diagnostics.py --checkpoint CHECKPOINT --source-checkpoint SOURCE` | `output/audit/<scenario>/<attempt>/regional-NNN/` |
| `audit_representation_response.py` | `output/audit/scenario-11/002/representation-response-001/` |
| `audit_response_profiles.py` | Supplements `representation-response-001/` |
| `audit_lag_comparison.py` | Supplements `representation-response-001/` |
| `audit_response_reliability.py` | Stages the next `output/audit/scenario-11/002/response-reliability-NNN/` generation |

For example:

```text
.venv/Scripts/python.exe script/audit/audit_replay.py
.venv/Scripts/python.exe script/audit/temporal_diagnostics.py --checkpoint output/runs/scenario-11/002/model_best.pt --device cuda
```

Replay auditing and both model diagnostic commands accept `--output-dir`.
Temporal and regional commands choose the next unused numbered generation when
that option is omitted; explicit existing diagnostic destinations are rejected.
For checkpoints outside a scenario directory, defaults use
`output/audit/<run-directory>/<kind>-NNN/`.
The Scenario 11 analyses retain their fixed experiment inputs and refusal to
replace published artifacts. Reliability publication remains a manually directed
`write_report(stage)` followed by `publish(stage)` after figure inspection.
These functions are in `audit_response_reliability.py`.

`auto_fix_replay.py` and `auto-fix-replay.sh` read
`output/audit/audit-replay.txt` by default. Their generated replay data still go
to the configured replay root. Use `--dry-run` to inspect the retry selection
without generating replay data. The shell launcher uses the existing
`script/environment.sh` environment helper.

## CLIP feature similarity analysis

From the checkout root:

```powershell
.venv/Scripts/python.exe script/audit/analyze_clip_similarity.py
```

The script reads the explicit single `eid` string from `src/configs/run.json`
(`--run-config` overrides it), resolves `<eid>_visual_clip.npz` under
`utils.paths.visual_dir()` (`--visual-dir` overrides it), and exhausts
`FeatureArtifactReader(..., eid=eid)` before reporting verified results.
Null or multiple EIDs, incompatible legacy archives, and malformed features fail
explicitly. No encoder is loaded and no extraction/training artifacts are changed.

Selection follows archive order and takes the first same-trial pair with adjacent
source `schedule_index` values and a per-dimension absolute difference above
`--tolerance` (default `1e-7`). Trial outcomes and observations without features
break adjacency. Source indices, observation IDs, session-second timestamps,
timing provenance, original dimensions, norms, differing dimensions, and top
absolute differences are reported. Image equality uses verified source RGB hashes
retained in the archive; missing hashes yield unavailable, without requiring images.

The six metrics are raw cosine, raw L2, unit-normalized L2, EID-mean-centered
cosine, per-dimension Z-score L2, and regularized PCA-whitened L2. Raw stored
features are already L2 normalized. Reference statistics use a uniform observation
reservoir (`--reference-samples`, default 4096); pair distributions use a separate
uniform reservoir of eligible consecutive pairs (`--pair-samples`, default 1024),
including identical pairs. `--seed` defaults to 42. Both population and sample
counts are reported. Reservoirs bound memory independently of archive length.

Standard deviations use population variance and a `--std-floor` of `1e-6`.
PCA uses sample covariance, defaults to at most `min(64, samples-1, dimensions)`
components, and accepts `--pca-components`. Eigenvalues at or below
`max(--variance-floor, --pca-ridge * largest_eigenvalue)` are discarded;
retained scores are divided by `sqrt(eigenvalue + ridge)`. The defaults are
`1e-12` and `1e-4`, respectively. The effective rank and retained count are
reported. Insufficient samples or stable variance yield unavailable whitening;
zero centered norms yield unavailable centered cosine (`--norm-floor`, `1e-12`).

Terminal tables and the JSON report contain selected metric values and sampled
minimum, median, maximum, and selected-pair percentiles (fraction less than or
equal, including ties). JSON defaults to `output/audit/clip-similarity.json` (or
`<VINED_OUTPUT_DIR>/audit/clip-similarity.json`); `--json-output` selects another
report path. JSON uses null for unavailable metrics and refuses to
overwrite existing paths. Exit codes are 0 for a comparison, 2 for no differing
pair, and 1 for input/read/report errors; argument errors also return 2.
Distances on different scales do not establish stronger discrimination. This is
an EID-wide diagnostic: fit statistics on the training partition before any
future training integration.

## CLIP configuration and representation sensitivity

```powershell
.venv/Scripts/python.exe script/audit/audit_clip_sensitivity.py
```

This standalone audit reads the active EID and reuses `analyze_clip_similarity.py`
for primary pair selection. It verifies the feature archive, resolves the saved
replay through the extraction CLI resolver, and recovers original pixels through
the public replay reader with matching metadata and RGB hashes. It checks the
published replay identity but stops after recovering the required images; it does
not reverify unrelated replay payloads. Missing originals produce a blocker report.

The frozen checkpoint/revision, processor, padding, attention backend, and effective
precision are verified before alternatives are interpreted.
`--reproduction-atol` defaults to `1e-4` per coordinate with zero relative tolerance;
the report includes actual errors and the original batch/cache/package provenance.
`historical_baseline` reports strict reproduction against stored features using
the original extraction batch size and cache history through the public extraction
path. Numerical mismatch is reported without stopping experiments once provenance
is verified. Source, checkpoint, or preprocessing identity failures still block.
`baseline` is the newly generated two-image projected CLS baseline with reuse
disabled; all distance changes and ratios use it, never stored features.

`numerical_stability` probes the same physical pair at batch sizes 1, 2, 8, and
the original extraction size, at recorded precision and FP32. Larger numerical
probe batches repeat the unchanged pair; these are not secondary observations.
Every experimental representation also runs singleton, paired, reversed-pair,
and identical-image contexts. Each result records context drift, FP32 discrepancy,
and an empirical noise envelope. The baseline envelope additionally includes all
batch/precision probes. A fixed `1e-12` cosine arithmetic floor prevents treating
roundoff as measurable improvement; reproduction tolerances remain unchanged.

Experiments cover bicubic/bilinear/Lanczos fitting, neutral/edge padding, direct
resize, fixed contrast factors 0.75/1/1.25/1.5 around RGB 128, central 75%/50%
crops, and horizontal/vertical half-field crops. Original channel normalization
is retained. Hidden states after quarter/middle/three-quarter/final transformer
blocks yield CLS, patch mean, and ordered patch concatenation representations;
projected and pre-projection pooled CLS are also compared. All configurations run
at recorded precision and FP32. No production extraction or training code changes.

Ranking uses primary-pair cosine distance. Sanity results separately expose exact
pixel transformations, clipping, removed non-neutral content, aspect distortion,
and identical-image repeatability including a different batch context. The first
two additional same-hash and two differing-hash adjacent pairs are used by default
(`--secondary-pairs` controls each category). Secondary success conservatively
requires gains exceeding candidate plus matching baseline noise envelopes on every
differing pair and matching identical pairs. Same-precision baselines control these
decisions; the common recorded-precision baseline remains the ranking reference.
`stable_meaningful_candidate` additionally requires the primary gain to exceed
noise, plus geometry and repeatability checks. Recommendations use this flag.
`noise_limited_improvement` marks positive primary gains below the measured bound.
These empirical envelopes are sampled numerical bounds, not confidence intervals
or evidence that different feature spaces have equivalent similarity scales.
These small, deterministic samples are exploratory, not population evidence.

Outputs default to the fresh directory `output/audit/clip-sensitivity/`, containing
`report.json`, original source PNGs on success, and primary-pair preprocessing
previews. Existing output directories are refused; use `--output-dir` for another
run. Other options include `--run-config`, `--visual-dir`, `--replay-dir`, `--device`,
`--workers`, and `--tolerance`. Exit 0 means a completed comparison; exit 1 means
a provenance/input/inference blocker, recorded in JSON. A historical numerical
mismatch alone permits exit 0 with strict status and warnings retained in the
report. Argument errors return 2.
The report records vector dimensions/storage, inference-group runtimes, peak CUDA
allocation, recommendations, and downstream dimension compatibility. Group runtime
is shared by all representations from one forward group and must not be summed
across its result rows. Larger distance across feature spaces alone does not prove
a better representation. No temporal sampling, source artifacts, weights, or
training settings are changed.

## Multi-trial CLIP displacement sensitivity

```powershell
.venv/Scripts/python.exe script/audit/audit_clip_motion.py --source-cache output/audit/clip-motion-final/source-frames.npz --output-dir output/audit/clip-motion-visibility
```

This standalone audit compares four frozen FP32 CLIP ViT-L/14 configurations:
projected CLS (768) and layer 24 CLS (1024), each at `clip_input_contrast_gain`
1.0 and 1.5. Layer 24 means the token after block 24, before final pooling layer
normalization. Each vector is L2 normalized; cosine accumulates in float64.
The saved immutable revision, processor, bicubic full-view fit/pad and attention
backend are verified. No generator, source parameter, source artifact, production
feature, timing, weight or neural pipeline change is made.

The previous sensitivity audit's enhancement remains `128 + gain*(RGB8-128)`
in float64, followed by clipping and nearest-even rounding before resizing.
Any nonzero clipping fraction, geometric distortion, lost content, preparation
failure or inference failure makes that frame ineligible for that configuration.
Such frames are recorded with reasons and ranges and excluded from inference
and metrics; they do not abort other frames/configurations. Clipping fractions
count RGB channel values before rounding/resizing. Both endpoints must be
eligible for a pair. `stimulus_generator_contrast` is read-only source metadata;
the independent input gain only affects post-rendering CLIP preparation.

Source-identical images with nonzero azimuth displacement are classified as
`visually_unobservable_motion`, including zero-contrast controls. Positive motion
sensitivity requires source RGB differences, nonzero displacement, positive
original contrast, visible source state and unchanged stimulus parameters.
Blank and unobservable-motion controls remain in coverage/stability reports,
but are excluded from positive motion correlations. Prepared-image equality and
source pixel differences are tracked separately from displacement and FP32 noise.

Primary comparisons use the intersection of eligible pairs across all four
configurations, with identical per-contrast samples and displacement bins.
Per-configuration-only summaries retain their separate coverage and are not
four-way comparisons. Reports stratify by original contrast and trial, with
explicit exclusions, source RGB ranges, unbounded enhancement ranges, clipping
fractions, Pearson/Spearman (moving-only and visible stationary-plus-moving),
distance distributions, displacement bins, stationary separation and numerical
stability. Undefined correlations return null with a reason. Small motion means
at or below the median eligible positive displacement in the reported stratum.
Sampled noise includes repeat/reversed half-batch contexts, singleton image drift,
identical-source image distances and a 1e-12 cosine arithmetic floor.

Defaults select eight uniformly spaced eligible trials and at most 18 adjacent
pairs per trial, using stationary/left/right quotas and displacement quantiles,
never embedding values. Options: `--trial-limit`, `--pairs-per-trial`,
`--batch-size` (8), `--stability-frames` (12), `--stationary-atol-deg` (1e-10),
`--workers`, `--device`, `--run-config`, `--visual-dir`, `--replay-dir` and
`--output-dir`. Omit `--source-cache` to recover pixels through the existing
sensitivity audit/public replay reader and save `source-frames.npz`. A supplied
audit cache must match every original shape, dtype and RGB SHA256 from the fully
verified feature archive. The report records cache identity and verification mode.
Old reports contain no reusable vectors; inference is repeated because filtering
also changes batch context. New per-configuration vector NPZ files preserve
observation IDs and first/repeated/alternate vectors; report provenance records
checkpoint, preparation, sources, batch sizes and contexts.

Outputs: `report.json`, `report.md`, `comparison.png` (one row per source
contrast), and vector archives. Runtime groups are shared by both representations
at a given gain and include stability probes; do not sum shared group times.
The default output is `output/audit/clip-motion/`; existing destinations are
refused. `ok` means meaningful matched evaluation, `partial` means usable
per-configuration analysis but insufficient common motion, and `blocked` means
no meaningful motion evaluation or a global input/provenance/model failure.
Exit 0 accepts ok/partial, 1 reports blocked, and 2 indicates argument errors.
These exploratory, dependent samples and cosine metrics do not prove recoverable
position, signed displacement or neural prediction improvement. Subsequent
decoding should use held-out trials and source-contrast strata, with negative
controls retained and no automatic production integration.

## Physical motion and neural activity

```powershell
.venv/Scripts/python.exe script/audit/analyze_motion_neural.py
```

This standalone audit uses the explicit active EID, `dataset_generation` and
optional expected generation ID in `src/configs/run.json`. `--run-config`,
`--visual-dir`, `--config` and `--output-dir` override inputs/destination.
Relative dataset paths resolve from the checkout. Parameters live in
`script/audit/motion-neural.json`; a copy of effective parameters and the run
configuration is saved with every result. Existing destinations are refused.
Errors exit nonzero and leave partial diagnostics without a complete report.

The public dataset loader verifies the prepared generation. The audit separately
verifies the feature manifest/definition and complete `records.jsonl` hash,
generation binding, observation IDs, source indices and physical timestamps.
It reads physical stimulus metadata without loading CLIP vectors or regenerating
frames/counts. Replay RGB payloads are not reverified. Contrast remains the original
generator value; no CLIP input gain is applied. Azimuth is interpolated at existing
60 Hz neural bin centers using the saved alignment associations. This remains
reconstructed timing. Original trial and unit identities and all valid zero counts
are retained. Results cover all prepared splits and are exploratory, not held-out
predictive evidence.

`report.json`/`report.md` summarize contrasts, coverage, movement distributions,
trial-weighted lag/window associations, population PCA, limitations and future model
experiments. `input-audit.json` records schema/provenance checks. `units.csv` gives
the full ordered population. `observations/<trial-id>.npz` contains raw `[time,unit]`
counts, session and trial-relative seconds, original contrast, stage, visibility,
azimuth, signed/absolute displacement, velocity, direction and net/traveled movement
over 3/6/12 bins. Movement units are degrees; velocity is degrees/second.

`associations.npz` stores Pearson/Spearman and half-trial circular-shift references
with axes `[trial,comparison,target]`; explicit axis labels, lag/window coordinates
and `[trial,comparison]` pair counts are included. Targets are region-mean Hz/neuron
and trial-centered raw-count population PCs in rate units. `neuron-associations.npz`
uses the same axes for individual neurons and the absolute one-bin movement
variable; neuron columns map to `units.csv`. Undefined constant/short series are NaN.
`neurons.json` retains contrast-specific neuron summaries. `movement-bins.json`
contains displacement quartile rates and paired stationary effects with observation
and trial counts. `events.json` stores onset/cessation/reversal/small/large
event curves and per-trial uncertainty. `trials/` and `regions/` contain plots;
the observations permit inspection of every other trial and unit/region.

Lag is stimulus time to the beginning of the subsequent response window, 0–200 ms.
Response windows are forward means of 1/3/6/12 bins; movement history ends at the
stimulus time. Common comparison support stays within each independent stimulus
interval. Trial bootstrap intervals and sign consistency use independent trials,
not overlapping bins. Event episodes are averaged within trial before aggregation.
Only representative trial plots smooth rates (three-bin trailing means).
The shifted reference is descriptive, not a calibrated significance test.
Selected maxima do not establish physiological latency or predictive benefit.
Closed-loop visual/motor/state confounding and zero-contrast limitations are
retained explicitly. No model, training, evaluation or upstream component changes.

## Relocated historical results

The complete former `output/diagnostics/` tree now lives directly under
`output/audit/`, preserving its scenario/attempt structure, setup evidence, and
incomplete staging directories. The former `output/audit-replay.txt` is now
`output/audit/audit-replay.txt`.

Historical payloads and manifests were moved without changing their bytes or
hashes. Embedded historical source paths, script hashes and reproduction commands
describe the original run; they are not rewritten inside immutable artifacts.
Resolve historical `output/diagnostics/...` paths as `output/audit/...`, former
`script/audit_*.py` paths under `script/audit/`, and former
`src/{temporal,region}_diagnostics.py` paths under `script/audit/`.
Current documentation links and script defaults use the new locations.

The reusable `trainer.temporal_diagnostics` implementation and its HTML template
remain in `src/trainer/`; the standalone command lives in this folder.
