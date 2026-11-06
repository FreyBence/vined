# Training interface

## Controlled target delay

Training setup accepts `--target-delay-bins D` (default 0) and
`--target-support-bins S` (default 12). Nondefault values require strict encoding
and S>=H+D, S>=12. Resolved values are recorded in `training.temporal_context`,
used by objective selection and model construction, and checked on checkpoint
resume/restoration. Coverage counts use S rather than the historical fixed 12.
Default D=0/S=12 configuration and checkpoint identities remain unchanged.

The independent `script/experiment/target-delay/` workflow consumes the existing
configuration resolver, model builder, production trainer and checkpoint APIs.
Its [README](../../script/experiment/target-delay/README.md) defines run JSON,
common-support preflight, isolated output manifests, resume and stage commands.
It supports the configured neuron or regional count targets using production
aggregation and changes no production training profiles. Regional experiment
bias initialization and baselines use its recorded common-support training bins.

## Regional count encoding

Training JSON controls prediction granularity through `training.target_granularity`
(`neuron` in the base profiles, or `region`). The current `run.json` selects
regional prediction through its inline training overrides. For example:

```json
{"training": {"target_granularity": "region"}}
```

The selected scenario's existing `neural_region_selection` controls the population:
`visual_only` retains the configured visual areas and `all_recorded` retains all
recorded neurons. Regional aggregation runs after that selection and automatically
creates one channel per area actually represented in each session. Visual groups
use the existing roots in `src/configs/visual-regions.json`. All-recorded groups
use recorded Allen areas, merging layer subdivisions into their parent area.
There is no separate training region list. Channels are ordered alphabetically
within each session; absent areas do not create synthetic zero-count channels.
Different sessions may have different region sets and use normal channel padding.
Missing or unknown anatomical IDs for valid recorded neurons fail explicitly.

The normal default or multi-session training profile defines the default;
no separate regional profile is required. Granularity follows normal
training-profile merge precedence. Derived groups, source columns and
grouping policy are recorded in effective dataset metadata and checkpoints.
There are no `--target-granularity` or `--target-regions` CLI options, and
`training.target_regions` is not a supported override.
Direct `src/train.py` additionally supports `--encoder-checkpoint PATH` for
encoder-only initialization. Regional targets support encoding only.

`trainer.targets.transform_neural_targets(views, metadata, granularity, *, regions=None)`
returns runtime split views and metadata, summing original int64 counts over valid
recorded source neurons. It preserves physical bins, visual features, sample/trial
identity and memberships, and zeros temporal padding. Runtime channel identities
are region rows; `metadata.regional_targets` retains ordered source units,
source neuron count, root names, source column lists and contributing neuron
counts per region. The published dataset remains unchanged. Neuron mode returns
the original views. `regional_training_means(train)` computes count-weighted means
only on complete H=12 eligible training bins and rejects nontraining samples.
The optional explicit `regions` argument restores previously recorded root-based
mappings for checkpoint diagnostics; training derives its groups automatically.

`regional_training_baseline` is recorded in effective dataset metadata and supplies
regional bias initialization and validation BPS null rates. Regional validation
BPS retains the existing per-channel/session averaging; the diagnostic command
below additionally reports pooled bits/spike. The singleton VISpm experiment has
one channel and one session, so these coincide. Loss reduction and accumulation
remain valid-cell weighted Poisson NLL with `log_input=True`, `full=False`.

`trainer.pretrained.initialize_regional_encoder(model, path, *, config)` strictly
restores and validates the complete source checkpoint, checks source population,
dataset/split provenance, context and encoder architecture, then copies and
verifies every non-output-head tensor. All source neuron-head keys are excluded;
the regional projection is new. Its exact transferred, initialized and excluded
parameter lists appear in `run.json` under `adaptation`; `pretrained` records the
source path/hash. The optimizer and scheduler are fresh. `initialize_regional_bias`
sets each intercept to `log(max(training_mean, 1e-8))`; projection weights retain
ordinary trainable linear initialization. Resume cannot be combined with encoder
initialization. `model_from_checkpoint` restores regional checkpoints strictly.

`src/configs/training/default.json` and `multi_session.json` supply base settings.
Their optional `model` objects override the selected session-count architecture
profile before explicit training configuration/CLI overrides are applied. The
current `run.json` supplies the controlled 100-epoch, batch-one, accumulation-four
AdamW configuration, regional granularity, and scenario-15/003 encoder dimensions
and dropout. The effective run `config.json` is the complete resolved configuration.

After training, run `python script/audit/region_diagnostics.py --checkpoint RUN/model_best.pt
--source-checkpoint SOURCE/model_best.pt --output-dir NEW_DIRECTORY`. This
validation-only diagnostic restores the selected regional and source models,
verifies training/validation membership and anatomical mappings, and compares
regional predictions, training-derived constant counts and summed neuron-model
expected counts on identical H=12 bins. It never evaluates test targets. It writes
`results.json`, `report.md`, original-bin `trajectories.npz`, rate trajectory PNGs
in spikes/second/source-neuron, and a training history PNG. Calibration error is
predicted minus observed mean count; pooled BPS uses the training regional null.
Global and trial-centered temporal standard deviations describe expected-count
variation separately from observed variability. Existing output directories are
rejected. No smoothing or further optimization occurs.

The CLI's `--output-dir` is optional. Without it, a new directory is selected at
`output/audit/<scenario>/<attempt>/regional-NNN/`, using the next unused number.
`VINED_OUTPUT_DIR` overrides the parent output root. The Python function still
requires an explicit destination.

## Frozen temporal prediction diagnostic

From the checkout root, run:

```text
.venv\Scripts\python.exe script/audit/temporal_diagnostics.py --checkpoint output/runs/scenario-15/006/model_best.pt --output-dir output/audit/scenario-15/006/temporal-001 --device cuda
```

Omit `--output-dir` to select the next unused
`output/audit/<scenario>/<attempt>/temporal-NNN/` directory automatically.
`VINED_OUTPUT_DIR` overrides the parent output root. Both diagnostic CLIs use
`output/audit/<run-directory>/<kind>-NNN/` for checkpoints outside scenario runs.

`trainer.temporal_diagnostics.run_diagnostic(checkpoint, output_dir, *, device="cpu", dataset_path=None)`
returns the persisted results dictionary. It supports regional encoding checkpoints
with recorded complete H=12 strict or full-trial context. `dataset_path` permits
relocating the exact recorded generation, never substituting another generation.
It verifies generation integrity, selected ordered source units, regional mapping,
train/validation membership, and recorded training baseline. The existing dataset
handoff verifies the complete generation; test samples are discarded before runtime
selection, aggregation, prediction or statistics. No optimizer is constructed.
Both splits use ordered single-trial visual-only inference under `eval()` and
`inference_mode()`, preserving checkpoint context and historical alignment policy.
Unsupported directions/granularities, missing train/validation, incompatible
identities, or nonfinite selected predictions raise `ValueError`. Existing output
destinations raise `FileExistsError`; failed runs can leave an incomplete new directory.

The new directory contains:

- `results.json`: schema version 1, checkpoint SHA-256/epoch, generation identity,
  full context/alignment labels, split memberships, source-unit regional mapping,
  source and payload hashes, definitions, frozen training statistics, trial/region
  records and split summaries. Its existence with `complete=true` marks completion.
- `predictions/{train|val}_NNNNNN.npz`: original-bin `observed` int64 `[T,R]`,
  `log_expected` and `expected` float64 `[T,R]`, `scoring_mask` bool `[T,R]`,
  `neuron_mask[R]`, `temporal_mask[T]`, `prediction_mask[T]`, integer model
  `temporal_positions[T]`, session-clock physical timestamps and bin edges `[T]`,
  and `time[T]` in seconds from stimulus onset. `T` excludes padding; `R` is the
  session's ordered region count. Scoring intersects scientific, model-prediction
  and H=12 eligibility. Stored off-support predictions are never scored.
- `report.md` and standalone offline `report.html`: split comparisons, original
  activity/scatter curves, selectable session/region/trial and trailing smoothing
  at 1, 3, 6 and 12 bins. Time-range controls affect curves; metrics retain full
  eligible trial support. Observed counts and expected counts have distinct labels.

All quantities use counts per original bin. Smoothing uses complete contiguous
eligible trailing means and never crosses trials; smoothed objectives are descriptive,
not official scores. Original-bin Poisson objective is `mean(exp(z)-y*z)`.
Auxiliary BPS uses the training session/region mean. Threshold recall uses
`y>=q95` and `p>=q95` from the bin-weighted eligible training counts, with linear
quantiles. Ranking recall separately compares within-trial top `ceil(.05*n)`
indices, breaking ties by earlier eligible bin. Calibration uses frozen training
prediction-decile edges and reports mean bias, mean ratio and bin reliability.
Training statistics are fixed before validation scoring.

`records` contains per-trial/region `scales` keyed by bin width. `splits` contains
session/region `groups` with equal-trial means and contributor counts, median
within-trial and between-trial-mean correlations, and separately labeled pooled
metrics including trial-centered variance/correlation. It additionally supplies
equal-region session and equal-session means. Constants and zero denominators
return null; empty support has an explicit reason and zero bins. No scientific
requirement is imposed on matching noisy observed variance or individual peaks.
Official evaluation and historical scientific artifacts remain unchanged.


## Scenario launcher

From the checkout root, `python src/run.py train 3` selects scenario 3 from
`src/configs/scenarios.json`. The Bash wrapper `bash script/run.sh train 3`
accepts the same arguments, uses the project virtual environment, and resolves
the checkout root through `script/environment.sh`. Both entry points support
`finetune` or `eval` in place of `train`. The launcher accepts
`train|finetune|eval` and an integer scenario ID; fine-tuning and evaluation
also accept `--attempt 001` or `--checkpoint PATH`. `--setup-only` verifies
the resolved operation without starting training or evaluation.
Configure a concrete published `dataset_generation`, optional
`expected_dataset_generation_id` and `eid`, and operation-specific entry settings
in `src/configs/run.json`. Paths are resolved from the checkout; null `eid` selects
all persisted sessions. Dataset generations remain explicit. Evaluation defaults
to `model_best.pt` in the latest numbered training attempt for the chosen scenario.
Selection uses attempt numbers and the training-selected checkpoint, never test
scores. A latest attempt without a selected checkpoint fails without falling back.

Scenario direction, temporal context, and neural region selection take precedence
over module settings. Optional scenario `train`/`eval` objects override matching
entry settings in `run.json`; scenario-level dataset/session fields override common
settings. Training otherwise retains its session-count model/training profiles and
optional `training_config` override file. Supported `train` entry settings are
`base_path`, `training_config`, `setup_only`, `seed`, `epochs`, `batch_size`,
`validation_batch_size`, `learning_rate`, `weight_decay`, `scheduler`, `eval_every`,
`checkpoint_selection`, `no_wandb`, `resume_checkpoint`, `multi_gpu`, `debug`,
`num_sessions`, `mask_mode`, `mask_ratio`, `mixed_training`, `enc_task_var`, and
`training_overrides`.
Null entry settings preserve the underlying module defaults.

`train.training_overrides` is an inline JSON object using the same schema and
validation as a `--training-config` file (for example `training`, `optimizer`,
and `model.encoder` settings). The launcher forwards it intact through
`--training-overrides`; region lists still come from scenario population selection.
Precedence is base profiles, optional training configuration file, inline
overrides, then explicit flat entry/CLI settings and supported search overrides.
For example, flat `train.epochs` overrides an inline `training.num_epochs`.
Scenario settings replace matching run entry settings, including the complete
`training_overrides` object when supplied. Fine-tuning's existing
`--prefer-training-config` reapplies its dedicated profile after these overrides.

`bash script/run.sh finetune 15 --attempt 002` initializes a new fine-tuning
run from scenario 15 attempt 002's `model_best.pt`. Omitting `--attempt` uses
the latest selected attempt; `--checkpoint PATH` selects an explicit source.
The source directory is preserved and outputs use a new numbered attempt.
The launcher always selects `src/configs/training/finetune.json` and passes
`--prefer-training-config`: fields present in that profile override defaults,
session model profiles and inherited run/scenario entry settings. Unspecified
fields retain their resolved defaults. Training entry settings supply fallback
runtime values, with optional run/scenario `finetune` objects overriding those
fallbacks. Dataset/session and temporal context come from the selected scenario
and common entry settings. Fine-tuning uses fresh optimizer/scheduler state;
resume, debug and multi-GPU launcher settings are rejected. Direct training
commands keep their existing precedence unless the new preference flag is used.

`--neural-region-selection all_recorded|visual_only` defaults to `all_recorded`.
`trainer.selection.select_neural_regions(views, metadata, selection, *, regions=None)` returns
runtime split views and metadata, preserving source neuron order and counts,
trial/sample/session identity, timestamps, masks, and persisted split membership.
The published dataset is unchanged. Missing region metadata or a selected session
with no matching neurons fails explicitly. Runtime samples record selected source
columns and original neuron count in `metadata.runtime_neural_selection`; effective
dataset metadata and checkpoint populations record the selected mapping and policy.
These runtime views are not newly published dataset generations.

`src/configs/visual-regions.json` is shared with `script/audit/find_visual_eids.py`:
LGd, VISp, VISl, VISli, VISpl, VISpor, VISpm, VISam, VISa, VISrl, and VISal.
Selection matches absolute Allen atlas IDs, including descendants and both
hemispheres, using the same hierarchy rule as relevance discovery. Missing atlas
IDs are not selected. Checkpoints record root acronyms and expanded atlas IDs;
evaluation restores the recorded policy and rejects changed atlas descendants.

## Dataset selection and configuration

From the checkout root, use the project Python with `src/train.py` or
`src/finetune.py`. Both consume one explicitly selected, verified published
training-dataset generation:

```text
python src/train.py --dataset-generation PATH --expected-dataset-generation-id ID --epochs 10 --seed 42 --no-wandb
```

`--data_path PATH` also selects a generation directory. Legacy caches, automatic
generation discovery, and runtime resplitting are unsupported. Verification
failures never fall back to another dataset. `--eid UUID` selects one persisted
session; omission selects all sessions. Optional `--num_sessions` must equal the
selected count. Fine-tuning requires `--eid` selecting exactly one session and
accepts `--pretrained-checkpoint PATH` for pretrained weights.

Split membership, sample/trial/session identities, physical timestamps, and
ordered unit tables come from the persisted dataset. Source counts/features
are unchanged. Runtime tensors use the training-dataset loader contract:
`spikes_data[B,T,N]`, `vision-clip[B,T,768]`, temporal/neuron validity masks,
and separate identity/provenance fields. `B` is batch size, `T` the configured
time size, and `N` the largest selected session population. No implicit population
sorting or filtering is allowed. Time maxima cannot truncate real observations.
Counts remain counts per persisted bin; physical time remains session seconds.

Only the training loader and, when enabled, validation loader are constructed.
Test observations never receive a training-owned loader. An empty test split
is supported. Validation requires a nonempty validation split with sessions
represented in optimization. To disable it, use `--eval-every 0
--checkpoint-selection final`.

## Effective configuration

Encoding/decoding entry points now select temporal context through
`--context-mode strict|full_trial` (effective default `full_trial`) and
`--context-bins 1|3|6|9|12` (required only for `strict`). Both `train.py` and
`finetune.py` use the shared argument boundary; shell launchers forward these
arguments. The shared launcher supplies context arguments from the selected
scenario; direct module commands do not read scenarios. Context fields in module
JSON profiles/overrides are not used to select context. Finite legacy model forward/backward limits conflict with
these explicit modes and are rejected. Inherited `mm` remains available when
context arguments are omitted; explicit context requests for `mm` are rejected.

Resolved `training.temporal_context` records mode, bins, fixed
`target_support_bins=12`, `boundary_policy="complete"`, and
`source="entry_arguments"` in run configuration/provenance and checkpoints.
`training.temporal_target_coverage` reports eligible/excluded temporal targets and
trials without targets for optimization and enabled validation. This metadata is
generated from arguments and the dataset, not accepted as JSON configuration.

Context-selected runs accept explicit interpolated or historical held-state
alignment provenance independently of temporal context. See alignment policy
compatibility below; new experiments must select their regenerated dataset explicitly. A train or required
validation split with no complete H=12 targets is rejected at setup. Real trials
are retained, including shorter trials, with unchanged identities and splits.

Optimization and validation select the same temporal targets for every strict
length and full-trial control: encoding excludes the first eleven bins; decoding
excludes the last eleven. Source observations remain available as input context.
Scientific validity and input corruption stay separate from this selection.
Selectors intersect H=12 before accumulation counts, losses, validation BPS null
means, cosine scores, and plots are calculated. Empty accumulation windows are
skipped visibly; an epoch without targets for an active objective fails. Neuron
validity and valid zero-spike counts retain their existing interpretation.

Training supplies dataset-owned runtime context views to the model and verifies
their direction. `trainer.pretrained.build_model` and `model_from_checkpoint`
restore constructor context arguments from recorded runtime metadata. Resume
requires identical effective context/support configuration; adaptation explicitly
checks context as well as architecture, direction, and losses. Older checkpoints
without this metadata retain legacy construction on explicit restoration, but
cannot silently initialize a new context-selected run. Direct inference with a
new restored model must supply its matching runtime context view. Evaluation
requires explicit matching CLI context and applies the same H=12 support.

`trainer.setup.resolve_setup(args, tune_config=None)` returns
`(config, train_loader, validation_loader_or_none, metadata)` after generation
verification and configuration validation. Entry points resolve session-count
model and training JSON profiles under `src/configs`; `--config_dir` can change
that root. Precedence is defaults, `--training-config JSON`, explicit CLI overrides,
then supported Ray trial overrides. Unspecified CLI options preserve JSON values.

`model/default.json`, `single_session.json`, `medium.json`, and `large.json`
contain base architecture settings; the selected training profile's `model`
overrides are applied on top. Selection uses the selected session count: one
session uses `single_session`, 2–10 uses `default`, 11–69 uses `medium`, and 70+
uses `large`. `training/default.json` supplies optimization, masking, tracking,
losses, validation, and runtime batching; 41+ sessions use `training/multi_session.json`.
These validation settings belong to training, not final evaluation.

Dataset and output path defaults come from `utils.paths.dataset_dir()` and
`utils.paths.output_dir()`, honoring `VINED_DATA_DIR` and `VINED_OUTPUT_DIR`.
Use `--dataset-generation` / `--data_path` and `--base_path` for explicit paths.
Training JSON files contain no filesystem paths or Hugging Face access settings;
the selected local dataset path is recorded in the effective `dataset` metadata.
Unused legacy dataset, padding, logging, and optimizer fields are omitted from
the shipped profiles. Neuron counts come from the selected dataset;
Runtime `data.max_time_length` is derived from the longest real trial across
the selected train/validation/test splits. It also sets
`model.encoder.embedder.max_F`; neither is a configurable trial-length limit.
Held-out sequence lengths size model capacity without evaluating targets.
Preserving neuron order, persisted split membership, and the fixed IBL loader
contract are implementation rules, not overridable profile fields. W&B run names
use the invocation directory name.

Supported CLI overrides include `--model_mode encoding|decoding|mm`,
`--epochs`, `--batch-size`, `--validation-batch-size`, `--learning-rate`,
`--weight-decay`, `--scheduler linear|cosine|none`, `--mask_mode temporal|causal`,
`--mask_ratio`, `--enc_task_var all|random|vision-clip`, `--mixed_training`,
`--eval-every`, `--checkpoint-selection`, `--seed`, and `--no-wandb`.
Encoding is the default objective. Its active loss is unit-weight Poisson NLL
on predicted log spike counts. Decoding retains unit-weight CLIP cosine loss;
`mm` retains both losses and its configured mixed or sampled masking schemes.
Other loss names/weights and optimizers besides AdamW are rejected.

Training JSON overrides support the corresponding `training`, `optimizer`,
`masking`, model embedding/transformer, and runtime metadata fields.
Training-owned `masking` is composed into `model.masker` for the model consumer;
the existing `model.masker` override form is also accepted, but specifying both
forms is an error. For example, `{"masking": {"ratio": 0.1}}` overrides corruption
without changing architecture. `trainer.setup.load_defaults(root, session_count)`
composes the profiles; `resolve_setup` derives runtime dimensions after overrides.
Unknown or unsupported overrides are errors. Neuron size and model time size
are derived consistently from the selected populations and trial lengths.
Search overrides are limited to `learning_rate`, `weight_decay`, `mask_ratio`,
`hidden_size`, `inter_size`, and `n_layers`; both optimizer and scheduler use
the resolved learning rate. Search requires validation with metric selection.

`training.checkpoint_selection` accepts `validation_metric` (maximize
`eval_avg_metric`), `validation_loss` (minimize `eval_loss`), or `final`.
Selection is independent of test observations. Validation-selected output is
`model_best.pt`; final output is `model_last.pt`. Validation metrics are BPS
for spike outputs and cosine similarity for visual outputs. The configuration
records active losses, training schemes, selection direction, generation and
split provenance, selected sessions, and ordered units. Configured seed controls
initialization, loader shuffling, masking, and epoch random generators.

`--setup-only` verifies the generation, resolves configuration, constructs only
required loaders, and prints a JSON setup summary without creating a model or
run artifacts. It also works through `bash script/train.sh`; prepend
`--finetune` for fine-tuning. Positional training wrappers accept additional
Python options, including explicit generation selection.

Invalid configuration, incompatible feature width (the current model requires
768), truncating time maxima, unavailable sessions, empty optimization or
required validation splits, and generation verification errors fail visibly.
Legacy implicit `--continue_pretrain` is rejected.

## Optimization and validity

Training uses boolean `temporal_mask[B,T]` and `neuron_mask[B,N]` to define
valid neural targets. Their conjunction includes valid zero-spike counts.
Visual targets use temporal validity. Stochastic target selection is computed
separately, then intersected with scientific validity. Invalid runtime inputs
are replaced with a model-safe fill without changing persisted arrays; they
never contribute to the training loss or its denominator.

For each configured loss component, an accumulation window minimizes the sum
of eligible losses divided by the number of eligible observations in that
window. Poisson NLL counts valid time/channel cells; cosine loss counts valid
visual vectors. `optimizer.gradient_accumulation_steps` controls the maximum
number of microbatches in a window. Partial batches and the final incomplete
window are retained and normalized by their actual targets. Windows selecting
no targets under stochastic masking are visibly skipped without an optimizer
or scheduler update.

Both entry points prepare model, optimizer, and training loader through
Accelerate. Scheduler length uses the prepared loader's accumulation-window
count and configured epochs; `optimizer.planned_updates` records that maximum.
Schedulers advance exactly once per successful optimizer update. Epoch logs
include per-component loss, valid target counts, learning rate, and cumulative
`optimizer_step`/`scheduler_step`; the checkpoint also carries these counters.

Single-device CPU/CUDA and ordinary PyTorch DDP are supported when the backend
is available. DDP sums valid target denominators across ranks and compensates
for gradient averaging, so losses retain the same global observation weighting.
Accelerate's default even-batch sharding can repeat existing training samples
to equalize ranks at the epoch boundary; their persisted identities and split
membership remain intact. Rank seeds control stochastic operations. Main-rank
validation invokes the unwrapped model between synchronized training epochs.
Other distributed execution plugins are rejected.

Session populations and persisted sample membership are checked before every
forward. The training adapter preserves singleton session embeddings through
an unsupervised runtime copy and returns only original-sample predictions;
the copy never increases the objective denominator. Training registers the
retained unimodal decoder dictionaries before constructing the optimizer,
so their existing modules move with the model and participate in optimization
and DDP. Checkpoints now also contain their session parameters under
`mod_stitcher_proj_dict.<modality>.stitch_decoder_dict.<session>.*`; existing
registered parameter names are preserved. `trainer.runtime.register_runtime_modules(model)`
provides the same registration for callers restoring the full state dictionary.

Malformed batches, mask disagreement, population/session or dimension mismatch,
non-finite model predictions/losses/gradients, and skipped numerical optimizer
updates abort visibly. There is no numerical recovery policy.

## Validation and checkpoint selection

Validation visits every sample in the selected persisted validation view exactly
once, without gradients. Singleton and mixed-session batches retain their
session mapping. Padding is excluded through explicit validity; valid zero-spike
bins remain targets. Encoding validation predicts all valid neural observations;
decoding predicts all valid visual vectors. Multimodal validation evaluates both
directions deterministically, independently of stochastic training masks.

`eval_loss` is the weighted sum of component means over all valid targets, with
`eval_valid_target_counts` and `eval_sample_count` exposing coverage. Neural
bits-per-spike compares predicted counts with each neuron's validation mean
count, using its valid bins. It averages over neurons with positive total spikes,
then equally over observed sessions. Zero-spike neurons remain in the loss and
are listed in `eval_metric_unavailable`. A session with no usable neural metric
makes the aggregate metric unavailable (`None`). Visual cosine similarity averages
valid vectors within each session, then equally over sessions. `eval_avg_metric`
is the equal mean of the active modality metrics. Sessions absent from validation
are listed in `eval_sessions_absent`; they do not enter validation averages.
Plots use observed sessions and valid trial-average time positions only.

`validation_metric` selects the largest `eval_avg_metric`; `validation_loss`
selects the smallest `eval_loss`. Strict improvement replaces `model_best.pt`;
ties retain the earlier epoch. `model_best_avg.pt` is a compatibility alias
with the same selected weights and selection record. An unavailable selection value aborts explicitly.
`final` selects `model_last.pt` after the configured final epoch. No test loader
participates in validation or selection. Training returns validation loss/metrics
from the selected epoch, and its reported best values refer to that same epoch.
For final selection, the returned validation report is empty.

Checkpoint `selection` records the rule, selected epoch, metric name/value, and
validation report (without prediction arrays). For final selection the metric
and value are `None`. `model_last.pt` still contains final weights when a
validation-selected `model_best.pt` comes from an earlier epoch.

## Run artifacts and continuation

Scenario training creates `--base_path/runs/scenario-<id>/<attempt>/`, with
automatically increasing numbers `001`, `002`, etc. Numbers are never reused and
concurrent invocations reserve different directories. Direct training/fine-tuning
can supply `--scenario-id` and `--scenario-name`; omission retains the legacy
`--base_path/runs/<attempt_id>/` layout (default base: project output).
`run.json` records the stable logical `run_id`, unique invocation `attempt_id`,
entry-point kind, and any parent checkpoint. Numbered runs additionally record
`scenario_id`, `scenario_name`, and `attempt`; UUID identities remain independent
of directory names. Resume preserves the logical run
identity and writes into a new invocation directory; original artifacts are
never overwritten. `--overwrite` and implicit `--continue_pretrain` are rejected.

`trainer.run_paths.new_attempt_directory(root, scenario_id)` reserves the next
numbered directory. `selected_attempt(root, scenario_id, attempt=None)` returns
the selected attempt name and its `model_best.pt` path. Roots are the runs
directory, not the common output root. Explicit attempts are normalized to at
least three digits. Legacy UUID directories are excluded from numbered selection;
they remain accessible through explicit checkpoint paths.

Each directory contains:

- `config.json`: complete effective model/training/optimizer/dataset configuration,
  including the planned optimizer-update budget.
- `provenance.json`: dataset generation and persisted split configuration/memberships,
  selected sessions, ordered unit rows/columns/dtypes, effective optimization
  configuration, and distributed device/precision layout.
- `history.json`: epoch training reports and validation reports (empty when not
  evaluated), including optimizer/scheduler steps and target coverage.
- `selection.json`: the selected rule, epoch, value, validation report, and the
  selected checkpoint filename when validation selects a model.
- `model_epoch_<epoch>.pt`: complete epoch-boundary snapshots at `training.save_every`
  (zero-based epoch numbering); `model_last.pt` is the final snapshot.
- `model_selected_epoch_<epoch>.pt`: retained snapshots on strict selection
  improvement; `model_best.pt` and `model_best_avg.pt` contain the latest selection.
- `params.pkl`: flat `hidden_size`, `inter_size`, and `n_layers` values retained for
  existing evaluation configuration readers; full configuration is in `config.json`.

JSON and checkpoint writes replace files atomically. Selected snapshots are
retained so continuation can recover the selection belonging to an earlier
checkpoint even if its original run later selected a different epoch.
Fine-tuning records the initial pretrained file path and SHA-256 in `run.json`.

`trainer.artifacts.load_training_checkpoint(path)` loads a trusted PyTorch
checkpoint and rejects unsupported or incomplete resume formats. Version 1 has
`format_version`, `run`, `config`, `compatibility`, `model`, `optimizer`,
`lr_sched`, `epoch`, `next_epoch`, `optimizer_step`, `scheduler_step`, `selection`,
`history`, and `rank_states`. The existing `model` state-dictionary key and
optimizer/scheduler keys remain available to downstream weight readers.
`epoch` is the completed zero-based epoch; `next_epoch` is its successor.
`rank_states` holds Python, NumPy, Torch CPU/CUDA randomness, loader/sampler
generator states, and the prepared loader's epoch counter for every rank.
Continuation is supported at complete epoch boundaries, with zero pending
accumulated gradients. Mid-epoch continuation is unsupported.

To resume, repeat the effective training configuration and explicitly choose
an epoch checkpoint:

```text
python src/train.py --dataset-generation PATH --training-config CONFIG.json --epochs 10 --resume-checkpoint output/runs/ATTEMPT/model_epoch_0.pt --no-wandb
```

The configured total duration, architecture, masking/objective/losses, optimizer,
scheduler, batch/accumulation settings, validation/selection settings, seed,
generation, split/sample/session identities, ordered units, world size, device
type, and precision must match. Dataset/output locations and WandB settings may
change. Resume restores full model/optimizer/scheduler state, next epoch,
step counters, history, selection, and rank-local random/loader state. It also
copies the prior selected snapshot and aliases into the new directory.
Keep the source epoch checkpoint and its referenced selected snapshot together.
Completed final checkpoints with no remaining epochs cannot continue, and
changing total duration is rejected because it changes the scheduler trajectory.
Legacy weight-only checkpoints cannot resume.

`src/finetune.py` accepts the same resume option with `--eid`; it restores the
same adaptation run rather than starting another adaptation. Combining
`--pretrained-checkpoint` and `--resume-checkpoint`, resuming inside Ray search,
or crossing training/fine-tuning kinds is rejected. Unsupported format or
configuration/session/population mismatch raises an explicit error before
optimization. Checkpoint deserialization requires trusted files.

The legacy evaluation entry points still derive checkpoint locations from their
older directory convention. They do not automatically discover these run
folders; downstream consumers should explicitly select a checkpoint and use
its effective configuration and session mapping. Evaluation implementation is
unchanged.

## Pretrained session adaptation and model restoration

Fine-tuning starts a new optimization trajectory from an explicitly selected
pretrained checkpoint:

```text
python src/finetune.py --dataset-generation TARGET_GENERATION --eid TARGET_SESSION --pretrained-checkpoint SOURCE/model_best.pt --training-config CONFIG.json --epochs 10 --no-wandb
```

Exactly one target session is required. Its persisted generation, sample/split
membership, and ordered population remain authoritative. Training and validation
use the same configured validity-aware optimization and selection contracts as
ordinary training, and adaptation never constructs a test loader. Optimizer,
scheduler, step counters, history, and run identity start afresh. All model
parameters remain trainable. Use `--resume-checkpoint` instead to continue an
existing adaptation run; it bypasses pretrained adaptation and restores full
training state. Neither operation guesses a checkpoint directory.

`trainer.pretrained.load_pretrained_model(path, *, config, metadata, modal_filter)`
returns a registered CPU model ready for optimization. The checkpoint must have
effective configuration, population provenance, and explicit
`compatibility.session_embedding_ids` mapping each encoder modality to its
ordered session embedding rows. Historical row order is never inferred from
current EID files. Missing identity provenance is an error.

Source and target must agree on architecture (including context, embedding and
transformer settings), prediction direction, active input/output modalities,
and model loss configuration. Runtime neural padding width and training
corruption settings may change for a new adaptation run. Shared parameter
names, shapes, and dtypes must match exactly; incompatible or non-finite loaded
parameters fail before optimization.

Session-specific input/output modules are retained only for the same session
with identical ordered unit rows and matching complete module tensor layouts.
Otherwise the module is initialized and its reason is recorded. In particular,
a different neural padding width can require a fresh neural head even when the
real unit identities match: inherited neural heads contain population-wide
LayerNorm and cannot be safely sliced. No neuron reordering or dimension-only
session matching is performed. Session embedding rows are retained by explicit
session identity when the population matches; new or changed populations get
fresh rows. The adapted embedding table represents the selected session only,
including sessions absent from the inherited global EID lists.

`run.json` and checkpoint `run` include the source pretrained path/SHA-256 and
an `adaptation` report listing retained and initialized session modules. Resume
preserves this source and adaptation provenance. Full checkpoint state includes
all registered target-session input/output parameters and its embedding mapping.

Downstream consumers can restore a model without constructing any dataset or
trainer loaders:

```python
from trainer.pretrained import model_from_checkpoint
model = model_from_checkpoint(checkpoint_path)
model.to(device)
```

`model_from_checkpoint(path)` reconstructs the saved configuration, neural
layout, and explicit session embedding mapping, strictly restores weights, and
returns a registered CPU model in evaluation mode. Its inputs/outputs remain
the existing multimodal model contract; callers must use the saved target
sessions and ordered populations. Scientific batches still require explicit
validity handling at the caller boundary. This helper supports ordinary training
and adaptation checkpoints with recorded mappings. Earlier snapshots lacking
these mappings cannot be adapted or restored through this helper; compatibility
checks also reject them for continuation. Existing legacy evaluation scripts
remain unchanged and require their older directory/model conventions.


## Alignment policy compatibility

New alignment generations use `L2-normalized linear interpolation at neural bin centers`
with `resampling.policy_version=1`, physical query timestamps and true bracketing
associations. Historical `timestamp-aware held-state over neural intervals v1`
generations retain their original identity. Neither policy is converted at load.
Unknown policies and contradictory association identities are rejected by dataset
construction/loading. New-version interpolation associations are checked against
physical bin centers. Source alignment generation identities remain in sample metadata.

Temporal-context setup accepts either known policy independently of context mode,
requires one policy across consumed training/validation samples, and records
`training.alignment_policies` for interpolation runs in effective configuration and
checkpoints. Historical held-state configurations keep their existing fields for
exact resume compatibility; their dataset generation binds the policy. Resume
retains existing generation/configuration compatibility checks. For the upcoming
Scenario 11 baseline explicitly select a newly interpolated dataset; selecting a
historical dataset intentionally retains its historical representation.
