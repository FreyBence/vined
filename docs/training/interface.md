# Training interface

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
time size, and `N` the largest selected session population. No population
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
arguments. Scenario files and context fields in JSON profiles/overrides are not
used to select context. Finite legacy model forward/backward limits conflict with
these explicit modes and are rejected. Inherited `mm` remains available when
context arguments are omitted; explicit context requests for `mm` are rejected.

Resolved `training.temporal_context` records mode, bins, fixed
`target_support_bins=12`, `boundary_policy="complete"`, and
`source="entry_arguments"` in run configuration/provenance and checkpoints.
`training.temporal_target_coverage` reports eligible/excluded temporal targets and
trials without targets for optimization and enabled validation. This metadata is
generated from arguments and the dataset, not accepted as JSON configuration.

Context-selected runs require held-state alignment provenance
(`timestamp-aware held-state over neural intervals v1`). Interpolated generations
must be regenerated through the shared alignment policy. A train or required
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
contain architecture settings. Selection uses the selected session count: one
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
`model.encoder.embedder.max_F` is derived from training's `data.max_time_length`.
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
`masking`, model embedding/transformer, and runtime time-size/metadata fields.
Training-owned `masking` is composed into `model.masker` for the model consumer;
the existing `model.masker` override form is also accepted, but specifying both
forms is an error. For example, `{"masking": {"ratio": 0.1}}` overrides corruption
without changing architecture. `trainer.setup.load_defaults(root, session_count)`
composes the profiles; `resolve_setup` derives runtime dimensions after overrides.
Unknown or unsupported overrides are errors. Neuron size and model time size
are derived consistently from the selected populations and `data.max_time_length`.
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

Every training or fine-tuning invocation creates an isolated
`--base_path/runs/<attempt_id>/` directory (default base: project output).
`run.json` records the stable logical `run_id`, unique invocation `attempt_id`,
entry-point kind, and any parent checkpoint. Resume preserves the logical run
identity and writes into a new invocation directory; original artifacts are
never overwritten. `--overwrite` and implicit `--continue_pretrain` are rejected.

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
