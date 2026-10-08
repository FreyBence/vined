# Model interface

The public encoding boundary is `MultiModal` construction, keyword forward
or `encode`, `MultiModalOutput`, and checkpoint identity validation. Modality
dictionary calls provide optional NEDS compatibility. Inputs are already aligned;
acquisition, alignment, optimization, and scientific evaluation belong to callers.

## Session-aware construction

`src/multi_modal/encoder_embeddings.py` exposes:

```python
EncoderEmbedding(n_channel, output_channel, config, stitching=False,
                 eid_list=None, mod=None, max_F=100, **kwargs)
```

`config` is the encoder configuration. Provide `eid_list` as a nonempty
`dict[str, int]` of explicit session IDs to positive neuron counts, including
for non-stitched embeddings. Session IDs cannot contain dots. Session embedding
rows follow this mapping's insertion order; no EID files are read. With
`stitching=True`, visual projections are session-specific and accept the configured
`config.embedder.visual_dim` (default 768). Neural input projections consume the
largest configured population width. Inputs and outputs preserve session-column order.

`src/multi_modal/mm.py` exposes:

```python
MultiModal(encoder_embeddings, avail_mod, avail_beh, model_mode, config,
           neuron_order=None, **kwargs)
```

`config` is the model configuration; `kwargs["eid_list"]` must match every
encoder's session populations. `encoder_embeddings` maps input modality names
to `EncoderEmbedding` modules. Existing modes are `encoding`, `decoding`, and
`mm`. The current primary modalities are `spike` and `vision-clip`.

For encoding, construct only the visual encoder, with
`avail_mod=["spike", "vision-clip"]`, `avail_beh=["vision-clip"]`, and
`model_mode="encoding"`. From the checkout root with `src` on the import path:

```python
from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.mm import MultiModal
from utils.config_utils import update_config

config = update_config("src/configs/multi_modal/mm.yaml")
config["masker"]["force_active"] = False
sessions = {"session-a": 3, "session-b": 5}
hidden = config.encoder.transformer.hidden_size
visual = EncoderEmbedding(
    hidden, hidden, config.encoder, stitching=True, eid_list=sessions,
    mod="vision-clip", max_F=config.encoder.embedder.max_F,
)
model = MultiModal(
    {"vision-clip": visual}, ["spike", "vision-clip"], ["vision-clip"],
    "encoding", config, eid_list=sessions,
).eval()
```

The `DictConfig` model configuration supplies `encoder.embedder` dimensions,
modality count, learned-position flag `pos`, projection options, and dropout;
`encoder.transformer` supplies hidden/intermediate widths, layers, heads,
`use_rope`, normalization, activation, initialization, and dropout.
`masker.force_active=False` avoids constructing a stochastic masker; enabled
legacy modes are `temporal` and `causal`. Use shipped defaults for unused legacy
configuration fields. Hidden width must divide evenly among positive attention
heads; rotary head width must be even.

Optional `neuron_order` is a session-keyed dictionary of ordered lists, one
entry per real neuron, containing caller-provided unit identifiers or records.
It is recorded for identity comparison, never used to reorder data. When omitted,
the model identifies channels only by session and column index; biological unit
identity must be checked by the caller using its population metadata.

All required projections and heads are registered at construction. Use ordinary
`model.parameters()`, `model.to(device_or_dtype)`, and `model.state_dict()`.
Construction does not automatically select CUDA. Singleton and mixed-session
batches receive the same session embedding for the same session.

## Neural output mapping

`src/models/stitcher.py` exposes:

```python
StitchDecoder(eid_list, n_channels, mod="spike", max_F=100, visual_dim=768)
decoder(x, eid)
decoder.neuron_mask(eid, *, device=None)
```

For neural output, `x` contains latent vectors of width `n_channels`, conventionally
`[B,T,n_channels]`; the retained multimodal path may pass flattened `[B*T,n_channels]`
vectors in batch-major order. `eid` supplies one session ID per sample. Each
session head computes its real population width and normalizes only those
channels; a one-neuron head omits LayerNorm. Results have shape `[B,T,N_max]`,
where `N_max` is the largest configured population. Real channels occupy the
leading session-specific columns; remaining columns are zero padding. Batch and
temporal order are preserved. Neural predictions represent log expected spike
counts per aligned bin; padded zeros have no scientific output meaning.

`neuron_mask` returns boolean `[B,N_max]`, with true for real neural channels;
it is available only for `mod="spike"`. Unknown sessions raise `ValueError`.

## Checkpoint identity

`model.checkpoint_identity()` returns a detached dictionary describing the model
configuration, direction, encoder layout, session populations, ordered session
embedding rows, neural column ordering, optional unit identity, and output semantics.
`model.validate_checkpoint_identity(identity)` raises `ValueError` on mismatch.
New-session adaptation constructs an explicitly configured target model; its
registered session modules and embedding rows are exposed for deliberate weight
initialization. Identity validation does not infer adaptation from matching shapes.

Session input modules are under
`model.encoder_embeddings[mod].embedder.mod_stitch_encoder.project_dict[session]`;
neural output modules are under
`model.mod_stitcher_proj_dict["spike"].stitch_decoder_dict[session]` in encoding
mode. The embedder's `session_emb` table uses its documented `eid_lookup` row
order and `eid_to_indx` mapping. Compatible modules can be initialized through
their ordinary PyTorch `load_state_dict`; the caller must establish session and
ordered-population compatibility before copying weights.

Normal `state_dict()` saves identity in its PyTorch `_metadata`, keeping its
parameter entries tensor-only. Preserve the returned state dictionary, including
metadata, when serializing with `torch.save`; `load_state_dict` checks identity
before loading weights. Converting it to an ordinary dictionary discards identity
and is rejected on loading. Historical states without identity and states with
incompatible session rows or head layouts are rejected rather than automatically
migrated. The caller owns optimizer state and checkpoint artifact serialization.

Existing training registration helpers are harmless when applied to these
already registered modules. Its duplicate-sample workaround remains compatible
but is unnecessary for session embeddings. Session heads now use real population
widths, so earlier heads normalized over padded populations can have incompatible
parameter layouts.

## Visual-only encoding

For a model configured with `model_mode="encoding"` and only `vision-clip`
encoder embeddings, call the model with keyword inputs, or call `model.encode`
with the same arguments:

```python
encode(visual_features, temporal_positions, temporal_mask, session_id, *,
       training_mask=None, neural_targets=None, neuron_mask=None,
       target_mask=None, return_latent=False) -> MultiModalOutput
```

```python
output = model(
    visual_features=features,       # floating [B,T,D_visual]
    temporal_positions=positions,  # int64 [B,T], model-grid coordinates
    temporal_mask=valid,           # boolean or 0/1 [B,T]
    session_id=sessions,           # sequence of B explicit session IDs
)
```

`B` must be nonzero; runtime length must satisfy `0 < T <= max_F`.
Tensors and masks must share the input device. Positions denote model-grid
indices, not seconds. No targets, neural input dictionary, masker, corruption,
or optimization settings are needed. The call does not sample masks, including
in training mode. Ordinary `eval()` disables dropout and gives deterministic
predictions for identical inputs and explicit masks.

`config.encoder.embedder.visual_dim` controls visual input and output widths;
`config.encoder.embedder.max_F` is the maximum supported model position count.
Each encoder's `max_F` and visual width must agree with the model configuration.
Learned positions and RoPE tables use this same maximum, independently of modality
count. Valid positions must be int64 indices in `[0,max_F)`; padding indices are
ignored. Longer sequences and incompatible feature widths are rejected explicitly.
Adding temporal padding within the maximum does not change valid predictions
beyond normal floating-point rounding. Upstream feature preparation and training
dataset constraints are separate; the current research dataset still uses 768 features.

`config.context.forward` and `backward` are model-grid position limits: `-1`
means unrestricted in that direction and a nonnegative integer sets the maximum
distance. `forward=0, backward=-1` is causal; `forward=-1, backward=0` permits
only present/future positions; both zero permit only matching temporal positions.
Both `-1` retain the non-causal default. Limits use the supplied positions rather
than concatenated token offsets, so they apply consistently across multimodal
sequences. Context and scientific validity are combined in attention in both
training and evaluation; training corruption never changes context limits.

`MultiModalOutput.neural_prediction` is `[B,T,N_max]` log expected spike counts.
`mod_preds["spike"]` exposes the same tensor. `temporal_mask[B,T]` and
`neuron_mask[B,N_max]` define scientifically interpretable output cells through
their conjunction. Invalid time positions and neural channels return zero fill,
which must not be interpreted as a valid log-count prediction. An entirely invalid
sample returns zero-filled predictions and retains false temporal validity.
Without targets, `loss` and target/loss dictionaries are `None`.

`B` is batch size, `T` is runtime temporal length, `D_visual` is visual width,
`D_model` is transformer hidden width, and `N_max` is the largest session population.
Convert valid log counts to expected counts with `torch.exp`; these are counts
per aligned bin, not spikes per second. Bin duration and physical timestamp
provenance remain caller metadata.

Optional keyword arguments are:

- `training_mask[B,T]`: explicit token corruption, true where visual content is
  replaced by the learned mask token. It is intersected with temporal validity,
  never changes scientific validity, and is applied deterministically even in
  evaluation when explicitly supplied.
- `neuron_mask[B,N_max]`: optional channel validity, constrained to the configured
  real session population. It cannot mark padded channels valid. Omission uses
  session populations. Channel ordering is never inferred from target values.
- `neural_targets[B,T,N_max]`: optional spike-count targets enabling convenience
  Poisson NLL with `log_input=True`. Valid zero counts are included. Non-finite or
  negative selected real targets raise `ValueError`; invalid target fills are
  ignored before loss arithmetic.
- `target_mask[B,T]` or `[B,T,N_max]`: explicit target selection, independent of
  corruption and intersected with temporal/channel validity. Requires targets;
  omission selects all valid neural cells. `mod_n_examples["spike"]` is the
  selected real-cell count. An empty selection returns differentiable zero loss.
- `return_latent=True`: exposes `latent_representation[B,T,D_model]` in the result.

Invalid visual positions and neural input padding are filled before projection;
invalid positions are excluded as attention keys and reset after temporal layers.
Padded positional indices are ignored. Corrupted real tokens remain valid
attention context. Masks with wrong shape, nonbinary values, inconsistent devices,
unknown sessions, or invalid channel declarations raise `ValueError`.

## Retained modality dictionary calls

`model(mod_dict)` remains available for training and optional decoding/multimodal
operation. Each input modality dictionary provides `inputs[B,T,D]`,
`inputs_timestamp[B,T]`, scalar `inputs_modality`, `eid`, and explicit temporal
validity through `temporal_mask` or legacy `inputs_attn_mask`. Optional `targets`
enable convenience losses only for supplied output modalities. Input and target
sessions must agree sample by sample; in unimodal modes, temporal validity must
also agree. Encoding no longer requires a `spike` dictionary when predicting
without targets. The call copies dictionaries rather than modifying caller data.

Legacy `eval_mask` is an explicit temporal corruption/target-selection request;
`[B,T,D]` requests retain their first-channel temporal convention. Separate
`training_mask` (or legacy `inputs_token_mask`) overrides input corruption, and
`target_mask` overrides target selection. Stochastic legacy `random_token` or
`mixed` requests run only during training with an enabled masker. Without an
explicit evaluation corruption request, evaluation uses uncorrupted inputs and
all valid targets. Scientific validity is carried separately into attention and
loss; neural padding is determined by session populations or an explicit
`neuron_mask`, never by `targets == -1`.

The result retains `loss`, per-modality `mod_loss`, `mod_n_examples`, `mod_preds`,
and `mod_targets`, and also exposes neural predictions and validity when present.
Visual convenience loss remains cosine loss. Only supplied target modalities
contribute to the combined convenience loss; training owns the active objective
and optimization policy. The existing training adapter remains compatible.

## Consumer compatibility and failures

The training adapter supplies the declared modality dictionaries and retains
`mod_preds`, `mod_targets`, losses, and counts. Its registration helper is
idempotent, and its unsupervised singleton copies do not change session dispatch.
Training stores the unmodified `state_dict`, preserving model identity metadata;
it also owns ordered-population provenance and optimizer/checkpoint selection.
Training's current dataset configuration restricts visual width to 768 even
though the model itself accepts configured widths.

For compatible training checkpoints, the existing
`trainer.pretrained.model_from_checkpoint(path)` restores saved configuration,
session mappings, and full weights as a registered CPU evaluation model. Call
`model.to(device)` and use the visual keyword interface for inference. This helper
is training-owned; the model does not import or depend on it.

Legacy `src/eval.py` delegates loading to
`utils.eval_utils.load_model_data_local`, which removes session-specific weights
and converts retained weights to a plain dictionary. This discards identity
metadata and is rejected by the model, including with `strict=False`. That legacy
loader is incompatible with this checkpoint boundary; use full identity-preserving
restoration for supported inference. Evaluation must select real output cells
using validity and preserve the checkpoint's session and neuron ordering.

Invalid session populations, mismatched encoder/model configuration, unsupported
lengths or widths, malformed masks/positions, incompatible session IDs, and
selected non-finite or negative neural targets raise `ValueError`. Missing or
mismatched checkpoint identity also raises `ValueError`; ordinary strict PyTorch
loading raises `RuntimeError` for missing/unexpected parameter keys or incompatible
parameter shapes after identity validation. Identity checks remain active under
`strict=False`. Optional legacy calls without targets return empty per-modality
loss/target dictionaries and `loss=None`; keyword encoding leaves them `None`.
