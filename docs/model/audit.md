# Model audit

## Summary

The NEDS-derived implementation retains the accepted session-specific visual projection, shared temporal transformer, and session-specific neural output structure. It does not yet satisfy the standalone encoding contract: temporal padding participates in attention, predictions require targets and masking metadata, and session context depends on external files and batch cardinality. Training compensates for decoder registration and singleton embeddings, but those guarantees are not owned by the model itself.

This snapshot compares `spec.md`, `architecture.md`, and `docs/dependencies.md` with the model source and the directly connected training interface. No model `interface.md` exists. Findings are based on code inspection; numerical behavior was not exercised.

## Findings

### A01 — Temporal padding participates in attention

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/multi_modal/mm.py`, `forward`, `forward_mask_encoder`, `forward_encoder`.

**Finding:** Invalid temporal positions are folded into `inputs_mask` and replaced by the same token used for training corruption. Their contextual embeddings remain, and every transformer layer receives `mask=None`. These invalid keys and values can influence valid predictions even when the loss excludes padded targets. Scientific validity is not propagated separately into temporal mixing.

**Expected:** Spec §§8–11, 28, 33 and architecture §§14–15 require padding isolation and separate validity/corruption semantics.

**Suggested disposition:** Refactor.

### A02 — Required neural heads are registered by training rather than the model

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Architecture

**Location:** `src/multi_modal/mm.py`, `init_unimodal_stitcher`.

**Finding:** Unimodal output decoders are stored in an ordinary dictionary and moved to an automatically chosen device during construction. A directly constructed model omits these parameters from `parameters()`, `state_dict()`, and normal device movement. The training interface explicitly compensates through `trainer.runtime.register_runtime_modules`; this protects that workflow but leaves standalone consumers dependent on downstream repair.

**Expected:** Spec §37 and architecture §28 require the model to own all learnable encoding components through normal registration.

**Suggested disposition:** Refactor.

### A03 — Singleton sessions lose their contextual embedding

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:** `src/multi_modal/encoder_embeddings.py`, `EncoderEmbeddingLayer.forward`.

**Finding:** Session indices are squeezed, and session embedding addition runs only when `mask.dim() > 0`. A session represented once produces a scalar index and receives no session embedding. Its representation therefore changes with batch composition. Training documents an unsupervised duplicate-sample workaround; the direct model path remains incorrect.

**Expected:** Spec §35 and architecture §10 require consistent session context for every sample, including singleton sessions.

**Suggested disposition:** Refactor.

### A04 — Session embedding identity comes from unrelated split files

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/multi_modal/encoder_embeddings.py`, module initialization and `EncoderEmbeddingLayer.__init__`.

**Finding:** Importing the module reads `data/train_eids.txt` and `data/test_eids.txt`. Their concatenated order defines session embedding rows, independently of the explicit `eid_list` used for projections. Missing files prevent import; a configured session absent from those files fails when its embedding is accessed. Reordering files can reinterpret embedding rows without changing parameter shapes. The singleton omission in A03 can also bypass this lookup failure.

**Expected:** Spec §§16, 24, 35, 38 and architecture §29 require explicit, recoverable session identity and compatible adaptation without unvalidated external ordering.

**Suggested disposition:** Refactor.

### A05 — Prediction requires targets and legacy masking execution

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Interface

**Location:** `src/multi_modal/mm.py`, `forward`, `forward_unimodal_output`, `forward_loss`; `src/multi_modal/encoder_embeddings.py`, `EncoderEmbedding.forward`.

**Finding:** Forward iterates over both inputs and targets, requires `eval_mask` and `training_mode`, obtains session identity from `mod_dict["spike"]`, and always computes loss before returning predictions. Encoding cannot run from visual inputs, validity, positions, and explicit session identity alone. If `eval_mask=None`, it invokes `self.masker`, which is absent when `force_active=False`; when force-active masking is enabled, evaluation can still sample stochastic masks. Explicit masks supplied by training avoid this stochastic path, but ordinary `eval()` is insufficient.

**Expected:** Spec §§26–30, 36 and architecture §§22, 26–27 require independently interpretable prediction, optional convenience loss, optional training corruption, and deterministic inference.

**Suggested disposition:** Refactor.

### A06 — Neural validity is inferred from target values

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:** `src/multi_modal/mm.py`, `forward_loss`; `src/models/stitcher.py`, `StitchDecoder.__init__`.

**Finding:** Neural loss identifies channel padding using `targets != -1` rather than an explicit neuron mask. Every session head emits the largest configured population width; the model does not return a neuron-validity representation. Training supplies explicit validity externally and computes its own eligible objective, but the model convenience loss depends on sentinel-filled targets. The final neural `LayerNorm` also includes the padded output channels, so they participate in normalization of real channels.

**Expected:** Spec §§18, 21–23, 41 require explicit neural validity, valid zero-count observations, and session-correct interpretation of padded output widths.

**Suggested disposition:** Refactor.

### A07 — Architectural configuration does not fully control supported inputs or context

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:** `src/models/stitcher.py`, `OUTPUT_DIM`, `StitchEncoder.forward`; `src/multi_modal/encoder_embeddings.py`, `EncoderEmbeddingLayer.forward`, `EncoderLayer.__init__`; `src/multi_modal/mm.py`, `forward_encoder`, `forward_unimodal_output`; `src/multi_modal/mm_utils.py`, `Attention`.

**Finding:** The stitched visual width is fixed at 768. Embedding expansion and stitcher allocation use `max_F` rather than the actual input length, and unimodal predictions are reshaped to that fixed time length. Shorter inputs are not consistently supported or explicitly rejected. RoPE tables use `Attention` defaults (100 positions times two modalities), independently of configured model length. Configured context is unused because construction/application is commented out, and attention always uses `is_causal=False`.

**Expected:** Spec §§5, 13, 31–34 require configurable feature width, explicit supported lengths, preserved temporal ordering, and effective temporal-context configuration.

**Suggested disposition:** Refactor.

### A08 — Obsolete helpers and unused projections obscure the active model

**Priority:** Medium  
**Confidence:** Likely  
**Category:** Dead code

**Location:** `src/multi_modal/mm_utils.py`, `FactorsProjection`, `CrossAttention`, `create_context_mask`; `src/models/stitcher.py`, `StitchEncoder.__init__` and `forward`.

**Finding:** Repository caller searches found no active use of `FactorsProjection` or `CrossAttention`; `create_context_mask` is imported only for commented code. The visual stitch encoder additionally allocates a trainable `stitcher_dict` linear layer per session that its visual forward branch bypasses. These are unnecessary for the accepted encoding path. Serialized whole-model compatibility has not been established for the unused helper classes, so they are possibly obsolete rather than confirmed safe to delete.

**Expected:** Optional inherited functionality should remain distinguishable from required encoding; unused definitions and parameters should not obscure the active component or checkpoint meaning.

**Suggested disposition:** Refactor; remove confirmed unused visual projections, and remove possibly obsolete helpers only after checking checkpoint compatibility.

## Conforming areas

- Registered visual session projections dispatch by explicit EID and restore original batch ordering; the shared residual transformer remains reusable across sessions.
- Encoding constructs visual encoder tokens without neural placeholder tokens in the shared temporal sequence.
- Neural convenience loss explicitly uses `PoissonNLLLoss(log_input=True)`; valid zero-spike targets are included by its current sentinel rule.
- Attention and embedding dropout follow framework train/eval semantics. The inference concern in A05 comes from masking, not dropout.
- Unknown stitcher session keys fail rather than silently borrowing another session's mapping.
- The model performs no acquisition, alignment, resampling, dataset splitting, optimization, or final scientific metric calculation. No component dependency cycle was identified.
