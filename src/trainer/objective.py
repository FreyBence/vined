"""Scientific validity and runtime masking at the retained model boundary."""

from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F


def validate_batch(batch, *, config, populations, split):
    required = {"spikes_data", "vision-clip", "neural", "visual", "temporal_mask", "neuron_mask",
                "time_attn_mask", "space_attn_mask", "vision-clip_valid", "spikes_timestamps",
                "eid", "session_id", "sample_id", "trial_id", "split", "neuron_identity"}
    if not isinstance(batch, dict) or not required.issubset(batch):
        raise ValueError("Malformed scientific batch: required data, validity, or identity fields are missing")
    spikes, vision = batch["spikes_data"], batch["vision-clip"]
    if not isinstance(spikes, torch.Tensor) or spikes.ndim != 3:
        raise ValueError("spikes_data must be a tensor [B,T,N]")
    b, t, n = spikes.shape
    if b < 1 or t != config.data.max_time_length or n != config.data.max_space_length:
        raise ValueError("Batch dimensions disagree with the configured scientific runtime layout")
    shapes = {"vision-clip": (b,t,768), "visual": (b,t,768), "neural": (b,t,n),
              "temporal_mask": (b,t), "neuron_mask": (b,n), "time_attn_mask": (b,t),
              "space_attn_mask": (b,n), "vision-clip_valid": (b,t), "spikes_timestamps": (b,t)}
    for name, shape in shapes.items():
        if not isinstance(batch[name], torch.Tensor) or tuple(batch[name].shape) != shape:
            raise ValueError(f"Invalid {name} shape; expected {shape}")
    temporal, neurons = batch["temporal_mask"], batch["neuron_mask"]
    if temporal.dtype != torch.bool or neurons.dtype != torch.bool or batch["vision-clip_valid"].dtype != torch.bool:
        raise ValueError("Scientific validity masks must be boolean")
    if (not torch.equal(batch["time_attn_mask"], temporal.to(batch["time_attn_mask"].dtype))
            or not torch.equal(batch["space_attn_mask"], neurons.to(batch["space_attn_mask"].dtype))
            or not torch.equal(batch["vision-clip_valid"], temporal)):
        raise ValueError("Model attention masks disagree with scientific validity")
    # The dataset boundary supports right padding only.
    if (not temporal.any(dim=1).all() or not neurons.any(dim=1).all()
            or (temporal[:,1:] & ~temporal[:,:-1]).any()
            or (neurons[:,1:] & ~neurons[:,:-1]).any()):
        raise ValueError("Validity must contain nonempty real observations followed by right padding")
    if (batch["neural"].dtype != torch.int64 or spikes.dtype != torch.float32
            or vision.dtype != torch.float32 or batch["visual"].dtype != torch.float32):
        raise ValueError("Scientific counts/model counts/features have incompatible dtypes")
    if any(batch[name].dtype != torch.int64 for name in ("time_attn_mask", "space_attn_mask", "spikes_timestamps")):
        raise ValueError("Model attention masks and temporal indices must be int64")
    if not torch.equal(batch["spikes_timestamps"], torch.arange(t, device=spikes.device).expand(b,t)):
        raise ValueError("Model temporal indices disagree with the configured time layout")
    valid = temporal.unsqueeze(-1) & neurons.unsqueeze(1)
    if ((batch["neural"][valid] < 0).any() or not torch.isfinite(spikes[valid]).all()
            or not torch.equal(spikes[valid].double(), batch["neural"][valid].double())
            or not torch.isfinite(vision[temporal]).all()
            or not torch.equal(vision[temporal], batch["visual"][temporal])):
        raise ValueError("Real scientific observations are invalid or disagree with model inputs")
    for name in ("eid", "session_id", "sample_id", "trial_id", "split", "neuron_identity"):
        if len(batch[name]) != b:
            raise ValueError(f"Batch identity field {name} has the wrong length")
    for index, session in enumerate(batch["session_id"]):
        if session != batch["eid"][index] or session not in populations or batch["split"][index] != split:
            raise ValueError("Batch session identity or split is incompatible with this loader")
        units = batch["neuron_identity"][index]
        if (not isinstance(units, pd.DataFrame)
                or not units.reset_index(drop=True).equals(populations[session].reset_index(drop=True))
                or int(neurons[index].sum()) != len(units)):
            raise ValueError("Batch ordered neural population is incompatible with the selected session")
    return valid


def prepare_inputs(batch, *, model, training_mode, enc_task_var=None):
    """Return model arguments and separate scientific/stochastic target selectors."""
    temporal, neurons = batch["temporal_mask"], batch["neuron_mask"]
    validity = {"spike": temporal.unsqueeze(-1) & neurons.unsqueeze(1), "vision-clip": temporal}
    b, t, n = batch["spikes_data"].shape
    ones = torch.ones((b,t,n), dtype=torch.int64, device=temporal.device)
    zeros = torch.zeros_like(ones)
    modes = [training_mode] * b
    if training_mode == "mixed":
        # Resolve the retained per-sample schemes before invoking the model so
        # stochastic selectors and accumulation denominators remain explicit.
        choices = ("encoding", "decoding", "self-spike", "self-vision", "random_token")
        modes = [choices[index] for index in torch.randint(len(choices), (b,)).tolist()]
    data, selectors = {}, {}
    random_mask = None
    if any(mode in ("self-spike", "self-vision", "random_token") for mode in modes):
        _, random_mask = model.masker(torch.zeros_like(batch["spikes_data"]), None)
    for mod, mod_index in model.mod_to_indx.items():
        if mod not in validity:
            raise ValueError(f"Unsupported model modality: {mod}")
        selected = zeros.clone()
        for index, mode in enumerate(modes):
            supervised = (mode == "encoding" and mod == "spike") or (mode == "decoding" and mod == "vision-clip")
            stochastic = mode == "random_token" or (mode == "self-spike" and mod == "spike") or (mode == "self-vision" and mod == "vision-clip")
            if supervised:
                selected[index] = ones[index]
            elif stochastic:
                selected[index] = random_mask[index]
            elif mode not in ("encoding", "decoding", "self-spike", "self-vision", "random_token"):
                raise ValueError(f"Unsupported training scheme: {mode}")
        target_selector = selected[...,0].bool() & temporal
        selectors[mod] = (target_selector.unsqueeze(-1) & validity[mod] if mod == "spike" else target_selector)
        values = batch["spikes_data"] if mod == "spike" else batch["vision-clip"]
        value_valid = validity[mod] if mod == "spike" else temporal.unsqueeze(-1)
        # Numeric fill is a model adapter, never the source of scientific validity.
        inputs = torch.where(value_valid, values, torch.zeros_like(values))
        targets = torch.where(value_valid, values, torch.full_like(values, -1.) if mod == "spike" else torch.zeros_like(values))
        data[mod] = dict(inputs=inputs, targets=targets, eid=list(batch["session_id"]), num_neuron=n,
                         inputs_modality=torch.tensor(mod_index, device=values.device),
                         targets_modality=torch.tensor(mod_index, device=values.device),
                         inputs_attn_mask=temporal.to(torch.int64),
                         inputs_timestamp=batch["spikes_timestamps"], targets_timestamp=batch["spikes_timestamps"],
                         eval_mask=selected, training_mode="random_token")
        if enc_task_var is not None and enc_task_var != "all":
            data[mod]["inputs_token_mask"] = zeros if mod == enc_task_var else ones
    return data, selectors


def forward_objective(model, data, selectors, *, components):
    """Call the unchanged model and recompute loss using explicit validity."""
    sessions = data["spike"]["eid"]
    counts = Counter(sessions)
    indices = list(range(len(sessions))) + [index for index, session in enumerate(sessions) if counts[session] == 1]
    # The inherited session embedder skips scalar singleton indices. Add an
    # unsupervised runtime copy, then retain exactly the original predictions.
    runtime = {}
    for mod, values in data.items():
        runtime[mod] = {key: (value[indices] if isinstance(value, torch.Tensor) and value.ndim > 0 else
                             [sessions[index] for index in indices] if key == "eid" else value)
                        for key, value in values.items()}
    outputs = model(runtime)
    if not torch.isfinite(outputs.loss) or any(not torch.isfinite(value) for value in outputs.mod_loss.values()):
        raise ValueError("Non-finite loss returned by the retained model")
    for mod in outputs.mod_preds:
        predictions = outputs.mod_preds[mod][:len(sessions)]
        targets = outputs.mod_targets[mod][:len(sessions)]
        if predictions.shape != targets.shape or not torch.isfinite(predictions).all():
            raise ValueError(f"Non-finite or dimensionally incompatible {mod} model predictions")
        outputs.mod_preds[mod], outputs.mod_targets[mod] = predictions, targets
    numerators, counts = {}, {}
    for mod in components:
        if mod not in outputs.mod_preds or mod not in selectors:
            raise ValueError(f"Missing required output modality: {mod}")
        predictions, targets, valid = outputs.mod_preds[mod], outputs.mod_targets[mod], selectors[mod]
        expected = predictions.shape if mod == "spike" else predictions.shape[:2]
        if valid.shape != expected:
            raise ValueError(f"Incompatible {mod} target selector")
        counts[mod] = valid.sum()
        if mod == "spike":
            terms = F.poisson_nll_loss(predictions[valid], targets[valid], log_input=True, reduction="none")
        else:
            terms = 1 - (F.normalize(predictions[valid], dim=-1) * F.normalize(targets[valid], dim=-1)).sum(-1)
        numerators[mod] = terms.sum() if terms.numel() else predictions.sum() * 0.
        if not torch.isfinite(numerators[mod]):
            raise ValueError(f"Non-finite {mod} objective")
    outputs.mod_loss = {mod: numerators[mod] / counts[mod].clamp_min(1) for mod in components}
    outputs.mod_n_examples = counts
    outputs.loss = sum(components[mod]["weight"] * outputs.mod_loss[mod] for mod in components)
    if not torch.isfinite(outputs.loss):
        raise ValueError("Non-finite composed objective")
    return outputs, numerators
