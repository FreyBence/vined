"""Training-owned model construction and identity-aware pretrained adaptation."""

import copy
import logging

import torch
from torch import nn

from multi_modal.encoder_embeddings import EncoderEmbedding
from multi_modal.mm import MultiModal
from trainer.artifacts import load_training_checkpoint, plain
from trainer.runtime import register_runtime_modules
from utils.config_utils import DictConfig


SESSION_LAYERS = ("stitcher_dict", "project_dict", "stitch_decoder_dict")


def build_model(*, config, metadata, modal_filter):
    context = config.training.get("temporal_context")
    if context is not None and (context.get("target_support_bins", 0) < 12
                                or context.get("boundary_policy") != "complete"):
        raise ValueError("Checkpoint/runtime context requires complete target support of at least 12 bins")
    hidden = config.model.encoder.transformer.hidden_size
    embeddings = {mod: EncoderEmbedding(
        hidden_size=hidden, n_channel=hidden, output_channel=hidden,
        stitching=True, eid_list=metadata["eid_list"], mod=mod,
        config=config.model.encoder, max_F=config.data.max_time_length,
    ) for mod in modal_filter["input"]}
    model = MultiModal(embeddings, avail_mod=["spike", "vision-clip"], avail_beh=["vision-clip"],
                       model_mode=config.training.objective, config=config.model,
                       **({"context_mode": config.training.temporal_context.mode,
                           "context_bins": config.training.temporal_context.bins,
                           "target_delay_bins": context.get("target_delay_bins", 0),
                           "target_support_bins": context.get("target_support_bins", 12)}
                          if "temporal_context" in config.training else {}),
                       **metadata)
    # Bind the existing embedding operation to the selected explicit sessions.
    # No dependence on positions in the inherited global EID files remains.
    sessions = list(metadata["selected_session_ids"])
    for embedding in model.encoder_embeddings.values():
        embedder = embedding.embedder
        embedder.session_emb = nn.Embedding(len(sessions), hidden)
        embedder.eid_lookup = list(sessions)
        embedder.eid_to_indx = {session: index for index, session in enumerate(sessions)}
    return register_runtime_modules(model).cpu()


def _architecture(config):
    model = copy.deepcopy(plain(config["model"]))
    model.pop("masker", None)  # Runtime corruption can change for a new optimization run.
    model["encoder"]["embedder"].pop("n_channels", None)  # Derived session padding width.
    return model


def load_pretrained_model(path, *, config, metadata, modal_filter):
    """Adapt a provenance-bearing checkpoint; return a fresh trainable target model."""
    if len(metadata["selected_session_ids"]) != 1 or metadata["num_sessions"] != 1:
        raise ValueError("Pretrained adaptation requires exactly one selected target session")
    checkpoint = load_training_checkpoint(path)
    source_config = checkpoint["config"]
    if (_architecture(source_config) != _architecture(config)
            or source_config["training"]["loss_components"] != plain(config.training.loss_components)
            or source_config["training"]["objective"] != config.training.objective
            or source_config["training"].get("temporal_context") != plain(config.training.get("temporal_context"))
            or source_config["training"]["modal_filter"] != plain(modal_filter)):
        raise ValueError("Pretrained architecture, prediction direction, temporal context or model loss configuration is incompatible")
    source_maps = checkpoint["compatibility"].get("session_embedding_ids")
    if source_maps is None:
        raise ValueError("Pretrained checkpoint lacks explicit session embedding identities; historical row order cannot be inferred")
    model = build_model(config=config, metadata=metadata, modal_filter=modal_filter)
    current, source = model.state_dict(), checkpoint["model"]
    def is_session(key):
        return any(part in key.split(".") for part in SESSION_LAYERS) or key.endswith(".session_emb.weight")
    shared = {key: value for key, value in source.items() if not is_session(key)}
    expected = {key for key in current if not is_session(key)}
    if set(shared) != expected or any(current[key].shape != value.shape or current[key].dtype != value.dtype
                                       for key, value in shared.items()):
        raise ValueError("Pretrained shared parameter names, shapes or dtypes are incompatible")
    current.update(shared)
    report = {}
    source_populations = checkpoint["compatibility"]["populations"]
    for session in metadata["selected_session_ids"]:
        source_population = source_populations.get(session)
        target_rows = plain(config.dataset.ordered_units[session])
        population_matches = (source_population is not None and source_population["rows"] == target_rows)
        keys = [key for key in current if any(f".{layer}.{session}." in key for layer in SESSION_LAYERS)]
        retained, initialized = [], []
        # Retain each complete session module only when its entire layout agrees.
        groups = {}
        for key in keys:
            group = next(key.split(f".{layer}.{session}.")[0] + f".{layer}.{session}"
                         for layer in SESSION_LAYERS if f".{layer}.{session}." in key)
            groups.setdefault(group, []).append(key)
        for group, group_keys in groups.items():
            compatible = population_matches and all(key in source and source[key].shape == current[key].shape
                                                     and source[key].dtype == current[key].dtype for key in group_keys)
            if compatible:
                for key in group_keys:
                    current[key] = source[key]
                retained.append(group)
            else:
                initialized.append(dict(module=group, reason="session or ordered population differs"
                                        if not population_matches else "runtime parameter layout differs"))
        for mod, embedding in model.encoder_embeddings.items():
            key = f"encoder_embeddings.{mod}.embedder.session_emb.weight"
            mapping = source_maps.get(mod, [])
            if len(set(mapping)) != len(mapping) or key not in source or source[key].shape[0] != len(mapping):
                raise ValueError("Pretrained session embedding table/mapping is malformed")
            if source[key].shape[1:] != current[key].shape[1:] or source[key].dtype != current[key].dtype:
                raise ValueError("Pretrained session embedding dimensions are incompatible")
            if population_matches and session in mapping:
                current[key][embedding.embedder.eid_to_indx[session]] = source[key][mapping.index(session)]
                retained.append(f"{key}/{session}")
            else:
                initialized.append(dict(module=f"{key}/{session}", reason="session or ordered population differs"))
        report[session] = dict(retained=retained, initialized=initialized)
    if any(not torch.isfinite(value).all() for value in current.values() if value.is_floating_point()):
        raise ValueError("Non-finite pretrained parameters")
    model.load_state_dict(current, strict=True)
    model.adaptation_report = report
    logging.info("Pretrained session adaptation: %s", report)
    return model


def model_from_checkpoint(path):
    """Restore inference weights and explicit session mappings without dataset loaders."""
    checkpoint = load_training_checkpoint(path)
    config = DictConfig(checkpoint["config"])
    mappings = checkpoint["compatibility"].get("session_embedding_ids")
    if mappings is None:
        raise ValueError("Checkpoint lacks explicit model session embedding identities")
    model = build_model(config=config, metadata=dict(config.dataset),
                        modal_filter=dict(config.training.modal_filter))
    if set(mappings) != set(model.encoder_embeddings):
        raise ValueError("Checkpoint session embedding modalities are incompatible")
    for mod, embedding in model.encoder_embeddings.items():
        sessions = mappings[mod]
        if not sessions or len(set(sessions)) != len(sessions):
            raise ValueError("Malformed checkpoint session embedding mapping")
        embedder = embedding.embedder
        embedder.session_emb = nn.Embedding(len(sessions), config.model.encoder.transformer.hidden_size)
        embedder.eid_lookup = list(sessions)
        embedder.eid_to_indx = {session: index for index, session in enumerate(sessions)}
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def initialize_regional_encoder(model, path, *, config):
    """Validate the complete source, transfer every non-head tensor, verify copies."""
    if model.target_granularity != "region" or model.model_mode != "encoding":
        raise ValueError("Encoder-only initialization requires a regional encoding model")
    checkpoint = load_training_checkpoint(path)
    source_config = checkpoint["config"]
    for field in ("dataset_generation_id", "selected_session_ids", "dataset_metadata"):
        if source_config["dataset"].get(field) != plain(config.dataset.get(field)):
            raise ValueError(f"Encoder initialization dataset provenance differs: {field}")
    if (_architecture(source_config) != _architecture(config)
            or source_config["training"]["objective"] != "encoding"
            or source_config["training"].get("temporal_context") != plain(config.training.temporal_context)):
        raise ValueError("Source encoder architecture/direction/context differs")
    # Strict restoration checks all source keys and the source model identity.
    source_model = model_from_checkpoint(path)
    if source_model.target_granularity != "neuron":
        raise ValueError("Expected a neuron-level source checkpoint")
    for session, mapping in config.dataset.regional_targets.items():
        source_units = source_config["dataset"]["ordered_units"].get(session)
        if source_units != plain(mapping["ordered_units"]):
            raise ValueError("Regional source population differs from source checkpoint")
    head_prefix = "mod_stitcher_proj_dict."
    source, current = source_model.state_dict(), model.state_dict()
    transfer = {k for k in current if not k.startswith(head_prefix)}
    if transfer != {k for k in source if not k.startswith(head_prefix)}:
        raise ValueError("Source and target encoder parameter sets differ")
    for key in transfer:
        if current[key].shape != source[key].shape or current[key].dtype != source[key].dtype:
            raise ValueError(f"Incompatible encoder tensor: {key}")
        if not torch.isfinite(source[key]).all():
            raise ValueError(f"Non-finite encoder tensor: {key}")
        current[key] = source[key].clone()
    model.load_state_dict(current, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(loaded[k], source[k]) for k in transfer):
        raise ValueError("Encoder transfer verification failed")
    model.adaptation_report = dict(
        transferred=sorted(transfer),
        newly_initialized=sorted(k for k in current if k.startswith(head_prefix)),
        excluded_source=sorted(k for k in source if k.startswith(head_prefix)),
        verified_tensors=len(transfer), optimizer="fresh AdamW", scheduler="fresh")
    return model


def initialize_regional_bias(model, metadata):
    """Initialize log-count intercepts from eligible optimization targets only."""
    if model.target_granularity != "region":
        return
    with torch.no_grad():
        for session, stats in metadata["regional_training_baseline"].items():
            head = model.mod_stitcher_proj_dict["spike"].stitch_decoder_dict[session]
            head.bias.copy_(torch.tensor(stats["mean_counts"], dtype=head.bias.dtype).clamp_min(1e-8).log())
