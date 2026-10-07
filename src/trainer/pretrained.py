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
    hidden = config.model.encoder.transformer.hidden_size
    embeddings = {mod: EncoderEmbedding(
        hidden_size=hidden, n_channel=hidden, output_channel=hidden,
        stitching=True, eid_list=metadata["eid_list"], mod=mod,
        config=config.model.encoder, max_F=config.data.max_time_length,
    ) for mod in modal_filter["input"]}
    model = MultiModal(embeddings, avail_mod=["spike", "vision-clip"], avail_beh=["vision-clip"],
                       model_mode=config.training.objective, config=config.model,
                       **config.method.model_kwargs, **metadata)
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
            or source_config["method"]["model_kwargs"] != plain(config.method.model_kwargs)
            or source_config["training"]["objective"] != config.training.objective
            or source_config["training"]["modal_filter"] != plain(modal_filter)):
        raise ValueError("Pretrained architecture, prediction direction or model loss configuration is incompatible")
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
