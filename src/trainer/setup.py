"""Shared configuration and verified dataset selection for training entry points."""

import math
from pathlib import Path

from loader.make_loader import make_loader
from training_dataset.handoff import load_dataset_splits
from training_dataset import TARGET_SUPPORT_BINS
from utils.config_utils import DictConfig, load_config, update_config
from utils.utils import set_seed
from trainer.selection import select_neural_regions


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
SEARCH_FIELDS = {
    "learning_rate": ("optimizer", "lr"),
    "weight_decay": ("optimizer", "wd"),
    "mask_ratio": ("model", "masker", "ratio"),
    "hidden_size": ("model", "encoder", "transformer", "hidden_size"),
    "inter_size": ("model", "encoder", "transformer", "inter_size"),
    "n_layers": ("model", "encoder", "transformer", "n_layers"),
}
SUPPORTED_OVERRIDES = {
    "seed", "wandb.use", "wandb.entity", "wandb.project",
    "training.objective", "training.mixed_training", "training.enc_task_var",
    "training.num_epochs", "training.train_batch_size", "training.test_batch_size",
    "training.eval_every", "training.save_every", "training.save_plot_every_n_epochs",
    "training.checkpoint_selection", "training.selection_metric", "training.selection_direction",
    "training.validation_metrics", "training.loss_components.spike.name",
    "training.loss_components.spike.weight", "training.loss_components.vision-clip.name",
    "training.loss_components.vision-clip.weight",
    "optimizer.name", "optimizer.lr", "optimizer.wd", "optimizer.eps", "optimizer.scheduler",
    "optimizer.warmup_pct", "optimizer.div_factor", "optimizer.gradient_accumulation_steps",
    "data.load_meta",
}


def add_setup_arguments(parser):
    parser.add_argument("--neural-region-selection", choices=["all_recorded", "visual_only"], default="all_recorded")
    parser.add_argument("--context-mode", choices=["strict", "full_trial"],
                        help="Temporal context for encoding/decoding (default: full_trial)")
    parser.add_argument("--context-bins", type=int, choices=[1, 3, 6, 9, 12],
                        help="Required for strict context; omitted for full_trial")
    parser.add_argument("--dataset-generation", help="Explicit published dataset directory; alternatively use --data_path")
    parser.add_argument("--expected-dataset-generation-id")
    parser.add_argument("--training-config", help="Training JSON override file")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--scheduler", choices=["linear", "cosine", "none"])
    parser.add_argument("--eval-every", type=int, help="0 disables validation")
    parser.add_argument("--checkpoint-selection", choices=["validation_metric", "validation_loss", "final"])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume-checkpoint", help="Continue a compatible epoch-boundary checkpoint in a new artifact directory")
    parser.add_argument("--setup-only", action="store_true", help="Verify selection, resolve config and construct loaders without optimization")


def _set(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")


def _finite(value, name, minimum=0, *, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"Invalid {name}: {value}")


def _known_overrides(base, override, prefix=""):
    if not isinstance(override, dict):
        raise ValueError("Training configuration must be a mapping")
    for key, value in override.items():
        name = f"{prefix}{key}"
        if key not in base:
            raise ValueError(f"Unsupported configuration field: {name}")
        if isinstance(base[key], dict):
            _known_overrides(base[key], value, name + ".")
        elif name in ("model.encoder.embedder.max_F", "model.encoder.embedder.n_channels"):
            raise ValueError(f"{name} is derived from the selected dataset/runtime layout")
        elif name not in SUPPORTED_OVERRIDES and not name.startswith(("model.masker.", "model.encoder.transformer.", "model.encoder.embedder.")):
            raise ValueError(f"Unsupported configuration override: {name}")


def load_defaults(config_root, session_count):
    """Compose module-owned JSON profiles without dataset-derived dimensions."""
    config_root = Path(config_root)
    model_name = ("single_session" if session_count == 1 else "medium" if 10 < session_count < 70
                  else "large" if session_count >= 70 else "default")
    training_name = "default" if session_count <= 40 else "multi_session"
    config = load_config(config_root / "training" / (training_name + ".json"))
    model = load_config(config_root / "model" / (model_name + ".json"))
    if ("temporal_context" in config.get("training", {})
            or any(key in model for key in ("temporal_context", "context_mode", "context_bins"))):
        raise ValueError("Temporal context must be selected through CLI arguments, not JSON profiles")
    model["masker"] = config.pop("masking")
    config["model"] = model
    return DictConfig(config)


def _training_override(override):
    """Accept training-owned masking settings at the runtime model boundary."""
    if "masking" in override:
        masking = override.pop("masking")
        model = override.setdefault("model", {})
        if not isinstance(model, dict) or "masker" in model:
            raise ValueError("Specify masking once, using masking or model.masker")
        model["masker"] = masking
    return override


def resolve_setup(args, tune_config=None):
    """Return effective configuration, train/val loaders, and generation metadata."""
    generation_path = args.dataset_generation or args.data_path
    session_ids = None if args.eid in (None, "None") else [args.eid]
    train, val, _test, metadata = load_dataset_splits(
        generation_path, session_ids=session_ids,
        expected_generation_id=args.expected_dataset_generation_id,
    )
    train, val, _test, metadata = select_neural_regions(
        (train, val, _test), metadata, getattr(args, "neural_region_selection", "all_recorded"))
    if not train:
        raise ValueError("Selected dataset has no optimization samples")
    if args.num_sessions is not None and args.num_sessions != metadata["num_sessions"]:
        raise ValueError("--num_sessions must match the explicitly selected dataset sessions")
    count = metadata["num_sessions"]
    config_root = Path(args.config_dir).resolve()
    config = load_defaults(config_root, count)
    if args.training_config:
        override = _training_override(load_config(args.training_config))
        _known_overrides(config, override)
        config = update_config(config, override)
    cli_fields = {
        "seed": ("seed",), "epochs": ("training", "num_epochs"),
        "batch_size": ("training", "train_batch_size"),
        "validation_batch_size": ("training", "test_batch_size"),
        "learning_rate": ("optimizer", "lr"), "weight_decay": ("optimizer", "wd"),
        "scheduler": ("optimizer", "scheduler"), "eval_every": ("training", "eval_every"),
        "checkpoint_selection": ("training", "checkpoint_selection"),
        "model_mode": ("training", "objective"), "mask_mode": ("model", "masker", "mode"),
        "mask_ratio": ("model", "masker", "ratio"), "enc_task_var": ("training", "enc_task_var"),
    }
    for name, path in cli_fields.items():
        value = getattr(args, name, None)
        if value is not None:
            _set(config, path, value)
    if args.mixed_training:
        config["training"]["mixed_training"] = True
    for name, value in (tune_config or {}).items():
        if name not in SEARCH_FIELDS:
            raise ValueError(f"Unsupported search override: {name}")
        _set(config, SEARCH_FIELDS[name], value)
    if args.search or args.no_wandb:
        config["wandb"]["use"] = False

    training, optimizer, model, data = (config[key] for key in ("training", "optimizer", "model", "data"))
    objective = training["objective"]
    if objective not in ("encoding", "decoding", "mm"):
        raise ValueError(f"Unsupported objective: {objective}")
    mode = getattr(args, "context_mode", None)
    bins = getattr(args, "context_bins", None)
    if objective == "mm":
        if mode is not None or bins is not None:
            raise ValueError("Explicit temporal context requires encoding or decoding; mm retains its legacy path")
    else:
        mode = mode or "full_trial"
        if mode == "strict":
            if type(bins) is not int or bins not in (1, 3, 6, 9, 12):
                raise ValueError("Strict context requires --context-bins 1|3|6|9|12")
        elif mode != "full_trial" or bins is not None:
            raise ValueError("Full-trial context requires omission of --context-bins")
        if any(model.get("context", {}).get(key, -1) != -1 for key in ("forward", "backward")):
            raise ValueError("CLI context conflicts with finite legacy attention limits")
        training["temporal_context"] = dict(mode=mode, bins=bins, target_support_bins=TARGET_SUPPORT_BINS,
                                           boundary_policy="complete", source="entry_arguments")
    if args.modality not in (["ap", "vision-clip"], ["spike", "vision-clip"]):
        raise ValueError("Supported modalities are spike (or ap) and vision-clip, in that order")
    if training["mixed_training"] and objective != "mm":
        raise ValueError("Mixed training is supported only for the mm objective")
    if training["enc_task_var"] not in ("all", "random", "vision-clip"):
        raise ValueError("Unsupported encoding task variable")
    if getattr(args, "continue_pretrain", False):
        raise ValueError("Legacy implicit resume is unsupported; use --resume-checkpoint PATH")
    if getattr(args, "resume_checkpoint", None) and getattr(args, "pretrained_checkpoint", None):
        raise ValueError("Choose resume or pretrained adaptation, not both")
    if getattr(args, "resume_checkpoint", None) and args.search:
        raise ValueError("Resume is not supported inside a hyperparameter search")
    if getattr(args, "overwrite", False):
        raise ValueError("Runs use isolated directories; --overwrite is unsupported")
    _positive(config["seed"], "seed", zero=True)
    if config["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    for key in ("num_epochs", "train_batch_size", "test_batch_size", "save_every"):
        _positive(training[key], key)
    _positive(training["save_plot_every_n_epochs"], "save_plot_every_n_epochs", zero=True)
    _positive(training["eval_every"], "eval_every", zero=True)
    _positive(optimizer["gradient_accumulation_steps"], "gradient_accumulation_steps")
    if optimizer["name"] != "adamw":
        raise ValueError("Only the AdamW optimizer is supported")
    for key in ("lr", "eps", "div_factor"):
        _finite(optimizer[key], key, strict=True)
    _finite(optimizer["wd"], "weight decay")
    if optimizer["scheduler"] not in ("linear", "cosine", "none"):
        raise ValueError("Unsupported scheduler")
    _finite(optimizer["warmup_pct"], "warmup_pct", strict=True)
    if optimizer["warmup_pct"] >= 1:
        raise ValueError("warmup_pct must be less than one")
    if model["model_class"] != "MultiModal" or not model["masker"]["force_active"]:
        raise ValueError("Only MultiModal with explicit active masking is supported")
    if model["masker"]["mode"] not in ("temporal", "causal"):
        raise ValueError("Unsupported masking mode")
    for key in ("ratio", "zero_ratio", "random_ratio", "expand_prob"):
        _finite(model["masker"][key], key)
        if model["masker"][key] > 1:
            raise ValueError(f"{key} must be in [0,1]")
    transformer = model["encoder"]["transformer"]
    for key in ("hidden_size", "inter_size", "n_layers", "n_heads"):
        _positive(transformer[key], key)
    if transformer["hidden_size"] % transformer["n_heads"]:
        raise ValueError("hidden_size must be divisible by n_heads")
    for settings in (transformer, model["encoder"]["embedder"]):
        _finite(settings["dropout"], "dropout")
        if settings["dropout"] >= 1:
            raise ValueError("dropout must be less than one")
    for value in (training["mixed_training"], data["load_meta"], config["wandb"]["use"]):
        if not isinstance(value, bool):
            raise ValueError("mixed_training, load_meta, and wandb.use must be booleans")
    components = training["loss_components"]
    if components != {"spike": {"name": "poisson_nll", "weight": 1.0},
                      "vision-clip": {"name": "cosine", "weight": 1.0}}:
        raise ValueError("Only the retained model losses with unit weights are currently supported")
    inputs, outputs = {"encoding": (["vision-clip"], ["spike"]),
                       "decoding": (["spike"], ["vision-clip"]),
                       "mm": (["spike", "vision-clip"], ["spike", "vision-clip"])}[objective]
    training["active_loss_components"] = {key: components[key] for key in outputs}
    training["training_schemes"] = (["mixed"] if training["mixed_training"] else
                                   ["encoding", "decoding", "self-spike", "self-vision", "random_token"]
                                   if objective == "mm" else [objective])
    required_metrics = ["bps"] if outputs == ["spike"] else ["cosine_similarity"] if outputs == ["vision-clip"] else ["bps", "cosine_similarity"]
    if training["validation_metrics"] != ["bps", "cosine_similarity"] and training["validation_metrics"] != required_metrics:
        raise ValueError("Unsupported validation metrics for this objective")
    training["validation_metrics"] = required_metrics
    if training["checkpoint_selection"] not in ("validation_metric", "validation_loss", "final"):
        raise ValueError("Unsupported checkpoint selection")
    if training["selection_metric"] != "eval_avg_metric" or training["selection_direction"] != "max":
        raise ValueError("Only eval_avg_metric maximization is supported for validation_metric selection")
    if training["eval_every"] and not val:
        raise ValueError("Validation is configured but the selected validation split is empty")
    if not training["eval_every"] and training["checkpoint_selection"] != "final":
        raise ValueError("Validation-based checkpoint selection requires validation")
    if args.search and (not training["eval_every"] or training["checkpoint_selection"] != "validation_metric"):
        raise ValueError("Retained Ray search requires validation and validation_metric checkpoint selection")
    training["selection"] = ({"metric": "eval_loss", "direction": "min"}
                             if training["checkpoint_selection"] == "validation_loss" else
                             {"metric": "eval_avg_metric", "direction": "max"}
                             if training["checkpoint_selection"] == "validation_metric" else
                             {"metric": None, "direction": None})

    consumed = tuple(train.samples) + (tuple(val.samples) if training["eval_every"] else ())
    if objective != "mm":
        if any(sample.metadata["alignment"].get("visual_resampling") !=
               "timestamp-aware held-state over neural intervals v1" for sample in consumed):
            raise ValueError("Temporal-context runs require held-state aligned data; rebuild interpolated generations")
        coverage = {}
        for name, view in (("train", train), ("val", val if training["eval_every"] else None)):
            if view is None:
                continue
            eligible = sum(max(0, sample.sequence_length - TARGET_SUPPORT_BINS + 1) for sample in view.samples)
            if not eligible:
                raise ValueError(f"No complete H=12 targets in {name} split")
            coverage[name] = dict(eligible_temporal_targets=eligible,
                                 excluded_temporal_targets=sum(sample.sequence_length for sample in view.samples) - eligible,
                                 trials_without_targets=sum(sample.sequence_length < TARGET_SUPPORT_BINS for sample in view.samples))
        training["temporal_target_coverage"] = coverage
    if any(sample.visual.shape[1] != 768 for sample in consumed):
        raise ValueError("The current model requires 768-dimensional visual features")
    # Include held-out trial lengths so checkpoint capacity preserves every
    # selected trial; this does not evaluate held-out targets.
    data["max_time_length"] = max(sample.sequence_length
                                  for view in (train, val, _test) for sample in view.samples)
    optimization_sessions = {sample.session_id for sample in train.samples}
    if any(sample.session_id not in optimization_sessions for sample in consumed):
        raise ValueError("Validation sessions must have session-specific parameters learned from training")
    data.update(max_space_length=max(metadata["eid_list"].values()), dataset_name="ibl")
    model["encoder"]["embedder"]["max_F"] = data["max_time_length"]
    config["dataset"] = dict(metadata)
    config["dataset"]["ordered_units"] = {
        session: next(sample.neuron_identity.to_dict(orient="records")
                      for sample in (*train.samples, *val.samples, *_test.samples)
                      if sample.session_id == session)
        for session in metadata["selected_session_ids"]
    }
    set_seed(config["seed"])
    loader_kwargs = dict(target=["vision-clip"], load_meta=data["load_meta"],
                         pad_to_right=True, pad_value=-1., max_time_length=data["max_time_length"],
                         max_space_length=data["max_space_length"], dataset_name=data["dataset_name"],
                         stitching=True, seed=config["seed"], eids=metadata["eids"])
    train_loader = make_loader(train, batch_size=training["train_batch_size"], mode="train", shuffle=True, **loader_kwargs)
    val_loader = (make_loader(val, batch_size=training["test_batch_size"], mode="val", shuffle=False, **loader_kwargs)
                  if training["eval_every"] else None)
    config["training"]["modal_filter"] = {"input": inputs, "output": outputs}
    return DictConfig(config), train_loader, val_loader, metadata


def setup_summary(config, train_loader, val_loader):
    return {"dataset_generation_id": config.dataset.dataset_generation_id,
            "dataset_path": config.dataset.dataset_path,
            "sessions": config.dataset.selected_session_ids,
            "train_batches": len(train_loader), "validation_batches": len(val_loader) if val_loader else 0,
            "seed": config.seed, "training": dict(config.training),
            "optimizer": dict(config.optimizer), "model": dict(config.model)}
