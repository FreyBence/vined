"""Training run identity, provenance and epoch-boundary continuation."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import uuid

import numpy as np
import torch
import torch.distributed as distributed


FORMAT_VERSION = 1


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported run metadata type: {type(value).__name__}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def new_run_directory(base_path, accelerator):
    identity = [None, None]
    if accelerator.is_main_process:
        identity[0] = uuid.uuid4().hex
        try:
            (Path(base_path).resolve() / "runs" / identity[0]).mkdir(parents=True, exist_ok=False)
        except OSError as error:
            identity[1] = str(error)
    if accelerator.num_processes > 1:
        distributed.broadcast_object_list(identity, src=0)
    if identity[1] is not None:
        raise ValueError(f"Cannot create isolated run directory: {identity[1]}")
    return str(Path(base_path).resolve() / "runs" / identity[0])


def load_training_checkpoint(path):
    """Load a trusted, versioned training checkpoint; legacy weights cannot resume."""
    checkpoint = torch.load(path, map_location="cpu")
    required = {"format_version", "run", "config", "compatibility", "model", "optimizer",
                "lr_sched", "epoch", "next_epoch", "optimizer_step", "scheduler_step",
                "selection", "history", "rank_states"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint lacks the training state/provenance required for resume")
    if checkpoint["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported training checkpoint format")
    if checkpoint["next_epoch"] != checkpoint["epoch"] + 1:
        raise ValueError("Only complete epoch-boundary checkpoints can resume")
    return checkpoint


def compatibility(trainer, kind):
    config = plain(trainer.config)
    # Location and logging do not change the optimization trajectory.
    config.pop("dirs", None)
    config.pop("wandb", None)
    config["dataset"].pop("dataset_path", None)
    config["training"].pop("log_dir", None)
    populations = {session: {"columns": list(table.columns),
                             "dtypes": [str(dtype) for dtype in table.dtypes],
                             "rows": plain(table.to_dict(orient="records"))}
                   for session, table in trainer.populations.items()}
    views = {name: [{"sample_id": sample.sample_id, "session_id": sample.session_id,
                     "trial_id": sample.trial_id, "split": sample.split}
                    for sample in loader.dataset.scientific_split.samples]
             for name, loader in (("train", trainer.train_dataloader), ("val", trainer.eval_dataloader))
             if loader is not None}
    return dict(kind=kind, config=config, populations=populations, memberships=views,
                session_embedding_ids={mod: list(embedding.embedder.eid_lookup)
                                       for mod, embedding in trainer.accelerator.unwrap_model(trainer.model).encoder_embeddings.items()},
                world_size=trainer.accelerator.num_processes,
                device_type=trainer.accelerator.device.type,
                mixed_precision=trainer.accelerator.mixed_precision)


def _generators(loader):
    if loader is None:
        return {}
    candidates = {"loader": getattr(loader, "generator", None),
                  "sampler": getattr(loader.sampler, "generator", None),
                  "batch_sampler": getattr(getattr(loader.batch_sampler, "sampler", None), "generator", None)}
    return {name: generator for name, generator in candidates.items() if generator is not None}


def capture_rank_states(trainer):
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if trainer.accelerator.device.type == "cuda" else [],
                 generators={name: {key: generator.get_state() for key, generator in _generators(loader).items()}
                             for name, loader in (("train", trainer.train_dataloader), ("val", trainer.eval_dataloader))},
                 loader_epoch=getattr(trainer.train_dataloader, "iteration", None))
    if trainer.accelerator.num_processes == 1:
        trainer.rank_states = [state]
    else:
        states = [None] * trainer.accelerator.num_processes
        distributed.all_gather_object(states, state)
        trainer.rank_states = states


def initialize_run(trainer, *, kind, resume_checkpoint=None, pretrained_checkpoint=None):
    error = None
    try:
        _initialize_run(trainer, kind=kind, resume_checkpoint=resume_checkpoint,
                        pretrained_checkpoint=pretrained_checkpoint)
    except Exception as caught:
        error = caught
    trainer._check_all_ranks(error)


def _initialize_run(trainer, *, kind, resume_checkpoint=None, pretrained_checkpoint=None):
    trainer.history = []
    trainer.selected_checkpoint = None
    trainer.run_compatibility = compatibility(trainer, kind)
    attempt_id = Path(trainer.log_dir).name
    trainer.run = dict(run_id=attempt_id, attempt_id=attempt_id, kind=kind, parent=None)
    if resume_checkpoint:
        checkpoint = load_training_checkpoint(resume_checkpoint)
        if checkpoint["compatibility"] != trainer.run_compatibility:
            differences = [key for key in trainer.run_compatibility
                           if checkpoint["compatibility"].get(key) != trainer.run_compatibility[key]]
            raise ValueError(f"Incompatible resume checkpoint: {', '.join(differences)}")
        if not 0 <= checkpoint["next_epoch"] < trainer.config.training.num_epochs:
            raise ValueError("Checkpoint has no remaining epochs in the configured duration")
        if checkpoint["lr_sched"].get("last_epoch") != checkpoint["scheduler_step"]:
            raise ValueError("Checkpoint scheduler state does not match the saved step")
        if checkpoint["optimizer_step"] != checkpoint["scheduler_step"]:
            raise ValueError("Checkpoint optimizer/scheduler steps disagree")
        model = trainer.accelerator.unwrap_model(trainer.model)
        current = model.state_dict()
        if set(current) != set(checkpoint["model"]) or any(
                (current[key].shape != value.shape or current[key].dtype != value.dtype) for key, value in checkpoint["model"].items()):
            raise ValueError("Resume model parameter names/shapes are incompatible")
        model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.lr_scheduler.load_state_dict(checkpoint["lr_sched"])
        trainer.start_epoch = checkpoint["next_epoch"]
        trainer.optimizer_steps = checkpoint["optimizer_step"]
        trainer.scheduler_steps = checkpoint["scheduler_step"]
        trainer.history = copy.deepcopy(checkpoint["history"])
        trainer.selected_checkpoint = copy.deepcopy(checkpoint["selection"])
        if "adaptation" in checkpoint["run"]:
            trainer.run["adaptation"] = copy.deepcopy(checkpoint["run"]["adaptation"])
        if "pretrained" in checkpoint["run"]:
            trainer.run["pretrained"] = copy.deepcopy(checkpoint["run"]["pretrained"])
        trainer.run.update(run_id=checkpoint["run"]["run_id"],
                           parent=dict(attempt_id=checkpoint["run"]["attempt_id"],
                                       checkpoint=str(Path(resume_checkpoint).resolve()), epoch=checkpoint["epoch"]))
        if len(checkpoint["rank_states"]) != trainer.accelerator.num_processes:
            raise ValueError("Checkpoint random-state rank count is incompatible")
        state = checkpoint["rank_states"][trainer.accelerator.process_index]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if trainer.accelerator.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda"])
        for name, loader in (("train", trainer.train_dataloader), ("val", trainer.eval_dataloader)):
            generators = _generators(loader)
            if set(generators) != set(state["generators"][name]):
                raise ValueError("Checkpoint loader generator layout is incompatible")
            for key, generator in generators.items():
                generator.set_state(state["generators"][name][key])
        if state["loader_epoch"] is not None:
            trainer.train_dataloader.iteration = state["loader_epoch"]
        if trainer.selected_checkpoint and trainer.accelerator.is_main_process:
            name = trainer.selected_checkpoint["checkpoint"]
            source = Path(resume_checkpoint).resolve().parent / name
            selected = load_training_checkpoint(source)
            if selected["selection"] != trainer.selected_checkpoint or selected["run"]["run_id"] != trainer.run["run_id"]:
                raise ValueError("Selected checkpoint does not match the resumed selection state")
            for destination in (name, "model_best.pt", "model_best_avg.pt"):
                shutil.copyfile(source, Path(trainer.log_dir) / destination)
    if pretrained_checkpoint and not resume_checkpoint:
        trainer.run["adaptation"] = copy.deepcopy(getattr(trainer.accelerator.unwrap_model(trainer.model), "adaptation_report", {}))
        source = Path(pretrained_checkpoint).resolve()
        trainer.run["pretrained"] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if trainer.accelerator.is_main_process:
        atomic_json(Path(trainer.log_dir) / "run.json", trainer.run)
        atomic_json(Path(trainer.log_dir) / "config.json", trainer.config)
        atomic_json(Path(trainer.log_dir) / "provenance.json", trainer.run_compatibility)
        # Preserve the flat transformer settings consumed by legacy evaluation.
        with (Path(trainer.log_dir) / "params.pkl").open("wb") as handle:
            pickle.dump({key: trainer.config.model.encoder.transformer[key]
                         for key in ("hidden_size", "inter_size", "n_layers")}, handle)
        atomic_json(Path(trainer.log_dir) / "history.json", trainer.history)


def save_training_checkpoint(trainer, name, epoch):
    if not trainer.accelerator.is_main_process:
        return
    checkpoint = dict(format_version=FORMAT_VERSION, run=trainer.run, config=plain(trainer.config),
                      compatibility=trainer.run_compatibility, epoch=epoch, next_epoch=epoch + 1,
                      model=trainer.accelerator.unwrap_model(trainer.model).state_dict(),
                      optimizer=trainer.optimizer.state_dict(), lr_sched=trainer.lr_scheduler.state_dict(),
                      optimizer_step=trainer.optimizer_steps, scheduler_step=trainer.scheduler_steps,
                      selection=trainer.selected_checkpoint, history=trainer.history, rank_states=trainer.rank_states)
    filename = f"model_epoch_{epoch}.pt" if name == "epoch" else f"model_{name}.pt"
    atomic_checkpoint(Path(trainer.log_dir) / filename, checkpoint)
    atomic_json(Path(trainer.log_dir) / "history.json", trainer.history)
    atomic_json(Path(trainer.log_dir) / "selection.json", trainer.selected_checkpoint)
    print(f"Saved {filename} at epoch {epoch} in {trainer.log_dir}")
