"""E2 representation-aware resolution, initialization, and optimization."""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from math import ceil

import numpy as np
from common import *


def resolve(arm):
    from trainer.setup import CONFIG_ROOT, add_setup_arguments, resolve_setup

    parser = argparse.ArgumentParser()
    add_setup_arguments(parser)
    args = parser.parse_args([])
    derived = read(OUT / f"derived/{arm}.json")
    defaults = read(HERE.parent / "temporal-horizon/configs/training.json")
    overrides = deepcopy(defaults["training_overrides"])
    overrides["model"]["encoder"]["embedder"]["visual_dim"] = (
        768 if arm == "A" else 4096
    )
    args.__dict__.update(defaults["entry_arguments"])
    args.__dict__.update(
        dataset_generation=derived["dataset_path"],
        expected_dataset_generation_id=derived["dataset_generation"],
        data_path=None,
        eid=SESSION,
        config_dir=str(CONFIG_ROOT),
        training_config=None,
        training_overrides=json.dumps(overrides),
        modality=["ap", "vision-clip"],
        context_mode="strict",
        context_bins=12,
        target_delay_bins=0,
        target_support_bins=12,
        mixed_training=False,
        search=False,
        continue_pretrain=False,
        overwrite=False,
        model_mode=None,
    )
    config, train, val, metadata = resolve_setup(args)
    config["optimizer"]["planned_updates"] = config.training.num_epochs * ceil(
        len(train) / config.optimizer.gradient_accumulation_steps
    )
    if (
        config.seed != 42
        or config.training.num_epochs != 50
        or config.training.train_batch_size != 1
        or config.optimizer.gradient_accumulation_steps != 4
        or config.optimizer.lr != 0.0002
        or config.optimizer.wd != 0.01
        or config.optimizer.eps != 1e-8
        or config.optimizer.scheduler != "none"
        or config.training.eval_every != 1
        or config.training.checkpoint_selection != "validation_loss"
        or config.training.get("target_selection") is not None
        or config.training.target_granularity != "subregion"
        or metadata.get("neural_region_selection", "all_recorded") != "all_recorded"
        or config.training.objective != "encoding"
        or config.training.modal_filter
        != {"input": ["vision-clip"], "output": ["spike"]}
        or config.training.temporal_context.bins != 12
        or config.training.temporal_context.target_support_bins != 12
        or config.training.temporal_context.get("target_delay_bins", 0) != 0
    ):
        raise ValueError("Resolved training differs from the accepted E2 protocol")
    return config, train, val, metadata


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def controlled(config):
    value = deepcopy(plain(config))
    value.pop("dataset")
    value["model"]["encoder"]["embedder"].pop("visual_dim", None)
    return value


def preflight():
    import torch

    from trainer.objective import forward_objective, prepare_inputs, validate_batch
    from trainer.pretrained import build_model, initialize_regional_bias
    from utils.utils import set_seed

    if (
        not read(OUT / "qualification.json")["qualified"]
        or not read(OUT / "peripheral-qualification.json")["qualified"]
    ):
        raise ValueError("Numerical qualification must pass before training")
    support = deepcopy(
        read(ROOT / "output/experiment/temporal-horizon/manifest.json")["support"]
    )
    if support["median_train_length"] != 146 or support["primary_stop_bin"] != 146:
        raise ValueError("Expected training-only median 146")
    reference_state = None
    wide_state = None
    configs = {}
    reports = {}
    for arm in ARMS:
        config, train, val, metadata = resolve(arm)
        configs[arm] = plain(config)
        if (
            plain(metadata["regional_training_baseline"])
            != support["training_baseline"]
            or plain(metadata["regional_targets"])
            != support["compatibility"]["regional_targets"]
        ):
            raise ValueError("Baseline or anatomical mapping differs from E0")
        for name, loader in [("train", train), ("val", val)]:
            expected = support["compatibility"]["trials"][name]
            if [r.sample_id for r in loader.dataset.scientific_split.samples] != [
                r["sample_id"] for r in expected
            ]:
                raise ValueError("Resolved membership differs from reference")
        set_seed(42)
        model = build_model(
            config=config, metadata=metadata, modal_filter=config.training.modal_filter
        )
        initialize_regional_bias(model, metadata)
        state = model.state_dict()
        if reference_state is None:
            reference_state = {k: v.clone() for k, v in state.items()}
        shared = [
            k
            for k, v in state.items()
            if k in reference_state and reference_state[k].shape == v.shape
        ]
        for k in shared:
            state[k] = reference_state[k].clone()
        if arm == "G":
            if set(state) != set(wide_state) or any(
                v.shape != wide_state[k].shape for k, v in state.items()
            ):
                raise ValueError("B/G architectures differ")
            for key, value in wide_state.items():
                state[key] = value.clone()
        if arm == "B":
            wide_state = {k: v.clone() for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        count = sum(p.numel() for p in model.parameters())
        if count != {"A": 957325, "B": 1825933, "G": 1825933}[arm]:
            raise ValueError(f"Unexpected registered parameter count {arm}: {count}")
        directory = OUT / "runs" / arm
        directory.mkdir(parents=True, exist_ok=True)
        initial = directory / "initial.pt"
        hashes = {k: tensor_hash(v) for k, v in state.items()}
        if initial.exists():
            previous = torch.load(initial, map_location="cpu")
            if hashes != {k: tensor_hash(v) for k, v in previous.items()}:
                raise ValueError("Existing initial weights changed")
        else:
            torch.save(state, initial)
        # Directly inspect an actual training batch through the encoding objective.
        batch = next(iter(train))
        validate_batch(
            batch,
            config=config,
            populations={
                sample.session_id: sample.neuron_identity
                for sample in train.dataset.scientific_split.samples
            },
            split="train",
        )
        model.train()
        inputs, selectors = prepare_inputs(
            batch, model=model, training_mode="encoding", enc_task_var="all"
        )
        output, numerators = forward_objective(
            model, inputs, selectors, components=config.training.active_loss_components
        )
        loss = numerators["spike"] / selectors["spike"].sum()
        loss.backward()
        active = {
            name: p.numel()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        mask = selectors["spike"].any(-1)[0].numpy()
        length = int(batch["temporal_mask"][0].sum())
        from training_dataset import temporal_context_view

        view = temporal_context_view(
            batch["spikes_timestamps"],
            batch["temporal_mask"],
            direction="encoding",
            mode="strict",
            context_bins=12,
            target_support_bins=12,
        )
        checked_windows = {}
        for endpoint in (11, min(145, length - 1), length - 1):
            if endpoint >= 11:
                actual = view.context_indices[0, endpoint].tolist()
                if actual != list(range(endpoint - 11, endpoint + 1)):
                    raise ValueError("Strict context window changed")
                checked_windows[str(endpoint)] = actual
        first, stop = metrics.interval(
            max(length, 147), support["primary_stop_bin"], "primary"
        )
        primary_positions = np.arange(first, stop)
        if 145 not in primary_positions or 146 in primary_positions:
            raise ValueError("Primary endpoint policy changed")
        if not np.array_equal(np.flatnonzero(mask), np.arange(11, length)):
            raise ValueError("Target support differs from S12")
        report = dict(
            registered_parameters=count,
            encoding_gradient_parameters=sum(active.values()),
            active_parameters=active,
            parameter_shapes={k: list(v.shape) for k, v in state.items()},
            initial_hashes=hashes,
            initial_file_sha256=file_hash(initial),
            common_shared_keys=shared,
            unmatched_keys=[k for k in state if k not in shared],
            batch_visual_shape=list(batch["vision-clip"].shape)
            if "vision-clip" in batch
            else list(batch["visual"].shape),
            sequence_length=length,
            eligible_first=int(np.flatnonzero(mask)[0]),
            eligible_last=int(np.flatnonzero(mask)[-1]),
            primary_includes_145=bool(145 in primary_positions),
            primary_excludes_146=True,
            checked_context_windows=checked_windows,
            first_context=list(range(12)),
            last_context=list(range(length - 12, length)),
            loss=float(loss.detach()),
        )
        reports[arm] = report
        save(
            directory / "resolved.json",
            dict(
                arm=arm,
                configuration=configs[arm],
                derived=read(OUT / f"derived/{arm}.json"),
                support=support,
                initialization=report,
            ),
        )
        del model, train, val, metadata, output, loss, numerators, inputs, batch
    if any(controlled(configs[a]) != controlled(configs["A"]) for a in ARMS):
        raise ValueError("Shared training settings differ")
    if reports["B"]["initial_hashes"] != reports["G"]["initial_hashes"]:
        raise ValueError("B/G initialization differs")
    save(
        OUT / "preflight.json",
        dict(
            reports=reports,
            support=support,
            controlled_configuration=controlled(configs["A"]),
            sources=sources(),
        ),
    )
    print(
        {
            a: {
                k: reports[a][k]
                for k in ("registered_parameters", "encoding_gradient_parameters")
            }
            for a in ARMS
        }
    )


def train(arm):
    import torch

    from trainer.artifacts import initialize_run, load_training_checkpoint
    from trainer.make import make_multimodal_trainer
    from trainer.pretrained import build_model
    from trainer.runtime import make_accelerator, prepare_optimization
    from utils.utils import set_seed

    directory = OUT / "runs" / arm
    result_path = directory / "result.json"
    if result_path.exists():
        previous = read(result_path)
        if (
            previous["training_identity"]
            != fingerprint(read(directory / "resolved.json"))
            or file_hash(previous["checkpoint"]) != previous["checkpoint_sha256"]
            or sources() != read(OUT / "preflight.json")["sources"]
        ):
            raise ValueError("Completed training identity or checkpoint changed")
        return previous
    resolved = read(directory / "resolved.json")
    config, train_loader, val_loader, metadata = resolve(arm)
    if plain(config) != resolved["configuration"]:
        raise ValueError("Resolved configuration changed after preflight")
    if sources() != read(OUT / "preflight.json")["sources"]:
        raise ValueError("Implementation changed after preflight")
    model = build_model(
        config=config, metadata=metadata, modal_filter=config.training.modal_filter
    )
    initial = directory / "initial.pt"
    if file_hash(initial) != resolved["initialization"]["initial_file_sha256"]:
        raise ValueError("Initial weights changed")
    model.load_state_dict(torch.load(initial, map_location="cpu"), strict=True)
    active = set()
    handles = [
        p.register_hook(lambda grad, name=name: active.add(name))
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    accelerator = make_accelerator()
    if accelerator.num_processes != 1:
        raise ValueError("One process per E2 arm is required")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.wd,
        eps=config.optimizer.eps,
    )
    model, optimizer, train_loader, scheduler = prepare_optimization(
        accelerator, model, optimizer, train_loader, config
    )
    logdir = directory / "training"
    logdir.mkdir(exist_ok=False)
    trainer = make_multimodal_trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        optimizer=optimizer,
        log_dir=str(logdir),
        accelerator=accelerator,
        lr_scheduler=scheduler,
        avail_mod=["spike", "vision-clip"],
        avail_beh=["vision-clip"],
        modal_filter=config.training.modal_filter,
        mixed_training=False,
        enc_task_var=config.training.enc_task_var,
        config=config,
        multi_gpu=False,
        start_epoch=0,
        **metadata,
    )
    initialize_run(trainer, kind="train", resume_checkpoint=None)
    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    for handle in handles:
        handle.remove()
    selected = load_training_checkpoint(logdir / "model_best.pt")
    last = load_training_checkpoint(logdir / "model_last.pt")
    if last["next_epoch"] != 50 or selected["config"] != plain(config):
        raise ValueError("Training did not complete the fixed protocol")
    counts = {
        name: p.numel()
        for name, p in accelerator.unwrap_model(model).named_parameters()
    }
    if sorted(active) != sorted(resolved["initialization"]["active_parameters"]):
        raise ValueError(
            "Actual encoding gradient parameter set differs from preflight"
        )
    result = dict(
        checkpoint=str(logdir / "model_best.pt"),
        checkpoint_sha256=file_hash(logdir / "model_best.pt"),
        selected_epoch=selected["epoch"],
        selection=selected["selection"],
        history=last["history"],
        training_identity=fingerprint(resolved),
        seconds=time.perf_counter() - started,
        accelerator_device=str(accelerator.device),
        mixed_precision=accelerator.mixed_precision,
        training_rng_seed_after_model_construction=42,
        registered_parameters=sum(counts.values()),
        encoding_gradient_parameters=sum(counts[n] for n in active),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    save(result_path, result)
    accelerator.free_memory()
    return result
