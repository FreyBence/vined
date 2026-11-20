"""Isolated E0 preflight, explicit training, inference and paired comparison."""

import argparse
from pathlib import Path

import numpy as np
from analysis import DEFINITIONS, evaluate, paired
from common import (
    HERE,
    POLICY,
    artifacts,
    atomic_json,
    completed,
    controlled,
    file_hash,
    fingerprint,
    path,
    plain,
    read,
    resolve_arm,
    settings,
    sources,
    stage,
    support_record,
)


def snapshot(exp, horizon, config, support):
    return plain(
        {
            "arm": f"H{horizon}",
            "configuration": config,
            "experiment": exp,
            "support": support,
            "policy": POLICY,
        }
    )


def preflight(exp, out):
    support = support_record(exp)
    configurations, dimensions = {}, {}
    from trainer.pretrained import build_model, initialize_regional_bias

    for h in (6, 12):
        config, _, _, meta = resolve_arm(exp, h)
        if plain(meta["regional_training_baseline"]) != support["training_baseline"]:
            raise ValueError(
                "Production training baseline differs from full S=12 experiment baseline"
            )
        if (
            plain(meta["regional_targets"])
            != support["compatibility"]["regional_targets"]
        ):
            raise ValueError("Resolved training mapping differs from preflight")
        model = build_model(
            config=config, metadata=meta, modal_filter=config.training.modal_filter
        )
        initialize_regional_bias(model, meta)
        dimensions[str(h)] = {
            name: list(value.shape) for name, value in model.named_parameters()
        }
        configurations[h] = config
        del model
    if (
        controlled(configurations[6]) != controlled(configurations[12])
        or dimensions["6"] != dimensions["12"]
    ):
        raise ValueError(
            "E0 arms may differ only in temporal horizon and experiment identity"
        )
    record = plain(
        {
            "experiment": exp,
            "support": support,
            "policy": POLICY,
            "controlled_configuration": controlled(configurations[6]),
        }
    )
    artifacts.bind_experiment(out, record)
    for h, config in configurations.items():
        directory = out / "runs" / f"H{h}"
        directory.mkdir(parents=True, exist_ok=True)
        current = snapshot(exp, h, config, support)
        target = directory / "resolved.json"
        if target.exists() and read(target) != current:
            raise ValueError(
                "Existing E0 identity differs; use a new experiment output_root"
            )
        if not target.exists():
            atomic_json(target, current)
    atomic_json(
        out / "preflight.json",
        {
            "protocol_identity": fingerprint(record),
            "sources": sources(),
            "only_configuration_difference": "training.temporal_context.bins",
            "equal_parameter_shapes": True,
            "parameter_count": sum(
                np.prod(v, dtype=np.int64).item() for v in dimensions["6"].values()
            ),
            "parameter_shapes": dimensions["6"],
            "median_train_length": support["median_train_length"],
            "full_counts": support["counts"],
            "primary_counts": support["primary_counts"],
            "mapping": support["compatibility"]["regional_targets"],
            "boundaries": support["boundary_examples"],
            "training_launched": False,
            "policy": POLICY,
        },
    )
    print(
        f"E0 preflight: median={support['median_train_length']}; primary={support['primary_counts']}"
    )


def load_resolved(exp, out, arm):
    resolved = read(out / "runs" / arm / "resolved.json")
    if resolved["experiment"] != exp:
        raise ValueError("Experiment differs from the preflight identity")
    manifest = read(out / "manifest.json")
    if manifest["support"] != resolved["support"] or manifest[
        "controlled_configuration"
    ] != controlled(resolved["configuration"]):
        raise ValueError("Run differs from controlled experiment identity")
    return resolved


def train(exp, out, arm):
    # Re-resolve both arms and inspect current data before any optimizer is created.
    preflight(exp, out)
    resolved = load_resolved(exp, out, arm)
    h = int(arm[1:])
    config, train_loader, val_loader, metadata = resolve_arm(exp, h)
    if plain(config) != resolved["configuration"]:
        raise ValueError("Training resolution differs from the saved preflight")
    directory = out / "runs" / arm
    with stage(directory, "train", fingerprint(resolved)) as attempt:
        atomic_json(attempt / "resolved.json", resolved)
        atomic_json(attempt / "sources.json", sources())
        import torch

        from trainer.artifacts import initialize_run, load_training_checkpoint
        from trainer.make import make_multimodal_trainer
        from trainer.pretrained import build_model, initialize_regional_bias
        from trainer.runtime import make_accelerator, prepare_optimization

        accelerator = make_accelerator()
        if accelerator.num_processes != 1:
            raise ValueError("E0 supports one process per arm invocation")
        model = build_model(
            config=config, metadata=metadata, modal_filter=config.training.modal_filter
        )
        initialize_regional_bias(model, metadata)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.wd,
            eps=config.optimizer.eps,
        )
        model, optimizer, train_loader, scheduler = prepare_optimization(
            accelerator, model, optimizer, train_loader, config
        )
        logdir = attempt / "training"
        logdir.mkdir()
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
        trainer.train()
        checkpoint = logdir / "model_best.pt"
        selected = load_training_checkpoint(checkpoint)
        last = load_training_checkpoint(logdir / "model_last.pt")
        if (
            last["next_epoch"] != config.training.num_epochs
            or not last["history"]
            or selected["config"] != plain(config)
            or selected["selection"]["epoch"] != selected["epoch"]
        ):
            raise ValueError(
                "Training did not produce a complete compatible selected checkpoint"
            )
        atomic_json(
            attempt / "result.json",
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_hash(checkpoint),
                "selected_epoch": selected["epoch"],
                "selection": selected["selection"],
                "history": last["history"],
                "training_identity": fingerprint(resolved),
            },
        )


def selected_training(exp, out, arm):
    resolved = load_resolved(exp, out, arm)
    manifest, attempt = completed(out / "runs" / arm, "train")
    result = read(attempt / "result.json")
    if (
        manifest["identity"] != fingerprint(resolved)
        or result["training_identity"] != manifest["identity"]
    ):
        raise ValueError("Selected training and preflight identity differ")
    if file_hash(Path(result["checkpoint"])) != result["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    return resolved, result, attempt


def infer(exp, out, arm, device):
    resolved, result, training_attempt = selected_training(exp, out, arm)
    if support_record(exp) != resolved["support"]:
        raise ValueError("Current dataset/support differs from training")
    from trainer.artifacts import load_training_checkpoint

    checkpoint = load_training_checkpoint(result["checkpoint"])
    if checkpoint["config"] != resolved["configuration"]:
        raise ValueError(
            "Checkpoint effective configuration differs from E0 resolution"
        )
    from evaluation.predictions import collect_predictions
    from evaluation.setup import resolve_setup

    setup = resolve_setup(
        checkpoint_path=result["checkpoint"],
        dataset_generation=path(exp["dataset_generation"]),
        expected_generation_id=exp["expected_dataset_generation_id"],
        session_ids=[exp["session_id"]],
        context_mode="strict",
        context_bins=int(arm[1:]),
        batch_size=exp["evaluation_batch_size"],
        device=device or exp["evaluation_device"],
        seed=resolved["configuration"]["seed"],
        prediction_direction="encoding",
        neural_region_selection="all_recorded",
    )
    identity = fingerprint(
        {
            "training": result,
            "configuration": setup.configuration,
            "definitions": DEFINITIONS,
        }
    )
    with stage(
        out / "runs" / arm, "evaluate", identity, allow_new_inputs=True
    ) as attempt:
        collection = collect_predictions(setup)
        records, index = [], []
        expected = resolved["support"]["compatibility"]["trials"]["test"]
        if [r.sample_id for r in collection.predictions] != [
            r["sample_id"] for r in expected
        ]:
            raise ValueError("Inference membership differs from the paired test split")
        prediction_dir = attempt / "predictions"
        prediction_dir.mkdir()
        for i, (row, reference) in enumerate(zip(collection.predictions, expected)):
            length, width = reference["sequence_length"], reference["neurons"]
            eligible = np.flatnonzero(row.encoding_target_mask)
            observed = row.observed_neural[:length, :width]
            if (
                not np.array_equal(eligible, np.arange(11, length))
                or fingerprint(observed[11:].tolist())
                != reference["target_counts_hash"]
                or fingerprint(row.physical_timestamps[:length].tolist())
                != reference["physical_time_hash"]
                or plain(row.neuron_identity.to_dict(orient="records"))
                != resolved["support"]["compatibility"]["populations"][row.session_id]
            ):
                raise ValueError(
                    "Inference targets, bins or subregion identities differ from preflight"
                )
            item = {
                "sample_id": row.sample_id,
                "session_id": row.session_id,
                "trial_id": row.trial_id,
                "observed": observed,
                "log_expected": row.predicted_neural[:length, :width],
                "original_bins": row.temporal_positions[:length],
                "physical_timestamps": row.physical_timestamps[:length],
                "bin_start_times": row.bin_start_times[:length],
                "bin_end_times": row.bin_end_times[:length],
                "eligible_mask": row.encoding_target_mask[:length],
                "primary_mask": (row.temporal_positions[:length] >= 11)
                & (
                    row.temporal_positions[:length]
                    < resolved["support"]["primary_stop_bin"]
                ),
            }
            records.append(item)
            filename = f"{i:06d}.npz"
            np.savez_compressed(
                prediction_dir / filename,
                **{k: v for k, v in item.items() if isinstance(v, np.ndarray)},
            )
            index.append(
                {k: v for k, v in item.items() if not isinstance(v, np.ndarray)}
                | {
                    "file": filename,
                    "neuron_identity": plain(
                        row.neuron_identity.to_dict(orient="records")
                    ),
                    "metadata": plain(row.metadata),
                }
            )
        native = evaluate(records, resolved["support"])
        if (
            native["primary"]["cell_weighted"]["cells"]
            != resolved["support"]["primary_counts"]["test"]["target_cells"]
        ):
            raise ValueError("Primary inference coverage differs from preflight")
        atomic_json(prediction_dir / "index.json", index)
        atomic_json(
            attempt / "metrics.json",
            {
                "native": native,
                "supplementary_parent": evaluate(
                    records, resolved["support"], parent=True
                ),
                "definitions": DEFINITIONS,
                "policy": POLICY,
            },
        )
        atomic_json(
            attempt / "provenance.json",
            {
                "setup": setup.summary(),
                "training": result,
                "training_attempt": str(training_attempt),
                "training_sources": read(training_attempt / "sources.json"),
                "evaluation_sources": sources(),
                "resolved_identity": fingerprint(resolved),
                "checkpoint_sha256": result["checkpoint_sha256"],
            },
        )


def load_evaluation(exp, out, arm):
    resolved, training, _ = selected_training(exp, out, arm)
    manifest, attempt = completed(out / "runs" / arm, "evaluate")
    provenance = read(attempt / "provenance.json")
    if provenance["checkpoint_sha256"] != training["checkpoint_sha256"] or provenance[
        "resolved_identity"
    ] != fingerprint(resolved):
        raise ValueError("Evaluation refers to another checkpoint or configuration")
    records = []
    index = read(attempt / "predictions/index.json")
    for item in index:
        with np.load(
            attempt / "predictions" / item["file"], allow_pickle=False
        ) as payload:
            records.append(dict(item, **{k: payload[k].copy() for k in payload.files}))
    return resolved, training, manifest, provenance, records


def compare(exp, out):
    a, b = [load_evaluation(exp, out, arm) for arm in ("H6", "H12")]
    if (
        controlled(a[0]["configuration"]) != controlled(b[0]["configuration"])
        or a[0]["support"] != b[0]["support"]
    ):
        raise ValueError(
            "Arms differ in configuration, baseline, dataset, split, population or support"
        )
    if (
        a[3]["training_sources"] != b[3]["training_sources"]
        or a[3]["evaluation_sources"] != b[3]["evaluation_sources"]
    ):
        raise ValueError(
            "Arms used different implementation sources; incomparable controlled experiment"
        )
    expected = a[0]["support"]["compatibility"]["trials"]["test"]
    if len(a[4]) != len(b[4]) or len(a[4]) != len(expected):
        raise ValueError("Incomplete trial pairing")
    for x, y, reference in zip(a[4], b[4], expected):
        for key in ("sample_id", "session_id", "trial_id", "neuron_identity"):
            if x[key] != y[key]:
                raise ValueError(f"Unpaired {key}")
        if x["sample_id"] != reference["sample_id"]:
            raise ValueError("Prediction index differs from preflight membership")
        for key in (
            "observed",
            "original_bins",
            "physical_timestamps",
            "bin_start_times",
            "bin_end_times",
            "eligible_mask",
            "primary_mask",
        ):
            if not np.array_equal(x[key], y[key]):
                raise ValueError(f"Unpaired original observations: {key}")
    support = a[0]["support"]
    scores = [evaluate(value[4], support) for value in (a, b)]
    parents = [evaluate(value[4], support, parent=True) for value in (a, b)]
    identity = fingerprint(
        {"H6": a[2]["identity"], "H12": b[2]["identity"], "definitions": DEFINITIONS}
    )
    with stage(
        out / "comparison", "compare", identity, allow_new_inputs=True
    ) as attempt:
        provenance = {
            arm: {
                "configuration": v[0]["configuration"],
                "checkpoint": v[1],
                "evaluation": v[3],
            }
            for arm, v in zip(("H6", "H12"), (a, b))
        }
        result = {
            "definitions": DEFINITIONS,
            "policy": POLICY,
            "median_train_length": support["median_train_length"],
            "native": {"H6": scores[0], "H12": scores[1], "paired": paired(*scores)},
            "supplementary_parent": {
                "H6": parents[0],
                "H12": parents[1],
                "paired": paired(*parents),
            },
            "identity_checks": {
                "exact_trial_bin_subregion_pairing": True,
                "target_cells": scores[0]["primary"]["cell_weighted"]["cells"],
                "identical_dataset_splits_population_baseline": True,
                "identical_sources": True,
            },
            "provenance": provenance,
            "winner": None,
        }
        atomic_json(attempt / "comparison.json", result)
        curves(attempt, a[1], b[1])
        report(attempt, result)
        print(f"E0 comparison: {attempt / 'report.md'}")


def curves(directory, a, b):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, value in (("H6", a), ("H12", b)):
        history = value["history"]
        atomic_json(directory / f"{label}-history.json", history)
        for section, key in (("training", "train_loss"), ("validation", "eval_loss")):
            rows = [r for r in history if (r.get(section) or {}).get(key) is not None]
            if rows:
                ax.plot(
                    [r["epoch"] for r in rows],
                    [r[section][key] for r in rows],
                    label=f"{label} {key}",
                )
        ax.axvline(
            value["selected_epoch"],
            linestyle="--",
            alpha=0.4,
            label=f"{label} selected epoch",
        )
    ax.set(xlabel="Epoch (production numbering)", ylabel="Full-support Poisson loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "training-validation-curves.png")
    plt.close(fig)


def report(directory, result):
    lines = [
        "# E0 controlled temporal horizon comparison",
        "",
        POLICY["interpretation"],
        "",
        f"Training-only median: {result['median_train_length']} complete bins. Primary support: {POLICY['primary']}.",
        "",
        (
            "Optimization and checkpoint selection use full eligible S=12 support. The reporting cutoff does not affect either. "
            "The shared constant baseline is fitted on full eligible training support, not median-restricted support."
        ),
        "",
        "| Support | H6 NLL/cell | H12 NLL/cell | H12−H6 |",
        "|---|---:|---:|---:|",
    ]
    for scope in ("primary", "full", "tail"):
        x, y = [
            result["native"][arm][scope]["cell_weighted"]["poisson_nll"]
            for arm in ("H6", "H12")
        ]
        lines.append(
            f"| {scope} | {x} | {y} | {None if x is None or y is None else y - x} |"
        )
    lines += [
        "",
        "| Arm | Primary gain/cell | Equal-trial gain mean ± SD | Trials beating constant | Selected epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("H6", "H12"):
        primary = result["native"][arm]["primary"]
        variability = primary["equal_trial"]["nll_improvement"]
        provenance = result["provenance"][arm]
        lines.append(
            f"| {arm} | {primary['cell_weighted']['nll_improvement']} | {variability['mean']} ± {variability['std']} | {primary['fraction_improved_trials']} | {provenance['checkpoint']['selected_epoch']} |"
        )
    lines += [
        "",
        "| Arm | W | Equal-trial correlation | Undefined trial-channel correlations | Equal-trial variance ratio | Undefined trial-channel ratios |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("H6", "H12"):
        for w, diagnostic in result["native"][arm]["primary"]["diagnostics"].items():
            c, v = diagnostic["correlation"], diagnostic["variance_ratio"]
            lines.append(
                f"| {arm} | {w} | {c['equal_trial']['mean']} | {c['cell_weighted']['undefined']} | {v['equal_trial']['mean']} | {v['cell_weighted']['undefined']} |"
            )
    pair = result["native"]["paired"]["primary"]
    lines += [
        "",
        f"Paired primary trial NLL difference: {pair['equal_trial_nll_difference']}. Fractions with lower NLL: H6={pair['fraction_trials_H6_lower_nll']}, H12={pair['fraction_trials_H12_lower_nll']}; tied trials={pair['tied_trials']}.",
        "",
    ]
    for arm, provenance in result["provenance"].items():
        checkpoint = provenance["checkpoint"]
        lines.append(
            f"{arm}: checkpoint `{checkpoint['checkpoint']}`, SHA-256 `{checkpoint['checkpoint_sha256']}`. Dataset generation `{provenance['configuration']['dataset']['dataset_generation_id']}`."
        )
    lines += [
        "",
        (
            "All records pair identical session, trial, original bin and native subregion. "
            "Low-activity and mathematically undefined channels remain represented. "
            "comparison.json contains cell-weighted and equal-trial summaries, trial variability, undefined reasons/counts, "
            "W=1/6/12 correlations and variance ratios, fraction improved trials, and paired trial/channel differences."
        ),
        "",
        (
            "Full eligible intervals, long-trial tails, and parent-region reconstruction are supplementary. "
            "Parent predictions sum intensities before taking logs. No supplementary result selects a primary winner."
        ),
        "",
        (
            "Configuration, dataset/source mappings, checkpoint hashes, selection epochs and complete training histories "
            "are included in provenance. training-validation-curves.png shows full-support losses."
        ),
        "",
        "No automatic winner is declared. Short runs do not establish convergence or statistical superiority.",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "train", "evaluate", "compare")
    )
    parser.add_argument(
        "--experiment-config", default=str(HERE / "configs/experiment.json")
    )
    parser.add_argument("--arm", choices=("H6", "H12"))
    parser.add_argument("--device", help="Inference device only")
    args = parser.parse_args()
    if (args.command in ("train", "evaluate")) != (args.arm is not None):
        parser.error("--arm is required only for train/evaluate")
    if args.device and args.command != "evaluate":
        parser.error("--device applies only to evaluate")
    exp, out = settings(args.experiment_config)
    if args.command == "preflight":
        preflight(exp, out)
    elif args.command == "train":
        train(exp, out, args.arm)
    elif args.command == "evaluate":
        infer(exp, out, args.arm, args.device)
    else:
        compare(exp, out)


if __name__ == "__main__":
    main()
