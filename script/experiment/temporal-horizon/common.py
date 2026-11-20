"""E0's isolated protocol; production components retain their general defaults."""

import argparse
import importlib.util
import sys
from copy import deepcopy
from math import ceil, floor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "e0_artifacts", HERE.parent / "target-delay/common.py"
)
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)
read, path, atomic_json = artifacts.read, artifacts.path, artifacts.atomic_json
plain, fingerprint, file_hash = (
    artifacts.plain,
    artifacts.fingerprint,
    artifacts.file_hash,
)
stage, completed = artifacts.stage, artifacts.completed

import numpy as np

POLICY = {
    "training": "full eligible S=12 support, t>=11",
    "checkpoint_selection": "minimum full-support validation Poisson loss",
    "primary": "11<=t<min(trial_length,floor(median_train_length))",
    "baseline": "per-subregion training mean on full eligible S=12 support; clamp minimum 1e-8 for NLL",
    "smoothing": "trailing W-bin means; every bin inside the declared interval; no cross-trial windows",
    "interpretation": "Exploratory short runs; numerical gains do not establish superiority or convergence.",
}


def settings(filename):
    exp = read(path(filename))
    required = {
        "context_mode": "strict",
        "horizons": [6, 12],
        "target_delay_bins": 0,
        "support_bins": 12,
        "target_granularity": "subregion",
        "neural_region_selection": "all_recorded",
        "information_filtering": False,
        "objective": "encoding",
        "response_windows": [1, 6, 12],
    }
    if any(
        exp.get(k) != v or type(exp.get(k)) is not type(v) for k, v in required.items()
    ):
        raise ValueError(
            "E0 requires strict H6/H12, D=0, S=12, unfiltered all-recorded subregion encoding, W=1/6/12"
        )
    if any(
        type(v) is not int for k in ("horizons", "response_windows") for v in exp[k]
    ):
        raise ValueError("Horizon and response windows must be integer bins")
    if (
        type(exp["evaluation_batch_size"]) is not int
        or exp["evaluation_batch_size"] < 1
    ):
        raise ValueError("Evaluation batch size must be a positive integer")
    if set(exp) != set(required) | {
        "dataset_generation",
        "expected_dataset_generation_id",
        "session_id",
        "training_defaults",
        "output_root",
        "evaluation_batch_size",
        "evaluation_device",
    }:
        raise ValueError("Unknown or missing E0 experiment setting")
    out = path(exp["output_root"])
    base = (ROOT / "output/experiment/temporal-horizon").resolve()
    if out != base and base not in out.parents:
        raise ValueError(
            "E0 outputs must stay under output/experiment/temporal-horizon"
        )
    return exp, out


def setup_arguments(exp, horizon):
    from trainer.setup import CONFIG_ROOT, add_setup_arguments

    parser = argparse.ArgumentParser()
    add_setup_arguments(parser)
    args = parser.parse_args([])
    defaults = read(path(exp["training_defaults"]))
    if set(defaults) != {"entry_arguments", "training_overrides"} or defaults[
        "entry_arguments"
    ] != {"num_sessions": 1, "no_wandb": True}:
        raise ValueError(
            "Configure E0 training exclusively through the shared training_overrides"
        )
    training = defaults["training_overrides"].get("training", {})
    if (
        training.get("objective") != "encoding"
        or training.get("target_granularity") != "subregion"
        or training.get("mixed_training") is not False
        or training.get("checkpoint_selection") != "validation_loss"
        or training.get("eval_every") != 1
    ):
        raise ValueError(
            "E0 training requires subregion encoding and full-support validation-loss selection every epoch"
        )
    args.__dict__.update(defaults["entry_arguments"])
    args.__dict__.update(
        dataset_generation=str(path(exp["dataset_generation"])),
        expected_dataset_generation_id=exp["expected_dataset_generation_id"],
        data_path=None,
        eid=exp["session_id"],
        config_dir=str(CONFIG_ROOT),
        training_config=None,
        training_overrides=None,
        modality=["ap", "vision-clip"],
        context_mode="strict",
        context_bins=horizon,
        target_delay_bins=0,
        target_support_bins=12,
        mixed_training=False,
        search=False,
        continue_pretrain=False,
        overwrite=False,
        model_mode=None,
    )
    # The wrapper envelope is not a production override file.
    import json

    args.training_overrides = json.dumps(defaults["training_overrides"])
    return args


def controlled(config):
    value = deepcopy(plain(config))
    value["training"]["temporal_context"].pop("bins")
    return value


def resolve_arm(exp, horizon):
    from trainer.setup import resolve_setup

    config, train, val, meta = resolve_setup(setup_arguments(exp, horizon))
    training = config.training
    context = training.temporal_context
    if (
        context.mode != "strict"
        or context.bins != horizon
        or context.target_support_bins != 12
        or context.get("target_delay_bins", 0) != 0
        or training.get("target_selection") is not None
        or meta.get("neural_region_selection", "all_recorded") != "all_recorded"
        or training.modal_filter != {"input": ["vision-clip"], "output": ["spike"]}
    ):
        raise ValueError(
            "Resolved E0 configuration violates temporal, population or modality policy"
        )
    config["optimizer"]["planned_updates"] = training.num_epochs * ceil(
        len(train) / config.optimizer.gradient_accumulation_steps
    )
    return config, train, val, meta


def support_record(exp):
    from trainer.targets import transform_neural_targets
    from training_dataset import temporal_context_view
    from training_dataset.handoff import load_dataset_splits

    train, val, test, meta = load_dataset_splits(
        path(exp["dataset_generation"]),
        session_ids=[exp["session_id"]],
        expected_generation_id=exp["expected_dataset_generation_id"],
    )
    native = transform_neural_targets((train, val, test), meta, "subregion")
    views, metadata = native[:3], native[3]
    median = float(np.median([s.sequence_length for s in train.samples]))
    stop = floor(median)
    # Integer t denotes bin [t,t+1); a bin straddling a fractional endpoint is excluded.
    primary_counts, trials, populations, sums, bins = {}, {}, {}, {}, {}
    boundaries = []
    for split, original, view in zip(
        ("train", "val", "test"), (train, val, test), views
    ):
        count = 0
        trials[split] = []
        for source, sample in zip(original.samples, view.samples):
            length = sample.sequence_length
            if (
                not np.array_equal(
                    sample.temporal_positions[:length], np.arange(length)
                )
                or not sample.temporal_mask[:length].all()
                or not sample.neuron_mask.all()
                or not np.isclose(sample.bin_size, 1 / 60, rtol=0, atol=1e-12)
                or not np.isclose(
                    sample.aligned_start, sample.stim_on, rtol=0, atol=1e-8
                )
                or sample.aligned_end > sample.stim_off + 1e-8
            ):
                raise ValueError(
                    "E0 requires original continuous onset-aligned 60 Hz trials and valid recorded groups"
                )
            units = plain(sample.neuron_identity.to_dict(orient="records"))
            if (
                sample.session_id in populations
                and populations[sample.session_id] != units
            ):
                raise ValueError("Subregion population/order changes within a session")
            populations[sample.session_id] = units
            trials[split].append(
                {
                    "sample_id": sample.sample_id,
                    "session_id": sample.session_id,
                    "trial_id": sample.trial_id,
                    "sequence_length": length,
                    "neurons": sample.neuron_count,
                    "target_counts_hash": fingerprint(
                        sample.neural[11:length].tolist()
                    ),
                    "physical_time_hash": fingerprint(
                        sample.physical_timestamps[:length].tolist()
                    ),
                    "source_alignment": plain(sample.metadata.get("source_alignment")),
                }
            )
            if split == "train":
                session = sample.session_id
                sums[session] = sums.get(session, 0) + sample.neural[11:length].sum(
                    0, dtype=np.int64
                )
                bins[session] = bins.get(session, 0) + max(0, length - 11)
            mapping = metadata["regional_targets"][sample.session_id]
            columns = [c for group in mapping["source_columns"] for c in group]
            if sorted(columns) != list(np.flatnonzero(source.neuron_mask)):
                raise ValueError(
                    "Subregions must partition every recorded source neuron exactly once"
                )
            expected = np.stack(
                [source.neural[:, c].sum(1) for c in mapping["source_columns"]], axis=-1
            )
            if not np.array_equal(expected, sample.neural):
                raise ValueError(
                    "Aggregated targets differ from original source counts"
                )
            for h in (6, 12):
                context = temporal_context_view(
                    sample.temporal_positions,
                    sample.temporal_mask,
                    direction="encoding",
                    mode="strict",
                    context_bins=h,
                    target_support_bins=12,
                )
                eligible = np.flatnonzero(context.target_mask[0].numpy())
                if not np.array_equal(eligible, np.arange(11, length)):
                    raise ValueError(
                        "Both arms must supervise exactly original bins 11..T-1"
                    )
                for t in (11, length - 1):
                    if t < 11 or t >= length:
                        continue
                    if context.context_indices[0, t].tolist() != list(
                        range(t - h + 1, t + 1)
                    ):
                        raise ValueError("Unexpected causal visual access")
            count += max(0, min(length, stop) - 11)
            if len(boundaries) < 3 or (split != boundaries[-1]["split"]):
                boundaries.append(
                    {
                        "split": split,
                        "trial": sample.trial_id,
                        "sample_id": sample.sample_id,
                        "length": length,
                        "primary_first": 11,
                        "primary_stop": min(length, stop),
                        "first_H6": list(range(6, 12)),
                        "first_H12": list(range(12)),
                        "physical_first": float(sample.bin_start_times[0]),
                        "physical_last": float(sample.bin_end_times[length - 1]),
                    }
                )
        primary_counts[split] = {
            "temporal_positions": count,
            "target_cells": sum(
                max(0, min(s.sequence_length, stop) - 11) * s.neuron_count
                for s in view.samples
            ),
        }
    if any(not rows for rows in trials.values()) or any(n <= 0 for n in bins.values()):
        raise ValueError("E0 requires nonempty splits and eligible training support")
    compatibility = plain(
        {
            "dataset_generation_id": metadata["dataset_generation_id"],
            "dataset_metadata": metadata["dataset_metadata"],
            "regional_targets": metadata["regional_targets"],
            "regional_target_policy": metadata["regional_target_policy"],
            "target_granularity": "subregion",
            "populations": populations,
            "trials": trials,
            "policy": POLICY,
        }
    )
    return {
        "compatibility": compatibility,
        "compatibility_id": fingerprint(compatibility),
        "training_baseline": {
            s: {
                "mean_counts": (sums[s] / n).tolist(),
                "eligible_bins": n,
                "spike_sums": sums[s].tolist(),
                "split": "train",
            }
            for s, n in bins.items()
        },
        "counts": {
            split: {
                "trials": len(rows),
                "single_bin_targets": sum(
                    max(0, r["sequence_length"] - 11) for r in rows
                ),
                "target_cells": sum(
                    max(0, r["sequence_length"] - 11) * r["neurons"] for r in rows
                ),
            }
            for split, rows in trials.items()
        },
        "median_train_length": median,
        "primary_stop_bin": stop,
        "primary_counts": primary_counts,
        "boundary_examples": boundaries,
        "policy": POLICY,
    }


def sources():
    files = list((ROOT / "src").rglob("*.py")) + list(HERE.glob("*.py"))
    files += list((ROOT / "src/configs").rglob("*.json"))
    files += [HERE.parent / "target-delay/common.py"]
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in sorted(files)
    }
