"""Target-delay configuration, provenance and non-destructive stage publication."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import traceback
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from trainer.artifacts import atomic_json, plain
from utils.provenance import file_hash, fingerprint
from utils.progress import configure_progress, iter_progress


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def path(value):
    result = Path(os.path.expandvars(value))
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Run IDs must contain only letters, numbers, underscore and hyphen")
    return value


def parser(description, *, run=False):
    configure_progress()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--experiment-config", default=str(HERE / "configs/experiment.json"))
    if run:
        p.add_argument("--run-config", required=True)
        p.add_argument("--run-id", help="New ID for an independent repeat; defaults to configuration ID")
    return p


def settings(args):
    exp = read(path(args.experiment_config))
    if (exp["context_bins"] != 12 or not exp["delays"] or
            any(type(d) is not int or d < 0 for d in exp["delays"]) or
            len(set(exp["delays"])) != len(exp["delays"])):
        raise ValueError("Expected H=12 and unique nonnegative integer delays")
    evaluation = read(path(exp["evaluation_config"]))
    ws = evaluation["response_windows"]
    if not ws or any(type(w) is not int or w < 1 for w in ws) or len(set(ws)) != len(ws):
        raise ValueError("Response windows must be unique positive integers")
    if evaluation["split"] != "test":
        raise ValueError("This workflow uses the entire persisted test split")
    out = path(exp["output_root"])
    required = (ROOT / "output/experiment/target-delay").resolve()
    if out != required and required not in out.parents:
        raise ValueError("Experiment output_root must be under output/experiment/target-delay")
    exp["support_bins"] = exp["context_bins"] + max(exp["delays"])
    return exp, evaluation, out


def run_settings(args):
    exp, evaluation, out = settings(args)
    run = read(path(args.run_config))
    if set(run) != {"run_id", "context_bins", "target_delay_bins"}:
        raise ValueError("Run configuration must explicitly contain only run_id, context_bins, target_delay_bins")
    if run["context_bins"] != exp["context_bins"] or run["target_delay_bins"] not in exp["delays"]:
        raise ValueError("Run H/D differs from the experiment grid")
    run_id = name(args.run_id or run["run_id"])
    return exp, evaluation, run, out / "runs" / run_id


def source_identity():
    files = list((ROOT / "src").rglob("*.py")) + list(HERE.glob("*.py"))
    return {str(f.relative_to(ROOT)).replace("\\", "/"): file_hash(f) for f in iter_progress(sorted(files), "experiment: recording source identity", total=len(files), unit="files")}


def setup_arguments(exp, run):
    from trainer.setup import add_setup_arguments, CONFIG_ROOT
    p = argparse.ArgumentParser()
    add_setup_arguments(p)
    args = p.parse_args([])
    defaults = read(path(exp["training_defaults"]))
    args.__dict__.update(defaults["entry_arguments"])
    args.__dict__.update(dataset_generation=str(path(exp["dataset_generation"])),
        expected_dataset_generation_id=exp["expected_dataset_generation_id"],
        data_path=str(path(exp["dataset_generation"])), eid=exp["session_id"],
        config_dir=str(CONFIG_ROOT), training_config=None,
        training_overrides=json.dumps(defaults["training_overrides"]),
        context_mode="strict", context_bins=run["context_bins"],
        target_delay_bins=run["target_delay_bins"], target_support_bins=exp["support_bins"],
        modality=["ap", "vision-clip"], mixed_training=False, search=False,
        continue_pretrain=False, overwrite=False, model_mode="encoding")
    return args


def support_record(exp, evaluation, *, target_granularity=None):
    from training_dataset.handoff import load_dataset_splits
    from training_dataset import temporal_context_view
    train, val, test, meta = load_dataset_splits(path(exp["dataset_generation"]),
        session_ids=[exp["session_id"]] if exp["session_id"] else None,
        expected_generation_id=exp["expected_dataset_generation_id"])
    from trainer.targets import transform_neural_targets
    if target_granularity is None:
        defaults = read(path(exp["training_defaults"]))
        target_granularity = defaults["training_overrides"]["training"]["target_granularity"]
    train, val, test, meta = transform_neural_targets((train, val, test), meta, target_granularity)
    trials, populations, means, sums, bins = {}, {}, {}, {}, {}
    first = exp["support_bins"] - 1
    curve_first = first + max(evaluation["response_windows"]) - 1
    for split, view in (("train", train), ("val", val), ("test", test)):
        trials[split] = []
        for sample in iter_progress(view.samples, f"preflight: {split} common support",
                                    total=len(view.samples), unit="trials"):
            t, n = sample.sequence_length, sample.neuron_count
            if (not np.isclose(sample.bin_size, 1/60, rtol=0, atol=1e-12)
                    or not np.isclose(sample.aligned_start, sample.stim_on, rtol=0, atol=1e-8)
                    or sample.aligned_end > sample.stim_off + 1e-8
                    or not np.array_equal(sample.temporal_positions[:t], np.arange(t))
                    or not sample.temporal_mask[:t].all() or not sample.neuron_mask[:n].all()):
                raise ValueError("Requires onset-aligned continuous 60 Hz trials, within stimulus offset, with real units")
            if not np.allclose(np.diff(sample.bin_start_times[:t]), 1/60, rtol=0, atol=1e-8):
                raise ValueError("Physical trial grid is not consecutive 60 Hz")
            # Reject unsupported short trials rather than silently change split membership.
            if t <= curve_first:
                raise ValueError(f"Trial {sample.sample_id} lacks common curve support (needs >{curve_first} bins)")
            masks = []
            for d in exp["delays"]:
                view_d = temporal_context_view(sample.temporal_positions, sample.temporal_mask,
                    direction="encoding", mode="strict", context_bins=exp["context_bins"],
                    target_delay_bins=d, target_support_bins=exp["support_bins"])
                mask = view_d.target_mask[0].numpy()
                expected = np.zeros_like(mask); expected[first:t] = True
                if not np.array_equal(mask, expected):
                    raise ValueError("Actual temporal support differs from common policy")
                indices = view_d.context_indices[0, first].tolist()
                if indices != list(range(first-d-exp["context_bins"]+1, first-d+1)):
                    raise ValueError("Delayed context mapping differs from the requested slice")
                masks.append(mask)
            units = plain(sample.neuron_identity.to_dict(orient="records"))
            if sample.session_id in populations and populations[sample.session_id] != units:
                raise ValueError("Neuron population/order changes within a session")
            populations[sample.session_id] = units
            trials[split].append(dict(sample_id=sample.sample_id, session_id=sample.session_id,
                trial_id=sample.trial_id, sequence_length=t, target_indices=list(range(first,t)),
                curve_endpoints=list(range(curve_first,t)), neurons=n,
                target_counts_hash=fingerprint(sample.neural[first:t,:n].tolist()),
                physical_time_hash=fingerprint(sample.physical_timestamps[:t].tolist()),
                source_alignment=plain(sample.metadata.get("source_alignment"))))
            if split == "train":
                session = sample.session_id
                sums[session] = sums.get(session, 0) + sample.neural[first:t,:n].sum(0, dtype=np.int64)
                bins[session] = bins.get(session, 0) + t-first
    if any(not rows for rows in trials.values()):
        raise ValueError("Nonempty persisted train/val/test splits are required")
    for session, units in populations.items():
        if session not in sums:
            raise ValueError("Every evaluated population must occur in training")
        means[session] = dict(mean_counts=(sums[session]/bins[session]).tolist(),
            eligible_bins=bins[session], spike_sums=sums[session].tolist(), split="train")
    policy = dict(context_bins=exp["context_bins"], delays=exp["delays"],
        support_bins=exp["support_bins"], first_target_index=first,
        response_windows=evaluation["response_windows"], first_curve_endpoint=curve_first,
        curve_policy="trailing mean rate, identical endpoints for every W; no crossing trials",
        training_targets="original single-bin spike counts", split="test")
    compatibility = plain(dict(dataset_generation_id=meta["dataset_generation_id"],
        dataset_metadata=meta["dataset_metadata"], target_granularity=target_granularity,
        regional_targets=meta.get("regional_targets"), regional_target_policy=meta.get("regional_target_policy"),
        populations=populations, trials=trials, policy=policy))
    return dict(compatibility=compatibility, compatibility_id=fingerprint(compatibility),
        training_baseline=means, counts={split: dict(trials=len(rows),
            single_bin_targets=sum(len(r["target_indices"]) for r in rows),
            curve_endpoints=sum(len(r["curve_endpoints"]) for r in rows),
            target_cells=sum(len(r["target_indices"])*r["neurons"] for r in rows))
            for split, rows in trials.items()},
        mapping_examples={str(d): dict(target=first, stimulus_slice=[first-d-exp["context_bins"]+1, first-d+1])
                          for d in exp["delays"]})


def bind_experiment(out, record):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        if read(manifest) != record:
            raise ValueError("Experiment manifest differs; select a new output_root for a changed experiment")
    else:
        atomic_json(manifest, record)


@contextmanager
def stage(directory, stage_name, identity, *, skip=False, allow_new_inputs=False):
    """Exclusive stage lock; unique attempts retain partial/failed output."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / (stage_name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Stage locked: {lock}. After a hard kill, verify no process remains before removing this lock.")
    os.close(fd)
    manifest_path = directory / (stage_name + ".json")
    try:
        previous = read(manifest_path) if manifest_path.exists() else None
        if previous and previous["identity"] != identity and not allow_new_inputs:
            raise ValueError("Incompatible previous stage; choose a new run ID/output root")
        if skip and previous and previous["identity"] == identity and previous["state"] == "completed":
            verify_files(Path(previous["attempt"]), previous["files"])
            yield None
            return
        attempt = directory / (stage_name + "-attempts") / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        manifest = dict(schema_version=1, identity=identity, state="running", attempt=str(attempt))
        atomic_json(manifest_path, manifest)
        try:
            yield attempt
            payloads = sorted(p for p in attempt.rglob("*") if p.is_file())
            files = {str(p.relative_to(attempt)): file_hash(p) for p in iter_progress(
                payloads, f"{stage_name}: recording artifacts", total=len(payloads), unit="files")}
            if not files:
                raise ValueError("Stage produced no artifacts")
            manifest.update(state="completed", files=files)
        except BaseException as error:
            manifest.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                            error=str(error), traceback=traceback.format_exc())
            atomic_json(attempt / "execution.json", manifest)
            atomic_json(manifest_path, manifest)
            raise
        atomic_json(attempt / "execution.json", manifest)
        atomic_json(manifest_path, manifest)
    finally:
        lock.unlink()


def verify_files(directory, files):
    if not files:
        raise ValueError("No completed artifacts")
    for filename, digest in iter_progress(files.items(), "experiment: verifying artifacts",
                                         total=len(files), unit="files"):
        if file_hash(directory / filename) != digest:
            raise ValueError(f"Artifact changed: {filename}")


def completed(directory, stage_name):
    manifest = read(directory / (stage_name + ".json"))
    if manifest["state"] != "completed":
        raise ValueError(f"{stage_name} is not completed in {directory}")
    attempt = Path(manifest["attempt"])
    verify_files(attempt, manifest["files"])
    return manifest, attempt
