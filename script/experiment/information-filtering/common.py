"""Experiment 3 adapters to the existing experiment and production boundaries."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def module(directory, filename, alias):
    spec = importlib.util.spec_from_file_location(alias, HERE.parent / directory / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


previous = module("subregion-aggregation", "common.py", "information_subregion_common")
delay = previous.delay
read, path, name = delay.read, delay.path, delay.name
stage, completed, atomic_json = delay.stage, delay.completed, delay.atomic_json
plain, fingerprint, file_hash = delay.plain, delay.fingerprint, delay.file_hash
bind_experiment, changed_settings = delay.bind_experiment, delay.changed_settings
analysis = previous.load_helper("analysis.py", "information_analysis")
load_helper, regional_sources = previous.load_helper, previous.regional_sources
ACTIVE_RUN = None


def source_identity():
    return previous.source_identity() | {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p)
                                        for p in HERE.glob("*.py")}


def parser(description, *, run=False):
    p = delay.parser(description, run=run)
    p.set_defaults(experiment_config=str(HERE / "configs/experiment.json"))
    return p


def settings(args):
    exp = read(path(args.experiment_config))
    ev = read(path(exp["evaluation_config"]))
    if exp["context_bins"] != 12 or exp["delays"] != [0, 6] or exp["support_bins"] != 24:
        raise ValueError("Require H=12, D=0/6 and original S=24 support")
    if ev["split"] != "val" or ev["response_windows"] != [1, 3, 6, 12]:
        raise ValueError("Experiment 3 model-development comparison requires validation and W=1,3,6,12")
    out = path(exp["output_root"])
    required = ROOT / "output/experiment/information-filtering"
    if out != required and required not in out.parents:
        raise ValueError("Output must remain under output/experiment/information-filtering")
    return exp, ev, out


def run_settings(args):
    global ACTIVE_RUN
    exp, ev, out = settings(args)
    run = read(path(args.run_config))
    if (set(run) != {"run_id", "strategy", "context_bins", "target_delay_bins"}
            or run["context_bins"] != 12 or run["target_delay_bins"] not in (0, 6)
            or run["strategy"] not in ("neuron_filtered", "subregion_filtered")):
        raise ValueError("Require one independent neuron/subregion filtered D=0/6 run")
    ACTIVE_RUN = run
    return exp, ev, run, out / "runs" / name(args.run_id or run["run_id"])


def original_experiment(exp):
    return read(path(exp["unfiltered_experiment"]))


def views(exp):
    from training_dataset.handoff import load_dataset_splits
    return load_dataset_splits(path(exp["dataset_generation"]), session_ids=[exp["session_id"]],
                              expected_generation_id=exp["expected_dataset_generation_id"])


def original_support(exp, ev, granularity):
    old = original_experiment(exp)
    old["support_bins"] = 24
    support = delay.support_record(old, dict(ev, split="test"), target_granularity=granularity)
    support["compatibility"]["policy"]["split"] = "val"
    support["compatibility_id"] = fingerprint(support["compatibility"])
    return support


def unfiltered_sources(exp):
    result = {}
    for label, directory in exp["unfiltered_runs"].items():
        directory = path(directory)
        training, attempt = completed(directory, "train")
        resolved = read(directory / "resolved.json")
        trained = read(attempt / "result.json")
        if (fingerprint(resolved) != training["identity"] or
                file_hash(path(trained["checkpoint"])) != trained["checkpoint_sha256"] or
                resolved["run"]["target_delay_bins"] != int(label[-2:]) or
                resolved["configuration"]["training"]["target_granularity"] != "subregion"):
            raise ValueError(f"Incompatible Experiment 2 training artifact: {label}")
        # The original resolved file/hash remains the authoritative reference.
        # Avoid recursively embedding Experiment 1 evaluation/checkpoint records
        # already nested inside each Experiment 2 support object.
        resolved = {k: resolved[k] for k in ("run", "configuration", "experiment")} | {
            "support": {"compatibility": resolved["support"]["compatibility"]}}
        result[label] = dict(directory=str(directory), training_identity=training["identity"],
            checkpoint=trained["checkpoint"], checkpoint_sha256=trained["checkpoint_sha256"],
            resolved_sha256=file_hash(directory / "resolved.json"), resolved=resolved)
    return result


def selection_artifact(exp):
    manifest, attempt = completed(path(exp["output_root"]) / "selection", "score")
    data = read(attempt / "selection.json")
    policy = read(path(exp["filtering_config"]))
    if data["configuration"] != policy or data["dataset_generation_id"] != exp["expected_dataset_generation_id"]:
        raise ValueError("Selection policy/generation changed; publish a new selection first")
    if data["scoring_sources"] != scoring_sources():
        raise ValueError("Selection implementation changed; rerun score before consuming it")
    return data, attempt, manifest


def scoring_sources():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in
            [HERE / "score.py", HERE / "common.py", ROOT / "src/trainer/targets.py"]}


def setup_arguments(exp, run):
    data, attempt, _ = selection_artifact(exp)
    args = delay.setup_arguments(exp, run)
    args.target_selection_json = str(attempt / (run["strategy"] + ".json"))
    return args


def filtered_support(full, selected_views, metadata):
    support = deepcopy(full)
    comp = support["compatibility"]
    comp.update(regional_targets=plain(metadata["regional_targets"]),
                regional_target_policy=plain(metadata["regional_target_policy"]))
    comp["populations"] = {}
    for view in selected_views:
        for sample, row in zip(view.samples, comp["trials"][view.split]):
            if row["sample_id"] != sample.sample_id:
                raise ValueError("Selection changed trial order")
            n, t = sample.neuron_count, sample.sequence_length
            comp["populations"][sample.session_id] = plain(sample.neuron_identity.to_dict(orient="records"))
            row.update(neurons=n, target_counts_hash=fingerprint(sample.neural[23:t, :n].tolist()))
        support["counts"][view.split]["target_cells"] = sum(
            len(r["target_indices"]) * r["neurons"] for r in comp["trials"][view.split])
    support["training_baseline"] = {}
    for session in metadata["regional_targets"]:
        samples = [s for s in selected_views[0].samples if s.session_id == session]
        counts = sum(s.sequence_length-23 for s in samples)
        sums = sum((s.neural[23:s.sequence_length].sum(0, dtype=np.int64) for s in samples))
        support["training_baseline"][session] = dict(mean_counts=(sums/counts).tolist(),
            spike_sums=sums.tolist(), eligible_bins=counts, split="train")
    support["compatibility_id"] = fingerprint(comp)
    return support


def support_record(exp, ev, *, target_granularity="subregion"):
    from trainer.targets import transform_neural_targets
    data, _, manifest = selection_artifact(exp)
    full = original_support(exp, ev, "subregion")
    parent = original_support(exp, ev, "region")
    sources = unfiltered_sources(exp)
    for source in sources.values():
        old = deepcopy(source["resolved"]["support"]["compatibility"])
        old["policy"]["split"] = "val"
        if old != full["compatibility"]:
            raise ValueError("Original unfiltered population, trial/bin support or dataset differs")
    raw = views(exp)
    selection = data["selections"][ACTIVE_RUN["strategy"]]
    *selected, metadata = transform_neural_targets(raw[:3], raw[3], "subregion", selection=selection)
    record = filtered_support(full, selected, metadata)
    record.update(parent_support=parent, selection_identity=manifest["identity"],
        selection=data, unfiltered_sources=sources,
        full_subregion_mapping=full["compatibility"]["regional_targets"])
    return record


def verify_configuration(config, exp, run):
    for label, source in unfiltered_sources(exp).items():
        old, new = deepcopy(source["resolved"]["configuration"]), plain(config)
        for value in (old, new):
            value.pop("dataset", None)
            value["data"].pop("max_space_length", None)
            value["training"].pop("target_selection", None)
            value["training"]["temporal_context"].pop("target_delay_bins", None)
        if old != new:
            raise ValueError(f"Controlled resolved settings differ from {label}: " +
                             str([k for k in set(old)|set(new) if old.get(k) != new.get(k)]))
    if (config.seed != 42 or config.training.num_epochs != 50 or config.training.save_every != 5
            or config.training.eval_every != 1 or config.training.checkpoint_selection != "validation_loss"):
        raise ValueError("Required training controls differ")


def publish_preflight(directory, config, record):
    from trainer.pretrained import build_model, initialize_regional_bias
    model = build_model(config=config, metadata=config.dataset, modal_filter=config.training.modal_filter)
    initialize_regional_bias(model, config.dataset)
    heads = model.mod_stitcher_proj_dict["spike"].stitch_decoder_dict
    widths = {s: h.out_features for s, h in heads.items()}
    if widths != dict(config.dataset.eid_list):
        raise ValueError("Selected output dimensions differ")
    data = record["selection"]
    from trainer.targets import transform_neural_targets
    raw = views(record["unfiltered_sources"]["unfiltered_00"]["resolved"]["experiment"])
    *parents, _ = transform_neural_targets(raw[:3], raw[3], "region")
    mappings = record["parent_support"]["compatibility"]["regional_targets"]
    checked = 0
    for original_view, parent_view in zip(raw[:3], parents):
        for original, parent in zip(original_view.samples, parent_view.samples):
            mapping = mappings[original.session_id]
            exact = np.stack([original.neural[:, c].sum(1) for c in mapping["source_columns"]], -1)
            if not np.array_equal(exact, parent.neural):
                raise ValueError("Full-region integer ground truth reconstruction differs")
            checked += 1
    atomic_json(directory / "selection.json", data)
    atomic_json(directory / "manual-verification.json", dict(output_dimensions=widths,
        selection_identity=record["selection_identity"], selected_mapping=record["compatibility"]["regional_targets"],
        full_region_trials_verified=checked, common_first_target_bin=23,
        training_only_selection=True, delay_independent_selection=True,
        excluded_constants=data["excluded_baselines"][ACTIVE_RUN["strategy"]],
        original_ground_truth="exact sums of original full-population int64 source counts",
        support_matches_experiment_2=True, training_launched=False))
