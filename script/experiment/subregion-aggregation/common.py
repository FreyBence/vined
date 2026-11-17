"""Experiment 2 configuration and exact Experiment 1 compatibility boundary."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_helper(filename, alias):
    spec = importlib.util.spec_from_file_location(alias, HERE.parent / "target-delay" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


delay = load_helper("common.py", "target_delay_common")
read, path, name = delay.read, delay.path, delay.name
stage, completed, atomic_json = delay.stage, delay.completed, delay.atomic_json
plain, fingerprint, file_hash = delay.plain, delay.fingerprint, delay.file_hash
bind_experiment, changed_settings = delay.bind_experiment, delay.changed_settings
import numpy as np


def source_identity():
    sources = delay.source_identity()
    sources.update({str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in HERE.glob("*.py")})
    return sources


def parser(description, *, run=False):
    p = delay.parser(description, run=run)
    p.set_defaults(experiment_config=str(HERE / "configs/experiment.json"))
    return p


def settings(args):
    exp = read(path(args.experiment_config))
    evaluation = read(path(exp["evaluation_config"]))
    if exp["context_bins"] != 12 or exp["delays"] != [0, 2, 4, 6, 8, 12]:
        raise ValueError("Retain Experiment 1's H=12/D=12 common support policy")
    if evaluation["response_windows"] != [1, 3, 6, 12] or evaluation["split"] != "test":
        raise ValueError("Expected original test evaluation windows W=1,3,6,12")
    exp["support_bins"] = 24
    out = path(exp["output_root"])
    required = ROOT / "output/experiment/subregion-aggregation"
    if out != required and required not in out.parents:
        raise ValueError("Output must be under output/experiment/subregion-aggregation")
    return exp, evaluation, out


def run_settings(args):
    exp, evaluation, out = settings(args)
    run = read(path(args.run_config))
    if (set(run) != {"run_id", "context_bins", "target_delay_bins"} or
            run["context_bins"] != 12 or type(run["target_delay_bins"]) is not int or
            run["target_delay_bins"] not in (0, 6)):
        raise ValueError("Expected a subregion run with H=12 and D=0 or 6")
    return exp, evaluation, run, out / "runs" / name(args.run_id or run["run_id"])


setup_arguments = delay.setup_arguments


def regional_sources(exp):
    result = {}
    for label, directory in exp["regional_runs"].items():
        directory = path(directory)
        training, train_attempt = completed(directory, "train")
        evaluation, eval_attempt = completed(directory, "evaluate")
        resolved = read(directory / "resolved.json")
        trained = read(train_attempt / "result.json")
        metrics = read(eval_attempt / "metrics.json")
        support = read(eval_attempt / "support.json")
        expected_delay = {"region_00": 0, "region_06": 6}[label]
        if (fingerprint(resolved) != training["identity"] or
                metrics["training_identity"] != training["identity"] or
                metrics["checkpoint_sha256"] != trained["checkpoint_sha256"] or
                file_hash(Path(trained["checkpoint"])) != trained["checkpoint_sha256"] or
                support != resolved["support"] or
                fingerprint(support["compatibility"]) != support["compatibility_id"] or
                metrics["compatibility_id"] != support["compatibility_id"] or
                resolved["run"]["target_delay_bins"] != expected_delay or
                resolved["configuration"]["training"]["target_granularity"] != "region"):
            raise ValueError(f"{label}: source training/evaluation/checkpoint association differs")
        result[label] = dict(directory=str(directory), training_attempt=str(train_attempt),
            evaluation_attempt=str(eval_attempt), resolved_path=str(directory / "resolved.json"),
            resolved_sha256=file_hash(directory / "resolved.json"), checkpoint=trained["checkpoint"],
            checkpoint_sha256=trained["checkpoint_sha256"], training_identity=training["identity"],
            evaluation_identity=evaluation["identity"], evaluation_manifest=evaluation,
            support=support, resolved=resolved)
    if result["region_00"]["support"] != result["region_06"]["support"]:
        raise ValueError("Regional runs have different populations, trial/bin support or source generations")
    return result


def verify_configuration(config, exp, run):
    """Allow only output widths/mappings and granularity/delay to change."""
    for label, source in regional_sources(exp).items():
        old, new = deepcopy(source["resolved"]["configuration"]), plain(config)
        for value in (old, new):
            value.pop("dataset", None)
            value["data"].pop("max_space_length", None)
            value["training"].pop("target_granularity", None)
            value["training"]["temporal_context"].pop("target_delay_bins", None)
        if old != new:
            keys = [k for k in set(old) | set(new) if old.get(k) != new.get(k)]
            raise ValueError(f"{label}: resolved controlled settings differ in {keys}")
    if (config.seed != 42 or config.training.num_epochs != 50 or
            config.training.save_every != 5 or config.training.eval_every != 1 or
            config.training.checkpoint_selection != "validation_loss" or
            config.training.temporal_context.target_delay_bins != run["target_delay_bins"]):
        raise ValueError("Required controlled training settings differ")


def support_record(exp, evaluation, *, target_granularity="subregion"):
    from training_dataset.handoff import load_dataset_splits
    from trainer.targets import transform_neural_targets
    sources = regional_sources(exp)
    reference = sources["region_00"]["support"]
    # The legacy helper independently checks all source trial IDs, physical times,
    # splits, target-count hashes and ordered units against the original run.
    parent = delay.support_record(exp, evaluation, target_granularity="region")
    if parent != reference:
        raise ValueError("Current regional source data/support differs from Experiment 1")
    native = delay.support_record(exp, evaluation, target_granularity=target_granularity)
    views = load_dataset_splits(path(exp["dataset_generation"]), session_ids=[exp["session_id"]],
        expected_generation_id=exp["expected_dataset_generation_id"])
    *subviews, meta = transform_neural_targets(views[:3], views[3], "subregion")
    *regviews, regmeta = transform_neural_targets(views[:3], views[3], "region")
    diagnostics, reconstruction = {}, []
    for session, mapping in meta["regional_targets"].items():
        parent_mapping = regmeta["regional_targets"][session]
        columns = [c for group in mapping["source_columns"] for c in group]
        intended = [c for group in parent_mapping["source_columns"] for c in group]
        if len(columns) != len(set(columns)) or sorted(columns) != sorted(intended):
            raise ValueError("Subregions do not partition the complete regional neuron population")
        groups = []
        for j, label in enumerate(mapping["regions"]):
            cols = mapping["source_columns"][j]
            values = np.concatenate([s.neural[23:s.sequence_length, j] for s in subviews[0].samples if s.session_id == session])
            groups.append(dict(label=label, parent_region=mapping["parent_regions"][j],
                source_columns=cols, neurons=[mapping["ordered_units"][c] for c in cols], neuron_count=len(cols),
                aggregation="sum", units="spike_count_per_1/60_second_bin",
                training_mean_count=float(values.mean()), training_rate_hz=float(values.mean()*60),
                training_per_neuron_rate_hz=float(values.mean()*60/len(cols)),
                training_zero_fraction=float(np.mean(values == 0)), small_group=len(cols) < 5,
                mapping_limitation="area-only label; no recorded layer" if label == mapping["parent_regions"][j] else None))
        diagnostics[session] = dict(source_neurons=len(columns), target_groups=len(groups), groups=groups,
            omissions=[], duplications=[], excluded_neurons=[])
    for subview, regview in zip(subviews, regviews):
        for sub, reg in zip(subview.samples, regview.samples):
            mapping = meta["regional_targets"][sub.session_id]
            names = regmeta["regional_targets"][reg.session_id]["regions"]
            reconstructed = np.stack([sub.neural[:, np.asarray(mapping["parent_regions"]) == label].sum(1) for label in names], axis=-1)
            if sub.sample_id != reg.sample_id or not np.array_equal(reconstructed, reg.neural):
                raise ValueError(f"Subregion count reconstruction fails: {sub.sample_id}")
            reconstruction.append(dict(sample_id=sub.sample_id, trial_id=sub.trial_id, split=sub.split,
                bins=sub.sequence_length, original_bin_indices=list(range(sub.sequence_length)), exact=True))
    original_directory = Path(sources["region_00"]["evaluation_attempt"]) / "predictions"
    stored = {r["sample_id"]: r for r in read(original_directory / "index.json")}
    if list(stored) != [s.sample_id for s in regviews[2].samples]:
        raise ValueError("Original saved regional prediction trial membership differs")
    for sample in regviews[2].samples:
        with np.load(original_directory / stored[sample.sample_id]["file"], allow_pickle=False) as data:
            t = sample.sequence_length
            if (not np.array_equal(data["observed"], sample.neural[:t]) or
                    not np.array_equal(data["physical_timestamps"], sample.physical_timestamps[:t]) or
                    not np.array_equal(data["target_indices"], np.arange(23, t))):
                raise ValueError(f"Original saved regional ground truth/bin support differs: {sample.sample_id}")
    native.update(parent_support=parent, regional_sources=sources,
        aggregation_diagnostics=diagnostics, reconstruction=reconstruction)
    return native


def publish_preflight(directory, config, record):
    """Publish inspectable mapping, reconstruction, dimensions and training-only plots."""
    from trainer.pretrained import build_model, initialize_regional_bias
    model = build_model(config=config, metadata=config.dataset, modal_filter=config.training.modal_filter)
    initialize_regional_bias(model, config.dataset)
    heads = model.mod_stitcher_proj_dict["spike"].stitch_decoder_dict
    widths = {s: dict(output_channels=head.out_features, source_neurons=record["aggregation_diagnostics"][s]["source_neurons"],
                     target_labels=config.dataset.regional_targets[s]["regions"], head_type=type(head).__name__)
              for s, head in heads.items()}
    if any(widths[s]["output_channels"] != config.dataset.eid_list[s] for s in widths):
        raise ValueError("Model output dimensions differ from subregion mapping")
    atomic_json(directory / "mapping.json", record["aggregation_diagnostics"])
    atomic_json(directory / "reconstruction.json", record["reconstruction"])
    atomic_json(directory / "manual-verification.json", dict(model_dimensions=widths,
        reconstructed_trials=len(record["reconstruction"]), exact_integer_reconstruction=True,
        common_support=record["parent_support"]["counts"], first_target_bin=23,
        full_population_partition=True, unexpected_neuron_exclusions=[],
        ground_truth_matches_original_saved_evaluation=True, training_launched=False,
        limitations="Recorded area-only and noncortical labels remain unsplit. No layer labels are invented."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [g for d in record["aggregation_diagnostics"].values() for g in d["groups"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(axes, ("neuron_count", "training_per_neuron_rate_hz"),
                             ("All contributing neurons", "Training common-support spikes/s/neuron")):
        ax.barh([g["label"] for g in groups], [g[key] for g in groups])
        ax.set(title=title)
    fig.tight_layout(); fig.savefig(directory / "mapping-diagnostics.png", dpi=130); plt.close(fig)
