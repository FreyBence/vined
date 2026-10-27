"""Validation-only comparison of a regional run and its neuron-level source."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer.artifacts import atomic_json, load_training_checkpoint, plain
from trainer.pretrained import model_from_checkpoint
from trainer.selection import select_neural_regions
from trainer.targets import transform_neural_targets, regional_training_means
from training_dataset import temporal_context_view
from training_dataset.handoff import load_dataset_splits


def _predict(model, sample, device):
    length = sample.sequence_length
    positions = torch.tensor(sample.temporal_positions[:length], device=device)[None]
    mask = torch.tensor(sample.temporal_mask[:length], device=device)[None]
    view = temporal_context_view(positions, mask, direction="encoding",
                                 mode=model.context_mode, context_bins=model.context_bins)
    output = model(visual_features=torch.tensor(sample.visual[:length], device=device)[None],
                   temporal_positions=positions, temporal_mask=mask,
                   session_id=[sample.session_id], context_view=view)
    if not torch.equal(output.target_eligibility, view.target_mask):
        raise ValueError("Prediction and observed temporal support differ")
    return output.neural_prediction[0].double().cpu().numpy(), view.target_mask[0].cpu().numpy()


def regional_diagnostics(checkpoint, source_checkpoint, output_dir, *, device="cuda"):
    """Save same-bin validation metrics and unsmoothed stimulus-relative rates."""
    state = load_training_checkpoint(checkpoint)
    source_state = load_training_checkpoint(source_checkpoint)
    history_path = Path(checkpoint).resolve().parent / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if history[:len(state["history"])] != state["history"]:
        raise ValueError("Run history does not extend the selected checkpoint history")
    config, source_config = state["config"], source_state["config"]
    dataset = config["dataset"]
    if dataset.get("target_granularity") != "region" or config["training"]["objective"] != "encoding":
        raise ValueError("Expected regional encoding checkpoint")
    if state["selection"]["epoch"] != state["epoch"]:
        raise ValueError("Use the training-selected regional checkpoint")
    if source_state["selection"]["epoch"] != source_state["epoch"]:
        raise ValueError("Use the training-selected neuron checkpoint")
    for key in ("dataset_generation_id", "selected_session_ids", "dataset_metadata"):
        if dataset[key] != source_config["dataset"][key]:
            raise ValueError(f"Different source and regional datasets: {key}")
    if config["training"]["temporal_context"] != source_config["training"]["temporal_context"]:
        raise ValueError("Temporal context/support differs")
    source_path = Path(source_checkpoint).resolve()
    expected = state["run"]["pretrained"]
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("Neuron comparison checkpoint differs from initialization source")
    train, val, test, metadata = load_dataset_splits(dataset["dataset_path"],
        session_ids=dataset["selected_session_ids"], expected_generation_id=dataset["dataset_generation_id"])
    selection = source_config["dataset"].get("neural_region_selection", "all_recorded")
    policy = source_config["dataset"].get("neural_selection_policy", {})
    train, val, test, metadata = select_neural_regions((train, val, test), metadata, selection,
                                                       regions=policy.get("regions"))
    for sample in (*train.samples, *val.samples):
        if plain(sample.neuron_identity.to_dict(orient="records")) != source_config["dataset"]["ordered_units"][sample.session_id]:
            raise ValueError("Source neuron order differs from checkpoint")
    region_sets = {tuple(mapping["regions"]) for mapping in dataset["regional_targets"].values()}
    if len(region_sets) != 1:
        raise ValueError("This validation comparison requires the same ordered regions across selected sessions")
    regions = list(next(iter(region_sets)))
    rt, rv, _, regional_meta = transform_neural_targets((train, val, test), metadata, "region",
        regions=None if "regional_target_policy" in dataset else regions)
    if plain(regional_meta["regional_targets"]) != dataset["regional_targets"]:
        raise ValueError("Regional anatomical mapping differs from trained targets")
    baseline = regional_training_means(rt)
    if baseline != dataset["regional_training_baseline"]:
        raise ValueError("Training baseline differs from recorded statistics")
    for name, view in (("train", rt), ("val", rv)):
        actual = [{"sample_id": s.sample_id, "session_id": s.session_id,
                   "trial_id": s.trial_id, "split": s.split} for s in view.samples]
        if actual != state["compatibility"]["memberships"][name] or actual != source_state["compatibility"]["memberships"][name]:
            raise ValueError("Persisted training/validation memberships differ")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    model = model_from_checkpoint(checkpoint).to(device).eval()
    source_model = model_from_checkpoint(source_checkpoint).to(device).eval()
    rows, arrays = [], {}
    with torch.inference_mode():
        for i, (sample, regional) in enumerate(zip(val.samples, rv.samples)):
            log_counts, valid = _predict(model, regional, device)
            neuron_logs, source_valid = _predict(source_model, sample, device)
            if not np.array_equal(valid, source_valid):
                raise ValueError("Models score different temporal bins")
            mapping = dataset["regional_targets"][sample.session_id]
            source_rates = np.stack([np.exp(neuron_logs[:, columns]).sum(-1)
                                     for columns in mapping["source_columns"]], axis=-1)
            observed = regional.neural[:sample.sequence_length]
            if not np.array_equal(observed, np.stack([sample.neural[:sample.sequence_length, columns].sum(-1)
                                       for columns in mapping["source_columns"]], axis=-1)):
                raise ValueError("Regional target does not equal original neuron sum")
            rates = np.exp(log_counts)
            constant = np.broadcast_to(baseline[sample.session_id]["mean_counts"], rates.shape)
            if not np.isfinite(rates).all() or not np.isfinite(source_rates).all():
                raise ValueError("Non-finite predicted expected counts")
            rows.append(dict(session_id=sample.session_id, sample_id=sample.sample_id,
                trial_id=sample.trial_id, eligible_bins=int(valid.sum()),
                bin_size=sample.bin_size, source_neuron_counts=mapping["region_neuron_counts"]))
            arrays[f"trial_{i}_time"] = sample.physical_timestamps[:sample.sequence_length] - sample.stim_on
            arrays[f"trial_{i}_eligible"] = valid
            for key, values in (("observed", observed), ("regional", rates),
                                ("aggregated_neuron", source_rates), ("baseline", constant)):
                arrays[f"trial_{i}_{key}"] = values
                arrays[f"trial_{i}_{key}_spikes_per_second_per_neuron"] = (
                    values / sample.bin_size / np.asarray(mapping["region_neuron_counts"])[None])
            print(f"Validation trial {i + 1}/{len(val.samples)}: {sample.trial_id}", flush=True)
    observed = np.concatenate([arrays[f"trial_{i}_observed"][arrays[f"trial_{i}_eligible"]]
                               for i in range(len(rows))])
    constant = np.concatenate([arrays[f"trial_{i}_baseline"][arrays[f"trial_{i}_eligible"]]
                               for i in range(len(rows))])
    null_nll = constant - observed * np.log(np.maximum(constant, 1e-300))
    scores = {}
    for name in ("regional", "baseline", "aggregated_neuron"):
        rates = np.concatenate([arrays[f"trial_{i}_{name}"][arrays[f"trial_{i}_eligible"]]
                                 for i in range(len(rows))])
        nll = rates - observed * np.log(np.maximum(rates, 1e-300))
        residuals = [arrays[f"trial_{i}_{name}"][arrays[f"trial_{i}_eligible"]] -
                     arrays[f"trial_{i}_{name}"][arrays[f"trial_{i}_eligible"]].mean(axis=0)
                     for i in range(len(rows)) if rows[i]["eligible_bins"]]
        observed_residuals = [arrays[f"trial_{i}_observed"][arrays[f"trial_{i}_eligible"]] -
                     arrays[f"trial_{i}_observed"][arrays[f"trial_{i}_eligible"]].mean(axis=0)
                     for i in range(len(rows)) if rows[i]["eligible_bins"]]
        rate_values = np.concatenate([arrays[f"trial_{i}_{name}_spikes_per_second_per_neuron"][arrays[f"trial_{i}_eligible"]]
                                     for i in range(len(rows))])
        scores[name] = dict(poisson_nll=float(nll.mean()),
            pooled_bits_per_spike=float((null_nll.sum() - nll.sum()) / observed.sum() / np.log(2)),
            predicted_mean_counts=rates.mean(0).tolist(), observed_mean_counts=observed.mean(0).tolist(),
            mean_calibration_error=(rates.mean(0) - observed.mean(0)).tolist(),
            absolute_mean_calibration_error=np.abs(rates.mean(0) - observed.mean(0)).tolist(),
            relative_mean_calibration_error=(rates.mean(0) / observed.mean(0) - 1).tolist(),
            predicted_mean_spikes_per_second_per_neuron=rate_values.mean(0).tolist(),
            predicted_std_counts=rates.std(0).tolist(), observed_std_counts=observed.std(0).tolist(),
            within_trial_predicted_std_counts=np.concatenate(residuals).std(0).tolist(),
            within_trial_observed_std_counts=np.concatenate(observed_residuals).std(0).tolist(),
            temporal_count_correlation=[float(np.corrcoef(rates[:,r], observed[:,r])[0,1])
                if rates[:,r].std() > 1e-12 and observed[:,r].std() > 0 else None for r in range(len(regions))])
    result = dict(split="val", dataset_generation_id=dataset["dataset_generation_id"],
        checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        source_checkpoint=str(source_path), source_checkpoint_sha256=expected["sha256"],
        selected_epoch=state["epoch"], training_epochs_completed=len(history),
        regions=regions, eligible_bins=int(len(observed)),
        baseline=baseline, metrics=scores, samples=rows,
        definitions=dict(poisson_nll="mean lambda - y*log(lambda); full=False, matching training",
            bits_per_spike="pooled (training-null NLL - model NLL) / observed spikes / ln(2)",
            calibration_error="predicted mean count minus observed mean count",
            variability="population std across eligible bins; within-trial std centers each trial first",
            rate="count / persisted bin_size / recorded source neuron count; no smoothing",
            support="identical complete H=12 encoding targets; full-trial context"))
    np.savez_compressed(destination / "trajectories.npz", **arrays)
    atomic_json(destination / "results.json", result)
    _plots(destination, arrays, rows, regions, history)
    lines = [f"# {' / '.join(regions)} regional validation experiment", "", "Validation-only results; no test-based selection.", "",
             "| Predictor | Poisson NLL | Bits/spike | Mean count | Mean calibration error |", "|---|---:|---:|---:|---:|"]
    for name, metric in scores.items():
        lines.append(f"| {name} | {metric['poisson_nll']:.6f} | {metric['pooled_bits_per_spike']:.6f} | {metric['predicted_mean_counts'][0]:.6f} | {metric['mean_calibration_error'][0]:.6f} |")
    lines += ["", f"Observed mean: {observed.mean():.6f} counts/bin. Eligible validation bins: {len(observed)}.",
              "", "Selected epoch (zero based): " + str(state["epoch"]), "",
              "The effective training configuration, full training history, source initialization parameter lists and pre-training checks are in the parent run directory: `config.json`, `history.json`, `run.json` and `preflight.json`. `results.json` records checkpoint hashes, exact validation membership, baseline sufficient statistics and metric definitions; `trajectories.npz` retains original-bin counts and spikes/second/source-neuron rates.", "",
              f"The regional model {'outperformed' if scores['regional']['poisson_nll'] < scores['baseline']['poisson_nll'] else 'did not outperform'} the training-derived constant baseline on selected-checkpoint validation NLL, and {'outperformed' if scores['regional']['poisson_nll'] < scores['aggregated_neuron']['poisson_nll'] else 'did not outperform'} the summed neuron model.", "",
              f"Regional temporal standard deviation: {scores['regional']['predicted_std_counts'][0]:.6f} counts/bin; observed: {scores['regional']['observed_std_counts'][0]:.6f}. Within-trial predicted standard deviation: {scores['regional']['within_trial_predicted_std_counts'][0]:.6f}. Temporal count correlation: {scores['regional']['temporal_count_correlation'][0]}. Mean relative calibration error: {100 * scores['regional']['relative_mean_calibration_error'][0]:.3f}%.", "",
              "Regional aggregation can cancel heterogeneous neuronal responses. Observed variability includes Poisson noise, while predictions describe expected counts. Validation was used for checkpoint selection, so these are selection-set diagnostics rather than unbiased held-out performance. Full-trial attention is noncausal. One session and one regional run do not establish a general benefit of aggregation."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _plots(destination, arrays, rows, regions, history):
    order = np.argsort([row["eligible_bins"] for row in rows])
    selected = [int(order[0]), int(order[len(order)//2]), int(order[-1])]
    for r, region in enumerate(regions):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
        for ax, i in zip(axes, selected):
            valid = arrays[f"trial_{i}_eligible"]
            time = arrays[f"trial_{i}_time"][valid]
            scale = 1 / rows[i]["bin_size"] / rows[i]["source_neuron_counts"][r]
            for name, label in (("observed", "Observed"), ("regional", "Region model"),
                                ("aggregated_neuron", "Summed neuron model"), ("baseline", "Training constant")):
                ax.plot(time, arrays[f"trial_{i}_{name}"][valid,r] * scale,
                        label=label, linewidth=0.8, alpha=0.65 if name == "observed" else 1)
            ax.set(title=f"{region}: original trial {rows[i]['trial_id']}", ylabel="spikes/s/neuron")
        axes[0].legend(ncol=4)
        axes[-1].set_xlabel("Seconds from stimulus onset (original 60 Hz bins)")
        fig.savefig(destination / f"{region}_trajectories.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4), constrained_layout=True)
    # Histories contain scalar validation metrics alongside the training objective.
    for section, key in (("training", "train_loss"), ("validation", "eval_loss")):
        values = [(row["epoch"]+1, row[section][key]) for row in history if key in row[section]]
        if values:
            ax.plot(*zip(*values), label=key)
    ax.set(xlabel="Epoch", ylabel="Mean Poisson NLL (full=False)")
    ax.legend()
    fig.savefig(destination / "training_history.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = regional_diagnostics(args.checkpoint, args.source_checkpoint, args.output_dir, device=args.device)
    print(json.dumps(result["metrics"], indent=2))
