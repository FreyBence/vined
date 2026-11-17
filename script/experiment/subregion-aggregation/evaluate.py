"""Evaluate one subregion model natively and on exact reconstructed parent counts."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from common import (parser, run_settings, completed, read, stage, atomic_json, file_hash,
                    fingerprint, source_identity, load_helper, regional_sources)

analysis = load_helper("analysis.py", "target_delay_analysis")


def save_predictions(directory, saved, rows):
    directory.mkdir()
    index = []
    for i, (row, source) in enumerate(zip(saved, rows)):
        filename = f"{i:06d}.npz"
        np.savez_compressed(directory / filename, **{k: v for k, v in row.items() if isinstance(v, np.ndarray)})
        index.append({k: v for k, v in row.items() if not isinstance(v, np.ndarray)} |
                     dict(file=filename, metadata=source.metadata))
    atomic_json(directory / "index.json", index)
    for entry in read(directory / "index.json"):
        with np.load(directory / entry["file"], allow_pickle=False) as data:
            if data["observed"].shape != data["expected"].shape:
                raise ValueError("Invalid saved prediction dimensions")


def reconstruct(rows, support):
    """Sum intensities, never log intensities; verify the full original arrays."""
    source = support["regional_sources"]["region_00"]
    directory = Path(source["evaluation_attempt"]) / "predictions"
    index = {r["sample_id"]: r for r in read(directory / "index.json")}
    result = []
    from scipy.special import logsumexp
    for row in rows:
        mapping = support["compatibility"]["regional_targets"][row.session_id]
        parent_mapping = support["parent_support"]["compatibility"]["regional_targets"][row.session_id]
        names = parent_mapping["regions"]
        columns = [np.flatnonzero(np.asarray(mapping["parent_regions"]) == label) for label in names]
        observed = np.stack([row.observed_neural[:, c].sum(1, dtype=np.int64) for c in columns], axis=-1)
        logp = np.stack([logsumexp(row.predicted_neural[:, c].astype(np.float64), axis=1) for c in columns], axis=-1)
        with np.load(directory / index[row.sample_id]["file"], allow_pickle=False) as original:
            length = len(original["observed"])
            if (not np.array_equal(observed[:length], original["observed"]) or
                    not np.array_equal(row.physical_timestamps[:length], original["physical_timestamps"]) or
                    not np.array_equal(np.flatnonzero(row.encoding_target_mask), original["target_indices"])):
                raise ValueError(f"Original regional target/bin reconstruction differs: {row.sample_id}")
        identity = pd.DataFrame(dict(acronym=names, uuids=[f"{row.session_id}:region:{r}" for r in names],
            source_neuron_count=parent_mapping["region_neuron_counts"]))
        result.append(replace(row, observed_neural=observed, predicted_neural=logp,
            neuron_identity=identity, neuron_mask=np.ones(len(names), dtype=bool)))
    return result


def native_diagnostics(rows, metrics, support):
    result = []
    for session, mapping in support["compatibility"]["regional_targets"].items():
        for j, label in enumerate(mapping["regions"]):
            selected = [r for r in rows if r.session_id == session]
            y = np.concatenate([r.observed_neural[r.encoding_target_mask, j] for r in selected])
            p = np.concatenate([np.exp(r.predicted_neural[r.encoding_target_mask, j].astype(float)) for r in selected])
            metric = next(n for n in metrics["neurons"] if n["session_id"] == session and n["column"] == j)
            temporal = {}
            for w in ("1", "3", "6", "12"):
                trials = [t["windows"][w]["neurons"][j] for t in metrics["trials"] if t["session_id"] == session]
                ratios = [t["variance_ratio"] for t in trials]
                temporal[w] = dict(
                    nearly_constant_prediction_trials=sum(t["predicted_std"] <= 1e-6 for t in trials),
                    defined_correlations=sum(t["correlation"]["value"] is not None for t in trials),
                    unavailable_correlations=sum(t["correlation"]["value"] is None for t in trials),
                    fraction_positive_correlation=float(np.mean([t["correlation"]["value"] > 0 for t in trials if t["correlation"]["value"] is not None])) if any(t["correlation"]["value"] is not None for t in trials) else None,
                    fraction_low_variance=sum(v is not None and v < 1e-4 for v in ratios)/len(trials),
                    thresholds="descriptive only: near-constant std<=1e-6 spikes/s/group; low predicted/observed variance<1e-4; no filtering")
            result.append(dict(session_id=session, label=label, parent_region=mapping["parent_regions"][j],
                neuron_count=mapping["region_neuron_counts"][j], single_bin=metric["single_bin"],
                windows=metric["windows"], observed_mean_count=float(y.mean()), predicted_mean_count=float(p.mean()),
                observed_range=[int(y.min()), int(y.max())], predicted_range=[float(p.min()), float(p.max())],
                observed_zero_fraction=float(np.mean(y == 0)), predicted_std_count=float(p.std()),
                temporal_variation=temporal,
                description="Inspect within-trial variance/correlation at each W; no filtering or selection from test responses"))
    return result


def native_plots(directory, saved, diagnostics, evaluation, mapping):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist([r["neuron_count"] for r in diagnostics], bins=20)
    axes[0].set(xlabel="Contributing neurons", ylabel="Subregions", title="Contributing group sizes")
    axes[1].hist([r["observed_mean_count"]*60/r["neuron_count"] for r in diagnostics], bins=20)
    axes[1].set(xlabel=f"Observed spikes/s/neuron ({evaluation['split']} diagnostic)", ylabel="Subregions")
    fig.tight_layout(); fig.savefig(directory / "counts_rates.png", dpi=130); plt.close(fig)
    indices = np.linspace(0, len(saved)-1, min(len(saved), evaluation["representative_trials"]), dtype=int)
    for i in indices:
        row = saved[i]
        names = mapping[row["session_id"]]["regions"]
        for start in range(0, len(names), 8):
            columns = list(range(start, min(start+8, len(names))))
            fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2.2*len(columns)), squeeze=False)
            for ax, j in zip(axes[:, 0], columns):
                ix = row["target_indices"]
                ax.plot(row["time"][ix], row["observed"][ix, j]*60, color="gray", alpha=.35, label="Observed raw W=1")
                ax.plot(row["time"][ix], row["expected"][ix, j]*60, label="Predicted raw W=1")
                ep = row["curve_endpoints"]
                for field in ("observed", "expected"):
                    ax.plot(row["time"][ep], analysis.rolling(row[field][:, j], 12, ep)*60, label=field+" W=12 trailing mean")
                ax.set(title=f"{names[j]} trial {row['trial_id']}", ylabel="spikes/s/subregion", xlabel="Seconds from onset")
            axes[0, 0].legend(fontsize=7, ncol=2)
            fig.tight_layout(); fig.savefig(directory / f"trial_{i:06d}_groups_{start:03d}.png", dpi=130); plt.close(fig)


def main():
    p = parser(__doc__, run=True)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--device")
    args = p.parse_args()
    exp, evaluation, run, directory = run_settings(args)
    training, train_attempt = completed(directory, "train")
    resolved = read(directory / "resolved.json")
    if (fingerprint(resolved) != training["identity"] or resolved["experiment"] != exp or
            resolved["evaluation"] != evaluation or resolved["run"] != run):
        raise ValueError("Current run settings differ from recorded training")
    # Reverify referenced regional artifacts before consuming stored observations.
    sources = regional_sources(exp)
    if sources != resolved["support"]["regional_sources"]:
        raise ValueError("Regional source artifacts changed since preflight")
    trained = read(train_attempt / "result.json")
    checkpoint = Path(trained["checkpoint"])
    if file_hash(checkpoint) != trained["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    effective = dict(evaluation, target_granularity="subregion")
    if args.device:
        effective["device"] = args.device
    identity = fingerprint(dict(training=training["identity"], checkpoint=trained["checkpoint_sha256"], evaluation=effective))
    with stage(directory, "evaluate", identity, skip=args.skip_completed, allow_new_inputs=True) as attempt:
        if attempt is None:
            return
        from evaluation.setup import resolve_setup
        from evaluation.predictions import collect_predictions
        setup = resolve_setup(checkpoint_path=checkpoint,
            dataset_generation=resolved["configuration"]["dataset"]["dataset_path"],
            expected_generation_id=exp["expected_dataset_generation_id"], session_ids=[exp["session_id"]],
            context_mode="strict", context_bins=12, batch_size=effective["batch_size"],
            device=effective["device"], seed=effective["seed"], prediction_direction="encoding",
            neural_region_selection="all_recorded")
        if (setup.configuration["target_granularity"] != "subregion" or
                setup.configuration["temporal_context"].get("target_delay_bins", 0) != run["target_delay_bins"] or
                setup.configuration["temporal_context"]["target_support_bins"] != 24):
            raise ValueError("Checkpoint granularity, delay or support differs")
        rows = collect_predictions(setup).predictions
        support = resolved["support"]
        if [r.sample_id for r in rows] != [r["sample_id"] for r in support["compatibility"]["trials"]["test"]]:
            raise ValueError("Evaluated trial membership differs")
        native, saved_native = analysis.analyze(rows, support, effective)
        native["definitions"] = {k: v.replace("neuron", "subregion") for k, v in native["definitions"].items()}
        native["definitions"]["channel"] = "recorded anatomical subregion; summed counts"
        diagnostics = native_diagnostics(rows, native, support)
        parent_rows = reconstruct(rows, support)
        metrics, saved = analysis.analyze(parent_rows, support["parent_support"], effective)
        metrics.update(run=run, run_id=directory.name, training_identity=training["identity"],
            checkpoint_sha256=trained["checkpoint_sha256"], compatibility_id=support["parent_support"]["compatibility_id"],
            evaluation_configuration=dict(effective, target_granularity="region"),
            execution_sources=dict(training=read(train_attempt / "resolved.json")["sources"], evaluation=source_identity()))
        if metrics["single_bin"]["cells"] != support["parent_support"]["counts"]["test"]["target_cells"]:
            raise ValueError("Parent evaluation coverage differs")
        save_predictions(attempt / "predictions", saved, parent_rows)
        save_predictions(attempt / "native-predictions", saved_native, rows)
        atomic_json(attempt / "metrics.json", metrics)
        atomic_json(attempt / "native-metrics.json", native)
        atomic_json(attempt / "subregion-diagnostics.json", diagnostics)
        atomic_json(attempt / "support.json", support["parent_support"])
        atomic_json(attempt / "native-support.json", support)
        atomic_json(attempt / "provenance.json", setup.summary())
        analysis.plots(attempt / "plots", saved, effective["response_windows"], effective["representative_trials"], label=directory.name)
        native_plots(attempt / "native-plots", saved_native, diagnostics, effective, support["compatibility"]["regional_targets"])
        (attempt / "report.md").write_text(
            f"# {directory.name}\n\nParent-region single-bin NLL: {metrics['single_bin']['poisson_nll']:.6g}.\n\n"
            "Parent targets exactly reconstruct Experiment 1 observations. Native losses describe a different target space and must not be ranked against regional losses. "
            "Native diagnostics retain every group, undefined correlations and low-activity targets. All results are exploratory; no test-derived filtering is applied.\n", encoding="utf-8")
        with (attempt / "report.md").open("a", encoding="utf-8") as stream:
            stream.write("\n| Subregion | Neurons | W=12 mean correlation | W=12 variance ratio | Near-constant trials |\n| --- | ---: | ---: | ---: | ---: |\n")
            for d in diagnostics:
                stream.write(f"| {d['label']} | {d['neuron_count']} | {d['windows']['12']['correlation']['mean']} | {d['windows']['12']['variance_ratio']['mean']} | {d['temporal_variation']['12']['nearly_constant_prediction_trials']} |\n")


if __name__ == "__main__":
    main()
