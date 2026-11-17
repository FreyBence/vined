"""Validation inference on native targets and the fixed full original region targets."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
from common import (parser, settings, run_settings, completed, read, stage, atomic_json,
                    fingerprint, file_hash, source_identity, unfiltered_sources, original_support,
                    selection_artifact, views, analysis, module, path)

legacy = module("subregion-aggregation", "evaluate.py", "information_evaluation_helpers")


def reconstruct(rows, raw, parent_support, selected_mapping, excluded):
    """Ground truth uses original observations; prediction uses outputs + train constants."""
    from trainer.targets import transform_neural_targets
    *parents, _ = transform_neural_targets(raw[:3], raw[3], "region")
    original = {s.sample_id:s for s in raw[1].samples}
    regional = {s.sample_id:s for s in parents[1].samples}
    result, decomposition = [], []
    for row in rows:
        sample, full = original[row.sample_id], regional[row.sample_id]
        session = row.session_id
        mapping = selected_mapping[session]
        parent_mapping = parent_support["compatibility"]["regional_targets"][session]
        names = parent_mapping["regions"]
        length, padded = sample.sequence_length, len(row.temporal_mask)
        if (row.trial_id != sample.trial_id or row.session_id != sample.session_id or
                not np.array_equal(row.physical_timestamps[:length], sample.physical_timestamps[:length]) or
                not np.array_equal(np.flatnonzero(row.encoding_target_mask), np.arange(23,length))):
            raise ValueError("Common trial/time/bin support differs")
        native_truth = np.stack([sample.neural[:length,c].sum(1) for c in mapping["source_columns"]], -1)
        if not np.array_equal(native_truth, row.observed_neural[:length,:len(mapping["regions"])]):
            raise ValueError("Native selected target is not the declared source count sum")
        observed = np.zeros((padded, len(names)), dtype=np.int64)
        observed[:length] = full.neural[:length]
        expected = np.zeros((padded,len(names)), dtype=float)
        model_expected = np.exp(row.predicted_neural.astype(float))
        for j, parent in enumerate(names):
            included = np.flatnonzero(np.asarray(mapping["parent_regions"]) == parent)
            constant = excluded[session]["parent_mean_counts"].get(parent, 0.)
            learned = model_expected[:,included].sum(1) if len(included) else np.zeros(padded)
            expected[:,j] = learned+constant
            decomposition.append(dict(sample_id=row.sample_id, region=parent, included_groups=len(included),
                excluded_constant_count=constant, mean_learned_count=float(learned[23:length].mean()),
                constant_fraction_of_prediction=float(constant / max(expected[23:length,j].mean(), 1e-8)),
                constant_only=len(included)==0))
        # Zero-count region expectations are floored only to represent log intensity.
        logp = np.log(np.maximum(expected,1e-8))
        truth_hash = fingerprint(observed[23:length].tolist())
        reference = next(r for r in parent_support["compatibility"]["trials"]["val"] if r["sample_id"] == row.sample_id)
        if truth_hash != reference["target_counts_hash"]:
            raise ValueError("Full original region ground truth differs from common support")
        result.append(replace(row, observed_neural=observed, predicted_neural=logp,
            neuron_identity=full.neuron_identity, neuron_mask=np.ones(len(names),dtype=bool),
            metadata=deepcopy(full.metadata)))
    return result, decomposition


def baseline_audit(metrics):
    trials = metrics["trials"]
    gains = [t["single_bin"]["nll_improvement"] for t in trials]
    return dict(reference="constant counts fitted on original train common support",
        weighting="global NLL and gain: eligible cell weighted; mean trial gain: equal trials",
        global_poisson_nll=metrics["single_bin"]["poisson_nll"],
        global_paired_gain=metrics["single_bin"]["nll_improvement"],
        equal_trial_mean_gain=float(np.mean(gains)), median_trial_gain=float(np.median(gains)),
        fraction_trials_improved=float(np.mean(np.asarray(gains)>0)),
        eligible_elements=metrics["single_bin"]["cells"])


def main():
    p = parser(__doc__)
    p.add_argument("--run", choices=[f"{s}_{d:02}" for s in ("unfiltered","neuron_filtered","subregion_filtered") for d in (0,6)], required=True)
    p.add_argument("--skip-completed", action="store_true")
    p.add_argument("--device")
    args = p.parse_args()
    exp, ev, out = settings(args)
    run_id = args.run
    delay = int(run_id[-2:])
    strategy = run_id[:-3]
    if strategy == "unfiltered":
        source = unfiltered_sources(exp)[run_id]
        directory = Path(source["directory"])
        training, train_attempt = completed(directory, "train")
        resolved = source["resolved"]
        support = original_support(exp, ev, "subregion")
        old = deepcopy(resolved["support"]["compatibility"]); old["policy"]["split"] = "val"
        if old != support["compatibility"]:
            raise ValueError("Unfiltered source common support or original data differs")
        parent_support = original_support(exp, ev, "region")
        excluded = {s:dict(parent_mean_counts={}) for s in support["compatibility"]["regional_targets"]}
        output = out / "runs" / run_id
    else:
        directory = out / "runs" / run_id
        training, train_attempt = completed(directory, "train")
        resolved = read(directory / "resolved.json")
        if fingerprint(resolved) != training["identity"] or resolved["experiment"] != exp:
            raise ValueError("Filtered training configuration differs")
        support = resolved["support"]
        data, _, manifest = selection_artifact(exp)
        if manifest["identity"] != support["selection_identity"] or data != support["selection"]:
            raise ValueError("Selection differs from trained population")
        parent_support = support["parent_support"]
        excluded = data["excluded_baselines"][strategy]
        output = directory
    trained = read(train_attempt / "result.json")
    checkpoint = path(trained["checkpoint"])
    if file_hash(checkpoint) != trained["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    if resolved["run"]["target_delay_bins"] != delay:
        raise ValueError("Run label and trained delay differ")
    effective = dict(ev, device=args.device or ev["device"])
    sources = source_identity()
    identity = fingerprint(dict(training=training["identity"], checkpoint=trained["checkpoint_sha256"],
        evaluation=effective, support=support["compatibility_id"], sources=sources))
    with stage(output, "evaluate", identity, skip=args.skip_completed, allow_new_inputs=True) as attempt:
        if attempt is None:
            return
        from evaluation.setup import resolve_setup
        from evaluation.predictions import collect_predictions
        setup = resolve_setup(checkpoint_path=checkpoint, dataset_generation=path(exp["dataset_generation"]),
            expected_generation_id=exp["expected_dataset_generation_id"], session_ids=[exp["session_id"]],
            context_mode="strict", context_bins=12, batch_size=effective["batch_size"], device=effective["device"],
            seed=42, prediction_direction="encoding", neural_region_selection="all_recorded", split="val")
        context = setup.configuration["temporal_context"]
        if (context["target_support_bins"] != 24 or context.get("target_delay_bins",0) != delay
                or setup.configuration["target_granularity"] != "subregion"):
            raise ValueError("Checkpoint delay/granularity/support differs")
        rows = collect_predictions(setup).predictions
        if [r.sample_id for r in rows] != [r["sample_id"] for r in support["compatibility"]["trials"]["val"]]:
            raise ValueError("Validation membership differs")
        native, saved_native = analysis.analyze(rows, support, effective)
        native["definitions"] = {key: value.replace("neuron", "subregion")
                                 for key, value in native["definitions"].items()}
        native["definitions"]["channel"] = "actual selected anatomical-group summed counts"
        native.update(interpretation="Target-dependent selected-population diagnostics; not cross-population NLL ranking",
            baseline_audit=baseline_audit(native))
        diagnostics = legacy.native_diagnostics(rows, native, support)
        for diagnostic in diagnostics:
            diagnostic["description"] = "Validation selected-population diagnostic; no validation-derived screening"
        parent_rows, decomposition = reconstruct(rows, views(exp), parent_support,
            support["compatibility"]["regional_targets"], excluded)
        metrics, saved = analysis.analyze(parent_rows, parent_support, effective)
        metrics.update(run=dict(run_id=run_id, target_delay_bins=delay, strategy=strategy), run_id=run_id,
            training_identity=training["identity"], checkpoint_sha256=trained["checkpoint_sha256"],
            compatibility_id=parent_support["compatibility_id"], evaluation_configuration=effective,
            baseline_audit=baseline_audit(metrics), execution_sources=sources,
            interpretation="Validation model-development results; excluded activity predicted by training constants")
        if metrics["single_bin"]["cells"] != parent_support["counts"]["val"]["target_cells"]:
            raise ValueError("Incomplete full-region eligible elements")
        legacy.save_predictions(attempt / "predictions", saved, parent_rows)
        legacy.save_predictions(attempt / "native-predictions", saved_native, rows)
        for filename, value in [("metrics.json",metrics),("native-metrics.json",native),
                ("subregion-diagnostics.json",diagnostics),("decomposition.json",decomposition),
                ("support.json",parent_support),("native-support.json",support),("provenance.json",setup.summary())]:
            atomic_json(attempt / filename,value)
        analysis.plots(attempt / "plots", saved, ev["response_windows"], ev["representative_trials"], label=run_id+" validation")
        # Existing plots include explicit all-group native traces and count/rate histograms.
        legacy.native_plots(attempt / "native-plots", saved_native, diagnostics, ev, support["compatibility"]["regional_targets"])
        (attempt / "report.md").write_text(f"# {run_id} — validation\n\n"
            f"Full-region NLL: {metrics['single_bin']['poisson_nll']:.8g}; "
            f"gain over training constant: {metrics['single_bin']['nll_improvement']:.8g}.\n\n"
            "Native scores are target-dependent diagnostics. Excluded activity uses training constants; "
            "this does not recover omitted temporal activity. Single-bin NLL and smoothed-curve similarity "
            "answer different questions. See metrics, subregion diagnostics and decomposition for every target and trial.\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
