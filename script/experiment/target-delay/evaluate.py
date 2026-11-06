"""Evaluate every trial of a completed delay run, without updating the model."""
from pathlib import Path
import numpy as np
from common import (parser, run_settings, completed, read, stage, atomic_json, file_hash,
                    fingerprint, source_identity)
from analysis import analyze, plots
from utils.progress import iter_progress


def main():
    p = parser(__doc__,run=True)
    p.add_argument("--skip-completed",action="store_true")
    p.add_argument("--device",help="Override inference device without changing scientific definitions")
    args = p.parse_args()
    exp, evaluation, run, directory = run_settings(args)
    manifest, train_attempt = completed(directory,"train")
    resolved = read(directory/"resolved.json")
    if fingerprint(resolved) != manifest["identity"]:
        raise ValueError("Resolved run snapshot differs from completed training identity")
    if (resolved["experiment"] != exp or resolved["evaluation"] != evaluation or resolved["run"] != run
            or resolved["sources"] != source_identity()):
        raise ValueError("Current experiment/configuration/source differs from the recorded run")
    result = read(train_attempt/"result.json")
    checkpoint = Path(result["checkpoint"])
    if file_hash(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    effective = dict(evaluation, target_granularity=resolved["configuration"]["training"]["target_granularity"])
    if args.device:
        effective["device"] = args.device
    identity = fingerprint(dict(training=manifest["identity"], checkpoint=result["checkpoint_sha256"], evaluation=effective))
    with stage(directory,"evaluate",identity,skip=args.skip_completed,allow_new_inputs=True) as attempt:
        if attempt is None:
            return
        from evaluation.setup import resolve_setup
        from evaluation.predictions import collect_predictions
        setup = resolve_setup(checkpoint_path=checkpoint, dataset_generation=resolved["configuration"]["dataset"]["dataset_path"],
            expected_generation_id=exp["expected_dataset_generation_id"],
            session_ids=[exp["session_id"]] if exp["session_id"] else None,
            context_mode="strict",context_bins=run["context_bins"],
            batch_size=effective["batch_size"],device=effective["device"],seed=effective["seed"],
            prediction_direction="encoding",neural_region_selection="all_recorded")
        if setup.configuration["target_granularity"] != effective["target_granularity"]:
            raise ValueError("Checkpoint target granularity differs from the run")
        context = setup.configuration["temporal_context"]
        if (context.get("target_delay_bins",0) != run["target_delay_bins"] or
                context["target_support_bins"] != exp["support_bins"]):
            raise ValueError("Checkpoint delay/support differs from experiment")
        collection = collect_predictions(setup)
        expected = resolved["support"]["compatibility"]["trials"]["test"]
        if [r.sample_id for r in collection.predictions] != [r["sample_id"] for r in expected]:
            raise ValueError("Evaluated trial membership differs from preflight")
        metrics,saved = analyze(collection.predictions,resolved["support"],effective)
        metrics.update(run=run, run_id=directory.name, compatibility_id=resolved["support"]["compatibility_id"],
            checkpoint_sha256=result["checkpoint_sha256"], evaluation_configuration=effective,
            training_identity=manifest["identity"])
        predictions = attempt/"predictions"; predictions.mkdir()
        index = []
        for i,(row,source) in enumerate(iter_progress(zip(saved,collection.predictions),
                "evaluation: saving predictions", total=len(saved), unit="trials")):
            filename=f"{i:06d}.npz"
            np.savez_compressed(predictions/filename,**{k:v for k,v in row.items() if isinstance(v,np.ndarray)})
            index.append({k:v for k,v in row.items() if not isinstance(v,np.ndarray)} | dict(file=filename,metadata=source.metadata))
        atomic_json(predictions/"index.json",index)
        atomic_json(attempt/"provenance.json",setup.summary())
        atomic_json(attempt/"metrics.json",metrics)
        atomic_json(attempt/"support.json",resolved["support"])
        plots(attempt/"plots",saved,effective["response_windows"],effective["representative_trials"],label=directory.name)
        (attempt/"report.md").write_text(
            f"# Target-delay evaluation: {directory.name}\n\n"
            f"Entire test split: {len(saved)} trials. Single-bin NLL: {metrics['single_bin']['poisson_nll']:.6g}. "
            f"NLL improvement over training mean: {metrics['single_bin']['nll_improvement']:.6g}.\n\n"
            "See metrics.json for per-neuron/trial/region/contrast and W summaries; predictions retain every original bin. "
            "Plots use deterministic uniformly spaced trials, with all retained curve endpoints. "
            "These exploratory delay comparisons do not establish significance or independent test confirmation.\n",encoding="utf-8")
        # Readback and exact count coverage are required before stage completion.
        if metrics["single_bin"]["cells"] != resolved["support"]["counts"]["test"]["target_cells"]:
            raise ValueError("Evaluation cell coverage differs from preflight")
        stored = read(predictions/"index.json")
        for item in iter_progress(stored, "evaluation: prediction readback", total=len(stored), unit="trials"):
            with np.load(predictions/item["file"],allow_pickle=False) as payload:
                if payload["observed"].shape != payload["expected"].shape:
                    raise ValueError("Unreadable prediction artifact")


if __name__ == "__main__":
    main()
