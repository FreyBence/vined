"""Evaluate a fixed checkpoint on its verified scientific test split."""

import argparse
from contextlib import redirect_stdout
import json
import sys
from evaluation.setup import resolve_setup
from evaluation.predictions import collect_predictions
from evaluation.metrics import MetricConfig
from evaluation.artifacts import publish_evaluation
from utils.paths import output_dir
from utils.progress import configure_progress, logger
from trainer.run_paths import scenario_directory, attempt_name


def main():
    configure_progress()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Explicit selected training checkpoint")
    parser.add_argument("--scenario-id", type=int, help="Scenario for matching evaluation artifact folders")
    parser.add_argument("--attempt", help="Training attempt number associated with this evaluation")
    parser.add_argument("--prediction-direction", choices=["encoding", "decoding"])
    parser.add_argument("--neural-region-selection", choices=["all_recorded", "visual_only"])
    parser.add_argument("--context-mode", required=True, choices=["strict", "full_trial"],
                        help="Required intended context; must match checkpoint")
    parser.add_argument("--context-bins", type=int, choices=[1, 3, 6, 9, 12],
                        help="Required for strict; omitted for full_trial; must match checkpoint")
    parser.add_argument("--dataset-generation", "--data_path", dest="dataset_generation", required=True)
    parser.add_argument("--expected-dataset-generation-id")
    parser.add_argument("--eid", action="append", help="Select a persisted session; repeat for multiple sessions")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--setup-only", action="store_true", help="Verify inputs and restore model without inference")
    parser.add_argument("--aggregation", choices=["session_weighted", "neuron_weighted"], default="session_weighted")
    parser.add_argument("--bps-baseline", choices=["test_mean_count"], default="test_mean_count")
    parser.add_argument("--psth-grouping", choices=["all_trials"], default="all_trials")
    parser.add_argument("--output-dir", default=str(output_dir() / "evaluation"))
    parser.add_argument("--no-predictions", action="store_true", help="Publish metrics/provenance without raw prediction arrays")
    parser.add_argument("--save-plot", "--save_plot", dest="save_plot", action="store_true")
    args = parser.parse_args()
    if (args.scenario_id is None) != (args.attempt is None):
        parser.error("--scenario-id and --attempt must be provided together")
    if args.scenario_id is not None:
        destination = scenario_directory(args.output_dir, args.scenario_id) / attempt_name(args.attempt)
        if not args.setup_only and destination.exists():
            raise FileExistsError(f"Evaluation already exists at {destination}; choose another --output-dir to repeat it")
    # Dataset/model diagnostics belong on stderr; stdout is a compact summary.
    logger.info("evaluation: verifying dataset and restoring checkpoint %s", args.checkpoint)
    with redirect_stdout(sys.stderr):
        setup = resolve_setup(checkpoint_path=args.checkpoint, dataset_generation=args.dataset_generation,
                              context_mode=args.context_mode, context_bins=args.context_bins,
                              expected_generation_id=args.expected_dataset_generation_id,
                              session_ids=args.eid, batch_size=args.batch_size, device=args.device, seed=args.seed,
                              prediction_direction=args.prediction_direction,
                              neural_region_selection=args.neural_region_selection)
    logger.info("evaluation: setup complete on %s (%d test samples)", args.device, len(setup.dataset.samples))
    if args.scenario_id is not None:
        run = setup.provenance["training_run"]
        if run.get("scenario_id") != args.scenario_id or run.get("attempt") != attempt_name(args.attempt):
            raise ValueError("Checkpoint scenario/attempt identity differs from the requested evaluation")
    if args.setup_only:
        print(json.dumps(setup.summary(), indent=2, allow_nan=False))
        return

    collection = collect_predictions(setup)
    artifact = publish_evaluation(collection, args.output_dir,
                                  metric_config=MetricConfig(args.aggregation, args.bps_baseline, args.psth_grouping),
                                  persist_predictions=not args.no_predictions, save_plots=args.save_plot,
                                  scenario_id=args.scenario_id, attempt=args.attempt)
    summary = {
        "evaluation_id": artifact.result["evaluation_id"],
        "global_metrics": artifact.result["global_metrics"],
        "artifact_path": str(artifact.path),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
