"""Run training or evaluation with scenario settings taking precedence."""

import argparse
from pathlib import Path
import subprocess
import sys

from utils.config_utils import load_config
from utils.paths import REPO_ROOT, output_dir
from trainer.run_paths import selected_attempt


CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
COMMON = {"dataset_generation", "expected_dataset_generation_id", "eid"}
TRAIN = {
    "base_path", "training_config", "setup_only", "seed", "epochs", "batch_size",
    "validation_batch_size", "learning_rate", "weight_decay", "scheduler", "eval_every",
    "checkpoint_selection", "no_wandb", "resume_checkpoint", "multi_gpu", "debug",
    "num_sessions", "mask_mode", "mask_ratio", "mixed_training", "enc_task_var",
}
EVAL = {
    "checkpoint", "setup_only", "batch_size", "device", "seed", "aggregation",
    "bps_baseline", "psth_grouping", "output_dir", "no_predictions", "save_plot",
}
LEGACY_FLAGS = {"base_path", "num_sessions", "mask_mode", "mask_ratio",
                "mixed_training", "enc_task_var", "multi_gpu"}
PATH_FIELDS = {"dataset_generation", "checkpoint", "base_path", "training_config",
               "resume_checkpoint", "output_dir"}
BOOL_FIELDS = {"setup_only", "no_wandb", "multi_gpu", "debug", "mixed_training",
               "no_predictions", "save_plot"}


def build_command(operation, scenario_id, *, attempt=None, checkpoint=None):
    """Resolve a scenario and its selected training attempt without test scoring."""
    if operation not in ("train", "eval") or type(scenario_id) is not int:
        raise ValueError("Choose train or eval and an integer scenario ID")
    if operation != "eval" and (attempt is not None or checkpoint is not None):
        raise ValueError("--attempt and --checkpoint apply only to evaluation")
    if attempt is not None and checkpoint is not None:
        raise ValueError("Choose --attempt or --checkpoint, not both")
    scenarios = load_config(CONFIG_ROOT / "scenarios.json")["scenarios"]
    ids = [item["id"] for item in scenarios]
    if any(type(value) is not int for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Scenario IDs must be unique integers")
    scenario = next((item for item in scenarios if item["id"] == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Unknown scenario ID {scenario_id}; available IDs: {ids}")
    direction = scenario["prediction_direction"]
    context = scenario["temporal_context"]
    selection = scenario["neural_region_selection"]
    if direction not in ("encoding", "decoding") or selection not in ("all_recorded", "visual_only"):
        raise ValueError("Unsupported scenario direction or neural region selection")
    mode, bins = context.get("mode"), context.get("bins")
    if (mode not in ("strict", "full_trial")
            or (mode == "strict" and (type(bins) is not int or bins not in (1, 3, 6, 9, 12)))
            or (mode == "full_trial" and bins is not None)):
        raise ValueError("Scenario requires strict context with bins 1|3|6|9|12, or full_trial without bins")
    run = load_config(CONFIG_ROOT / "run.json")
    if set(run) - COMMON - {"train", "eval"}:
        raise ValueError("Unsupported run configuration fields")
    settings = load_config(CONFIG_ROOT / "evaluation/default.json") if operation == "eval" else {}
    settings.update({key: value for key, value in run.items() if key in COMMON})
    settings.update(run.get(operation, {}))
    settings.update({key: value for key, value in scenario.items() if key in COMMON})
    settings.update(scenario.get(operation, {}))
    allowed = COMMON | (TRAIN if operation == "train" else EVAL)
    if set(settings) - allowed:
        raise ValueError(f"Unsupported {operation} settings: {sorted(set(settings) - allowed)}")
    for key in ("dataset_generation",):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"Set {key} in src/configs/run.json or the selected scenario's {operation} object")
    command = [sys.executable, str(REPO_ROOT / "src" / f"{operation}.py"),
               "--context-mode", mode, "--neural-region-selection", selection,
               "--model_mode" if operation == "train" else "--prediction-direction", direction]
    if operation == "train":
        command.extend(["--scenario-id", str(scenario_id), "--scenario-name", scenario["name"]])
    else:
        explicit = checkpoint if checkpoint is not None else settings.get("checkpoint")
        if attempt is not None or explicit is None:
            training_settings = {**run.get("train", {}), **scenario.get("train", {})}
            base = training_settings.get("base_path") or output_dir()
            base = Path(base).expanduser()
            base = base if base.is_absolute() else REPO_ROOT / base
            number, chosen = selected_attempt(base / "runs", scenario_id, attempt)
            settings["checkpoint"] = str(chosen)
            command.extend(["--scenario-id", str(scenario_id), "--attempt", number])
        else:
            if not isinstance(explicit, str) or not explicit.strip():
                raise ValueError("checkpoint must be a nonempty path or null for the latest scenario attempt")
            settings["checkpoint"] = explicit
    if bins is not None:
        command.extend(["--context-bins", str(bins)])
    for key, value in settings.items():
        if value is None:
            continue
        flag = "--" + (key if key in LEGACY_FLAGS else key.replace("_", "-"))
        if key in BOOL_FIELDS:
            if type(value) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
            if value:
                command.append(flag)
        elif key == "eid":
            values = value if isinstance(value, list) else [value]
            if not values or any(not isinstance(item, str) or not item for item in values):
                raise ValueError("eid must be a session ID or nonempty list of session IDs")
            if operation == "train" and len(values) != 1:
                raise ValueError("Training accepts one eid or null for all persisted sessions")
            for item in values:
                command.extend([flag, item])
        else:
            if isinstance(value, (dict, list, bool)):
                raise ValueError(f"{key} must be a scalar configuration value")
            if key in PATH_FIELDS:
                path = Path(value).expanduser()
                value = str(path if path.is_absolute() else REPO_ROOT / path)
            command.extend([flag, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["train", "eval"])
    parser.add_argument("scenario_id", type=int)
    parser.add_argument("--attempt", help="Evaluate a numbered training attempt; default: latest")
    parser.add_argument("--checkpoint", help="Explicit checkpoint override, including older UUID runs")
    args = parser.parse_args()
    try:
        command = build_command(args.operation, args.scenario_id, attempt=args.attempt, checkpoint=args.checkpoint)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
