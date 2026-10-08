"""Isolated evaluation publication and verified, inference-free readback."""

from dataclasses import dataclass, fields
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import MetricConfig, compute_metrics
from evaluation.predictions import EvaluationPrediction, PredictionCollection
from trainer.artifacts import plain
from utils.paths import REPO_ROOT
from utils.provenance import file_hash, fingerprint, write_json


ARRAY_FIELDS = ("temporal_positions", "physical_timestamps", "bin_start_times", "bin_end_times",
                "temporal_mask", "neuron_mask", "observed_neural", "observed_visual",
                "neural_prediction_mask", "visual_prediction_mask", "encoding_target_mask", "decoding_target_mask",
                "predicted_neural", "predicted_visual")


@dataclass(frozen=True)
class EvaluationArtifact:
    path: Path
    manifest: dict
    result: dict
    collection: PredictionCollection | None


def _software():
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                  capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        revision_id = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except OSError:
        revision_id, dirty = None, None
    return dict(python=sys.version, git_revision=revision_id, git_dirty=dirty,
                cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                packages={name: version(name) for name in ("numpy", "torch", "pandas", "matplotlib")},
                sources={str(path.relative_to(REPO_ROOT)).replace("\\", "/"): file_hash(path)
                         for path in sorted((REPO_ROOT / "src/evaluation").glob("*.py"))})


def _plots(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    sessions = result["session_results"]
    names = list(result["global_metrics"])
    figure, axes = plt.subplots(len(names), 1, figsize=(9, 3 * len(names)), squeeze=False)
    for axis, name in zip(axes[:, 0], names):
        for index, (session, data) in enumerate(sessions.items()):
            value = data["metrics"][name]["value"]
            if value is None:
                axis.text(index, 0.9, "unavailable", transform=axis.get_xaxis_transform(),
                          ha="center", va="top", fontsize=8)
            else:
                axis.bar(index, value, color="steelblue")
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_xticks(range(len(sessions)), [key[:8] for key in sessions])
        axis.set_xlim(-0.6, len(sessions) - 0.4)
        axis.set_title(name)
        axis.set_xlabel("Session (ID prefix)")
    figure.tight_layout()
    figure.savefig(directory / "session_metrics.png", dpi=150)
    plt.close(figure)
    regions = result.get("bits_per_spike_by_region", {})
    if regions:
        figure, axis = plt.subplots(figsize=(max(6, len(regions) * 0.6), 3))
        for index, (label, score) in enumerate(regions.items()):
            if score["value"] is None:
                axis.text(index, 0.9, "unavailable", transform=axis.get_xaxis_transform(), ha="center", fontsize=8)
            else:
                axis.bar(index, score["value"], color="steelblue")
        axis.set_xticks(range(len(regions)), list(regions), rotation=45, ha="right")
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set_ylabel("Bits per spike")
        axis.set_title("BPS by trained brain region")
        figure.tight_layout()
        figure.savefig(directory / "region_bits_per_spike.png", dpi=150)
        plt.close(figure)
    for session, data in sessions.items():
        # One explicitly identified representative profile per session avoids
        # producing an unbounded figure set for large neural populations.
        neuron = next((item for item in data["neurons"] if item["psth"] is not None), None)
        if neuron is None:
            continue
        profile = neuron["psth"]
        centers = np.mean(profile["relative_bin_edges"], axis=1)
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.plot(centers, profile["observed"], label="Observed")
        axis.plot(centers, profile["predicted"], label="Predicted")
        axis.set_xlabel("Time from stimulus onset (seconds)")
        axis.set_ylabel("Mean count per bin")
        axis.set_title(f"Session {session[:8]}, neuron column {neuron['column']}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / f"{session}_neuron_{neuron['column']}_psth.png", dpi=150)
        plt.close(figure)


def publish_evaluation(collection, output_dir, *, metric_config=None,
                       persist_predictions=True, save_plots=False):
    """Write a new complete artifact, verify it, and never replace an existing run."""
    config = MetricConfig() if metric_config is None else metric_config
    result = compute_metrics(collection, config=config)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    evaluation_id = uuid.uuid4().hex
    final, staging = root / evaluation_id, root / (".staging-" + evaluation_id)
    staging.mkdir(exist_ok=False)
    try:
        software = _software()
        manifest = dict(schema_version=2, kind="evaluation", complete=True, evaluation_id=evaluation_id,
                        configuration=plain(collection.configuration), provenance=plain(collection.provenance),
                        metric_configuration=plain(result["configuration"]), software=software,
                        predictions_persisted=bool(persist_predictions), samples=[], units={})
        (staging / "units").mkdir()
        if persist_predictions:
            (staging / "predictions").mkdir()
        for index, row in enumerate(collection.predictions):
            session = row.session_id
            if session not in manifest["units"]:
                relative = f"units/{len(manifest['units']):06d}.parquet"
                row.neuron_identity.to_parquet(staging / relative, index=False)
                manifest["units"][session] = relative
            scalar = {field.name: plain(getattr(row, field.name)) for field in fields(row)
                      if field.name not in (*ARRAY_FIELDS, "neuron_identity")}
            entry = dict(scalar=scalar, arrays={}, path=None)
            if persist_predictions:
                arrays = {name: getattr(row, name) for name in ARRAY_FIELDS if getattr(row, name) is not None}
                relative = f"predictions/{index:06d}.npz"
                np.savez_compressed(staging / relative, **arrays)
                entry.update(path=relative, arrays={name: dict(shape=list(value.shape), dtype=str(value.dtype))
                                                     for name, value in arrays.items()})
            manifest["samples"].append(entry)
        result.update(evaluation_id=evaluation_id, evaluation_configuration=plain(collection.configuration),
                      software=software, prediction_artifact="predictions" if persist_predictions else None)
        if save_plots:
            _plots(result, staging / "plots")
        result["plots"] = [str(path.relative_to(staging)).replace("\\", "/")
                           for path in sorted((staging / "plots").glob("*.png"))]
        write_json(staging / "result.json", result)
        manifest["files"] = {str(path.relative_to(staging)).replace("\\", "/"): file_hash(path)
                             for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest["manifest_id"] = fingerprint(manifest)
        write_json(staging / "manifest.json", manifest)
        load_evaluation(staging, expected_evaluation_id=evaluation_id)
        if final.exists():
            raise FileExistsError(final)
        staging.rename(final)
    except Exception:
        if staging.parent == root and staging.name == ".staging-" + evaluation_id:
            shutil.rmtree(staging)
        raise
    return EvaluationArtifact(final, manifest, result, collection if persist_predictions else None)


def _read_json(path):
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)


def load_evaluation(path, *, expected_evaluation_id=None, expected_checkpoint_sha256=None,
                    expected_dataset_generation_id=None):
    """Verify one explicit evaluation directory; never infer or repair missing files."""
    path = Path(path).resolve()
    manifest = _read_json(path / "manifest.json")
    identity = manifest.pop("manifest_id", None)
    if (manifest.get("schema_version") != 2 or manifest.get("kind") != "evaluation"
            or manifest.get("complete") is not True or fingerprint(manifest) != identity):
        raise ValueError("Incomplete, unsupported or corrupt evaluation manifest")
    manifest["manifest_id"] = identity
    for expected, actual in ((expected_evaluation_id, manifest["evaluation_id"]),
                             (expected_checkpoint_sha256, manifest["provenance"]["checkpoint_sha256"]),
                             (expected_dataset_generation_id, manifest["provenance"]["dataset_generation_id"])):
        if expected is not None and expected != actual:
            raise ValueError("Evaluation artifact identity is incompatible with requested input")
    actual_files = {str(item.relative_to(path)).replace("\\", "/") for item in path.rglob("*") if item.is_file()}
    if actual_files != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Evaluation artifact has missing or unaccounted payload files")
    for relative, digest in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or target.is_symlink() or file_hash(target) != digest:
            raise ValueError(f"Corrupt evaluation payload: {relative}")
    result = _read_json(path / "result.json")
    if (result["evaluation_id"] != manifest["evaluation_id"]
            or result["provenance"] != manifest["provenance"]
            or result["configuration"] != manifest["metric_configuration"]
            or result["evaluation_configuration"] != manifest["configuration"]):
        raise ValueError("Evaluation result and manifest contracts disagree")
    if any(relative not in manifest["files"] for relative in manifest["units"].values()):
        raise ValueError("Unaccounted unit identity table")
    units = {session: pd.read_parquet(path / relative) for session, relative in manifest["units"].items()}
    rows, seen = [], set()
    for entry in manifest["samples"]:
        scalar = entry["scalar"]
        if scalar["sample_id"] in seen or scalar["session_id"] not in units:
            raise ValueError("Duplicate sample or unknown neural population in evaluation artifact")
        seen.add(scalar["sample_id"])
        if not manifest["predictions_persisted"]:
            if entry["path"] is not None or entry["arrays"]:
                raise ValueError("Unexpected prediction payload in metrics-only artifact")
            continue
        if entry["path"] not in manifest["files"]:
            raise ValueError("Unaccounted prediction archive")
        with np.load(path / entry["path"], allow_pickle=False) as archive:
            if set(archive.files) != set(entry["arrays"]):
                raise ValueError("Prediction archive schema differs from manifest")
            arrays = {name: archive[name].copy() for name in archive.files}
        for name, value in arrays.items():
            if name not in ARRAY_FIELDS or dict(shape=list(value.shape), dtype=str(value.dtype)) != entry["arrays"][name]:
                raise ValueError("Prediction array dimensions or dtype differ from manifest")
        for name in ARRAY_FIELDS:
            if name not in arrays:
                if name not in ("predicted_neural", "predicted_visual"):
                    raise ValueError("Missing scientific prediction array")
                arrays[name] = None
        rows.append(EvaluationPrediction(**scalar, neuron_identity=units[scalar["session_id"]].copy(deep=True), **arrays))
    collection = PredictionCollection(tuple(rows), manifest["configuration"], manifest["provenance"]) if rows else None
    memberships = {}
    for entry in manifest["samples"]:
        scalar = entry["scalar"]
        memberships.setdefault(scalar["session_id"], []).append(scalar["sample_id"])
    if (not memberships or set(memberships) != set(result["session_results"])
            or any(ids != result["session_results"][session]["sample_ids"] for session, ids in memberships.items())):
        raise ValueError("Result membership differs from recorded source observations")
    if manifest["predictions_persisted"]:
        if collection is None:
            raise ValueError("Empty persisted prediction collection")
        # Validate readback semantics and metric association using the stored policy.
        config = manifest["metric_configuration"]
        recomputed = compute_metrics(collection, config=MetricConfig(config["aggregation"], config["bps_baseline"], config["psth_grouping"],
                                                                     region_bps=config.get("region_bps", False)))
        if (recomputed["configuration"] != manifest["metric_configuration"]
                or recomputed["session_results"] != result["session_results"]
                or recomputed.get("bits_per_spike_by_region") != result.get("bits_per_spike_by_region")
                or recomputed["global_metrics"] != result["global_metrics"]):
            raise ValueError("Persisted metrics do not match associated scientific predictions")
    return EvaluationArtifact(path, manifest, result, collection)
