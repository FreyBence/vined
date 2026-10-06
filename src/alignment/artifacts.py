"""Standalone unsplit aligned generations with verified atomic publication."""

from dataclasses import dataclass
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import tempfile
from uuid import UUID

import numpy as np
import pandas as pd

from utils.provenance import file_hash, fingerprint, source_hashes, write_json
from .inputs import prepare_inputs, _id
from .temporal import AlignedTrial, align_trials


@dataclass(frozen=True)
class AlignmentGeneration:
    generation_id: str
    path: Path
    trials: tuple[AlignedTrial, ...]
    manifest: dict


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _implementation():
    return dict(sources=source_hashes(
        "src/alignment/__init__.py", "src/alignment/inputs.py",
        "src/alignment/temporal.py", "src/alignment/artifacts.py", "src/prepare_alignment.py"),
        packages={name: version(name) for name in ("numpy", "pandas", "pyarrow")})


def _validate(trials):
    if not trials:
        raise ValueError("Cannot publish an empty alignment generation")
    units = trials[0].neuron_identity
    if str(UUID(trials[0].session_id)) != trials[0].session_id:
        raise ValueError("Alignment session ID must be a canonical UUID")
    identity = ["eid", "pid", "probe", "collection", "revision", "source_unit_id", "recording_index"]
    if (not set(identity).issubset(units) or not len(units)
            or units.duplicated([key for key in identity if key != "pid"]).any()):
        raise ValueError("Missing or duplicate scoped neuron identities")
    ids = [_id(trial.trial_id) for trial in trials]
    if ids != sorted(set(ids)):
        raise ValueError("Aligned trials must have unique IDs in canonical order")
    for trial in trials:
        t, n = trial.bin_count, len(units)
        scalars = [trial.stim_on, trial.stim_off, trial.aligned_start,
                   trial.aligned_end, trial.bin_size, trial.discarded_tail_duration]
        if (trial.session_id != trials[0].session_id or not units.eid.eq(trial.session_id).all()
                or not trial.neuron_identity.equals(units) or not np.isfinite(scalars).all()
                or not isinstance(t, int) or isinstance(t, bool) or t < 1 or trial.bin_size <= 0):
            raise ValueError(f"Trial {trial.trial_id}: invalid aligned identity or timing")
        starts, ends, centers = trial.bin_start_times, trial.bin_end_times, trial.bin_center_times
        tolerance = 4 * max(abs(np.spacing(trial.stim_on)), abs(np.spacing(trial.stim_off)))
        if (any(array.shape != (t,) or array.dtype != np.dtype("float64")
                or not np.isfinite(array).all() for array in (starts, ends, centers))
                or np.any(ends <= starts) or np.any(np.diff(centers) <= 0)
                or not np.array_equal(starts[1:], ends[:-1])
                or np.any(np.abs(ends - starts - trial.bin_size) > 2 * tolerance)
                or np.any(np.abs(centers - (starts + (ends - starts) / 2)) > tolerance)
                or trial.aligned_start != starts[0] or trial.aligned_end != ends[-1]
                or abs(starts[0] - trial.stim_on) > tolerance
                or trial.stim_off <= trial.stim_on or ends[-1] > trial.stim_off + tolerance
                or trial.discarded_tail_duration < 0
                or trial.discarded_tail_duration >= trial.bin_size + tolerance
                or abs(max(0., trial.stim_off - ends[-1]) - trial.discarded_tail_duration) > tolerance):
            raise ValueError(f"Trial {trial.trial_id}: inconsistent complete neural bin grid")
        if (trial.neural_activity.shape != (t, n) or trial.neural_activity.dtype != np.dtype("int64")
                or np.any(trial.neural_activity < 0)
                or trial.visual_features.shape != (t, 768)
                or trial.visual_features.dtype != np.dtype("float32")
                or not np.isfinite(trial.visual_features).all()
                or not np.allclose(np.linalg.norm(trial.visual_features, axis=1), 1, atol=1e-5)):
            raise ValueError(f"Trial {trial.trial_id}: invalid aligned arrays")
        metadata = trial.alignment_metadata
        if metadata.get("clock") != "session_seconds" or metadata.get("neural_intervals") != "[start,end)":
            raise ValueError("Missing aligned physical-time provenance")
        if not set(("neural_configuration", "neural_recordings", "neural_trial_sources",
                    "neural_generation_id", "neural_request_id", "neural_source_interval",
                    "neural_source_bin_slice", "visual_definition", "visual_completion",
                    "visual_outcome", "visual_resampling", "resampling")).issubset(metadata):
            raise ValueError("Incomplete alignment source/resampling provenance")
        resampling = metadata["resampling"]
        times = np.asarray(resampling["source_times"])
        left = np.asarray(resampling["left_source_indices"])
        right = np.asarray(resampling["right_source_indices"])
        weights = np.asarray(resampling["interpolation_weights"])
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
                or len(resampling["source_observation_ids"]) != len(times)
                or len(resampling["source_schedule_indices"]) != len(times)
                or left.shape != (t,) or right.shape != (t,) or weights.shape != (t,)
                or left.dtype.kind not in "iu" or right.dtype.kind not in "iu"
                or np.any(left < 0) or np.any(right < left) or np.any(right >= len(times))
                or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1)):
            raise ValueError("Invalid visual interpolation provenance")


def _load_alignment(path, *, expected_generation_id=None):
    """Load precisely one generation, verifying all payloads without source access."""
    import json
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "generation_id"}
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "aligned_trials"
            or manifest.get("complete") is not True
            or fingerprint(payload) != manifest.get("generation_id")
            or expected_generation_id is not None and expected_generation_id != manifest["generation_id"]):
        raise ValueError("Incomplete, incompatible, or mismatched alignment generation")
    entries = manifest["trials"]
    files = {"units.parquet"} | {f"trials/{index:06d}.npz" for index in range(len(entries))}
    if set(manifest["files"]) != files:
        raise ValueError("Alignment file accounting mismatch")
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or file_hash(target) != digest:
            raise ValueError(f"Alignment payload hash mismatch: {name}")
    units = pd.read_parquet(path / "units.parquet")
    trials = []
    for index, entry in enumerate(entries):
        with np.load(path / f"trials/{index:06d}.npz", allow_pickle=False) as arrays:
            required = {"bin_start_times", "bin_center_times", "bin_end_times", "neural_activity", "visual_features"}
            if set(arrays.files) != required:
                raise ValueError("Unsupported aligned array schema")
            metadata = deepcopy(entry["alignment_metadata"])
            resampling = metadata["resampling"]
            for name in ("source_times", "interpolation_weights"):
                resampling[name] = np.asarray(resampling[name], dtype=np.float64)
            for name in ("left_source_indices", "right_source_indices"):
                resampling[name] = np.asarray(resampling[name], dtype=np.int64)
            trials.append(AlignedTrial(
                manifest["eid"], entry["trial_id"], entry["stim_on"], entry["stim_off"],
                entry["aligned_start"], entry["aligned_end"], entry["bin_size"],
                entry["bin_count"], entry["discarded_tail_duration"],
                arrays["bin_start_times"], arrays["bin_center_times"], arrays["bin_end_times"],
                arrays["neural_activity"], units.copy(deep=True), arrays["visual_features"], metadata))
    _validate(trials)
    if manifest["requested_trial_ids"] != [trial.trial_id for trial in trials]:
        raise ValueError("Alignment trial accounting mismatch")
    return AlignmentGeneration(manifest["generation_id"], path, tuple(trials), manifest)


def load_alignment(path, *, expected_generation_id=None):
    """Verify a selected generation; malformed schema raises ValueError."""
    try:
        return _load_alignment(path, expected_generation_id=expected_generation_id)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError(f"Malformed alignment generation: {exc}") from exc


def publish_alignment(trials, output_dir):
    """Publish supplied aligned trials only after a successful consumer readback."""
    trials = tuple(trials)
    _validate(trials)
    implementation = _implementation()
    parent = Path(output_dir).resolve() / trials[0].session_id
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alignment-", dir=parent) as temporary:
        staging = Path(temporary) / "generation"
        (staging / "trials").mkdir(parents=True)
        trials[0].neuron_identity.to_parquet(staging / "units.parquet", index=False)
        entries = []
        for index, trial in enumerate(trials):
            np.savez_compressed(staging / f"trials/{index:06d}.npz",
                bin_start_times=trial.bin_start_times, bin_center_times=trial.bin_center_times,
                bin_end_times=trial.bin_end_times, neural_activity=trial.neural_activity,
                visual_features=trial.visual_features)
            entries.append({name: _json(getattr(trial, name)) for name in (
                "trial_id", "stim_on", "stim_off", "aligned_start", "aligned_end", "bin_size",
                "bin_count", "discarded_tail_duration", "alignment_metadata")})
        files = {name: file_hash(staging / name) for name in
                 ["units.parquet"] + [f"trials/{index:06d}.npz" for index in range(len(trials))]}
        manifest = dict(schema_version=1, kind="aligned_trials", complete=True,
                        eid=trials[0].session_id, requested_trial_ids=[trial.trial_id for trial in trials],
                        trials=entries, implementation=implementation, files=files)
        manifest["generation_id"] = fingerprint(manifest)
        write_json(staging / "manifest.json", manifest)
        verified = load_alignment(staging, expected_generation_id=manifest["generation_id"])
        destination = parent / manifest["generation_id"]
        if destination.exists():
            raise FileExistsError(f"Alignment generation exists; load it explicitly: {destination}")
        if _implementation() != implementation:
            raise ValueError("Alignment implementation changed during publication")
        staging.rename(destination)
    return AlignmentGeneration(verified.generation_id, destination, verified.trials, verified.manifest)


def generate_alignment(neural, visual_path, output_dir, **input_options):
    """Prepare explicit modality inputs, align, and publish one session."""
    return publish_alignment(align_trials(prepare_inputs(neural, visual_path, **input_options)), output_dir)
