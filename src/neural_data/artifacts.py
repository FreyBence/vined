"""Standalone neural generations with checked, immutable publication."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from utils.provenance import file_hash, fingerprint, source_hashes, write_json
from .counting import DEFAULT_BIN_SIZE, CountWindow, NeuralCounts, count_intervals, count_trials
from .provenance import json_value
from .sources import load_population


@dataclass
class NeuralGeneration:
    generation_id: str
    path: Path
    data: NeuralCounts
    manifest: dict


def _implementation():
    return dict(sources=source_hashes(
        "src/neural_data/sources.py", "src/neural_data/counting.py",
        "src/neural_data/provenance.py", "src/neural_data/artifacts.py",
        "src/neural_data/__init__.py", "src/prepare_neural_data.py",
        "src/session_data/access.py", "src/session_data/spikes.py", "src/utils/provenance.py"),
        packages={name: version(name) for name in
                  ("numpy", "pandas", "pyarrow", "iblatlas", "ibllib", "ONE-api")})


def _validate(data):
    identity = ["eid", "probe", "collection", "revision", "source_unit_id"]
    if (not set(identity + ["recording_index"]).issubset(data.units)
            or data.units.duplicated(identity).any()
            or not data.units["eid"].eq(data.eid).all()):
        raise ValueError("Invalid persisted unit identities")
    n_units = len(data.units)
    for index, record in enumerate(data.recordings):
        rows = data.units.loc[data.units.recording_index == index, "source_unit_id"].tolist()
        if (record["source"]["eid"] != data.eid or rows != record["selected_unit_ids"]
                or not record.get("content_sha256")):
            raise ValueError("Recording/unit accounting or source content identity mismatch")
    if not data.units.recording_index.isin(range(len(data.recordings))).all():
        raise ValueError("Unknown unit recording index")
    ids = [window.request_id for window in data.windows]
    if len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("Duplicate or invalid request IDs")
    for window in data.windows:
        edges = window.bin_edges
        shape = (max(0, len(edges) - 1), n_units)
        if (edges.ndim != 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)
                or window.counts.shape != shape or window.counts.dtype != np.dtype("int64")
                or np.any(window.counts < 0) or window.valid.shape != shape
                or window.valid.dtype != np.dtype("bool") or window.coverage_status.shape != shape
                or window.coverage_status.dtype != np.dtype("uint8")
                or np.any(window.coverage_status > 5) or window.observed_duration.shape != shape
                or not np.array_equal(window.valid, window.coverage_status == 1)):
            raise ValueError(f"Invalid neural window: {window.request_id}")
        if len(edges):
            if len(edges) < 2 or tuple(edges[[0, -1]]) != tuple(window.interval):
                raise ValueError("Window edges differ from requested interval")
        elif window.reason != "missing_or_nonfinite_timing" or np.isfinite(window.interval).all():
            raise ValueError("Missing window has no unavailable timing outcome")
        if window.status not in ("available", "partial", "unavailable", "empty_selection"):
            raise ValueError("Unknown request outcome")


def load_generation(path, *, expected_generation_id=None):
    """Load an explicitly selected generation, independent of current source files."""
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    generation_id = manifest.get("generation_id")
    payload = {key: value for key, value in manifest.items() if key != "generation_id"}
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or fingerprint(payload) != generation_id
            or (expected_generation_id is not None and generation_id != expected_generation_id)):
        raise ValueError("Incomplete, incompatible, or mismatched neural generation")
    expected_files = {"units.parquet"} | {f"windows/{index:06d}.npz"
                                         for index in range(len(manifest["windows"]))}
    if set(manifest["files"]) != expected_files:
        raise ValueError("Neural generation file accounting mismatch")
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or file_hash(target) != digest:
            raise ValueError(f"Neural artifact content mismatch: {name}")
    units = pd.read_parquet(path / "units.parquet")
    windows = []
    for index, outcome in enumerate(manifest["windows"]):
        with np.load(path / f"windows/{index:06d}.npz", allow_pickle=False) as arrays:
            windows.append(CountWindow(outcome["request_id"], tuple(arrays["interval"]),
                arrays["bin_edges"], arrays["counts"], arrays["valid"],
                arrays["coverage_status"], arrays["observed_duration"],
                outcome["status"], outcome["reason"]))
    configuration = manifest["configuration"].copy()
    configuration["unit_coverage"] = {int(k): v for k, v in configuration["unit_coverage"].items()}
    data = NeuralCounts(manifest["eid"], units, tuple(manifest["recordings"]),
                        tuple(windows), configuration, tuple(manifest["trial_sources"]))
    _validate(data)
    if ([w.request_id for w in windows] != manifest["requested_ids"]
            or len(data.recordings) != len(manifest["requested_recordings"])
            or dict(Counter(w.status for w in windows)) != manifest["outcomes"]):
        raise ValueError("Neural generation request accounting mismatch")
    return NeuralGeneration(generation_id, path, data, manifest)


def generate_neural(access, eid, recordings, output_dir, *, intervals=None,
                    request_ids=None, event=None, offsets=None, bin_size=DEFAULT_BIN_SIZE,
                    quality=None, anatomy=None, unit_coverage=None,
                    trial_collection="alf", trial_revision=None, workers=1):
    """Process one session and atomically publish a new fully accounted generation.

    Exactly one of intervals or event is required. Existing generations are never
    returned in place of a new run; an identical destination raises FileExistsError.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (intervals is None) == (event is None):
        raise ValueError("Select either absolute intervals or a trial event")
    if event is not None and (offsets is None or request_ids is not None):
        raise ValueError("Trial processing needs offsets and uses original trial IDs")
    if intervals is not None and offsets is not None:
        raise ValueError("Absolute intervals cannot have trial offsets")
    recordings = tuple(recordings)
    implementation = _implementation()
    population = load_population(access, eid, recordings, quality=quality, anatomy=anatomy)
    if event is None:
        data = count_intervals(population, intervals, bin_size=bin_size,
                               request_ids=request_ids, unit_coverage=unit_coverage, workers=workers)
        requested_count = len(intervals)
    else:
        trials = access.load_trials(population.eid, collection=trial_collection, revision=trial_revision)
        data = count_trials(population, trials, event=event, offsets=offsets,
                            bin_size=bin_size, unit_coverage=unit_coverage, workers=workers)
        requested_count = len(trials.data)
    del population
    if len(data.windows) != requested_count or len(data.recordings) != len(recordings):
        raise ValueError("Not all requested neural sources and intervals were accounted for")
    _validate(data)
    destination_root = Path(output_dir).resolve() / data.eid
    destination_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".neural-", dir=destination_root) as temporary:
        staging = Path(temporary) / "generation"
        (staging / "windows").mkdir(parents=True)
        data.units.to_parquet(staging / "units.parquet", index=False)
        files = {"units.parquet": file_hash(staging / "units.parquet")}
        outcomes = []

        def write_window(item):
            index, window = item
            name = f"windows/{index:06d}.npz"
            np.savez_compressed(staging / name, interval=np.asarray(window.interval, dtype=np.float64),
                bin_edges=window.bin_edges, counts=window.counts, valid=window.valid,
                coverage_status=window.coverage_status, observed_duration=window.observed_duration)
            return name, file_hash(staging / name), dict(
                request_id=window.request_id, status=window.status, reason=window.reason)

        if workers > 1 and len(data.windows) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(data.windows)),
                                    thread_name_prefix="neural-write") as pool:
                for name, digest, outcome in pool.map(write_window, enumerate(data.windows)):
                    files[name] = digest
                    outcomes.append(outcome)
        else:
            for item in enumerate(data.windows):
                name, digest, outcome = write_window(item)
                files[name] = digest
                outcomes.append(outcome)
        if _implementation() != implementation:
            raise ValueError("Neural processing implementation changed during generation")
        manifest = json_value(dict(schema_version=1, complete=True, eid=data.eid,
            requested_recordings=[asdict(request) for request in recordings],
            requested_ids=[w.request_id for w in data.windows], recordings=data.recordings,
            trial_sources=data.trial_sources, configuration=data.configuration,
            implementation=implementation, files=files, windows=outcomes,
            outcomes=dict(Counter(w.status for w in data.windows))))
        generation_id = fingerprint(manifest)
        manifest["generation_id"] = generation_id
        write_json(staging / "manifest.json", manifest)
        # The consumer loader checks the complete artifact before publication.
        del data
        verified = load_generation(staging, expected_generation_id=generation_id)
        destination = destination_root / generation_id
        if destination.exists():
            raise FileExistsError(f"Neural generation already exists; load it explicitly: {destination}")
        staging.rename(destination)
    verified.path = destination
    return verified
