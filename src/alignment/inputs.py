"""Pair prepared modalities without rebinning counts or resampling features."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from neural_data import CountWindow, NeuralCounts, load_generation
from visual_features import EncodedObservation, FeatureArtifactReader


@dataclass(frozen=True)
class PreparedTrial:
    session_id: str
    trial_id: int
    stim_on: float
    stim_off: float
    neural: CountWindow
    # All observations, including unselected and source-unavailable records.
    observations: tuple[EncodedObservation, ...]
    visual_outcome: dict


@dataclass(frozen=True)
class AlignmentInputs:
    session_id: str
    bin_size: float
    units: object
    recordings: tuple
    trials: tuple[PreparedTrial, ...]
    neural_configuration: dict
    neural_trial_sources: tuple
    neural_generation_id: str | None
    visual_definition: dict
    visual_completion: dict


def _id(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= np.iinfo(np.int64).max:
        raise ValueError("Trial/request IDs must be nonnegative int64 integers")
    return int(value)


def _window(window, units, onset, offset, bin_size):
    """Check the usable stimulus grid; a shortened final source bin is permitted."""
    edges = np.asarray(window.bin_edges)
    shape = (max(0, len(edges) - 1), units)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
            or window.counts.shape != shape or window.counts.dtype != np.dtype("int64")
            or np.any(window.counts < 0)
            or window.valid.shape != shape or window.valid.dtype != np.dtype("bool")
            or window.coverage_status.shape != shape
            or window.coverage_status.dtype != np.dtype("uint8")
            or np.any(window.coverage_status > 5)
            or not np.array_equal(window.valid, window.coverage_status == 1)
            or window.observed_duration.shape != shape):
        raise ValueError(f"Trial {window.request_id}: malformed or unavailable neural counts")
    if tuple(edges[[0, -1]]) != tuple(window.interval):
        raise ValueError(f"Trial {window.request_id}: neural edges disagree with source interval")
    # Endpoint ULP tolerance, rather than a relative tolerance on session time.
    tolerance = 4 * max(abs(np.spacing(onset)), abs(np.spacing(offset)), np.max(np.abs(np.spacing(edges))))
    ratio = (offset - onset) / bin_size
    nearest = round(ratio)
    bins = nearest if abs(onset + nearest * bin_size - offset) <= tolerance else int(np.floor(ratio))
    if bins < 1:
        raise ValueError(f"Trial {window.request_id}: stimulus contains no complete neural bin")
    expected = onset + np.arange(bins + 1) * bin_size
    start = int(np.searchsorted(edges, onset - tolerance))
    supplied = edges[start:start + bins + 1]
    if (len(supplied) != len(expected) or np.any(np.abs(supplied - expected) > tolerance)
            or np.any(np.diff(expected) <= 0)):
        raise ValueError(f"Trial {window.request_id}: incompatible neural grid; prepare onset-anchored "
                         "counts covering every complete stimulus bin upstream")
    usable = slice(start, start + bins)
    if (not window.valid[usable].all()
            or not np.isfinite(window.observed_duration[usable]).all()
            or np.any(np.abs(window.observed_duration[usable] - np.diff(supplied)[:, None]) > tolerance)):
        raise ValueError(f"Trial {window.request_id}: complete observed neural coverage is required; "
                         "resolve unknown/assumed/partial/invalid support upstream")
    return usable


def prepare_inputs(neural, visual_path, *, request_trial_ids=None,
                   expected_neural_generation_id=None, expected_visual_generation_id=None):
    """Return paired input snapshots after complete public artifact verification.

    `neural` is NeuralCounts or an explicit generation directory. Absolute count
    requests require an explicit request-ID -> original-trial-ID mapping. The
    visual archive and neural requests must contain exactly the same trial set.
    No acquisition, filtering, gap repair, interpolation, or file publication occurs.
    """
    generation_id = None
    if isinstance(neural, (str, Path)):
        loaded = load_generation(neural, expected_generation_id=expected_neural_generation_id)
        generation_id, neural = loaded.generation_id, loaded.data
    elif expected_neural_generation_id is not None:
        raise ValueError("An expected neural generation ID requires a generation directory")
    if not isinstance(neural, NeuralCounts):
        raise TypeError("neural must be NeuralCounts or an explicit generation directory")
    neural = deepcopy(neural)
    configuration = neural.configuration
    if (configuration.get("representation") != "unsmoothed_spike_counts"
            or configuration.get("clock") != "source-session"
            or configuration.get("units") != "seconds"
            or configuration.get("counting_intervals") != "[start,end)"):
        raise ValueError("Unsupported neural representation or physical clock")
    bin_size = configuration.get("bin_size")
    if isinstance(bin_size, bool) or not isinstance(bin_size, (int, float)) or not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("Prepare neural counts with an explicit positive uniform bin_size upstream")
    identity = ["eid", "probe", "collection", "revision", "source_unit_id"]
    units = neural.units
    if (not set(identity + ["pid", "recording_index"]).issubset(units)
            or not len(units) or not units.eid.eq(neural.eid).all()
            or units.duplicated(identity).any()
            or not units.recording_index.isin(range(len(neural.recordings))).all()):
        raise ValueError("A nonempty neural population with scoped ordered unit identities is required")
    for index, record in enumerate(neural.recordings):
        if (record["source"]["eid"] != neural.eid or not record.get("content_sha256")
                or units.loc[units.recording_index == index, "source_unit_id"].tolist() != record["selected_unit_ids"]):
            raise ValueError("Neural recording source/unit identity mismatch")
    windows = {_id(window.request_id): window for window in neural.windows}
    if len(windows) != len(neural.windows) or not windows:
        raise ValueError("Neural requests must have unique IDs and cannot be empty")
    if request_trial_ids is None:
        if configuration.get("trial_window", {}).get("identity") != "original trial table row index":
            raise ValueError("Absolute neural requests require request_trial_ids mapping to original trial IDs")
        mapping = {key: key for key in windows}
    else:
        mapping = {_id(key): _id(value) for key, value in request_trial_ids.items()}
    if set(mapping) != set(windows) or len(set(mapping.values())) != len(mapping):
        raise ValueError("request_trial_ids must map every neural request to one unique original trial")
    by_trial = {mapping[key]: window for key, window in windows.items()}
    observations, outcomes = {}, {}
    with FeatureArtifactReader(visual_path, eid=neural.eid) as reader:
        definition = reader.definition
        records = definition["replay_definition"]["inputs"]["trials"]
        timing_records = {_id(record["trial_id"]): record for record in records}
        for item in reader:
            metadata = deepcopy(item.metadata)
            tid = _id(metadata["trial_id"])
            if metadata["eid"] != neural.eid:
                raise ValueError("Visual/neural session mismatch")
            if isinstance(item, EncodedObservation):
                feature = None if item.feature is None else item.feature.copy()
                if feature is not None:
                    feature.setflags(write=False)
                observations.setdefault(tid, []).append(
                    EncodedObservation(metadata, item.selected, feature, item.status))
            else:
                if metadata.get("kind") != "trial_outcome" or tid in outcomes:
                    raise ValueError("Unexpected or duplicate visual trial outcome")
                outcomes[tid] = metadata
        completion = reader.completion
    if completion is None:
        raise ValueError("Visual generation did not complete verification")
    if expected_visual_generation_id is not None and completion["generation_id"] != expected_visual_generation_id:
        raise ValueError("Unexpected visual generation; select the intended archive explicitly")
    if set(outcomes) != set(by_trial) or set(timing_records) != set(outcomes):
        raise ValueError("Visual and neural original trial sets differ; supply matching prepared inputs")
    trials = []
    for tid in sorted(by_trial):
        record, outcome = timing_records[tid], outcomes[tid]
        events = record.get("events", {})
        bounds = np.asarray([events.get("onset", np.nan), events.get("offset", np.nan)], dtype=float)
        if not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
            raise ValueError(f"Trial {tid}: missing or invalid exact stimulus onset/offset upstream")
        if list(bounds) != record.get("requested_domain") or list(bounds) != outcome.get("requested_domain"):
            raise ValueError(f"Trial {tid}: conflicting stimulus bounds and visual domain")
        rows = tuple(observations.get(tid, ()))
        times = np.asarray([row.metadata["session_time"] for row in rows], dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"Trial {tid}: malformed or non-monotonic visual timestamps")
        _window(by_trial[tid], len(units), *bounds, float(bin_size))
        trials.append(PreparedTrial(neural.eid, tid, *map(float, bounds), by_trial[tid], rows, outcome))
    return AlignmentInputs(neural.eid, float(bin_size), units, neural.recordings,
                           tuple(trials), configuration, neural.trial_sources,
                           generation_id, definition, completion)
