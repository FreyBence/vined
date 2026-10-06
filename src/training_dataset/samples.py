"""Count-preserving trial packaging, independent of model and split objectives."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Union
from uuid import UUID

import numpy as np
import pandas as pd

from alignment import AlignedTrial, load_alignment
from utils.provenance import fingerprint


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum or value > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must be an integer in [{minimum}, int64 maximum]")
    return int(value)


@dataclass(frozen=True)
class SampleConfig:
    """Optional right padding; None retains each trial's original axis length."""

    max_time_length: Optional[int] = None
    max_neuron_count: Optional[int] = None

    def __post_init__(self):
        for name in ("max_time_length", "max_neuron_count"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(value, name, minimum=1))


@dataclass(frozen=True)
class AlignmentSource:
    path: Union[str, Path]
    expected_generation_id: Optional[str] = None


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    session_id: str
    trial_id: int
    neural: np.ndarray
    visual: np.ndarray
    physical_timestamps: np.ndarray
    bin_start_times: np.ndarray
    bin_end_times: np.ndarray
    temporal_positions: np.ndarray
    temporal_mask: np.ndarray
    neuron_identity: pd.DataFrame
    neuron_mask: np.ndarray
    sequence_length: int
    neuron_count: int
    stim_on: float
    stim_off: float
    aligned_start: float
    aligned_end: float
    bin_size: float
    discarded_tail_duration: float
    metadata: dict
    split: Optional[str] = None


def _coincident(actual, expected, tolerance):
    # Preserve the supplied grid, tolerating only source-clock endpoint roundoff.
    return np.all(np.abs(actual - expected) <= tolerance)


def _validate_trial(trial):
    if not isinstance(trial, AlignedTrial):
        raise TypeError("Samples require alignment.AlignedTrial objects")
    try:
        if not isinstance(trial.session_id, str) or str(UUID(trial.session_id)) != trial.session_id:
            raise ValueError("session_id must be a canonical UUID")
    except (ValueError, AttributeError) as error:
        raise ValueError("session_id must be a canonical UUID") from error
    _integer(trial.trial_id, "trial_id")
    length = _integer(trial.bin_count, "bin_count", minimum=1)
    neural, visual = trial.neural_activity, trial.visual_features
    if (not isinstance(neural, np.ndarray) or neural.dtype != np.dtype("int64")
            or neural.ndim != 2 or neural.shape[0] != length or neural.shape[1] < 1
            or np.any(neural < 0)):
        raise ValueError("neural_activity must be nonnegative int64 [bin_count, neurons]")
    if (not isinstance(visual, np.ndarray) or visual.dtype != np.dtype("float32")
            or visual.ndim != 2 or visual.shape[0] != length or visual.shape[1] < 1
            or not np.isfinite(visual).all()
            or not np.allclose(np.linalg.norm(visual, axis=1), 1, atol=1e-5)):
        raise ValueError("visual_features must be normalized finite float32 [bin_count, feature_width]")
    timing = [trial.stim_on, trial.stim_off, trial.aligned_start, trial.aligned_end,
              trial.bin_size, trial.discarded_tail_duration]
    if not all(isinstance(value, (float, np.floating)) for value in timing):
        raise ValueError("Trial timing must contain floating-point session seconds")
    if (not np.isfinite(timing).all() or trial.bin_size <= 0
            or trial.stim_off <= trial.stim_on or trial.aligned_end <= trial.aligned_start
            or trial.discarded_tail_duration < 0):
        raise ValueError("Invalid aligned bounds, bin duration, or discarded tail")
    starts, centers, ends = trial.bin_start_times, trial.bin_center_times, trial.bin_end_times
    tolerance = 4 * max(abs(np.spacing(trial.stim_on)), abs(np.spacing(trial.stim_off)))
    for times in (starts, centers, ends):
        if (not isinstance(times, np.ndarray) or times.dtype != np.dtype("float64")
                or times.shape != (length,) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Bin timestamps must be strictly increasing finite float64 [bin_count]")
    if (np.any(ends <= starts) or np.any(centers <= starts) or np.any(centers >= ends)
            or not _coincident(centers, starts + (ends - starts) / 2, tolerance)
            or not np.array_equal(starts[1:], ends[:-1])
            or not _coincident(ends - starts, trial.bin_size, 2 * tolerance)
            or not _coincident(starts[0], trial.stim_on, tolerance)
            or trial.aligned_start != starts[0] or trial.aligned_end != ends[-1]
            or ends[-1] > trial.stim_off + tolerance
            or not _coincident(max(0., trial.stim_off - ends[-1]),
                               trial.discarded_tail_duration, tolerance)
            or trial.discarded_tail_duration >= trial.bin_size + tolerance):
        raise ValueError("Bin timestamps do not match the supplied complete stimulus grid")
    units = trial.neuron_identity
    keys = ["eid", "probe", "collection", "revision", "source_unit_id", "recording_index"]
    if (not isinstance(units, pd.DataFrame) or len(units) != neural.shape[1]
            or not set(keys + ["pid"]).issubset(units.columns)
            or not units["eid"].eq(trial.session_id).all()
            or units[[key for key in keys if key != "revision"]].isna().any().any()
            or units.duplicated(keys).any()):
        raise ValueError("neuron_identity must provide one unique session-scoped unit row per column")
    metadata = trial.alignment_metadata
    required = {"clock", "neural_intervals", "neural_configuration", "neural_recordings",
                "neural_trial_sources", "neural_generation_id", "neural_request_id",
                "neural_source_interval", "neural_source_bin_slice", "visual_definition",
                "visual_completion", "visual_outcome", "visual_resampling", "resampling"}
    if (not isinstance(metadata, dict) or not required.issubset(metadata)
            or metadata["clock"] != "session_seconds" or metadata["neural_intervals"] != "[start,end)"):
        raise ValueError("alignment_metadata must contain physical-time and source provenance")
    resampling = metadata["resampling"]
    required = {"source_times", "source_observation_ids", "source_schedule_indices",
                "left_source_indices", "right_source_indices", "interpolation_weights"}
    if not isinstance(resampling, dict) or not required.issubset(resampling):
        raise ValueError("Missing visual interpolation provenance")
    times = np.asarray(resampling["source_times"])
    left, right = (np.asarray(resampling[name]) for name in
                   ("left_source_indices", "right_source_indices"))
    weights = np.asarray(resampling["interpolation_weights"])
    if (times.dtype.kind not in "fiu" or times.ndim != 1 or not len(times)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or np.asarray(resampling["source_observation_ids"]).shape != times.shape
            or np.asarray(resampling["source_schedule_indices"]).shape != times.shape
            or left.dtype.kind not in "iu" or right.dtype.kind not in "iu"
            or left.shape != (length,) or right.shape != (length,)
            or np.any(left < 0) or np.any(right < left) or np.any(right >= len(times))
            or weights.dtype.kind not in "fiu" or weights.shape != (length,)
            or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1)):
        raise ValueError("Invalid visual interpolation provenance")


def _sample(trial, config, source):
    length, neurons = trial.neural_activity.shape
    time_size = config.max_time_length if config.max_time_length is not None else length
    neuron_size = config.max_neuron_count if config.max_neuron_count is not None else neurons
    if time_size < length or neuron_size < neurons:
        raise ValueError(f"Padding maxima cannot shorten trial {trial.session_id}/{trial.trial_id} "
                         f"with shape {(length, neurons)}")
    neural = np.zeros((time_size, neuron_size), dtype=np.int64)
    neural[:length, :neurons] = trial.neural_activity
    visual = np.zeros((time_size, trial.visual_features.shape[1]), dtype=np.float32)
    visual[:length] = trial.visual_features
    physical_times = []
    for original in (trial.bin_center_times, trial.bin_start_times, trial.bin_end_times):
        padded = np.full(time_size, np.nan, dtype=np.float64)
        padded[:length] = original
        physical_times.append(padded)
    temporal_mask = np.arange(time_size) < length
    neuron_mask = np.arange(neuron_size) < neurons
    positions = np.full(time_size, -1, dtype=np.int64)
    positions[:length] = np.arange(length, dtype=np.int64)
    configuration = asdict(config)
    sample_id = fingerprint(dict(session_id=trial.session_id, trial_id=int(trial.trial_id),
                                 representation="aligned-trial-v1", padding=configuration))
    return TrainingSample(
        sample_id=sample_id, session_id=trial.session_id, trial_id=int(trial.trial_id),
        neural=neural, visual=visual, physical_timestamps=physical_times[0],
        bin_start_times=physical_times[1], bin_end_times=physical_times[2],
        temporal_positions=positions, temporal_mask=temporal_mask,
        neuron_identity=deepcopy(trial.neuron_identity), neuron_mask=neuron_mask,
        sequence_length=length, neuron_count=neurons,
        stim_on=trial.stim_on, stim_off=trial.stim_off,
        aligned_start=trial.aligned_start, aligned_end=trial.aligned_end, bin_size=trial.bin_size,
        discarded_tail_duration=trial.discarded_tail_duration,
        metadata=dict(alignment=deepcopy(trial.alignment_metadata),
                      source_alignment=deepcopy(source),
                      construction=dict(representation="aligned-trial-v1", padding=configuration)),
    )


def _build(trials, config, sources):
    if config is None:
        config = SampleConfig()
    if not isinstance(config, SampleConfig):
        raise TypeError("config must be SampleConfig")
    trials = tuple(trials)
    if not trials:
        raise ValueError("Sample construction requires at least one aligned trial")
    seen, populations = set(), {}
    for trial in trials:
        _validate_trial(trial)
        identity = (trial.session_id, int(trial.trial_id))
        if identity in seen:
            raise ValueError(f"Duplicate aligned trial: {identity}")
        seen.add(identity)
        if trial.session_id in populations:
            units, bin_size = populations[trial.session_id]
            if (not units.reset_index(drop=True).equals(trial.neuron_identity.reset_index(drop=True))
                    or bin_size != trial.bin_size):
                raise ValueError(f"Inconsistent neuron population or bin size for {trial.session_id}")
        else:
            populations[trial.session_id] = (trial.neuron_identity, trial.bin_size)
    return tuple(_sample(trial, config, sources.get(trial.session_id))
                 for trial in sorted(trials, key=lambda item: (item.session_id, item.trial_id)))


def build_samples(trials: Iterable[AlignedTrial], *, config: Optional[SampleConfig] = None):
    """Package in-memory aligned trials; no source-generation identity is invented."""
    return _build(trials, config, {})


def load_samples(sources: Iterable[AlignmentSource], *, config: Optional[SampleConfig] = None):
    """Verify explicit alignment generations, then construct all samples or fail."""
    sources = tuple(sources)
    if not sources or not all(isinstance(source, AlignmentSource) for source in sources):
        raise ValueError("Provide a nonempty collection of AlignmentSource selections")
    trials, provenance = [], {}
    for source in sources:
        generation = load_alignment(source.path, expected_generation_id=source.expected_generation_id)
        session_id = generation.trials[0].session_id
        if session_id in provenance:
            raise ValueError(f"Select exactly one alignment generation for session {session_id}")
        provenance[session_id] = dict(generation_id=generation.generation_id,
                                      path=str(generation.path),
                                      schema_version=generation.manifest["schema_version"])
        trials.extend(generation.trials)
    return _build(trials, config, provenance)
