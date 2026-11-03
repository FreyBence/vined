"""Stimulus-bounded, count-preserving alignment on supplied neural bins."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils.progress import iter_progress
from dataclasses import dataclass

import numpy as np

from .inputs import AlignmentInputs, TrialUnavailable, _id, _window


HELD_STATE_POLICY = "timestamp-aware held-state over neural intervals v1"
INTERPOLATION_POLICY = "L2-normalized linear interpolation at neural bin centers"


@dataclass(frozen=True)
class AlignedTrial:
    session_id: str
    trial_id: int
    stim_on: float
    stim_off: float
    aligned_start: float
    aligned_end: float
    bin_size: float
    bin_count: int
    discarded_tail_duration: float
    bin_start_times: np.ndarray
    bin_center_times: np.ndarray
    bin_end_times: np.ndarray
    neural_activity: np.ndarray
    neuron_identity: object
    visual_features: np.ndarray
    alignment_metadata: dict


def _coverage(parts):
    """Merge valid coverage while preserving open boundaries and singleton holes."""
    intervals = []
    for part in parts:
        start, end = part["start"], part["end"]
        if (part["status"] != "valid" or not np.isfinite([start, end]).all()
                or end < start or type(part["start_inclusive"]) is not bool
                or type(part["end_inclusive"]) is not bool):
            raise TrialUnavailable("Visual reconstruction coverage is missing, invalid, or unavailable")
        intervals.append((start, end, part["start_inclusive"], part["end_inclusive"]))
    merged = []
    for start, end, left_closed, right_closed in sorted(intervals):
        if start == end and not (left_closed and right_closed):
            continue
        if not merged or start > merged[-1][1] or (
                start == merged[-1][1] and not (merged[-1][3] or left_closed)):
            merged.append([start, end, left_closed, right_closed])
        else:
            previous = merged[-1]
            if start == previous[0]:
                previous[2] |= left_closed
            if end > previous[1]:
                previous[1], previous[3] = end, right_closed
            elif end == previous[1]:
                previous[3] |= right_closed
    return merged


def _visual(trial, starts, ends):
    outcome = trial.visual_outcome
    rows = trial.observations
    if (outcome.get("eid") != trial.session_id or outcome.get("trial_id") != trial.trial_id
            or outcome.get("requested_domain") != [trial.stim_on, trial.stim_off]):
        raise ValueError("Visual outcome identity or stimulus domain mismatch")
    if (outcome.get("status") != "complete" or outcome.get("input_coverage_status") != "available"
            or outcome.get("unattempted_count") != 0 or not rows
            or outcome.get("scheduled_count") != len(rows)
            or outcome.get("emitted_count") != len(rows)):
        raise TrialUnavailable("Complete accounted visual observations are required; resolve missing data upstream")
    timing = outcome.get("timing") or {}
    if (timing.get("clock") != "session_seconds" or timing.get("domain") != "visible_interval"
            or timing.get("time_kind") != "reconstructed" or timing.get("anchor") != "stimulus_onset"):
        raise ValueError("Unsupported visual temporal representation")
    cadence = timing.get("cadence_hz")
    if isinstance(cadence, bool) or not isinstance(cadence, (int, float)) or not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("Visual cadence evidence is required to check consecutive observations")
    segments = _coverage(outcome.get("input_coverage", []))
    if not any((start < trial.stim_on or start == trial.stim_on and left_closed)
               and end >= trial.stim_off
               for start, end, left_closed, right_closed in segments):
        raise TrialUnavailable("Visual reconstruction does not continuously cover the stimulus interval")
    times, features, indices, identities = [], [], [], []
    previous_time = None
    for index, row in enumerate(rows):
        meta = row.metadata
        time = meta.get("session_time", np.nan)
        if (meta.get("eid") != trial.session_id or meta.get("trial_id") != trial.trial_id
                or meta.get("schedule_index") != index
                or meta.get("trial_table_fingerprint") != outcome.get("trial_table_fingerprint")
                or meta.get("timing") != timing
                or not np.isfinite(time) or time < trial.stim_on or time > trial.stim_off
                or (previous_time is not None and time <= previous_time)):
            raise ValueError("Malformed, mismatched, or unavailable visual source observation")
        if meta.get("status") != "valid":
            raise TrialUnavailable("Visual source observation is unavailable or invalid")
        tolerance = 4 * max(abs(np.spacing(time)), abs(np.spacing(trial.stim_on)), abs(np.spacing(trial.stim_off)))
        expected = trial.stim_on + index / cadence
        terminal = index == len(rows) - 1 and time == trial.stim_off
        if abs(time - expected) > tolerance and not (terminal and time < expected):
            raise TrialUnavailable("Missing or irregular expected visual observation; regenerate upstream")
        previous_time = time
        if row.status == "encoded":
            feature = np.asarray(row.feature)
            if (not row.selected or feature.shape != (768,) or feature.dtype != np.dtype("float32")
                    or not np.isfinite(feature).all() or not np.isclose(np.linalg.norm(feature), 1, atol=1e-5)):
                raise ValueError("Malformed or non-normalized encoded visual feature")
            times.append(time)
            features.append(feature)
            indices.append(index)
            identities.append(meta["observation_id"])
        elif row.status != "not_selected" or row.selected or row.feature is not None:
            raise TrialUnavailable("Expected visual extraction is unavailable")
    next_time = trial.stim_on + len(rows) / cadence
    if previous_time != trial.stim_off and next_time < trial.stim_off:
        raise TrialUnavailable("Expected visual observations are missing at the stimulus end")
    if not times:
        raise TrialUnavailable("No encoded visual observations; regenerate upstream")
    times = np.asarray(times, dtype=np.float64)
    queries = starts + (ends - starts) / 2
    unsupported = (queries < times[0]) | (queries > times[-1])
    if np.any(unsupported):
        k = int(np.flatnonzero(unsupported)[0])
        raise TrialUnavailable(
            f"Bin {k} center {queries[k]:.17g} lacks interpolation support in "
            f"[{times[0]:.17g}, {times[-1]:.17g}]; extract bracketing source "
            "observations (including a valid terminal representation when needed) upstream; "
            "neural bins cannot be shortened or extrapolated")
    right = np.searchsorted(times, queries, side="left")
    exact = times[right] == queries
    left = np.where(exact, right, right - 1)
    weights = np.zeros(len(queries), dtype=np.float64)
    interpolated = ~exact
    weights[interpolated] = ((queries[interpolated] - times[left[interpolated]]) /
                             (times[right[interpolated]] - times[left[interpolated]]))
    # All scheduled rows were validated above, including unselected observations.
    # Irregular encoded spacing from explicit feature subsampling is not a gap.
    values = np.asarray(features, dtype=np.float64)
    result = (1 - weights[:, None]) * values[left] + weights[:, None] * values[right]
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).eps):
        raise TrialUnavailable("Interpolated visual vector has no stable L2 direction; inspect source features upstream")
    result = (result / norms).astype(np.float32)
    provenance = dict(source_times=times.copy(), source_observation_ids=tuple(identities),
                      source_schedule_indices=tuple(indices), left_source_indices=left,
                      right_source_indices=right, interpolation_weights=weights,
                      query_times=queries.copy(), policy=INTERPOLATION_POLICY, policy_version=1,
                      normalization="L2 after float64 linear interpolation; float32 output",
                      support="complete valid scheduled observations; no extrapolation",
                      timing_classification=deepcopy(timing),
                      verified_display_bin_alignment=False)
    return result, provenance


def align_trials(inputs, *, workers=1):
    """Return all aligned trials, or raise without returning a partial collection."""
    _validate_workers(workers)
    if not isinstance(inputs, AlignmentInputs):
        raise TypeError("inputs must be AlignmentInputs returned by prepare_inputs")
    if not np.isfinite(inputs.bin_size) or inputs.bin_size <= 0 or not inputs.trials:
        raise ValueError("A positive neural bin size and paired trials are required")
    identities = [_id(trial.trial_id) for trial in inputs.trials]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate original trial IDs")
    trials = sorted(inputs.trials, key=lambda item: item.trial_id)
    operation = partial(_align_trial, inputs)
    label = f"alignment {inputs.session_id}: resampling"
    if workers == 1:
        return tuple(iter_progress(map(operation, trials), label, total=len(trials), unit="trials"))
    with ThreadPoolExecutor(max_workers=min(workers, len(trials))) as executor:
        return tuple(iter_progress(executor.map(operation, trials), label,
                                   total=len(trials), unit="trials"))


def _align_trial(inputs, trial):
    try:
        if (trial.session_id != inputs.session_id or not inputs.units.eid.eq(inputs.session_id).all()
                or not np.isfinite([trial.stim_on, trial.stim_off]).all()
                or trial.stim_off <= trial.stim_on):
            raise ValueError("Session mismatch or invalid stimulus bounds")
        usable = _window(trial.neural, len(inputs.units), trial.stim_on, trial.stim_off, inputs.bin_size)
        edges = trial.neural.bin_edges[usable.start:usable.stop + 1].copy()
        starts, ends = edges[:-1], edges[1:]
        centers = starts + (ends - starts) / 2
        if np.any(centers <= starts) or np.any(centers >= ends):
            raise ValueError("Neural bin centers are unrepresentable at source clock precision")
        visual, resampling = _visual(trial, starts, ends)
        tail = trial.stim_off - float(ends[-1])
        tolerance = 4 * max(abs(np.spacing(trial.stim_off)), abs(np.spacing(ends[-1])))
        if tail < -tolerance or tail >= inputs.bin_size + tolerance:
            raise ValueError("Neural endpoint does not define a complete-bin stimulus interval")
        metadata = dict(
            clock="session_seconds", neural_intervals="[start,end)",
            visual_resampling=INTERPOLATION_POLICY,
            neural_generation_id=inputs.neural_generation_id,
            neural_request_id=trial.neural.request_id,
            neural_configuration=deepcopy(inputs.neural_configuration),
            neural_trial_sources=deepcopy(inputs.neural_trial_sources),
            neural_recordings=deepcopy(inputs.recordings),
            neural_source_interval=trial.neural.interval,
            neural_source_bin_slice=(usable.start, usable.stop),
            visual_definition=_visual_provenance(inputs, trial),
            visual_completion=dict(generation_id=inputs.visual_completion["generation_id"],
                definition_id=inputs.visual_completion.get("definition_id"),
                replay_generation_id=inputs.visual_completion["replay_completion"]["generation_id"],
                scope="verified_source_generation_identity"),
            visual_outcome=deepcopy(trial.visual_outcome), resampling=resampling,
        )
        return AlignedTrial(
            trial.session_id, trial.trial_id, trial.stim_on, trial.stim_off,
            float(starts[0]), float(ends[-1]), inputs.bin_size, len(starts), max(0., tail),
            starts, centers, ends, trial.neural.counts[usable].copy(),
            inputs.units.copy(deep=True), visual, metadata)
    except TrialUnavailable as exc:
        raise TrialUnavailable(f"Trial {trial.trial_id}: {exc}") from exc



def _validate_workers(workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")


def _visual_provenance(inputs, trial):
    definition = inputs.visual_definition
    # The complete source definition remains in the generation-level manifest.
    record = next(item for item in definition["replay_definition"]["inputs"]["trials"]
                  if item["trial_id"] == trial.trial_id)
    return dict(scope="trial", replay_definition_id=definition["replay_definition_id"],
                representation=deepcopy(definition["representation"]),
                selection=deepcopy(definition["selection"]), trial=deepcopy(record))


def _align_available(inputs, workers):
    """Collect trial outcomes in canonical order; unexpected failures propagate."""
    excluded = list(inputs.excluded_trials)
    def operation(trial):
        try:
            return _align_trial(inputs, trial)
        except TrialUnavailable as exc:
            return dict(session_id=inputs.session_id, trial_id=trial.trial_id,
                        stage="temporal", reason=str(exc))
    ordered = sorted(inputs.trials, key=lambda item: item.trial_id)
    retained = []
    def consume(results):
        for result in iter_progress(results, f"alignment {inputs.session_id}: resampling",
                                    total=len(ordered), unit="trials"):
            if isinstance(result, AlignedTrial):
                retained.append(result)
            else:
                excluded.append(result)
    if workers == 1 or not ordered:
        consume(map(operation, ordered))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(ordered))) as executor:
            consume(executor.map(operation, ordered))
    excluded.sort(key=lambda item: item["trial_id"])
    if not retained:
        from utils.progress import logger
        for item in excluded:
            logger.warning("alignment %s trial %s excluded: %s", inputs.session_id,
                           item["trial_id"], item["reason"])
        raise ValueError(f"No usable trials remain for {inputs.session_id}; excluded {len(excluded)} trials")
    return tuple(retained), tuple(excluded)
