"""Stimulus-bounded, count-preserving alignment on supplied neural bins."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .inputs import AlignmentInputs, _id, _window


HELD_STATE_POLICY = "timestamp-aware held-state over neural intervals v1"


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
            raise ValueError("Visual reconstruction coverage is missing, invalid, or unavailable")
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
        raise ValueError("Complete accounted visual observations are required; resolve missing data upstream")
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
        raise ValueError("Visual reconstruction does not continuously cover the stimulus interval")
    times, features, indices, identities = [], [], [], []
    update_times, encoded_indices = [], []
    previous_time = None
    for index, row in enumerate(rows):
        meta = row.metadata
        time = meta.get("session_time", np.nan)
        if (meta.get("eid") != trial.session_id or meta.get("trial_id") != trial.trial_id
                or meta.get("schedule_index") != index
                or meta.get("trial_table_fingerprint") != outcome.get("trial_table_fingerprint")
                or meta.get("timing") != timing or meta.get("status") != "valid"
                or not np.isfinite(time) or time < trial.stim_on or time > trial.stim_off
                or (previous_time is not None and time <= previous_time)):
            raise ValueError("Malformed, mismatched, or unavailable visual source observation")
        tolerance = 4 * max(abs(np.spacing(time)), abs(np.spacing(trial.stim_on)), abs(np.spacing(trial.stim_off)))
        expected = trial.stim_on + index / cadence
        terminal = index == len(rows) - 1 and time == trial.stim_off
        if abs(time - expected) > tolerance and not (terminal and time < expected):
            raise ValueError("Missing or irregular expected visual observation; regenerate upstream")
        previous_time = time
        update_times.append(time)
        encoded_indices.append(-1)
        if row.status == "encoded":
            feature = np.asarray(row.feature)
            if (not row.selected or feature.shape != (768,) or feature.dtype != np.dtype("float32")
                    or not np.isfinite(feature).all() or not np.isclose(np.linalg.norm(feature), 1, atol=1e-5)):
                raise ValueError("Malformed or non-normalized encoded visual feature")
            times.append(time)
            features.append(feature)
            indices.append(index)
            identities.append(meta["observation_id"])
            encoded_indices[-1] = len(features) - 1
        elif row.status != "not_selected" or row.selected or row.feature is not None:
            raise ValueError("Expected visual extraction is unavailable")
    next_time = trial.stim_on + len(rows) / cadence
    if previous_time != trial.stim_off and next_time < trial.stim_off:
        raise ValueError("Expected visual observations are missing at the stimulus end")
    if not times:
        raise ValueError("No encoded visual observations; regenerate upstream")
    times = np.asarray(times, dtype=np.float64)
    updates = np.asarray(update_times, dtype=np.float64)
    support_ends = np.r_[updates[1:], trial.stim_off]
    tolerance = 4 * max(abs(np.spacing(trial.stim_on)), abs(np.spacing(trial.stim_off)))
    # Account only for source-clock roundoff at a bin boundary. Never use a
    # future observation for a bin merely because its center follows an update.
    active = np.searchsorted(updates, starts + tolerance, side="right") - 1
    if np.any(active < 0):
        raise ValueError("No active visual state at a neural interval start")
    if np.any(ends > support_ends[active] + tolerance):
        raise ValueError("A display update falls inside a neural bin; matching physical intervals are required")
    source = np.asarray(encoded_indices, dtype=np.int64)[active]
    if np.any(source < 0):
        raise ValueError("Active visual state was not encoded; extract every required update upstream")
    for a, b in zip(starts, ends):
        if not any((a > start or a == start and left_closed) and b <= end
                   for start, end, left_closed, right_closed in segments):
            raise ValueError("Held visual state would cross unsupported coverage")
    result = np.asarray(features, dtype=np.float32)[source].copy()
    # Retain the existing association fields for downstream dataset consumers:
    # identical endpoints and zero weights denote selection, never interpolation.
    provenance = dict(source_times=times.copy(), source_observation_ids=tuple(identities),
                      source_schedule_indices=tuple(indices), left_source_indices=source.copy(),
                      right_source_indices=source.copy(), interpolation_weights=np.zeros(len(starts)),
                      policy=HELD_STATE_POLICY, source_interval_end_times=support_ends[indices].copy(),
                      timing_classification=deepcopy(timing),
                      update_bin_boundaries_coincident=bool(np.all(np.abs(updates[active] - starts) <= tolerance)),
                      verified_display_bin_alignment=False)
    return result, provenance


def align_trials(inputs):
    """Return all aligned trials, or raise without returning a partial collection."""
    if not isinstance(inputs, AlignmentInputs):
        raise TypeError("inputs must be AlignmentInputs returned by prepare_inputs")
    if not np.isfinite(inputs.bin_size) or inputs.bin_size <= 0 or not inputs.trials:
        raise ValueError("A positive neural bin size and paired trials are required")
    identities = [_id(trial.trial_id) for trial in inputs.trials]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate original trial IDs")
    aligned = []
    for trial in sorted(inputs.trials, key=lambda item: item.trial_id):
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
                visual_resampling=HELD_STATE_POLICY,
                neural_generation_id=inputs.neural_generation_id,
                neural_request_id=trial.neural.request_id,
                neural_configuration=deepcopy(inputs.neural_configuration),
                neural_trial_sources=deepcopy(inputs.neural_trial_sources),
                neural_recordings=deepcopy(inputs.recordings),
                neural_source_interval=trial.neural.interval,
                neural_source_bin_slice=(usable.start, usable.stop),
                visual_definition=deepcopy(inputs.visual_definition),
                visual_completion=deepcopy(inputs.visual_completion),
                visual_outcome=deepcopy(trial.visual_outcome), resampling=resampling,
            )
            aligned.append(AlignedTrial(
                trial.session_id, trial.trial_id, trial.stim_on, trial.stim_off,
                float(starts[0]), float(ends[-1]), inputs.bin_size, len(starts), max(0., tail),
                starts, centers, ends, trial.neural.counts[usable].copy(),
                inputs.units.copy(deep=True), visual, metadata))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Trial {trial.trial_id}: {exc}") from exc
    return tuple(aligned)
