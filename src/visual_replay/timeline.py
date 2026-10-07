"""Cadence-independent task state and explicit source-time sampling.

No image rendering, video time, or neural time grid is involved here.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .profiles import behavior_profile


@dataclass(frozen=True)
class StimulusState:
    eid: str
    trial_id: int
    session_time: float
    status: str
    reason: str | None
    stage: str
    visible: bool | None
    azimuth_deg: float | None
    parameters: dict
    profile_id: str
    trial_table_fingerprint: str

    @property
    def known_blank(self):
        return self.status == "valid" and (
            self.visible is False or self.parameters.get("contrast") == 0
        )


@dataclass(frozen=True)
class CoverageInterval:
    start: float
    end: float
    start_inclusive: bool
    end_inclusive: bool
    status: str
    reason: str | None


@dataclass(frozen=True)
class ObservationSchedule:
    eid: str
    trial_id: int
    requested_domain: tuple | None
    times: np.ndarray
    timing: dict
    coverage: tuple[CoverageInterval, ...]
    status: str
    reason: str | None


def _time(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Session time must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Session time must be a finite number")
    return value


class _Wheel:
    def __init__(self, times, positions):
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        self.reason = None
        if (times.ndim != 1 or positions.shape != times.shape
                or not np.isfinite(times).all() or not np.isfinite(positions).all()
                or np.any(np.diff(times) < 0)):
            self.reason = "Invalid wheel samples"
            times = positions = np.empty(0, dtype=np.float64)
        # Equal-time samples have no ordered physical interval. Keep the last
        # acquired value, consistently with legacy replay, and expose the count.
        keep = np.r_[np.diff(times) != 0, True] if len(times) else np.array([], dtype=bool)
        self.duplicates_removed = len(times) - int(keep.sum())
        self.times, self.positions = times[keep], positions[keep]
        self.times.setflags(write=False)
        self.positions.setflags(write=False)

    def at(self, time):
        if self.reason:
            return None, self.reason
        if not len(self.times) or time < self.times[0] or time > self.times[-1]:
            return None, "Wheel observations do not cover this time"
        index = int(np.searchsorted(self.times, time, side="left"))
        if index < len(self.times) and self.times[index] == time:
            return float(self.positions[index]), None
        left, right = index - 1, index
        fraction = (time - self.times[left]) / (self.times[right] - self.times[left])
        return float(self.positions[left] + fraction * (self.positions[right] - self.positions[left])), None


class TrialTimeline:
    """Prepared task timeline; evaluate(time) is independent of request history."""

    def __init__(self, definition, record, wheel):
        self.eid = definition["eid"]
        self.trial_id = record["trial_id"]
        self.record = deepcopy(record)
        self.configuration = deepcopy(definition["configuration"])
        self.profile_id = definition["profile"]["id"]
        self.trial_table_fingerprint = definition["trial_table_fingerprint"]
        self.wheel_policy = dict(self.configuration["movement"],
                                 duplicate_timestamp_policy="last acquired position",
                                 duplicates_removed=wheel.duplicates_removed)
        self._wheel = wheel
        self._baseline, self._baseline_reason = None, None
        self._frozen_position, self._freeze_reason = None, None
        if record["status"] == "prepared":
            self._baseline, self._baseline_reason = wheel.at(record["events"]["closed_loop"])
            if record["outcome"] == "error":
                self._frozen_position, self._freeze_reason = self._position(record["events"]["freeze"])

    def _position(self, time):
        if self._baseline_reason:
            return None, "Unavailable coupling baseline: " + self._baseline_reason
        wheel, reason = self._wheel.at(time)
        if reason:
            return None, reason
        parameters = self.record["parameters"]
        displacement = ((wheel - self._baseline) * self.configuration["movement"]["wheel_sign"]
                        * parameters["wheel_radius_mm"] * parameters["gain_deg_per_mm"])
        position = parameters["initial_azimuth_deg"] + displacement
        # Source angular wrapping, not a clamp to successful trajectories.
        return (position + 180) % 360 - 180, None

    def evaluate(self, session_time):
        time = _time(session_time)
        record = self.record
        parameters = deepcopy(record.get("parameters", {}))

        def state(stage, visible, position=None, status="valid", reason=None):
            return StimulusState(self.eid, self.trial_id, time, status, reason,
                                 stage, visible, position, parameters,
                                 self.profile_id, self.trial_table_fingerprint)

        if record["status"] != "prepared":
            return state("unknown", None, status=record["status"], reason=record["reason"])
        events = record["events"]
        onset, offset = events["onset"], events["offset"]
        interval = record.get("trial_interval")
        # The offset event itself establishes hiding; beyond known trial bounds
        # we cannot assert that another trial has not shown a new stimulus.
        if time == offset:
            return state("hidden", False)
        if time < onset or time >= offset:
            if interval is not None and interval[0] <= time < interval[1]:
                return state("hidden", False)
            return state("unknown", None, status="unavailable",
                         reason="Outside source-established trial visibility domain")
        # Right-continuous transitions: hide > freeze > closed loop > onset.
        # All events are evaluated even if a schedule never samples them.
        if record["outcome"] != "no_go" and time >= events["freeze"]:
            if record["outcome"] == "reward":
                return state("freeze_at_center", True, 0.0)
            if self._freeze_reason:
                return state("freeze_in_place", True, status="unavailable", reason=self._freeze_reason)
            return state("freeze_in_place", True, self._frozen_position)
        if time < events["closed_loop"]:
            initial = (parameters["initial_azimuth_deg"] + 180) % 360 - 180
            return state("stationary_visible", True, initial)
        position, reason = self._position(time)
        return state("closed_loop", True, position,
                     status="unavailable" if reason else "valid", reason=reason)

    def coverage(self):
        """Exact availability partition of the half-open requested domain.

        Open spans and explicit boundary points preserve valid observations at
        the ends of recorded wheel coverage without extrapolating beyond it.
        These are reconstruction-availability intervals, not frame supports.
        """
        if self.record["status"] != "prepared":
            return ()
        events = self.record["events"]
        start, end = self.record["requested_domain"]
        boundaries = {start, end, events["closed_loop"]}
        movement_end = events.get("freeze", end)
        boundaries.add(movement_end)
        wheel = self._wheel
        if len(wheel.times):
            candidates = np.r_[wheel.times[0], wheel.times[-1]]
            boundaries.update(float(x) for x in candidates
                              if events["closed_loop"] <= x <= movement_end)
        ordered = sorted(x for x in boundaries if start <= x <= end)
        result = []
        for left, right in zip(ordered, ordered[1:]):
            point = self.evaluate(left)
            middle_time = left + (right - left) / 2
            middle = self.evaluate(middle_time)
            result.append(CoverageInterval(left, left, True, True, point.status, point.reason))
            result.append(CoverageInterval(left, right, False, False, middle.status, middle.reason))
        return tuple(result)


def prepare_trial_timelines(plan):
    """Return timelines in requested original-trial order, sharing wheel evidence."""
    definition = plan.definition
    profile = behavior_profile(definition["profile"]["id"])
    if definition["profile"] != profile:
        raise ValueError("Resolved behavior profile differs from the supported profile")
    movement = definition["configuration"]["movement"]
    if (movement["interpolation"] != "linear" or movement["extrapolation"] != "unavailable"
            or movement["baseline"] != "closed_loop_event"):
        raise ValueError("Unsupported resolved movement policy")
    wheel = _Wheel(plan.wheel_timestamps, plan.wheel_positions)
    records = definition["trials"]
    if [record["trial_id"] for record in records] != definition["requested_trial_ids"]:
        raise ValueError("Trial records do not account for the requested original IDs")
    return tuple(TrialTimeline(definition, record, wheel) for record in records)


def schedule_trial(timeline, *, cadence_hz=None):
    """Schedule source-time points without altering task state or coverage.

    An explicit cadence override is recorded as configuration, never as a
    measured refresh rate. Invalid/unavailable input trials retain an empty
    schedule and their outcome, rather than becoming absent trials.
    """
    timing = deepcopy(timeline.configuration["observation"])
    if timing["time_kind"] != "reconstructed" or timing["domain"] != "visible_interval":
        raise ValueError("Unsupported resolved observation schedule")
    cadence = _time(timing["cadence_hz"] if cadence_hz is None else cadence_hz)
    if cadence <= 0:
        raise ValueError("Observation cadence must be positive")
    timing.update(cadence_hz=cadence, anchor="stimulus_onset", support_intervals=None)
    if cadence_hz is not None:
        timing["cadence_override"] = "Explicit schedule_trial request; reconstructed sampling"
    record = timeline.record
    times = np.empty(0, dtype=np.float64)
    domain = record.get("requested_domain")
    if record["status"] == "prepared":
        start, end = map(_time, domain)
        if end <= start:
            raise ValueError("Stimulus offset must follow onset")
        maximum = timing.get("max_frames_per_trial", 10_000)
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("max_frames_per_trial must be a positive integer")
        frame_count = (end - start) * cadence
        if not math.isfinite(frame_count) or frame_count > maximum:
            raise ValueError(
                f"Requested visibility duration {end - start:g} seconds at {cadence:g} Hz "
                f"exceeds max_frames_per_trial={maximum}; check source onset/offset "
                "or explicitly raise the configured limit for a known long trial"
            )
    coverage = timeline.coverage()
    status, reason = record["status"], record["reason"]
    if status == "prepared":
        start, end = domain
        count = math.ceil((end - start) * cadence)
        times = start + np.arange(count, dtype=np.float64) / cadence
        times = times[times < end]
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("Cadence cannot be represented as distinct float64 session times")
        available = any(item.status == "valid" for item in coverage)
        missing = any(item.status != "valid" for item in coverage)
        status = "partial" if available and missing else "unavailable" if missing else "available"
        reason = "Requested domain contains unavailable reconstruction" if missing else None
    times.setflags(write=False)
    return ObservationSchedule(timeline.eid, timeline.trial_id,
                               None if domain is None else tuple(domain), times,
                               timing, coverage, status, reason)
