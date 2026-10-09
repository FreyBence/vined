"""Unnormalized spike counts on explicit source-session intervals."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from utils.progress import iter_progress
from dataclasses import asdict, dataclass

import numpy as np

from .sources import Coverage, _coverage
from .provenance import content_hash

UNKNOWN, VALID, ASSUMED, PARTIAL, INVALID, UNAVAILABLE = range(6)
DEFAULT_BIN_SIZE = 1 / 60  # Seconds; matches the confirmed experimental projection rate.
COVERAGE_STATES = dict(enumerate(("unknown", "valid", "assumed", "partial", "invalid", "unavailable")))


@dataclass
class CountWindow:
    request_id: int
    interval: tuple
    bin_edges: np.ndarray
    counts: np.ndarray
    valid: np.ndarray
    coverage_status: np.ndarray
    observed_duration: np.ndarray
    status: str
    reason: str | None


@dataclass
class NeuralCounts:
    eid: str
    units: object
    recordings: tuple
    windows: tuple
    configuration: dict
    trial_sources: tuple = ()


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([float(start), float(end)])
    return merged


def _support(coverage):
    _coverage(Coverage(**coverage))
    invalid = _union(coverage["invalid"])
    if coverage["observed"] is None:
        return None, invalid
    support = _union(coverage["observed"])
    for left, right in invalid:
        remaining = []
        for start, end in support:
            if right <= start or left >= end:
                remaining.append([start, end])
            else:
                if start < left:
                    remaining.append([start, left])
                if right < end:
                    remaining.append([right, end])
        support = remaining
    return support, invalid


def _coverage_bins(edges, coverage):
    support, invalid = _support(coverage)
    statuses = np.full(len(edges) - 1, UNKNOWN, dtype=np.uint8)
    duration = np.full(len(statuses), np.nan)
    for row, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        bad = any(left < end and right > start for left, right in invalid)
        if support is None:
            if bad:
                statuses[row] = INVALID
            continue
        duration[row] = sum(max(0., min(end, right) - max(start, left))
                            for left, right in support)
        if any(left <= start and right >= end for left, right in support):
            statuses[row] = VALID if coverage["qualification"] == "observed" else ASSUMED
        elif duration[row] > 0:
            statuses[row] = PARTIAL
        else:
            statuses[row] = INVALID if bad else UNAVAILABLE
    return statuses, duration


def _edges(start, end, bin_size):
    if bin_size is None:
        return np.array([start, end], dtype=np.float64)
    # Compute from the origin, not repeated addition, and expose the shortened
    # final bin. A numerically coincident final grid point is the requested end.
    count = (end - start) / bin_size
    if not np.isfinite(count) or count > np.iinfo(np.intp).max - 1:
        raise ValueError("Requested bin grid is not representable")
    steps = int(np.ceil(count))
    interior = start + np.arange(1, steps, dtype=np.float64) * bin_size
    tolerance = 4 * max(abs(np.spacing(start)), abs(np.spacing(end)))
    interior = interior[interior < end - tolerance]
    edges = np.r_[start, interior, end]
    if np.any(np.diff(edges) <= 0) or (steps > 1 and start + bin_size == start):
        raise ValueError("Bin size is below source-time floating-point resolution")
    return edges


def _ids(values, count):
    values = np.asarray(values)
    if (values.shape != (count,) or values.dtype.kind not in "iu"
            or np.any(values < 0) or len(np.unique(values)) != count
            or np.any(values > np.iinfo(np.int64).max)):
        raise ValueError("Request/trial IDs must be unique nonnegative int64 values")
    return values.astype(np.int64)


def count_intervals(population, intervals, *, bin_size=DEFAULT_BIN_SIZE, request_ids=None,
                    unit_coverage=None, workers=1):
    """Count each requested window; no trial or visual eligibility is imposed.

    Coverage overrides are keyed by unit-axis position and may only restrict
    recording coverage (known unit evidence may qualify unknown recording support).
    Nonfinite windows produce explicit unavailable outcomes. Other invalid input
    raises ValueError rather than returning a partially processed result.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.size == 0 and intervals.shape == (0,):
        intervals = intervals.reshape(0, 2)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("Intervals must have shape [requests, 2]")
    if bin_size is not None and (not np.isfinite(bin_size) or bin_size <= 0):
        raise ValueError("Bin size must be finite and positive")
    finite = np.isfinite(intervals).all(axis=1)
    if np.any(intervals[finite, 1] <= intervals[finite, 0]):
        raise ValueError("Finite interval ends must be after their starts")
    ids = _ids(np.arange(len(intervals)) if request_ids is None else request_ids, len(intervals))
    times = np.asarray(population.spikes["times"])
    units = np.asarray(population.spikes["clusters"])
    n_units = len(population.units)
    if (times.ndim != 1 or times.dtype.kind not in "fiu" or not np.isfinite(times).all()
            or np.any(times[1:] < times[:-1]) or units.shape != times.shape
            or units.dtype.kind not in "iu" or np.any(units < 0) or np.any(units >= n_units)):
        raise ValueError("Counting requires validated sorted spike/unit associations")
    recording_indices = np.asarray(population.units["recording_index"])
    if (recording_indices.dtype.kind not in "iu" or np.any(recording_indices < 0)
            or np.any(recording_indices >= len(population.recordings))):
        raise ValueError("Unit recording references are invalid")
    coverages = [deepcopy(record["coverage"]) for record in population.recordings]
    for record, coverage in zip(population.recordings, coverages):
        if record["source"]["eid"] != population.eid:
            raise ValueError("Recording and population sessions differ")
        _support(coverage)
    overrides = {}
    for index, coverage in (unit_coverage or {}).items():
        if not isinstance(index, (int, np.integer)) or not 0 <= index < n_units:
            raise ValueError("Unit coverage key is not a unit-axis position")
        override = _coverage(coverage)
        base = coverages[recording_indices[index]]
        # Combine restrictions without extending a recording's known support.
        if base["observed"] is not None:
            if override["observed"] is None:
                override["observed"] = deepcopy(base["observed"])
            else:
                override["observed"] = [(max(a, c), min(b, d))
                    for a, b in base["observed"] for c, d in override["observed"]
                    if max(a, c) < min(b, d)]
            override["qualification"] = ("assumed" if "assumed" in
                (base["qualification"], coverage.qualification) else base["qualification"])
        override["invalid"] = list(base["invalid"]) + list(override["invalid"])
        override["provenance"] = repr((base["provenance"], coverage.provenance))
        overrides[int(index)] = override

    def count_window(request):
        request_id, (start, end), available = request
        edges = _edges(start, end, bin_size) if available else np.empty(0, dtype=float)
        n_bins = max(0, len(edges) - 1)
        counts = np.zeros((n_bins, n_units), dtype=np.int64)
        statuses = np.full((n_bins, n_units), UNKNOWN, dtype=np.uint8)
        duration = np.full((n_bins, n_units), np.nan)
        if available:
            # Left searches at both ends implement [start, end), retaining
            # coincident events and avoiding a full session scan per window.
            positions = np.searchsorted(times, edges, side="left")
            for row, (left, right) in enumerate(zip(positions[:-1], positions[1:])):
                counts[row] = np.bincount(units[left:right].astype(np.intp), minlength=n_units)
            for recording_index, coverage in enumerate(coverages):
                columns = recording_indices == recording_index
                status, observed = _coverage_bins(edges, coverage)
                statuses[:, columns] = status[:, None]
                duration[:, columns] = observed[:, None]
            for index, coverage in overrides.items():
                statuses[:, index], duration[:, index] = _coverage_bins(edges, coverage)
        valid = statuses == VALID
        if not available:
            outcome, reason = "unavailable", "missing_or_nonfinite_timing"
        elif n_units == 0:
            outcome, reason = "empty_selection", "no_selected_units"
        elif valid.all():
            outcome, reason = "available", None
        elif valid.any():
            outcome, reason = "partial", "incomplete_neural_coverage"
        else:
            outcome, reason = "unavailable", ",".join(COVERAGE_STATES[int(code)] for code in np.unique(statuses))
        return CountWindow(int(request_id), (float(start), float(end)), edges,
                           counts, valid, statuses, duration, outcome, reason)

    requests = zip(ids, intervals, finite)
    if workers > 1 and len(intervals) > 1:
        # Read shared validated inputs; each worker allocates only its own window.
        with ThreadPoolExecutor(max_workers=min(workers, len(intervals)),
                                thread_name_prefix="neural-count") as pool:
            windows = tuple(iter_progress(pool.map(count_window, requests),
                f"neural-data {population.eid}: counting", total=len(intervals), unit="windows"))
    else:
        windows = tuple(iter_progress(map(count_window, requests),
            f"neural-data {population.eid}: counting", total=len(intervals), unit="windows"))
    return NeuralCounts(population.eid, population.units.copy(deep=True),
        deepcopy(population.recordings), tuple(windows), dict(
            representation="unsmoothed_spike_counts", clock="source-session", units="seconds",
            counting_intervals="[start,end)", bin_size=bin_size,
            selection=deepcopy(population.selection), unit_coverage=overrides,
            coverage_policy="Only fully observed bins are valid; assumed/unknown bins remain qualified"))


def count_trials(population, trials, *, event=None, offsets=None, interval_fields=None,
                 bin_size=DEFAULT_BIN_SIZE, unit_coverage=None, workers=1):
    """Count a LoadedTrials table, preserving its original integer row index."""
    if not trials.sources or any(source.eid != population.eid for source in trials.sources):
        raise ValueError("Trial source identity must match the neural session")
    ids = _ids(trials.data.index.to_numpy(), len(trials.data))
    if (event is None) == (interval_fields is None):
        raise ValueError("Select either a trial event or trial interval fields")
    if interval_fields is not None:
        if (offsets is not None or not isinstance(interval_fields, (tuple, list))
                or len(interval_fields) != 2 or any(not isinstance(name, str) or not name for name in interval_fields)
                or interval_fields[0] == interval_fields[1]):
            raise ValueError("Trial interval fields require two distinct column names and no offsets")
        if any(name not in trials.data for name in interval_fields):
            raise ValueError(f"Required trial interval fields are unavailable: {interval_fields}")
        intervals = trials.data[list(interval_fields)].to_numpy(dtype=float, na_value=np.nan)
        definition = dict(interval_fields=list(interval_fields))
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (2,) or not np.isfinite(offsets).all() or offsets[1] <= offsets[0]:
            raise ValueError("Trial offsets must be a finite increasing pair")
        if event not in trials.data:
            raise ValueError(f"Required trial event is unavailable: {event}")
        origins = trials.data[event].to_numpy(dtype=float, na_value=np.nan)
        intervals = origins[:, None] + offsets
        definition = dict(event=event, offsets=offsets.tolist())
    result = count_intervals(population, intervals,
                             bin_size=bin_size, request_ids=ids, unit_coverage=unit_coverage,
                             workers=workers)
    result.configuration["trial_window"] = dict(**definition, identity="original trial table row index")
    result.configuration["trial_content_sha256"] = content_hash(trials.data)
    result.trial_sources = tuple(asdict(source) for source in trials.sources)
    return result
