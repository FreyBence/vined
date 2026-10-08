"""Validity-aware neural encoding metrics and explicit scientific aggregation."""

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from evaluation.predictions import PredictionCollection


NEURAL_METRICS = ("trial_r2", "psth_r2", "bits_per_spike")


@dataclass(frozen=True)
class MetricConfig:
    aggregation: str = "session_weighted"
    bps_baseline: str = "test_mean_count"
    psth_grouping: str = "all_trials"

    def __post_init__(self):
        if self.aggregation not in ("session_weighted", "neuron_weighted"):
            raise ValueError("Unsupported neural metric aggregation")
        if self.bps_baseline != "test_mean_count" or self.psth_grouping != "all_trials":
            raise ValueError("Unsupported BPS baseline or PSTH grouping")


def _result(value=None, reason=None, **coverage):
    if value is not None and not np.isfinite(value):
        value, reason = None, "nonfinite_calculation"
    return dict(value=None if value is None else float(value),
                status="available" if value is not None else "unavailable", reason=reason, **coverage)


def trial_r2(observed, predicted):
    """R² over a flat vector of valid observations, with explicit degeneracy."""
    y, p = np.asarray(observed, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError("R² expects matching one-dimensional valid observations")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        return _result(reason="nonfinite_observations_or_predictions", observations=len(y))
    if len(y) < 2:
        return _result(reason="insufficient_observations", observations=len(y))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        variance = np.sum((y - y.mean()) ** 2)
        if variance == 0:
            return _result(reason="constant_target", observations=len(y))
        return _result(1 - np.sum((y - p) ** 2) / variance, observations=len(y))


def poisson_bps(observed, log_expected_count):
    """Poisson likelihood gain versus the valid test mean count, in bits/spike."""
    y, logp = np.asarray(observed, dtype=np.float64), np.asarray(log_expected_count, dtype=np.float64)
    if y.ndim != 1 or y.shape != logp.shape or np.any(y < 0):
        raise ValueError("BPS expects matching valid nonnegative count vectors")
    coverage = dict(observations=len(y))
    if not np.isfinite(y).all() or not np.isfinite(logp).all():
        return _result(reason="nonfinite_observations_or_predictions", **coverage)
    if not len(y):
        return _result(reason="insufficient_observations", **coverage)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        total = y.sum()
        baseline = y.mean()
        if not np.isfinite(total) or not np.isfinite(baseline):
            return _result(reason="nonfinite_calculation", **coverage)
        coverage.update(observed_spikes=float(total), baseline_expected_count=float(baseline))
        if total == 0:
            return _result(reason="zero_observed_spikes", **coverage)
        # Factorial terms cancel. Keep log predictions to avoid log(exp(.)) underflow.
        improvement = np.sum(y * (logp - np.log(baseline)) - (np.exp(logp) - baseline))
        return _result(improvement / total / np.log(2), **coverage)


def _aggregate(items):
    available = [item["value"] for item in items if item["value"] is not None]
    return _result(np.mean(available) if available else None,
                   reason=None if available else "no_available_contributors",
                   total_contributors=len(items), available_contributors=len(available),
                   unavailable_contributors=len(items) - len(available),
                   partial=len(available) != len(items) or any(item.get("partial", False) for item in items))


def _psth(rows, neuron):
    coordinates, values = {}, {}
    for row in rows:
        if not row.neuron_mask[neuron]:
            continue
        for index in np.flatnonzero(row.temporal_mask):
            position = int(row.temporal_positions[index])
            coordinate = np.array([row.bin_start_times[index] - row.stim_on,
                                   row.bin_end_times[index] - row.stim_on])
            if (not np.isfinite(coordinate).all() or position < 0
                    or (position in coordinates and not np.allclose(
                        coordinates[position], coordinate, atol=1e-8, rtol=0))):
                return _result(reason="incompatible_stimulus_relative_bins"), None
            coordinates[position] = coordinate
            with np.errstate(over="ignore", invalid="ignore"):
                pair = (float(row.observed_neural[index, neuron]),
                        float(np.exp(np.float64(row.predicted_neural[index, neuron]))))
            values.setdefault(position, []).append(pair)
    positions = sorted(values)
    if any(not np.isfinite(values[position]).all() for position in positions):
        return _result(reason="nonfinite_observations_or_predictions"), None
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.array([np.mean(values[position], axis=0) for position in positions]).reshape(-1, 2)
    if not np.isfinite(means).all():
        return _result(reason="nonfinite_calculation"), None
    result = trial_r2(means[:, 0], means[:, 1])
    profile = dict(temporal_positions=positions,
                   relative_bin_edges=[coordinates[p].tolist() for p in positions],
                   trial_counts=[len(values[p]) for p in positions],
                   observed=means[:, 0].tolist(), predicted=means[:, 1].tolist())
    return result, profile


def _neuron_metrics(rows, neuron):
    y, logp, affected = [], [], []
    for row in rows:
        if not row.neuron_mask[neuron]:
            continue
        valid_y = row.observed_neural[row.temporal_mask, neuron]
        valid_p = row.predicted_neural[row.temporal_mask, neuron]
        y.extend(valid_y)
        logp.extend(valid_p)
        with np.errstate(over="ignore", invalid="ignore"):
            finite_counts = np.isfinite(np.exp(np.asarray(valid_p, dtype=np.float64))).all()
        if not np.isfinite(valid_y).all() or not np.isfinite(valid_p).all() or not finite_counts:
            affected.append(row.sample_id)
    y, logp = np.asarray(y, dtype=np.float64), np.asarray(logp, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        counts = np.exp(logp)
    psth_result, profile = _psth(rows, neuron)
    metrics = dict(trial_r2=trial_r2(y, counts), psth_r2=psth_result, bits_per_spike=poisson_bps(y, logp))
    if not np.isfinite(logp).all():
        metrics["trial_r2"] = _result(reason="nonfinite_observations_or_predictions", observations=len(y))
        metrics["psth_r2"], profile = _result(reason="nonfinite_observations_or_predictions"), None
    for item in metrics.values():
        item["sample_ids"] = [row.sample_id for row in rows if row.neuron_mask[neuron]]
        item["nonfinite_sample_ids"] = affected
    return dict(column=neuron, metrics=metrics, psth=profile)


def _visual_metric(rows):
    scores, affected = [], []
    for row in rows:
        y = np.asarray(row.observed_visual[row.temporal_mask], dtype=np.float64)
        p = np.asarray(row.predicted_visual[row.temporal_mask], dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            norms = np.linalg.norm(y, axis=-1) * np.linalg.norm(p, axis=-1)
            cosines = np.sum(y * p, axis=-1) / norms
        if not np.isfinite(cosines).all():
            affected.append(row.sample_id)
        scores.extend(cosines)
    if affected:
        return _result(reason="nonfinite_or_zero_norm_visual_vectors", valid_vectors=len(scores),
                       affected_sample_ids=affected)
    return _result(np.mean(scores) if scores else None,
                   reason=None if scores else "insufficient_observations", valid_vectors=len(scores),
                   affected_sample_ids=[])


def compute_metrics(collection: PredictionCollection, *, config=None):
    """Return JSON-compatible definitions, per-unit/session scores and global means."""
    config = MetricConfig() if config is None else config
    if not isinstance(config, MetricConfig) or not collection.predictions:
        raise ValueError("Metrics require a nonempty collection and MetricConfig")
    sessions = {}
    for row in collection.predictions:
        if (row.temporal_mask.dtype != np.bool_ or row.neuron_mask.dtype != np.bool_
                or row.observed_neural.shape != (len(row.temporal_mask), len(row.neuron_mask))
                or row.observed_visual.shape[0] != len(row.temporal_mask)
                or not np.array_equal(row.neuron_mask, np.arange(len(row.neuron_mask)) < len(row.neuron_identity))
                or not np.isfinite(row.bin_size) or row.bin_size <= 0
                or (row.predicted_neural is not None and row.predicted_neural.shape != row.observed_neural.shape)
                or (row.predicted_visual is not None and row.predicted_visual.shape != row.observed_visual.shape)):
            raise ValueError(f"Malformed prediction masks, dimensions or bin duration for {row.sample_id}")
        sessions.setdefault(row.session_id, []).append(row)
    results = {}
    for session, rows in sessions.items():
        units = rows[0].neuron_identity
        if any(not row.neuron_identity.equals(units) or row.bin_size != rows[0].bin_size for row in rows):
            raise ValueError(f"Inconsistent ordered population or bin duration for session {session}")
        mode = collection.configuration["model_mode"]
        neural, visual = mode in ("encoding", "mm"), mode in ("decoding", "mm")
        if mode not in ("encoding", "decoding", "mm"):
            raise ValueError("Unsupported metric prediction direction")
        if any((neural and row.predicted_neural is None) or (visual and row.predicted_visual is None) for row in rows):
            raise ValueError("Missing required prediction modality")
        neurons = [_neuron_metrics(rows, neuron) for neuron in range(len(units))] if neural else []
        # Complete scoped unit records remain available alongside positional metric columns.
        from trainer.artifacts import plain
        for neuron in neurons:
            neuron["identity"] = plain(units.iloc[neuron["column"]].to_dict())
        metrics = {name: _aggregate([neuron["metrics"][name] for neuron in neurons])
                   for name in NEURAL_METRICS} if neural else {}
        if visual:
            metrics["visual_cosine"] = _visual_metric(rows)
        results[session] = dict(trial_count=len(rows), neuron_count=len(units),
                                sample_ids=[row.sample_id for row in rows],
                                neurons=neurons, metrics=metrics, bin_size=rows[0].bin_size)
    names = next(iter(results.values()))["metrics"]
    global_metrics = {}
    for name in names:
        if name != "visual_cosine" and config.aggregation == "neuron_weighted":
            items = [neuron["metrics"][name] for session in results.values() for neuron in session["neurons"]]
        else:
            items = [session["metrics"][name] for session in results.values()]
        global_metrics[name] = _aggregate(items)
        global_metrics[name]["unavailable_sessions"] = [key for key, value in results.items()
                                                        if value["metrics"][name]["value"] is None]
    definitions = dict(asdict(config), trial_r2_axes="valid trial/time cells pooled per session neuron",
                       psth_axes="mean across all session trials per matching stimulus-relative position, then R2 over positions",
                       bps_formula="sum(y*(log_count-log(test_mean_count))-(exp(log_count)-test_mean_count))/sum(y)/log(2)",
                       bps_baseline_source="valid test observations per session neuron; test statistics explicitly permitted",
                       units="counts per persisted bin; no Hz conversion or firing-rate gate",
                       invalidity_policy="metric-specific unavailable values; average available contributors with explicit coverage",
                       visual_aggregation="valid vectors pooled within session, then equal session weighting",
                       negative_values="retained", numeric_precision="float64")
    return dict(configuration=definitions, session_results=results, global_metrics=global_metrics,
                provenance=deepcopy(collection.provenance))
