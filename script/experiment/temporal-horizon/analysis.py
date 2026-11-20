"""E0 metrics on explicit original-bin intervals; no activity-based exclusions."""

import numpy as np

WINDOWS = (1, 6, 12)
DEFINITIONS = {
    "poisson_nll": "mean(exp(logp)-y*logp); log_input=True, full=False",
    "improvement": "training-constant NLL minus model NLL; positive favors model",
    "correlation": "per-trial per-channel Pearson of trailing W-bin count means",
    "variance_ratio": "population variance(prediction)/population variance(target), same W means",
    "cell_weighted": "NLL sums / target cells; diagnostic means weighted by valid window endpoints",
    "equal_trial": "trial NLL means; diagnostic mean over defined channels within trial, then over trials",
    "variability": "population standard deviation and quartiles across defined trial summaries",
    "undefined": "null with explicit reason; retained in coverage counts, never treated as zero",
    "paired_difference": "H12 minus H6; negative NLL difference favors H12; no automatic winner",
}


def distribution(values, weights=None):
    pairs = [
        (v, 1 if weights is None else weights[i])
        for i, v in enumerate(values)
        if v is not None and np.isfinite(v)
    ]
    vals = [p[0] for p in pairs]
    return {
        "defined": len(vals),
        "undefined": len(values) - len(vals),
        "mean": float(np.average(vals, weights=[p[1] for p in pairs]))
        if vals
        else None,
        "std": float(np.std(vals)) if vals else None,
        "median": float(np.median(vals)) if vals else None,
        "q25": float(np.quantile(vals, 0.25)) if vals else None,
        "q75": float(np.quantile(vals, 0.75)) if vals else None,
    }


def likelihood(y, logp, baseline):
    baseline = np.maximum(baseline, 1e-8)
    loss = np.exp(logp) - y * logp
    null = baseline - y * np.log(baseline)
    if not np.isfinite(loss).all() or not np.isfinite(null).all():
        raise ValueError("Nonfinite likelihood; E0 cannot compare these predictions")
    cells = y.size
    model_sum, null_sum = float(loss.sum()), float(null.sum())
    return {
        "cells": cells,
        "nll_sum": model_sum,
        "baseline_nll_sum": null_sum,
        "poisson_nll": model_sum / cells if cells else None,
        "nll_improvement": (null_sum - model_sum) / cells if cells else None,
        "reason": None if cells else "empty_interval",
    }


def total(rows):
    cells = sum(r["cells"] for r in rows)
    loss = sum(r["nll_sum"] for r in rows)
    null = sum(r["baseline_nll_sum"] for r in rows)
    return {
        "cells": cells,
        "poisson_nll": loss / cells if cells else None,
        "baseline_poisson_nll": null / cells if cells else None,
        "nll_improvement": (null - loss) / cells if cells else None,
        "reason": None if cells else "empty_interval",
    }


def temporal(y, p, w):
    if len(y) < w:
        return {
            "endpoints": 0,
            "correlation": None,
            "variance_ratio": None,
            "correlation_reason": "insufficient_bins",
            "variance_ratio_reason": "insufficient_bins",
        }
    # Valid trailing means cannot read outside this already sliced interval.
    ys = np.lib.stride_tricks.sliding_window_view(y, w).mean(-1)
    ps = np.lib.stride_tricks.sliding_window_view(p, w).mean(-1)
    yvar, pvar = float(np.var(ys)), float(np.var(ps))
    reason = (
        "insufficient_endpoints"
        if len(ys) < 2
        else "constant_target"
        if yvar == 0
        else "constant_prediction"
        if pvar == 0
        else None
    )
    corr = float(np.corrcoef(ys, ps)[0, 1]) if reason is None else None
    return {
        "endpoints": len(ys),
        "correlation": corr,
        "correlation_reason": reason,
        "variance_ratio": pvar / yvar if yvar else None,
        "variance_ratio_reason": None if yvar else "constant_target",
    }


def interval(length, stop, support):
    if support == "primary":
        return 11, min(length, stop)
    if support == "full":
        return 11, length
    if support == "tail":
        return max(11, stop), length
    raise ValueError(support)


def evaluate(records, support, *, parent=False):
    """Records retain full original time axes, native ordered identities and log counts."""
    result = {}
    for scope in ("primary", "full", "tail"):
        trials, channels = [], {}
        for row in records:
            session = row["session_id"]
            mapping = support["compatibility"]["regional_targets"][session]
            labels = mapping["regions"]
            baseline = np.array(support["training_baseline"][session]["mean_counts"])
            y, logp = (
                row["observed"].astype(np.float64),
                row["log_expected"].astype(np.float64),
            )
            start, stop = interval(len(y), support["primary_stop_bin"], scope)
            y, logp = y[start:stop], logp[start:stop]
            if parent:
                names = sorted(set(mapping["parent_regions"]))
                groups = [
                    np.asarray(mapping["parent_regions"]) == name for name in names
                ]
                y = np.stack([y[:, g].sum(1) for g in groups], axis=1)
                # Sum intensities, never log outputs. Baselines sum before clamping.
                rates = np.exp(logp)
                logp = np.stack([np.log(rates[:, g].sum(1)) for g in groups], axis=1)
                baseline = np.array([baseline[g].sum() for g in groups])
                labels = names
            p = np.exp(logp)
            if not np.isfinite(p).all():
                raise ValueError("Nonfinite expected counts")
            metrics = []
            for j, label in enumerate(labels):
                item = {
                    "session_id": session,
                    "channel": label,
                    "likelihood": likelihood(y[:, j], logp[:, j], baseline[j]),
                    "windows": {str(w): temporal(y[:, j], p[:, j], w) for w in WINDOWS},
                }
                metrics.append(item)
                channels.setdefault((session, label), []).append(item)
            trials.append(
                {
                    "session_id": session,
                    "trial_id": row["trial_id"],
                    "sample_id": row["sample_id"],
                    "first_bin": start,
                    "stop_bin": max(start, stop),
                    "likelihood": likelihood(y, logp, baseline),
                    "channels": metrics,
                }
            )
        diagnostics = {}
        for w in WINDOWS:
            items = [c["windows"][str(w)] for t in trials for c in t["channels"]]
            diagnostics[str(w)] = {}
            for metric in ("correlation", "variance_ratio"):
                per_trial = [
                    distribution([c["windows"][str(w)][metric] for c in t["channels"]])[
                        "mean"
                    ]
                    for t in trials
                ]
                reasons = {}
                for item in items:
                    reason = item[metric + "_reason"]
                    if reason:
                        reasons[reason] = reasons.get(reason, 0) + 1
                diagnostics[str(w)][metric] = {
                    "cell_weighted": distribution(
                        [i[metric] for i in items], [i["endpoints"] for i in items]
                    ),
                    "equal_trial": distribution(per_trial),
                    "undefined_reasons": reasons,
                }
        values = [t["likelihood"]["nll_improvement"] for t in trials]
        defined = [v for v in values if v is not None]
        result[scope] = {
            "cell_weighted": total([t["likelihood"] for t in trials]),
            "equal_trial": {
                k: distribution([t["likelihood"][k] for t in trials])
                for k in ("poisson_nll", "nll_improvement")
            },
            "fraction_improved_trials": float(np.mean(np.array(defined) > 0))
            if defined
            else None,
            "trials_without_support": len(values) - len(defined),
            "diagnostics": diagnostics,
            "trials": trials,
            "channels": [
                {
                    "session_id": s,
                    "channel": c,
                    "cell_weighted": total([item["likelihood"] for item in items]),
                    "windows": {
                        str(w): {
                            metric: distribution(
                                [i["windows"][str(w)][metric] for i in items]
                            )
                            for metric in ("correlation", "variance_ratio")
                        }
                        for w in WINDOWS
                    },
                }
                for (s, c), items in channels.items()
            ],
        }
    return result


def difference(a, b):
    return None if a is None or b is None else b - a


def paired(left, right):
    result = {}
    for scope in ("primary", "full", "tail"):
        a, b = left[scope], right[scope]
        if len(a["trials"]) != len(b["trials"]):
            raise ValueError("Incomplete paired trials")
        rows = []
        for x, y in zip(a["trials"], b["trials"]):
            if (
                x["session_id"],
                x["trial_id"],
                x["sample_id"],
                x["first_bin"],
                x["stop_bin"],
            ) != (
                y["session_id"],
                y["trial_id"],
                y["sample_id"],
                y["first_bin"],
                y["stop_bin"],
            ):
                raise ValueError("Trial comparison identity differs")
            windows = {}
            for w in WINDOWS:
                windows[str(w)] = {
                    metric: distribution(
                        [
                            difference(
                                c["windows"][str(w)][metric],
                                d["windows"][str(w)][metric],
                            )
                            for c, d in zip(x["channels"], y["channels"])
                        ]
                    )["mean"]
                    for metric in ("correlation", "variance_ratio")
                }
            rows.append(
                {
                    "session_id": x["session_id"],
                    "trial_id": x["trial_id"],
                    "cells": x["likelihood"]["cells"],
                    "windows": windows,
                    "nll_H12_minus_H6": difference(
                        x["likelihood"]["poisson_nll"], y["likelihood"]["poisson_nll"]
                    ),
                }
            )
        values = [r["nll_H12_minus_H6"] for r in rows]
        defined = [v for v in values if v is not None]
        result[scope] = {
            "trials": rows,
            "equal_trial_nll_difference": distribution(values),
            "cell_weighted_nll_difference": difference(
                a["cell_weighted"]["poisson_nll"], b["cell_weighted"]["poisson_nll"]
            ),
            "fraction_trials_H12_lower_nll": float(np.mean(np.array(defined) < 0))
            if defined
            else None,
            "fraction_trials_H6_lower_nll": float(np.mean(np.array(defined) > 0))
            if defined
            else None,
            "tied_trials": sum(v == 0 for v in defined),
            "windows": {
                str(w): {
                    metric: distribution([r["windows"][str(w)][metric] for r in rows])
                    for metric in ("correlation", "variance_ratio")
                }
                for w in WINDOWS
            },
            "channels": [
                {
                    "session_id": x["session_id"],
                    "channel": x["channel"],
                    "nll_H12_minus_H6": difference(
                        x["cell_weighted"]["poisson_nll"],
                        y["cell_weighted"]["poisson_nll"],
                    ),
                    "windows": {
                        str(w): {
                            metric: distribution(
                                [
                                    difference(
                                        tx["channels"][j]["windows"][str(w)][metric],
                                        ty["channels"][j]["windows"][str(w)][metric],
                                    )
                                    for tx, ty in zip(a["trials"], b["trials"])
                                    if tx["session_id"] == x["session_id"]
                                ]
                            )
                            for metric in ("correlation", "variance_ratio")
                        }
                        for w in WINDOWS
                    },
                }
                for j, (x, y) in enumerate(zip(a["channels"], b["channels"]))
            ],
        }
    return result
