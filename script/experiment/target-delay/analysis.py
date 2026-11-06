"""Offline matched-window metrics and plots for the target-delay experiment."""
import math
import numpy as np

from common import atomic_json, plain
from evaluation.metrics import trial_r2
from utils.progress import iter_progress


def correlation(y, p):
    if len(y) < 2:
        return dict(value=None, reason="insufficient_bins")
    if np.ptp(y) == 0 or np.ptp(p) == 0:
        return dict(value=None, reason="constant_series")
    return dict(value=float(np.corrcoef(y, p)[0, 1]), reason=None)


def curves(y, p, baseline):
    y, p, baseline = [np.asarray(a, dtype=np.float64) for a in (y,p,baseline)]
    if not y.size:
        return dict(correlation=dict(value=None, reason="insufficient_bins"), bins=0)
    error = p-y
    observed_std, predicted_std = float(y.std()), float(p.std())
    mse = float(np.mean(error**2))
    baseline_mse = float(np.mean((baseline-y)**2))
    return dict(bins=len(y), correlation=correlation(y,p), r2=trial_r2(y,p),
        mae=float(np.mean(np.abs(error))), rmse=math.sqrt(mse), mse=mse,
        baseline_mse=baseline_mse, mse_improvement=baseline_mse-mse,
        observed_mean=float(y.mean()), predicted_mean=float(p.mean()),
        mean_error=float(error.mean()), observed_std=observed_std, predicted_std=predicted_std,
        variance_ratio=predicted_std**2/observed_std**2 if observed_std else None,
        observed_peak_to_peak=float(np.ptp(y)), predicted_peak_to_peak=float(np.ptp(p)),
        baseline= dict(correlation=correlation(y,baseline), r2=trial_r2(y,baseline),
                       mae=float(np.mean(np.abs(baseline-y))), rmse=math.sqrt(baseline_mse)))


def likelihood(y, logp, baseline):
    y, logp, baseline = [np.asarray(a, dtype=np.float64) for a in (y,logp,baseline)]
    p = np.exp(logp)
    baseline = np.broadcast_to(np.maximum(baseline, 1e-8), y.shape)
    model_nll = float(np.sum(p-y*logp))
    null_nll = float(np.sum(baseline-y*np.log(baseline)))
    spikes = float(y.sum())
    return dict(cells=y.size, observed_spikes=spikes, nll_sum=model_nll, baseline_nll_sum=null_nll,
        poisson_nll=model_nll/y.size, baseline_poisson_nll=null_nll/y.size,
        nll_improvement=(null_nll-model_nll)/y.size,
        bits_per_spike=(null_nll-model_nll)/spikes/math.log(2) if spikes else None)


def aggregate_nll(records):
    cells = sum(r["cells"] for r in records)
    spikes = sum(r["observed_spikes"] for r in records)
    nll = sum(r["nll_sum"] for r in records)
    null = sum(r["baseline_nll_sum"] for r in records)
    return dict(cells=cells, observed_spikes=spikes, nll_sum=nll, baseline_nll_sum=null,
        poisson_nll=nll/cells, baseline_poisson_nll=null/cells,
        nll_improvement=(null-nll)/cells, bits_per_spike=(null-nll)/spikes/math.log(2) if spikes else None)


def distribution(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(available=len(values), mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,
        std=float(np.std(values)) if values else None,
        q25=float(np.quantile(values,.25)) if values else None,
        q75=float(np.quantile(values,.75)) if values else None)


def summarize(records):
    keys = ("mse", "baseline_mse", "mse_improvement", "rmse", "mae", "mean_error",
            "observed_std", "predicted_std", "variance_ratio")
    result = {k: distribution([r[k] for r in records]) for k in keys}
    result["correlation"] = distribution([r["correlation"]["value"] for r in records])
    result.update(contributors=len(records),
        unavailable_correlations=sum(r["correlation"]["value"] is None for r in records),
        fraction_improved_mse=float(np.mean([r["mse_improvement"] > 0 for r in records])) if records else None)
    return result


def rolling(array, w, endpoints):
    # Direct window means avoid cumulative-sum cancellation creating apparent
    # temporal variance/correlation in a constant-mean prediction.
    return np.stack([array[e-w+1:e+1].mean(0) for e in endpoints])


def contrast(metadata):
    """Only use explicitly persisted original trial contrast; never infer from CLIP."""
    values = []
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ("contrast", "contrastLeft", "contrastRight", "contrast_left", "contrast_right"):
                    if isinstance(value, (float, int)) and np.isfinite(value):
                        values.append((key,float(value)))
                elif isinstance(value, dict):
                    visit(value)
    visit(metadata)
    if not values:
        return "unavailable"
    # Keep side labels and exact recorded values; no normalization or gain.
    return str(sorted(set(values)))


def analyze(rows, support, evaluation):
    policy = support["compatibility"]["policy"]
    windows = evaluation["response_windows"]
    granularity = support["compatibility"].get("target_granularity", "neuron")
    channel_label = "region" if granularity == "region" else "neuron"
    first, curve_first = policy["first_target_index"], policy["first_curve_endpoint"]
    trial_results, neuron_data, neuron_curve_records, region_records, population_records = [], {}, {}, {}, {}
    saved = []
    for row in iter_progress(rows, "evaluation: temporal metrics", total=len(rows), unit="trials"):
        t, n = row.observed_neural.shape[0], len(row.neuron_identity)
        length = int(row.temporal_mask.sum())
        if not row.neuron_mask[:n].all():
            raise ValueError("Unexpected unavailable neuron")
        expected_mask = np.zeros(t, dtype=bool); expected_mask[first:length] = True
        if not np.array_equal(row.encoding_target_mask, expected_mask):
            raise ValueError("Inference support differs from the preflight policy")
        if not row.neural_prediction_mask[expected_mask].all():
            raise ValueError("Missing predictions on required support")
        y = row.observed_neural[:length,:n].astype(np.float64)
        logp = row.predicted_neural[:length,:n].astype(np.float64)
        expected = np.exp(logp)
        if not np.isfinite(expected[first:]).all() or not np.isfinite(logp[first:]).all():
            raise ValueError("Nonfinite neural predictions")
        mean = np.asarray(support["training_baseline"][row.session_id]["mean_counts"])
        baseline = np.broadcast_to(mean,y.shape)
        endpoints = np.arange(curve_first,length)
        units = row.neuron_identity
        region_column = "selected_region" if "selected_region" in units else "acronym" if "acronym" in units else None
        labels = [str(v) if isinstance(v,str) and v else "unassigned" for v in units[region_column]] if region_column else ["unassigned"]*n
        trial = dict(sample_id=row.sample_id, session_id=row.session_id, trial_id=row.trial_id,
            contrast=contrast(row.metadata), single_bin=likelihood(y[first:],logp[first:],mean), windows={})
        for j in range(n):
            key = (row.session_id,j)
            data = neuron_data.setdefault(key, dict(region=labels[j], y=[], logp=[], mean=mean[j]))
            data["y"].append(y[first:,j]); data["logp"].append(logp[first:,j])
        for w in windows:
            # All curves in spikes/s/neuron; W is offline only.
            obs, pred = [rolling(a,w,endpoints)*60 for a in (y,expected)]
            null = np.broadcast_to(mean*60,obs.shape)
            individual = []
            for j in range(n):
                metric = curves(obs[:,j],pred[:,j],null[:,j])
                individual.append(metric)
                neuron_curve_records.setdefault((row.session_id,j,w), []).append(metric)
            regions = {}
            for label in sorted(set(labels)):
                columns = np.flatnonzero(np.asarray(labels)==label)
                metric = curves(obs[:,columns].mean(1),pred[:,columns].mean(1),null[:,columns].mean(1))
                regions[label] = dict(neuron_count=len(columns), curve=metric,
                    neurons=summarize([individual[j] for j in columns]))
                region_records.setdefault((label,w), []).append(metric)
            population = curves(obs.mean(1),pred.mean(1),null.mean(1))
            population_records.setdefault(w,[]).append(population)
            trial["windows"][str(w)] = dict(neurons=individual, neuron_summary=summarize(individual),
                                          regions=regions, population=population)
        trial_results.append(trial)
        saved.append(dict(sample_id=row.sample_id, session_id=row.session_id,trial_id=row.trial_id,
            target_granularity=granularity, time=row.physical_timestamps[:length]-row.stim_on, observed=y.astype(np.int64),
            log_expected=logp, expected=expected, baseline=mean, target_indices=np.arange(first,length),
            curve_endpoints=endpoints, physical_timestamps=row.physical_timestamps[:length],
            bin_start_times=row.bin_start_times[:length], bin_end_times=row.bin_end_times[:length]))
    neurons = []
    for (session,j),data in iter_progress(neuron_data.items(), "evaluation: neuron summaries",
                                          total=len(neuron_data), unit="neurons"):
        neurons.append(dict(session_id=session,column=j,region=data["region"],
            single_bin=likelihood(np.concatenate(data["y"]),np.concatenate(data["logp"]),data["mean"]),
            windows={str(w): summarize(neuron_curve_records[session,j,w]) for w in windows}))
    single = aggregate_nll([t["single_bin"] for t in trial_results])
    contrasts = {}
    for label in sorted({t["contrast"] for t in trial_results}):
        selected = [t for t in trial_results if t["contrast"]==label]
        available = label != "unavailable" and len(selected) >= evaluation["minimum_contrast_trials"]
        contrasts[label] = dict(trials=len(selected), available=available,
            reason=None if available else "missing_original_contrast" if label == "unavailable" else "insufficient_trials",
            single_bin=aggregate_nll([t["single_bin"] for t in selected]) if available else None,
            windows={str(w):summarize([t["windows"][str(w)]["population"] for t in selected]) for w in windows} if available else {})
    regions = {}
    for region in sorted({n["region"] for n in neurons}):
        subset = [n for n in neurons if n["region"]==region]
        regions[region] = dict(neurons=len(subset), single_bin=aggregate_nll([n["single_bin"] for n in subset]),
            windows={str(w):summarize(region_records[region,w]) for w in windows})
    result = dict(schema_version=1, target_granularity=granularity, definitions=dict(nll="single-bin Poisson, log_input=True, full=False; cell weighted",
        baseline=f"training common-support mean count per {channel_label}, floored at 1e-8 only for NLL",
        curves="trailing W-bin mean counts times 60; identical endpoints for all W",
        summary=f"curve metrics macro averaged across trial-{channel_label} pairs; population curves average matched target channels",
        correlation="Pearson within each trial, null for constant/short series",
        uncertainty="descriptive trial spread; overlapping windows are not independent replicates"),
        single_bin=single, neurons=neurons,trials=trial_results, regions=regions,contrasts=contrasts,
        windows={str(w): dict(neurons=summarize([r for (s,j,k),rs in neuron_curve_records.items() if k==w for r in rs]),
                            population=summarize(population_records[w])) for w in windows})
    return result,saved


def plots(directory, saved, windows, count, *, label=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    indices = np.linspace(0,len(saved)-1,min(count,len(saved)),dtype=int)
    for i in iter_progress(indices, "evaluation: temporal plots", total=len(indices), unit="plots"):
        row = saved[i]
        fig, axes = plt.subplots(len(windows),1,figsize=(11,2.6*len(windows)),squeeze=False)
        for ax,w in zip(axes[:,0],windows):
            endpoints = row["curve_endpoints"]
            for field,title in (("observed","Observed"),("expected","Predicted")):
                rate = rolling(row[field],w,endpoints).mean(1)*60
                ax.plot(row["time"][endpoints],rate,label=title)
            ax.axhline(float(row["baseline"].mean()*60),color="gray",ls="--",label="Training mean")
            ax.set(title=f"{label} {row['session_id']} trial {row['trial_id']} W={w}",
                   ylabel="spikes/s/" + row.get("target_granularity", "neuron"),xlabel="Seconds from stimulus onset")
            ax.legend()
        fig.tight_layout(); fig.savefig(directory/f"trial_{i:06d}.png",dpi=130); plt.close(fig)
