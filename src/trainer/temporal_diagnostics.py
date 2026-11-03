"""Read-only encoding diagnostics on the checkpoint's original train/val splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from trainer.artifacts import atomic_json, load_training_checkpoint, plain
from trainer.pretrained import model_from_checkpoint
from trainer.selection import select_neural_regions
from trainer.targets import transform_neural_targets
from training_dataset import temporal_context_view
from training_dataset.handoff import load_dataset_splits


WIDTHS = (1, 3, 6, 12)


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_chunks(stream)


def _hash_chunks(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _corr(y, p):
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return None
    return float(np.corrcoef(y, p)[0, 1])


def _divide(a, b):
    return float(a / b) if b else None


def _metrics(y, p, threshold=None, baseline=None, edges=None, log_p=None):
    """One trial/region or explicitly pooled values; empty support stays unavailable."""
    if not len(y):
        return {"bins": 0, "unavailable_reason": "no complete eligible windows"}
    vy, vp = float(np.var(y)), float(np.var(p))
    objective = p - y * (np.log(p) if log_p is None else log_p)
    result = dict(bins=len(y), observed_mean=float(y.mean()), predicted_mean=float(p.mean()),
        observed_variance=vy, predicted_variance=vp, variance_ratio=_divide(vp, vy),
        pearson=_corr(y, p), mean_bias=float((p-y).mean()),
        calibration_ratio=_divide(p.mean(), y.mean()), poisson_objective=float(objective.mean()),
        predicted_min=float(p.min()), predicted_max=float(p.max()),
        predicted_percentiles=dict(zip(("p05", "p25", "p50", "p75", "p95", "p99"),
                                      np.percentile(p, [5, 25, 50, 75, 95, 99]).tolist())))
    if result["pearson"] is None:
        result["pearson_unavailable_reason"] = "constant sequence or fewer than two bins"
    if baseline is not None:
        null = baseline - y * np.log(max(baseline, 1e-300))
        result["training_null_bps"] = _divide(null.sum()-objective.sum(), y.sum()*np.log(2))
    if threshold is not None:
        elevated, positive = y >= threshold, p >= threshold
        tp = int((elevated & positive).sum())
        result.update(threshold=threshold, elevated_bins=int(elevated.sum()),
            predicted_positive_bins=int(positive.sum()), true_positive_bins=tp,
            elevated_recall=_divide(tp, elevated.sum()), predicted_positive_rate=float(positive.mean()),
            precision=_divide(tp, positive.sum()),
            false_positive_rate=_divide((positive & ~elevated).sum(), (~elevated).sum()),
            peak_signed_error=float((p-y)[elevated].mean()) if elevated.any() else None,
            peak_mae=float(np.abs(p-y)[elevated].mean()) if elevated.any() else None)
        k = int(np.ceil(.05 * len(y)))
        # Stable descending order breaks ties by earlier original eligible bin.
        ranked_y = np.argsort(-y, kind="stable")[:k]
        ranked_p = np.argsort(-p, kind="stable")[:k]
        result.update(ranking_k=k, ranking_hits=len(np.intersect1d(ranked_y, ranked_p)),
                      ranking_recall_5pct=len(np.intersect1d(ranked_y, ranked_p))/k)
    if edges is not None:
        groups = np.searchsorted(edges, p, side="right")
        result["calibration_bins"] = [dict(group=i, bins=int((groups == i).sum()),
            predicted_mean=float(p[groups == i].mean()) if (groups == i).any() else None,
            observed_mean=float(y[groups == i].mean()) if (groups == i).any() else None)
            for i in range(len(edges)+1)]
    return result


def _smooth(y, p, valid, width):
    """Trailing means with full contiguous eligible support; no onset padding."""
    if len(valid) < width:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)
    kernel = np.ones(width)
    keep = np.convolve(valid.astype(int), kernel, "valid") == width
    indices = np.arange(width-1, len(valid))[keep]
    return (np.convolve(y, kernel/width, "valid")[keep],
            np.convolve(p, kernel/width, "valid")[keep], indices)


def _average(rows):
    keys = sorted({k for row in rows for k, v in row.items()
                   if (isinstance(v, (int, float)) or v is None) and k != "bins"})
    result, contributors = {}, {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
        contributors[key] = len(values)
    return dict(metrics=result, available_contributors=contributors, records=len(rows))


def _summaries(records, arrays, statistics):
    result = {}
    for split in ("train", "val"):
        selected = [r for r in records if r["split"] == split]
        groups = {}
        for row in selected:
            groups.setdefault((row["session_id"], row["region"]), []).append(row)
        grouped = []
        for (session, region), rows in groups.items():
            stats = statistics[session][region]
            widths = {}
            for width in WIDTHS:
                ys, ps, logs, centered_y, centered_p = [], [], [], [], []
                for row in rows:
                    a, c = arrays[row["array_key"]], row["column"]
                    y, p, indices = _smooth(a["observed"][:, c], a["expected"][:, c], a["scoring_mask"][:, c], width)
                    ys.append(y); ps.append(p)
                    logs.append(a["log_expected"][indices, c])
                    if len(y):
                        centered_y.append(y-y.mean()); centered_p.append(p-p.mean())
                y, p = np.concatenate(ys), np.concatenate(ps)
                pooled = _metrics(y, p, stats["q95"] if width == 1 else None,
                    stats["mean"], stats["calibration_edges"] if width == 1 else None,
                    np.concatenate(logs) if width == 1 else None)
                # Ranking is meaningful within trials only. Pool numerator/denominator.
                if width == 1 and len(y):
                    pooled.pop("ranking_k"); pooled.pop("ranking_hits"); pooled.pop("ranking_recall_5pct")
                    trial_scores = [r["scales"]["1"] for r in rows if r["scales"]["1"]["bins"]]
                    pooled["within_trial_ranking_recall_5pct_k_weighted"] = _divide(
                        sum(m["ranking_hits"] for m in trial_scores), sum(m["ranking_k"] for m in trial_scores))
                if centered_y:
                    cy, cp = np.concatenate(centered_y), np.concatenate(centered_p)
                    pooled.update(trial_centered_observed_variance=float(np.var(cy)),
                        trial_centered_predicted_variance=float(np.var(cp)),
                        trial_centered_variance_ratio=_divide(np.var(cp), np.var(cy)),
                        trial_centered_pearson=_corr(cy, cp))
                trial_metrics = [r["scales"][str(width)] for r in rows]
                available = [m for m in trial_metrics if m["bins"]]
                widths[str(width)] = dict(pooled=pooled, equal_trial=_average(trial_metrics),
                    within_trial_pearson_median=float(np.median([m["pearson"] for m in available if m["pearson"] is not None]))
                        if any(m.get("pearson") is not None for m in available) else None,
                    trial_mean_pearson=_corr(np.array([m["observed_mean"] for m in available]),
                                             np.array([m["predicted_mean"] for m in available])))
            grouped.append(dict(session_id=session, region=region, trials=len(rows), scales=widths))
        sessions = []
        for session in sorted({g["session_id"] for g in grouped}):
            session_groups = [g for g in grouped if g["session_id"] == session]
            sessions.append(dict(session_id=session, scales={str(w): _average([
                g["scales"][str(w)]["equal_trial"]["metrics"] for g in session_groups]) for w in WIDTHS}))
        result[split] = dict(trials=len({r["array_key"] for r in selected}),
            eligible_trial_region_records=sum(r["scales"]["1"]["bins"] > 0 for r in selected),
            eligible_cells=sum(r["scales"]["1"]["bins"] for r in selected),
            groups=grouped, equal_region_sessions=sessions,
            equal_session={str(w): _average([s["scales"][str(w)]["metrics"] for s in sessions]) for w in WIDTHS})
    return result


def run_diagnostic(checkpoint, output_dir, *, device="cpu", dataset_path=None):
    """Write a new diagnostic directory; reject incompatible regional encoders."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Diagnostic destination already exists: {destination}")
    checkpoint = Path(checkpoint).resolve()
    digest = _hash(checkpoint)
    state = load_training_checkpoint(checkpoint)
    config, dataset = state["config"], state["config"]["dataset"]
    if config["training"]["objective"] != "encoding" or dataset.get("target_granularity") != "region":
        raise ValueError("This diagnostic requires an encoding checkpoint with regional count targets")
    context = config["training"].get("temporal_context")
    if context is None or context["target_support_bins"] != 12 or context["boundary_policy"] != "complete":
        raise ValueError("An explicit checkpoint context with complete H=12 scoring support is required")
    train, val, unused_test, metadata = load_dataset_splits(dataset_path or dataset["dataset_path"],
        session_ids=dataset["selected_session_ids"], expected_generation_id=dataset["dataset_generation_id"])
    # The existing handoff verifies the complete generation. No test sample enters
    # runtime selection, aggregation, inference, thresholds, metrics or reports.
    del unused_test
    if metadata["dataset_metadata"] != dataset["dataset_metadata"]:
        raise ValueError("Checkpoint split metadata differs from the verified generation")
    train, val, metadata = select_neural_regions((train, val), metadata,
        dataset.get("neural_region_selection", "all_recorded"),
        regions=dataset.get("neural_selection_policy", {}).get("regions"))
    if metadata.get("neural_selection_policy") != dataset.get("neural_selection_policy"):
        raise ValueError("Runtime neural selection differs from checkpoint")
    for view in (train, val):
        for sample in view.samples:
            expected = dataset["regional_targets"][sample.session_id]["ordered_units"]
            if plain(sample.neuron_identity.to_dict(orient="records")) != expected:
                raise ValueError("Ordered source units differ from checkpoint")
    legacy_regions = None
    if "regional_target_policy" not in dataset:
        region_sets = {tuple(m["regions"]) for m in dataset["regional_targets"].values()}
        if len(region_sets) != 1:
            raise ValueError("Legacy root-based restoration needs common recorded region ordering")
        legacy_regions = list(next(iter(region_sets)))
    train, val, metadata = transform_neural_targets((train, val), metadata, "region", regions=legacy_regions)
    if plain(metadata["regional_targets"]) != dataset["regional_targets"] or metadata["eid_list"] != dataset["eid_list"]:
        raise ValueError("Regional mapping differs from checkpoint")
    memberships = {}
    for view in (train, val):
        if not view.samples:
            raise ValueError(f"No samples in required split {view.split}")
        memberships[view.split] = [dict(sample_id=s.sample_id, session_id=s.session_id,
            trial_id=s.trial_id, split=s.split) for s in view.samples]
        if memberships[view.split] != state["compatibility"]["memberships"][view.split]:
            raise ValueError("Training/validation membership differs from checkpoint")
    model = model_from_checkpoint(checkpoint).to(device).eval()
    model.requires_grad_(False)
    policies, arrays, records = set(), {}, []
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "predictions").mkdir()
    with torch.inference_mode():
        for view in (train, val):
            for index, sample in enumerate(view.samples):
                length = sample.sequence_length
                positions = torch.tensor(sample.temporal_positions[:length], device=device)[None]
                valid = torch.tensor(sample.temporal_mask[:length], device=device)[None]
                context_view = temporal_context_view(positions, valid, direction="encoding",
                    mode=context["mode"], context_bins=context["bins"])
                output = model(visual_features=torch.tensor(sample.visual[:length], device=device)[None],
                    temporal_positions=positions, temporal_mask=valid, session_id=[sample.session_id],
                    context_view=context_view)
                if not torch.equal(output.target_eligibility, context_view.target_mask):
                    raise ValueError("Restored model and dataset scoring support differ")
                n = sample.neuron_count
                z = output.neural_prediction[0, :, :n].double().cpu().numpy()
                prediction_mask = output.prediction_mask[0].cpu().numpy()
                mask = context_view.target_mask[0].cpu().numpy() & prediction_mask
                channels = output.neuron_mask[0, :n].cpu().numpy()
                if not np.array_equal(channels, sample.neuron_mask[:n]):
                    raise ValueError("Restored model channel validity differs from recorded targets")
                if not np.all(prediction_mask[context_view.target_mask[0].cpu().numpy()]):
                    raise ValueError("Model predictions do not cover all recorded H=12 scoring bins")
                scoring = mask[:, None] & channels[None, :]
                expected = np.exp(z)
                if not np.isfinite(expected[scoring]).all() or np.any(expected[scoring] <= 0):
                    raise ValueError("Nonfinite or nonpositive expected counts on scoring support")
                policy = sample.metadata["alignment"]["resampling"]["policy"]
                policies.add(policy)
                key = f"{view.split}_{index:06d}"
                arrays[key] = dict(observed=sample.neural[:length, :n], log_expected=z, expected=expected,
                    time=sample.physical_timestamps[:length]-sample.stim_on,
                    physical_timestamps=sample.physical_timestamps[:length],
                    bin_start_times=sample.bin_start_times[:length], bin_end_times=sample.bin_end_times[:length],
                    temporal_positions=sample.temporal_positions[:length], temporal_mask=sample.temporal_mask[:length],
                    neuron_mask=sample.neuron_mask[:n], prediction_mask=prediction_mask, scoring_mask=scoring)
                np.savez_compressed(destination / "predictions" / f"{key}.npz", **arrays[key])
                for c, region in enumerate(dataset["regional_targets"][sample.session_id]["regions"]):
                    records.append(dict(split=view.split, session_id=sample.session_id, sample_id=sample.sample_id,
                        trial_id=sample.trial_id, region=region, column=c, array_key=key,
                        bin_size=sample.bin_size, alignment_policy=policy))
                print(f"{view.split}: {index+1}/{len(view)} trial {sample.trial_id}", flush=True)
    statistics = {}
    for row in records:
        if row["split"] != "train":
            continue
        statistics.setdefault(row["session_id"], {}).setdefault(row["region"], []).append(row)
    for session, regions in statistics.items():
        for region, rows in regions.items():
            ys, ps = [], []
            for r in rows:
                a, c = arrays[r["array_key"]], r["column"]
                mask = a["scoring_mask"][:, c]
                ys.append(a["observed"][mask, c]); ps.append(a["expected"][mask, c])
            y, p = np.concatenate(ys), np.concatenate(ps)
            if not len(y):
                raise ValueError("No eligible training bins for threshold fitting")
            regions[region] = dict(q95=float(np.quantile(y, .95, method="linear")), mean=float(y.mean()),
                calibration_edges=np.unique(np.quantile(p, np.arange(.1, 1, .1))).tolist(),
                eligible_bins=len(y), spike_sum=int(y.sum()))
    # All training-derived quantities are now fixed before any validation scoring.
    for row in records:
        stats = statistics.get(row["session_id"], {}).get(row["region"])
        if stats is None:
            raise ValueError("Validation session/region has no original training baseline")
        a, c = arrays[row["array_key"]], row["column"]
        row["scales"] = {}
        for width in WIDTHS:
            y, p, indices = _smooth(a["observed"][:, c], a["expected"][:, c], a["scoring_mask"][:, c], width)
            row["scales"][str(width)] = _metrics(y, p, stats["q95"] if width == 1 else None,
                stats["mean"], stats["calibration_edges"] if width == 1 else None,
                a["log_expected"][indices, c] if width == 1 else None)
    summaries = _summaries(records, arrays, statistics)
    if "regional_training_baseline" in dataset:
        for session, regions in statistics.items():
            names = dataset["regional_targets"][session]["regions"]
            saved = dataset["regional_training_baseline"][session]
            if ([regions[r]["mean"] for r in names] != saved["mean_counts"]
                    or any(regions[r]["eligible_bins"] != saved["eligible_bins"] for r in names)
                    or [regions[r]["spike_sum"] for r in names] != saved["spike_sums"]):
                raise ValueError("Reconstructed training baseline differs from checkpoint")
    label = ("full-trial" if context["mode"] == "full_trial" else f"strict H={context['bins']}") + " / " + (
        "held-state" if all("held-state" in p for p in policies) else "; ".join(sorted(policies)))
    result = dict(schema_version=1, complete=True, historical_semantics=label,
        checkpoint=dict(path=str(checkpoint), sha256=digest, epoch=state["epoch"], run=state["run"]),
        dataset_generation_id=dataset["dataset_generation_id"], dataset_path=str(metadata["dataset_path"]),
        context=context, alignment_policies=sorted(policies), memberships=memberships,
        regional_targets=dataset["regional_targets"], training_statistics=statistics,
        diagnostic_configuration=dict(device=str(device), torch_version=torch.__version__, numpy_version=np.__version__,
            inference="eval / inference_mode / visual-only / float32 / one ordered trial per call",
            smoothing="trailing complete eligible windows; means in counts per original bin",
            smoothing_bins=WIDTHS, q95="bin-weighted eligible training counts; linear quantile; y>=q95 and p>=q95",
            ranking="per-trial ceil(0.05*n); descending stable sort, earlier eligible bin wins ties",
            variance="population denominator; expected-count variance need not match observed variance",
            aggregation="trial/region first; equal trials, equal regions per session, equal sessions; pooled separately",
            calibration="training-prediction decile edges, deduplicated; tails open; bias=p-y; ratio=mean(p)/mean(y)",
            poisson="mean exp(z)-y*z, full=False, unit weight; smoothed objective is descriptive only",
            unavailable="null for zero denominators or constant correlations; empty support explicitly recorded",
            interpretation="descriptive comparison; no noise ceiling, practical cutoff, lag fitting or model selection",
            test_access="handoff integrity-verifies complete generation; test samples discarded before runtime processing"),
        records=records, splits=summaries)
    if _hash(checkpoint) != digest:
        raise ValueError("Source checkpoint changed during diagnostic")
    result["source_hashes"] = {str(p.relative_to(Path(__file__).parents[2])): _hash(p) for p in
        (Path(__file__), Path(__file__).with_name("temporal_report.html"),
         Path(__file__).parents[2] / "script/audit/temporal_diagnostics.py")}
    result["payload_hashes"] = {str(p.relative_to(destination)): _hash(p) for p in (destination / "predictions").glob("*.npz")}
    _write_report(destination, result, arrays)
    result["payload_hashes"].update({name: _hash(destination / name) for name in ("report.html", "report.md")})
    atomic_json(destination / "results.json", result)
    return result


def _write_report(destination, result, arrays):
    lines = ["# Frozen-checkpoint temporal prediction diagnostic", "", result["historical_semantics"], "",
        "Deterministic train/validation inference; selected checkpoint epoch " + str(result["checkpoint"]["epoch"]) + " (zero based).",
        "Validation selected this checkpoint and is not an unbiased final test. No test predictions or training updates.", "",
        "| Split / session / region | Width (bins) | Eligible windows | Mean y / p | Pooled variance ratio | Trial-centered variance ratio | Equal-trial Pearson | Trial-mean Pearson | Poisson objective | Training-null BPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "unavailable" if x is None else f"{x:.6g}"
    for split, summary in result["splits"].items():
        for group in summary["groups"]:
            for width, scale in group["scales"].items():
                p = scale["pooled"]
                lines.append(f"| {split} / {group['session_id']} / {group['region']} | {width} | {p['bins']} | "
                    f"{fmt(p.get('observed_mean'))} / {fmt(p.get('predicted_mean'))} | {fmt(p.get('variance_ratio'))} | "
                    f"{fmt(p.get('trial_centered_variance_ratio'))} | {fmt(scale['equal_trial']['metrics'].get('pearson'))} | "
                    f"{fmt(scale['trial_mean_pearson'])} | {fmt(p.get('poisson_objective'))} | {fmt(p.get('training_null_bps'))} |")
    interpretation = []
    for train in result["splits"]["train"]["groups"]:
        val = next((g for g in result["splits"]["val"]["groups"] if
                    (g["session_id"], g["region"]) == (train["session_id"], train["region"])), None)
        if val is None:
            continue
        t, v = train["scales"]["1"], val["scales"]["1"]
        t12, v12 = train["scales"]["12"], val["scales"]["12"]
        tc, vc = t["equal_trial"]["metrics"].get("pearson"), v["equal_trial"]["metrics"].get("pearson")
        interpretation += [f"{train['session_id']} / {train['region']}: training within-trial Pearson averages {fmt(tc)} "
            f"(median {fmt(t['within_trial_pearson_median'])}); validation averages {fmt(vc)} "
            f"(median {fmt(v['within_trial_pearson_median'])}). The validation-minus-training change is "
            f"{fmt(vc-tc if tc is not None and vc is not None else None)} correlation units. "
            f"Trial-mean correlations are {fmt(t['trial_mean_pearson'])} / {fmt(v['trial_mean_pearson'])} (train / validation).",
            f"Pooled predicted/observed variance ratios are {fmt(t['pooled'].get('variance_ratio'))} / "
            f"{fmt(v['pooled'].get('variance_ratio'))}; trial-centered ratios are "
            f"{fmt(t['pooled'].get('trial_centered_variance_ratio'))} / {fmt(v['pooled'].get('trial_centered_variance_ratio'))}. "
            "Compare both with the correlations rather than treating noisy observed variance as a target.",
            f"At 200 ms the equal-trial correlations are {fmt(t12['equal_trial']['metrics'].get('pearson'))} / "
            f"{fmt(v12['equal_trial']['metrics'].get('pearson'))}; the trial-centered bin-weighted correlations are "
            f"{fmt(t12['pooled'].get('trial_centered_pearson'))} / {fmt(v12['pooled'].get('trial_centered_pearson'))}. "
            "A smoothing gain confined to equal-trial summaries does not establish tracking on the longer trials or reproducible slower stimulus responses."]
    result["comparison_notes"] = interpretation
    lines += ["", "## Split interpretation", "", *[line+"\n" for line in interpretation],
        "Widths 3, 6 and 12 are diagnostic trailing means (50, 100, 200 ms at 60 Hz), using only complete eligible windows. Original counts and official evaluation are unchanged.", "",
        "Inspect the split comparison before choosing representation/learnability versus generalization/regularization work. Small training temporal correlations and compressed training variance indicate weak fitted temporal structure; a train–validation gap additionally indicates loss of that structure on validation. Smoothing can remove noise and does not demonstrate reproducible stimulus modulation by itself.", "",
        "Expected counts exclude unpredictable count variability; variance ratio one and recall of every stochastic peak are not required targets. No matched-condition reliability or causal claim is established. Full-trial attention is noncausal; historical held-state alignment is preserved, not corrected."]
    (destination / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    payload = dict(result=result, arrays={key: {k: a[k].tolist() for k in
        ("time", "observed", "expected", "scoring_mask")} for key, a in arrays.items()})
    template = Path(__file__).with_name("temporal_report.html").read_text(encoding="utf-8")
    encoded = json.dumps(plain(payload), allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    (destination / "report.html").write_text(template.replace("__REPORT_DATA__", encoded), encoding="utf-8")
