"""CPU-only train-trial cross-validation and fixed anatomical population screening."""
from math import ceil
import numpy as np
from common import (parser, settings, views, original_support, stage, atomic_json,
                    fingerprint, read, path, scoring_sources, plain)


def features(sample, projection, delay):
    """Last and H=12 mean projected visual token, causal at t-D."""
    z = sample.visual[:sample.sequence_length].astype(np.float64) @ projection
    targets = np.arange(23, sample.sequence_length)
    end = targets-delay
    cumulative = np.concatenate([np.zeros((1, z.shape[1])), z.cumsum(0)])
    return np.concatenate([z[end], (cumulative[end+1]-cumulative[end-11])/12], 1)


def rank(indices, scores):
    # Undefined scores rank last; original source-column/group ordering breaks ties.
    return sorted(indices, key=lambda j: (scores[j] is None,
                  -scores[j] if scores[j] is not None else 0, j))


def compute(exp, ev, policy):
    from trainer.targets import transform_neural_targets
    raw = views(exp)
    train = raw[0]
    *_, metadata = transform_neural_targets(raw[:3], raw[3], "subregion")
    if any(s.split != "train" for s in train.samples):
        raise ValueError("Scoring consumes training trials only")
    # Membership is grouped by original session/trial, independently of bin count.
    identities = sorted({(s.session_id, s.trial_id) for s in train.samples})
    if len(identities) < policy["folds"] or policy["folds"] < 2:
        raise ValueError("Insufficient independent training trials for grouped folds")
    shuffled = np.random.default_rng(policy["seed"]).permutation(len(identities))
    assignment = {identities[int(j)]: int(i % policy["folds"]) for i, j in enumerate(shuffled)}
    width = train.samples[0].visual.shape[1]
    projection = np.random.default_rng(policy["seed"]).normal(
        size=(width, policy["projection_dimensions"])) / np.sqrt(width)
    folds = [dict(fold=k, held_out=[dict(session_id=s, trial_id=t) for (s,t),f in assignment.items() if f == k],
        fit=[dict(session_id=s, trial_id=t) for (s,t),f in assignment.items() if f != k]) for k in range(policy["folds"])]
    results, selections, excluded_baselines = {}, {s: {} for s in ("neuron_filtered", "subregion_filtered")}, {s: {} for s in ("neuron_filtered", "subregion_filtered")}
    for session, mapping in metadata["regional_targets"].items():
        samples = [s for s in train.samples if s.session_id == session]
        columns = [c for group in mapping["source_columns"] for c in group]
        if sorted(columns) != list(range(samples[0].neuron_count)):
            raise ValueError("Expected full valid original neuron partition")
        n = samples[0].neuron_count
        raw_counts = np.concatenate([s.neural[23:s.sequence_length, :n] for s in samples]).astype(float)
        group_counts = np.stack([raw_counts[:, c].sum(1) for c in mapping["source_columns"]], -1)
        y = np.concatenate([raw_counts, group_counts], 1)
        f = np.concatenate([np.full(s.sequence_length-23, assignment[s.session_id, s.trial_id]) for s in samples])
        baseline_mean = raw_counts.mean(0)
        delays, fold_records = {}, []
        for d in policy["scoring_delays"]:
            x = np.concatenate([features(s, projection, d) for s in samples])
            gain_sums, model_sums, null_sums = np.zeros(y.shape[1]), np.zeros(y.shape[1]), np.zeros(y.shape[1])
            for k in range(policy["folds"]):
                fit, held = f != k, f == k
                if not fit.any() or not held.any():
                    raise ValueError("Every session needs fit and held-out trials in every fold")
                center, scale = x[fit].mean(0), x[fit].std(0)
                scale = np.maximum(scale, 1e-8)
                xf, xh = (x[fit]-center)/scale, (x[held]-center)/scale
                mean = y[fit].mean(0)
                weights = np.linalg.solve(xf.T@xf + policy["ridge_alpha"]*np.eye(xf.shape[1]),
                                          xf.T@(y[fit]-mean))
                expected = np.maximum(mean+xh@weights, policy["count_floor"])
                null = np.maximum(mean, policy["count_floor"])
                model_nll = (expected-y[held]*np.log(expected)).sum(0)
                baseline_nll = (null-y[held]*np.log(null)).sum(0)
                model_sums += model_nll; null_sums += baseline_nll
                gain_sums += baseline_nll-model_nll
                fold_records.append(dict(delay=d, fold=k, held_out_bins=int(held.sum()), fit_bins=int(fit.sum()),
                    feature_mean=center.tolist(), feature_scale=scale.tolist(), baseline_counts=mean.tolist(),
                    coefficients=weights.tolist(), model_nll_sum=model_nll.tolist(), baseline_nll_sum=baseline_nll.tolist()))
            delays[str(d)] = dict(gain_per_cell=(gain_sums/len(y)).tolist(),
                model_nll=(model_sums/len(y)).tolist(), baseline_nll=(null_sums/len(y)).tolist())
        gains = np.mean([d["gain_per_cell"] for d in delays.values()], 0)
        scores = [float(v) if y[:,j].sum() > 0 and np.isfinite(v) else None for j,v in enumerate(gains)]
        neuron_records, groups = [], []
        neuron_selection = {}
        for j, label in enumerate(mapping["regions"]):
            original = mapping["source_columns"][j]
            ordered = rank(original, scores)
            keep = max(1, ceil(len(original)*policy["neuron_fraction"]))
            neuron_selection[label] = sorted(ordered[:keep])
            for position, c in enumerate(ordered, 1):
                neuron_records.append(dict(source_column=c, identity=mapping["ordered_units"][c],
                    subregion=label, parent_region=mapping["parent_regions"][j], score=scores[c], rank=position,
                    selected=position <= keep, undefined_reason="no_training_spikes_or_nonfinite_score" if scores[c] is None else None,
                    mean_count=float(baseline_mean[c]), zero_fraction=float(np.mean(raw_counts[:,c] == 0))))
            groups.append(dict(label=label, parent_region=mapping["parent_regions"][j], score=scores[n+j],
                source_columns=original, neuron_count=len(original), selected_neuron_count=keep,
                neuron_fraction=keep/len(original), neuron_score_cutoff=scores[ordered[keep-1]],
                mean_count=float(group_counts[:,j].mean()), rate_hz=float(group_counts[:,j].mean()*60),
                zero_fraction=float(np.mean(group_counts[:,j] == 0)),
                undefined_reason="no_training_spikes_or_nonfinite_score" if scores[n+j] is None else None))
        ranked = rank(list(range(len(groups))), scores[n:])
        keep_groups = max(1, ceil(len(groups)*policy["subregion_fraction"]))
        selected_groups = set(ranked[:keep_groups])
        for position, j in enumerate(ranked, 1):
            groups[j].update(rank=position, selected_subregion=j in selected_groups)
        selections["neuron_filtered"][session] = neuron_selection
        selections["subregion_filtered"][session] = {g["label"]: g["source_columns"] for j,g in enumerate(groups) if j in selected_groups}
        for strategy, sessions in selections.items():
            included = {c for cs in sessions[session].values() for c in cs}
            excluded = sorted(set(columns)-included)
            by_parent = {p: float(sum(baseline_mean[c] for g in groups if g["parent_region"] == p
                                    for c in g["source_columns"] if c in excluded)) for p in set(mapping["parent_regions"])}
            # Additive constants must partition each full training mean exactly.
            for parent, constant in by_parent.items():
                all_cols = [c for g in groups if g["parent_region"] == parent for c in g["source_columns"]]
                if not np.isclose(constant+sum(baseline_mean[c] for c in all_cols if c in included), baseline_mean[all_cols].sum()):
                    raise ValueError("Excluded baseline partition differs")
            excluded_baselines[strategy][session] = dict(parent_mean_counts=by_parent,
                excluded_source_columns=excluded, selected_source_columns=sorted(included),
                excluded_unit_identities=[mapping["ordered_units"][c] for c in excluded],
                selected_unit_identities=[mapping["ordered_units"][c] for c in sorted(included)],
                excluded_neuron_mean_counts=[float(baseline_mean[c]) for c in excluded])
        results[session] = dict(mapping=mapping, neurons=neuron_records, subregions=groups,
            delays=delays, fold_fits=fold_records, training_neuron_mean_counts=baseline_mean.tolist(),
            training_bins=len(y), subregion_score_cutoff=scores[n+ranked[keep_groups-1]],
            retained_subregion_fraction=keep_groups/len(groups))
    return plain(dict(schema_version=1, configuration=policy, scoring_sources=scoring_sources(),
        dataset_generation_id=raw[3]["dataset_generation_id"], dataset_metadata=raw[3]["dataset_metadata"],
        train_trials=[dict(sample_id=s.sample_id, session_id=s.session_id, trial_id=s.trial_id,
            first_target_bin=23, eligible_bins=s.sequence_length-23) for s in train.samples],
        folds=folds, projection=projection.tolist(), projection_sha256=fingerprint(projection.tolist()),
        scores=results, selections=selections, excluded_baselines=excluded_baselines,
        interpretation="OOF mean Poisson NLL gain under fixed projected visual ridge-count readout; not mutual information or visual causality",
        limitations="Ridge minimizes squared count error, not Poisson loss; negative estimates are floored. Small readout can miss nonlinear tuning. Raw per-cell gain favors absolute count-scale improvements. Undefined/zero-activity scores rank last; deterministic column order breaks ties."))


def plots(directory, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for session, result in data["scores"].items():
        groups = sorted(result["subregions"], key=lambda g:g["rank"])
        fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(groups)*.2)))
        labels = [g["label"] for g in groups]
        axes[0].barh(labels, [g["score"] or 0 for g in groups])
        axes[0].set(xlabel="Training OOF NLL gain/cell (mean D=0/6)", title="Complete subregion ranking; undefined shown as zero")
        axes[1].barh(labels, [g["neuron_count"] for g in groups], label="Original")
        axes[1].barh(labels, [g["selected_neuron_count"] for g in groups], label="Neuron filtered")
        axes[1].legend(); axes[1].set(xlabel="Neurons", title="Fixed within-group screen")
        fig.tight_layout(); fig.savefig(directory / f"selection-{session}.png", dpi=130); plt.close(fig)


def main():
    p = parser(__doc__)
    p.add_argument("--skip-completed", action="store_true")
    args = p.parse_args()
    exp, ev, out = settings(args)
    policy = read(path(exp["filtering_config"]))
    if (policy["method"] != "projected_visual_ridge_count" or policy["scoring_delays"] != [0, 6]
            or policy["combination"] != "arithmetic_mean_oof_gain" or policy["seed"] != 42
            or not 0 < policy["neuron_fraction"] <= 1 or not 0 < policy["subregion_fraction"] <= 1
            or policy["ridge_alpha"] <= 0 or policy["count_floor"] <= 0
            or type(policy["projection_dimensions"]) is not int or policy["projection_dimensions"] < 1):
        raise ValueError("Unsupported scoring/selection policy")
    identity = fingerprint(dict(policy=policy, generation=exp["expected_dataset_generation_id"], sources=scoring_sources()))
    with stage(out / "selection", "score", identity, skip=args.skip_completed, allow_new_inputs=True) as attempt:
        if attempt is None:
            return
        # This independently verifies physical and temporal support before scoring.
        support = original_support(exp, ev, "subregion")
        data = compute(exp, ev, policy)
        data["support_compatibility_id"] = support["compatibility_id"]
        atomic_json(attempt / "selection.json", data)
        for strategy, mapping in data["selections"].items():
            atomic_json(attempt / (strategy+".json"), mapping)
        plots(attempt, data)
        atomic_json(attempt / "verification.json", dict(training_only_fit=True,
            grouped_folds=len(data["folds"]), selection_shared_between_delays=True,
            excluded_baseline_partition_verified=True, gpu_training_launched=False))


if __name__ == "__main__":
    main()
