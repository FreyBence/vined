"""Compare all four completed models on identical saved parent-region targets."""
import csv
from pathlib import Path
import numpy as np
from common import (parser, settings, regional_sources, completed, read, fingerprint,
                    stage, atomic_json, file_hash, path, load_helper, name)

analysis = load_helper("analysis.py", "target_delay_comparison_analysis")


def predictions(attempt):
    rows = []
    for entry in read(attempt / "predictions/index.json"):
        with np.load(attempt / "predictions" / entry["file"], allow_pickle=False) as data:
            rows.append(dict(entry, **{k: data[k] for k in data.files}))
    return rows


def load_artifacts(exp, out):
    sources = regional_sources(exp)
    artifacts = []
    for label in ("region_00", "region_06", "subregion_00", "subregion_06"):
        directory = Path(sources[label]["directory"]) if label in sources else out / "runs" / label
        training, train_attempt = completed(directory, "train")
        manifest, attempt = completed(directory, "evaluate")
        resolved = read(directory / "resolved.json")
        trained = read(train_attempt / "result.json")
        support = read(attempt / "support.json")
        metrics = read(attempt / "metrics.json")
        expected_support = sources["region_00"]["support"]
        if (fingerprint(resolved) != training["identity"] or metrics["training_identity"] != training["identity"] or
                metrics["checkpoint_sha256"] != trained["checkpoint_sha256"] or
                file_hash(Path(trained["checkpoint"])) != trained["checkpoint_sha256"] or
                metrics["compatibility_id"] != expected_support["compatibility_id"] or support != expected_support):
            raise ValueError(f"{label}: checkpoint/training/evaluation/support association differs")
        if label.startswith("subregion"):
            if (read(attempt / "native-support.json") != resolved["support"] or
                    resolved["support"]["regional_sources"] != sources):
                raise ValueError(f"{label}: original regional sources changed")
            from common import verify_configuration
            from utils.config_utils import DictConfig
            verify_configuration(DictConfig(resolved["configuration"]), exp, resolved["run"])
        if metrics["run"]["target_delay_bins"] != int(label[-2:]):
            raise ValueError("Configuration label/delay differs")
        rows = predictions(attempt)
        artifacts.append(dict(run_id=label, attempt=attempt, metrics=metrics, rows=rows,
            sources=dict(run_directory=str(directory), resolved=str(directory / "resolved.json"),
                resolved_sha256=file_hash(directory / "resolved.json"), training_attempt=str(train_attempt),
                checkpoint=trained["checkpoint"], checkpoint_sha256=trained["checkpoint_sha256"],
                evaluation_attempt=str(attempt), evaluation_manifest=manifest,
                training_identity=training["identity"], execution_sources=metrics.get("execution_sources", resolved["sources"]))))
    base = artifacts[0]
    for artifact in artifacts[1:]:
        left = artifact["metrics"]["evaluation_configuration"].copy()
        right = base["metrics"]["evaluation_configuration"].copy()
        for value in (left, right):
            value.pop("device", None); value.pop("batch_size", None)
        if (left != right or artifact["metrics"]["definitions"] != base["metrics"]["definitions"] or
                len(artifact["rows"]) != len(base["rows"])):
            raise ValueError("Evaluation metric definitions or trial counts differ")
        for a, b in zip(artifact["rows"], base["rows"]):
            if any(a[k] != b[k] for k in ("sample_id", "session_id", "trial_id")):
                raise ValueError("Saved prediction trial identity/order differs")
            for field in ("observed", "baseline", "target_indices", "curve_endpoints", "physical_timestamps", "bin_start_times", "bin_end_times"):
                if not np.array_equal(a[field], b[field]):
                    raise ValueError(f"Saved {field} differs for {a['sample_id']}")
    return artifacts


def matrix(artifacts):
    result, regions = [], []
    for artifact in artifacts:
        m = artifact["metrics"]
        for w, values in m["windows"].items():
            row = dict(run_id=artifact["run_id"], delay_bins=m["run"]["target_delay_bins"], window_bins=int(w),
                poisson_nll=m["single_bin"]["poisson_nll"], baseline_poisson_nll=m["single_bin"]["baseline_poisson_nll"],
                nll_improvement=m["single_bin"]["nll_improvement"], bits_per_spike=m["single_bin"]["bits_per_spike"])
            for key in ("correlation", "variance_ratio", "mse_improvement", "mae", "rmse", "mean_error", "observed_std", "predicted_std"):
                row[key] = values["neurons"][key]["mean"]
            row["unavailable_correlations"] = values["neurons"]["unavailable_correlations"]
            row["fraction_improved_mse"] = values["neurons"]["fraction_improved_mse"]
            result.append(row)
            for region, metric in m["regions"].items():
                curves = metric["windows"][w]
                regions.append(dict(run_id=artifact["run_id"], region=region, window_bins=int(w),
                    poisson_nll=metric["single_bin"]["poisson_nll"], nll_improvement=metric["single_bin"]["nll_improvement"],
                    correlation=curves["correlation"]["mean"], variance_ratio=curves["variance_ratio"]["mean"],
                    mse_improvement=curves["mse_improvement"]["mean"]))
    return result, regions


def paired(artifacts):
    by_id = {a["run_id"]: a["metrics"] for a in artifacts}
    pairs = [("A", "region_00", "subregion_00"), ("B", "region_06", "subregion_06"),
             ("C", "subregion_00", "subregion_06"), ("D-region-delay", "region_00", "region_06"),
             ("D-cross-0-6", "region_00", "subregion_06"), ("D-cross-6-0", "region_06", "subregion_00")]
    result = []
    for label, left, right in pairs:
        rows = []
        for a, b in zip(by_id[left]["trials"], by_id[right]["trials"]):
            if a["sample_id"] != b["sample_id"]:
                raise ValueError("Paired trial identities differ")
            windows = {}
            for w in a["windows"]:
                x, y = a["windows"][w]["population"], b["windows"][w]["population"]
                xc, yc = x["correlation"]["value"], y["correlation"]["value"]
                windows[w] = dict(mse_gain=x["mse"]-y["mse"],
                    correlation_gain=None if xc is None or yc is None else yc-xc,
                    variance_ratio_change=None if x["variance_ratio"] is None or y["variance_ratio"] is None else y["variance_ratio"]-x["variance_ratio"])
            rows.append(dict(sample_id=a["sample_id"], trial_id=a["trial_id"],
                nll_gain=a["single_bin"]["poisson_nll"]-b["single_bin"]["poisson_nll"], windows=windows))
        result.append(dict(comparison=label, left=left, right=right, positive_gain_favors=right, trials=rows,
            nll_gain=analysis.distribution([r["nll_gain"] for r in rows]),
            fraction_trials_better_nll=float(np.mean([r["nll_gain"] > 0 for r in rows])),
            windows={w: {key: analysis.distribution([r["windows"][w][key] for r in rows])
                         for key in ("mse_gain", "correlation_gain", "variance_ratio_change")} for w in ("1", "3", "6", "12")}))
    return result


def plots(directory, artifacts, rows, regional, pairs, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    for keys, filename in [(("correlation", "variance_ratio", "mse_improvement"), "temporal_metrics"),
                           (("poisson_nll", "baseline_poisson_nll", "nll_improvement"), "likelihood_baseline")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key in zip(axes, keys):
            for a in artifacts:
                selected = [r for r in rows if r["run_id"] == a["run_id"]]
                ax.plot([r["window_bins"] for r in selected], [r[key] for r in selected], "o-", label=a["run_id"])
            ax.set(xlabel="W (bins); NLL always W=1", title=key)
        axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(directory / (filename+".png"), dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, pair in zip(axes.flat, pairs):
        ax.hist([r["nll_gain"] for r in pair["trials"]], bins=25)
        ax.axvline(0, color="black"); ax.set(title=pair["comparison"], xlabel=f"NLL gain: {pair['right']} over {pair['left']}")
    fig.tight_layout(); fig.savefig(directory / "paired_trials.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row_axes, pair in zip(axes, pairs[:3]):
        for ax, key in zip(row_axes, ("mse_gain", "correlation_gain")):
            for w in ("1", "3", "6", "12"):
                values = [r["windows"][w][key] for r in pair["trials"] if r["windows"][w][key] is not None]
                ax.hist(values, bins=20, histtype="step", label=f"W={w}")
            ax.axvline(0, color="black"); ax.set(title=f"Comparison {pair['comparison']}: {key}", xlabel=f"Positive favors {pair['right']}")
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(directory / "paired_temporal_differences.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    regions = sorted({r["region"] for r in regional})
    for a in artifacts:
        ax.plot(regions, [next(r["nll_improvement"] for r in regional if r["run_id"] == a["run_id"] and r["region"] == region) for region in regions], "o-", label=a["run_id"])
    ax.axhline(0, color="gray"); ax.set(ylabel="Single-bin NLL gain over training mean", title="All recorded parent regions")
    ax.legend(); fig.tight_layout(); fig.savefig(directory / "region_improvements.png", dpi=130); plt.close(fig)
    selected = np.linspace(0, len(artifacts[0]["rows"])-1, min(count, len(artifacts[0]["rows"])), dtype=int)
    names_by_session = read(artifacts[0]["attempt"] / "support.json")["compatibility"]["regional_targets"]
    for i in selected:
        base = artifacts[0]["rows"][i]
        names = names_by_session[base["session_id"]]["regions"]
        # Identical full retained interval in every panel, every region, every run.
        for j, region in enumerate(names):
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            for ax, w in zip(axes, (1, 3, 6, 12)):
                ep = base["target_indices"] if w == 1 else base["curve_endpoints"]
                ax.plot(base["time"][ep], analysis.rolling(base["observed"][:, j], w, ep)*60, color="black", label="Observed")
                ax.axhline(base["baseline"][j]*60, ls="--", color="gray", label="Training mean")
                for a in artifacts:
                    row = a["rows"][i]
                    ax.plot(row["time"][ep], analysis.rolling(row["expected"][:, j], w, ep)*60, label=a["run_id"], alpha=.8)
                ax.set(title=f"{region}, trial {base['trial_id']}, W={w} ({'raw bins' if w == 1 else 'trailing mean'})",
                    xlabel="Seconds from stimulus onset", ylabel="spikes/s/region")
            axes[0].legend(ncol=3, fontsize=7); fig.tight_layout()
            fig.savefig(directory / f"overlay_trial_{i:06d}_{name(region)}.png", dpi=130); plt.close(fig)


def main():
    p = parser(__doc__)
    p.add_argument("--comparison-config")
    p.add_argument("--skip-completed", action="store_true")
    args = p.parse_args()
    exp, _evaluation, out = settings(args)
    cfg = read(path(args.comparison_config or exp["comparison_config"]))
    if cfg["run_ids"] != ["region_00", "region_06", "subregion_00", "subregion_06"] or cfg["reference_run"] != "region_00":
        raise ValueError("All four fixed configurations are required")
    artifacts = load_artifacts(exp, out)
    identity = fingerprint(dict(config=cfg, inputs=[a["sources"] for a in artifacts]))
    with stage(out / "comparison", "compare", identity, skip=args.skip_completed, allow_new_inputs=True) as attempt:
        if attempt is None:
            return
        rows, regional = matrix(artifacts)
        pairs = paired(artifacts)
        report = dict(schema_version=1, sources={a["run_id"]: a["sources"] for a in artifacts},
            matrix=rows, regions=regional, paired_trial_comparisons=pairs,
            evaluations={a["run_id"]: a["metrics"] for a in artifacts},
            native_subregions={a["run_id"]: read(a["attempt"] / "subregion-diagnostics.json") for a in artifacts if a["run_id"].startswith("subregion")},
            interpretation="Exploratory single-seed comparison on previously inspected test observations; both delays retained. No reliable interaction, causal visual response or independent test confirmation is established.")
        atomic_json(attempt / "comparison.json", report)
        for filename, data in (("matrix.csv", rows), ("regions.csv", regional)):
            with (attempt / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
        plots(attempt / "plots", artifacts, rows, regional, pairs, cfg["representative_trials"])
        lines = ["# Subregion aggregation: exploratory 2×2 comparison", "",
            "All models use the exact parent-region counts and original trial/bin support from Experiment 1 (first target bin 23). NLL is the original single-bin Poisson objective without factorial terms; W changes offline curves only. Native training losses across granularities are not compared.", "",
            "| Run | W | NLL | Gain over constant mean | Curve correlation | Variance ratio |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            lines.append(f"| {row['run_id']} | {row['window_bins']} | {row['poisson_nll']:.6g} | {row['nll_improvement']:.6g} | {row['correlation']} | {row['variance_ratio']} |")
        for pair in pairs[:3]:
            lines += ["", f"Comparison {pair['comparison']}: {pair['left']} versus {pair['right']}. "
                f"Mean paired single-bin NLL gain favoring {pair['right']}: {pair['nll_gain']['mean']:.6g}; "
                f"fraction of trials with positive gain: {pair['fraction_trials_better_nll']:.3f}. "
                f"W=12 population correlation gain: {pair['windows']['12']['correlation_gain']['mean']}. "
                "Compare the separate likelihood and temporal metrics; an improvement in one does not imply improvement in the other."]
            right = next(a for a in artifacts if a["run_id"] == pair["right"])["metrics"]
            left = next(a for a in artifacts if a["run_id"] == pair["left"])["metrics"]
            for w in ("1", "3", "6", "12"):
                x, y = left["windows"][w]["neurons"], right["windows"][w]["neurons"]
                lines.append(f"W={w}: mean trial-region correlation {x['correlation']['mean']} → {y['correlation']['mean']}; "
                    f"variance ratio {x['variance_ratio']['mean']} → {y['variance_ratio']['mean']}; "
                    f"fraction of trial-region curves beating the constant mean in MSE {x['fraction_improved_mse']} → {y['fraction_improved_mse']}. "
                    f"Unavailable correlations: {x['unavailable_correlations']} → {y['unavailable_correlations']}.")
        lines += ["", "Comparison D includes all four models in the matrix, every parent region in regions.csv and all six paired contrasts in comparison.json.", "",
            "The representative trials are uniformly spaced in the same persisted order. W=1 plots show raw single-bin signals over the complete eligible interval; smoothed curves are labeled. Undefined correlations remain unavailable. Small and low-activity groups remain included.", "",
            "Both D=0 and D=6 remain candidates. Small numerical differences do not establish superiority or a granularity×delay interaction from one seed. Checkpoint selection uses validation loss. These test observations were previously inspected and provide exploratory evidence only. The IBL closed-loop visual/motor confound precludes a purely visual causal interpretation.", "",
            "For Experiment 3, native diagnostics describe activity, sparsity, neuron counts, variance and trial consistency. They do not authorize test-derived selection rules. Fit future information-based selection on training data and use validation for exploratory choices; use independent observations for confirmation.", "", "Exact source artifacts:"]
        for a in artifacts:
            lines += ["", f"- {a['run_id']}: `{a['sources']['checkpoint']}` (SHA-256 `{a['sources']['checkpoint_sha256']}`); "
                f"resolved `{a['sources']['resolved']}`; training `{a['sources']['training_attempt']}`; evaluation `{a['sources']['evaluation_attempt']}`."]
        (attempt / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
