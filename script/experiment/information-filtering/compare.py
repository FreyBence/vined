"""Compare all six configurations on identical full-region validation observations."""
import csv
from itertools import combinations
from pathlib import Path
import numpy as np
from common import (parser, settings, read, path, completed, stage, atomic_json,
                    fingerprint, file_hash, source_identity, analysis, module, selection_artifact,
                    unfiltered_sources)

legacy = module("subregion-aggregation", "compare.py", "information_comparison_helpers")


def paired(reference, candidate):
    left, right = reference["metrics"], candidate["metrics"]
    trials = []
    for a,b in zip(left["trials"],right["trials"]):
        x,y = a["single_bin"], b["single_bin"]
        if a["sample_id"] != b["sample_id"] or x["cells"] != y["cells"]:
            raise ValueError("Paired trial coverage differs")
        windows = {}
        for w in a["windows"]:
            u,v = a["windows"][w]["population"],b["windows"][w]["population"]
            uc,vc = u["correlation"]["value"],v["correlation"]["value"]
            windows[w] = dict(mse_gain=u["mse"]-v["mse"],
                correlation_gain=None if uc is None or vc is None else vc-uc,
                variance_ratio_change=None if u["variance_ratio"] is None or v["variance_ratio"] is None else v["variance_ratio"]-u["variance_ratio"])
        trials.append(dict(sample_id=a["sample_id"],trial_id=a["trial_id"],cells=x["cells"],
            nll_gain=x["poisson_nll"]-y["poisson_nll"], nll_gain_sum=x["nll_sum"]-y["nll_sum"],windows=windows))
    elements = sum(t["cells"] for t in trials)
    gains = np.asarray([t["nll_gain"] for t in trials])
    global_gain = sum(t["nll_gain_sum"] for t in trials)/elements
    if not np.isclose(global_gain, left["single_bin"]["poisson_nll"]-right["single_bin"]["poisson_nll"], atol=1e-12):
        raise ValueError("Global paired NLL weighting is inconsistent")
    return dict(comparison=candidate["run_id"]+" vs "+reference["run_id"], left=reference["run_id"],right=candidate["run_id"],
        reference=reference["run_id"], candidate=candidate["run_id"], positive_gain_favors=candidate["run_id"],
        gain_definition="reference NLL - candidate NLL", global_cell_weighted_paired_gain=global_gain,
        global_reference_nll=left["single_bin"]["poisson_nll"],global_candidate_nll=right["single_bin"]["poisson_nll"],
        equal_trial_mean_gain=float(gains.mean()), median_trial_gain=float(np.median(gains)),
        fraction_trials_improved=float(np.mean(gains>0)), eligible_region_bin_elements=elements,
        nll_gain=analysis.distribution(gains), fraction_trials_better_nll=float(np.mean(gains>0)),trials=trials,
        windows={w:{k:analysis.distribution([t["windows"][w][k] for t in trials])
            for k in ("mse_gain","correlation_gain","variance_ratio_change")} for w in ("1","3","6","12")})


def artifacts(exp, ev, out, run_ids):
    result = []
    sources = unfiltered_sources(exp)
    selection, _, selection_manifest = selection_artifact(exp)
    for run_id in run_ids:
        directory = out / "runs" / run_id
        if not (directory / "evaluate.json").is_file():
            raise ValueError(f"{run_id}: completed validation evaluation is missing; explicitly train any missing filtered model, then run evaluate. No training is launched by comparison.")
        manifest, attempt = completed(directory,"evaluate")
        metrics = read(attempt / "metrics.json")
        support = read(attempt / "support.json")
        native_support = read(attempt / "native-support.json")
        training_dir = Path(sources[run_id]["directory"]) if run_id in sources else directory
        training, train_attempt = completed(training_dir,"train")
        trained = read(train_attempt / "result.json")
        resolved = read(training_dir / "resolved.json")
        if (fingerprint(resolved) != training["identity"] or metrics["training_identity"] != training["identity"] or
                metrics["checkpoint_sha256"] != trained["checkpoint_sha256"] or
                file_hash(path(trained["checkpoint"])) != trained["checkpoint_sha256"] or
                metrics["run_id"] != run_id or metrics["evaluation_configuration"]["split"] != "val" or
                metrics["compatibility_id"] != fingerprint(support["compatibility"])):
            raise ValueError(f"{run_id}: checkpoint/evaluation/support association differs")
        if run_id not in sources and (native_support["selection_identity"] != selection_manifest["identity"]
                or native_support["selection"] != selection):
            raise ValueError("Compared selection differs from active selection")
        current = metrics["evaluation_configuration"].copy()
        expected = ev.copy()
        for obj in (current,expected):
            obj.pop("device",None);obj.pop("batch_size",None)
        if current != expected:
            raise ValueError("Evaluation definitions/configuration changed")
        result.append(dict(run_id=run_id, attempt=attempt, metrics=metrics,
            rows=legacy.predictions(attempt), native=read(attempt / "native-metrics.json"),
            decomposition=read(attempt / "decomposition.json"), support=support,
            sources=dict(training_directory=str(training_dir),training_identity=training["identity"],
                resolved_sha256=file_hash(training_dir / "resolved.json"),checkpoint=trained["checkpoint"],
                checkpoint_sha256=trained["checkpoint_sha256"], evaluation_manifest=manifest)))
    base=result[0]
    for artifact in result[1:]:
        if artifact["support"] != base["support"] or len(artifact["rows"]) != len(base["rows"]):
            raise ValueError("Full-region support differs")
        for a,b in zip(artifact["rows"],base["rows"]):
            if any(a[k] != b[k] for k in ("sample_id","session_id","trial_id")):
                raise ValueError("Paired sample identities differ")
            for field in ("observed","baseline","target_indices","curve_endpoints","physical_timestamps","bin_start_times","bin_end_times"):
                if not np.array_equal(a[field],b[field]):
                    raise ValueError(f"Common {field} differs: {a['sample_id']}")
    return result


def write_csv(filename, rows):
    with filename.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(attempt, artifacts, pairs):
    lines=["# Experiment 3 — validation comparison", "",
        "All primary metrics use identical complete original regional targets. Gain = reference NLL − candidate NLL; positive favors the candidate. NLL omits log-factorial, uses original bins, and weights eligible region-bin cells. Equal-trial gain gives every trial one vote.","",
        "| Reference | Candidate | Global paired gain | Equal-trial mean | Median trial | Fraction improved | Elements |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    primary=[p for p in pairs if p["reference"].startswith("unfiltered") and p["candidate"][:-3] != "unfiltered" and p["reference"][-2:]==p["candidate"][-2:]]
    for p in primary:
        lines.append(f"| {p['reference']} | {p['candidate']} | {p['global_cell_weighted_paired_gain']:.7g} | {p['equal_trial_mean_gain']:.7g} | {p['median_trial_gain']:.7g} | {p['fraction_trials_improved']:.3f} | {p['eligible_region_bin_elements']} |")
    lines += ["", "## Interpretation", "",
        "Native NLL and baseline gains are target-dependent diagnostics and cannot establish superiority across populations. Omitted components receive training constants; decomposition.json quantifies the constant contribution. This construction cannot recover omitted activity.","",
        "Single-bin likelihood and temporal-curve similarity are distinct. Higher smoothed correlation is not evidence of better single-bin prediction. Window metrics report undefined correlations and trial spread; overlapping windows are dependent. All anatomical groups remain in global evaluation, including difficult nonvisual groups.","",
        "## Native selected-population structure", "",
        "These target-dependent diagnostics characterize each population separately; their NLLs cannot rank different targets.", "",
        "| Run | Native gain over its training constant | W=1 correlation | W=12 correlation | W=12 variance ratio |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for artifact in artifacts:
        native=artifact["native"]
        one=native["windows"]["1"]["neurons"]
        twelve=native["windows"]["12"]["neurons"]
        lines.append(f"| {artifact['run_id']} | {native['single_bin']['nll_improvement']:.7g} | {one['correlation']['mean']} | {twelve['correlation']['mean']} | {twelve['variance_ratio']['mean']} |")
    lines += ["", "## Paired temporal behavior", ""]
    for pair in primary:
        for w in ("1","12"):
            summaries=pair["windows"][w]
            lines.append(f"- {pair['comparison']}, W={w}: equal-trial population correlation change "
                f"{summaries['correlation_gain']['mean']}; variance-ratio change {summaries['variance_ratio_change']['mean']}; "
                f"MSE gain {summaries['mse_gain']['mean']}. Undefined paired correlations: "
                f"{len(pair['trials'])-summaries['correlation_gain']['available']}. "
                "Inspect the paired distributions before interpreting increased prediction variance as useful temporal structure.")
    lines += ["", "## Experiment 4 candidates", ""]
    for strategy in ("neuron_filtered","subregion_filtered"):
        selected=[p for p in primary if p["candidate"].startswith(strategy)]
        consistent=all(p["global_cell_weighted_paired_gain"]>0 and p["equal_trial_mean_gain"]>0 and p["median_trial_gain"]>0 and p["fraction_trials_improved"]>.5 for p in selected)
        lines.append(f"- {strategy}: "+("candidate for Experiment 4 based on positive global, equal-trial and median gains at both delays; independent confirmation remains necessary." if consistent else "no consistent advantage across weighting criteria and both delays; retain the unfiltered reference as a candidate."))
    lines += ["", "These are descriptive model-development comparisons on validation trials also used for checkpoint selection. No significance or causal claim is made, and small numerical gains need practical scientific assessment. Previously inspected test results are exploratory and did not define the screen. Do not force a winner.","",
        "## Difficult recorded groups", ""]
    for artifact in artifacts:
        ordered=sorted(artifact["metrics"]["regions"].items(),key=lambda kv:kv[1]["single_bin"]["nll_improvement"])
        lines.append("- "+artifact["run_id"]+": "+", ".join(f"{r} (gain {m['single_bin']['nll_improvement']:.4g})" for r,m in ordered[:5]))
    lines += ["", "## Visual cortical subregions", "",
        "Recorded VIS* target diagnostics, including retained counts, NLL baseline gains, all W correlations and variance ratios, are in visual-subregions.json. Excluded targets remain explicit in the selection manifest; their missing learned predictions are not silently scored as native outputs.",""]
    (attempt / "report.md").write_text("\n".join(lines),encoding="utf-8")


def gain_plots(directory, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary=[p for p in pairs if p["reference"].startswith("unfiltered") and p["candidate"][:-3] != "unfiltered" and p["reference"][-2:]==p["candidate"][-2:]]
    x=np.arange(len(primary));fig,ax=plt.subplots(figsize=(12,5))
    ax.bar(x-.18,[p["global_cell_weighted_paired_gain"] for p in primary],.36,label="Global cell weighted")
    ax.bar(x+.18,[p["equal_trial_mean_gain"] for p in primary],.36,label="Equal trial weighted")
    ax.set_xticks(x,[p["candidate"] for p in primary],rotation=15)
    ax.axhline(0,color="gray");ax.set(ylabel="Reference − candidate NLL",title="Reference: matched-delay unfiltered")
    ax.legend();fig.tight_layout();fig.savefig(directory / "weighted-gains.png",dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,pair in zip(axes.flat,primary):
        ax.hist([t["nll_gain"] for t in pair["trials"]],bins=25);ax.axvline(0,color="black")
        ax.set(title=pair["comparison"],xlabel="Trial NLL gain (positive favors candidate)")
    fig.tight_layout();fig.savefig(directory / "primary-paired-distributions.png",dpi=130);plt.close(fig)


def main():
    p=parser(__doc__);p.add_argument("--skip-completed",action="store_true");args=p.parse_args()
    exp,ev,out=settings(args);cfg=read(path(exp["comparison_config"]))
    required=[f"{s}_{d:02}" for s in ("unfiltered","neuron_filtered","subregion_filtered") for d in (0,6)]
    if cfg["run_ids"] != required or cfg["reference"] != "matched_delay_unfiltered" or cfg["gain"] != "reference_nll_minus_candidate_nll":
        raise ValueError("Require all six configurations and fixed gain convention")
    data=artifacts(exp,ev,out,cfg["run_ids"])
    identity=fingerprint(dict(config=cfg,sources=source_identity(),artifacts=[a["sources"] for a in data]))
    with stage(out / "comparison","compare",identity,skip=args.skip_completed,allow_new_inputs=True) as attempt:
        if attempt is None:return
        matrix,regions=legacy.matrix(data)
        by_id={a["run_id"]:a["metrics"] for a in data}
        for row in matrix:
            row["temporal_mse"] = by_id[row["run_id"]]["windows"][str(row["window_bins"])]["neurons"]["mse"]["mean"]
        pairs=[paired(a,b) for a,b in combinations(data,2)]
        write_csv(attempt / "metrics-matrix.csv",matrix);write_csv(attempt / "per-region.csv",regions)
        write_csv(attempt / "paired-summary.csv",[{k:v for k,v in p.items() if k not in ("trials","windows","nll_gain")} for p in pairs])
        atomic_json(attempt / "comparison.json",dict(configuration=cfg,paired=pairs,metrics_matrix=matrix,per_region=regions,
            training_constant_comparisons={a["run_id"]:a["metrics"]["baseline_audit"] for a in data},
            native_target_dependent={a["run_id"]:a["native"] for a in data},sources={a["run_id"]:a["sources"] for a in data}))
        atomic_json(attempt / "decomposition.json",{a["run_id"]:a["decomposition"] for a in data})
        atomic_json(attempt / "visual-subregions.json",{a["run_id"]:[g for g in read(a["attempt"] / "subregion-diagnostics.json") if g["label"].startswith("VIS")] for a in data})
        # Legacy plots use only the first six paired distributions; complete pairs remain in JSON/CSV.
        legacy.plots(attempt / "plots",data,matrix,regions,pairs[:6],cfg["representative_trials"])
        gain_plots(attempt / "plots",pairs)
        report(attempt,data,pairs)


if __name__ == "__main__":
    main()
