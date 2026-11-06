"""Compare verified evaluation artifacts; never construct a model or run inference."""
import csv
import json
import numpy as np

from common import (parser, settings, read, path, name, completed, fingerprint,
                    stage, atomic_json, file_hash)
from analysis import distribution, rolling
from utils.progress import iter_progress


def main():
    p = parser(__doc__)
    p.add_argument("--comparison-config")
    p.add_argument("--run-ids", nargs="+")
    p.add_argument("--skip-completed",action="store_true")
    args = p.parse_args()
    exp,evaluation,out = settings(args)
    cfg = read(path(args.comparison_config or exp["comparison_config"]))
    ids = args.run_ids or cfg["run_ids"]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Select at least two unique completed runs")
    artifacts = []
    for run_id in iter_progress(ids, "comparison: loading evaluations", total=len(ids), unit="runs"):
        directory=out/"runs"/name(run_id)
        training,train_attempt = completed(directory,"train")
        manifest,attempt = completed(directory,"evaluate")
        metrics=read(attempt/"metrics.json")
        trained=read(train_attempt/"result.json")
        resolved=read(directory/"resolved.json")
        support=read(attempt/"support.json")
        if (fingerprint(resolved) != training["identity"] or
                metrics["training_identity"] != training["identity"] or
                metrics["checkpoint_sha256"] != trained["checkpoint_sha256"] or
                metrics["compatibility_id"] != support["compatibility_id"] or
                support != resolved["support"] or fingerprint(support["compatibility"]) != support["compatibility_id"]):
            raise ValueError("Run/evaluation/support manifest association differs")
        artifacts.append(dict(run_id=run_id,manifest=manifest,attempt=attempt,metrics=metrics,
                              resolved=resolved,support=support))
    artifacts.sort(key=lambda a:(a["metrics"]["run"]["target_delay_bins"],a["run_id"]))
    reference=next((a for a in artifacts if a["run_id"]==cfg["reference_run"]),None)
    if reference is None:
        raise ValueError("Comparison must include configured reference_run")
    base=reference["resolved"]
    def controlled_configuration(resolved):
        config=json.loads(json.dumps(resolved["configuration"]))
        config["training"]["temporal_context"].pop("target_delay_bins",None)
        return config
    for artifact in iter_progress(artifacts, "comparison: checking compatibility",
                                  total=len(artifacts), unit="runs"):
        resolved=artifact["resolved"]
        if (artifact["support"] != reference["support"] or resolved["experiment"] != exp or
                resolved["sources"] != base["sources"] or
                controlled_configuration(resolved) != controlled_configuration(base) or
                artifact["metrics"]["definitions"] != reference["metrics"]["definitions"]):
            raise ValueError("Incompatible dataset generation, splits, units, target bins, definitions or controlled training settings")
        left=artifact["metrics"]["evaluation_configuration"].copy()
        right=reference["metrics"]["evaluation_configuration"].copy()
        for key in ("device","batch_size"):
            left.pop(key,None); right.pop(key,None)
        if left != right:
            raise ValueError("Evaluation definitions differ")
    identity=fingerprint(dict(config=cfg,inputs={a["run_id"]:a["manifest"]["identity"] for a in artifacts}))
    with stage(out/"comparison","compare",identity,skip=args.skip_completed,allow_new_inputs=True) as attempt:
        if attempt is None:
            return
        matrix=[]
        paired=[]
        ref_trials={t["sample_id"]:t for t in reference["metrics"]["trials"]}
        for artifact in iter_progress(artifacts, "comparison: paired metrics",
                                      total=len(artifacts), unit="runs"):
            metrics=artifact["metrics"]; delay=metrics["run"]["target_delay_bins"]
            for w,record in metrics["windows"].items():
                matrix.append(dict(run_id=artifact["run_id"],delay_bins=delay,delay_ms=delay/60*1000,window_bins=int(w),
                    poisson_nll=metrics["single_bin"]["poisson_nll"],
                    nll_improvement=metrics["single_bin"]["nll_improvement"],
                    bits_per_spike=metrics["single_bin"]["bits_per_spike"],
                    mean_neuron_correlation=record["neurons"]["correlation"]["mean"],
                    mean_mse_improvement=record["neurons"]["mse_improvement"]["mean"],
                    fraction_improved_mse=record["neurons"]["fraction_improved_mse"],
                    mean_variance_ratio=record["neurons"]["variance_ratio"]["mean"],
                    unavailable_correlations=record["neurons"]["unavailable_correlations"]))
            differences=[]
            for trial in metrics["trials"]:
                ref=ref_trials[trial["sample_id"]]
                differences.append(dict(sample_id=trial["sample_id"],trial_id=trial["trial_id"],
                    session_id=trial["session_id"],
                    nll_improvement_over_reference=ref["single_bin"]["poisson_nll"]-trial["single_bin"]["poisson_nll"],
                    windows={w:dict(population_mse_improvement_over_reference=
                        ref["windows"][w]["population"]["mse"]-trial["windows"][w]["population"]["mse"])
                        for w in metrics["windows"]}))
            values=[d["nll_improvement_over_reference"] for d in differences]
            paired.append(dict(run_id=artifact["run_id"],trials=differences,nll_difference=distribution(values),
                fraction_trials_better_nll=float(np.mean(np.asarray(values)>0))))
        report=dict(schema_version=1,compatibility_id=reference["support"]["compatibility_id"],
            reference_run=reference["run_id"],matrix=matrix,paired_trial_comparisons=paired,
            evaluations={a["run_id"]:dict(evaluation_path=str(a["attempt"]),
                manifest=a["manifest"],single_bin=a["metrics"]["single_bin"],
                regions=a["metrics"]["regions"],neurons=a["metrics"]["neurons"],
                contrasts=a["metrics"]["contrasts"],windows=a["metrics"]["windows"])
                for a in artifacts})
        atomic_json(attempt/"comparison.json",report)
        with (attempt/"matrix.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(matrix[0])); writer.writeheader(); writer.writerows(matrix)
        make_plots(attempt,artifacts,matrix,cfg)
        ranked=sorted(artifacts,key=lambda a:a["metrics"]["single_bin"]["poisson_nll"])
        best=ranked[0]
        lines=["# Target-delay exploratory comparison","",
            "All runs use matched dataset generations, trial memberships, ordered target channels, physical target bins and training settings. "
            "Only D differs. NLL uses original single-bin counts; W changes only offline curves.","",
            "| Run | D (bins) | Single-bin NLL | Gain over training mean | Bits/spike |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for a in artifacts:
            m=a["metrics"]["single_bin"]
            lines.append(f"| {a['run_id']} | {a['metrics']['run']['target_delay_bins']} | {m['poisson_nll']:.6g} | {m['nll_improvement']:.6g} | {m['bits_per_spike']} |")
        consistency=next(p for p in paired if p["run_id"]==best["run_id"])
        lines += ["",f"The lowest pooled single-bin NLL in this grid is {best['run_id']} "
            f"(D={best['metrics']['run']['target_delay_bins']} bins). Its fraction of trials with lower NLL than "
            f"{reference['run_id']} is {consistency['fraction_trials_better_nll']:.3f}.","",
            "Treat this as a candidate delay range for follow-up only. Inspect neighboring delays in the full matrix, "
            "baseline-relative likelihood, paired trial spread, per-target-channel distributions, regional curves and contrast coverage together. "
            "If the chosen delay does not improve on the training-mean predictor, the comparison does not demonstrate useful encoding. "
            "Smoothing can raise correlation without improving single-bin likelihood; small predicted variance can hide poor amplitude recovery.","",
            "Undefined correlations remain unavailable. Region and population averages may cancel opposing neuron responses. "
            "Contrast summaries require original persisted metadata and the configured minimum number of trials; unavailable groups are explicit.","",
            "This comparison uses one controlled initialization seed and exploratory model selection on the selected test split. "
            "It supports neither statistical significance nor independent test confirmation. Overlapping response windows are dependent; "
            "no causal latency or physiological optimum is established. A separate untouched evaluation is needed after delay selection."]
        (attempt/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def make_plots(attempt,artifacts,matrix,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots=attempt/"plots"; plots.mkdir()
    channel = artifacts[0]["metrics"].get("target_granularity", "neuron")
    windows=sorted({r["window_bins"] for r in matrix})
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,key,title in zip(axes,("mean_neuron_correlation","mean_mse_improvement","mean_variance_ratio"),
                            (f"Mean within-trial {channel} correlation","MSE gain over training mean (Hz²)","Predicted/observed variance")):
        data=np.asarray([[next(r[key] for r in matrix if r["run_id"]==a["run_id"] and r["window_bins"]==w)
                          for w in windows] for a in artifacts],dtype=float)
        im=ax.imshow(np.ma.masked_invalid(data),aspect="auto")
        ax.set(xticks=range(len(windows)),xticklabels=windows,yticks=range(len(artifacts)),
               yticklabels=[a["run_id"] for a in artifacts],xlabel="W (bins)",title=title)
        fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(plots/"delay_window_matrix.png",dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key,title in zip(axes,("poisson_nll","nll_improvement","bits_per_spike"),("Single-bin NLL","Gain over training mean","Bits/spike")):
        ax.plot([a["metrics"]["run"]["target_delay_bins"] for a in artifacts],
                [a["metrics"]["single_bin"][key] for a in artifacts],"o-")
        ax.set(xlabel="D (bins)",title=title); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(plots/"likelihood_by_delay.png",dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for a in artifacts:
        axes[0].hist([n["single_bin"]["nll_improvement"] for n in a["metrics"]["neurons"]],bins=35,histtype="step",label=a["run_id"])
        axes[1].hist([t["single_bin"]["nll_improvement"] for t in a["metrics"]["trials"]],bins=25,histtype="step",label=a["run_id"])
    axes[0].set(title=f"Per-{channel} NLL gain",xlabel="Gain/count cell",ylabel=channel.title()+"s")
    axes[1].set(title="Per-trial NLL gain",xlabel="Gain/count cell",ylabel="Trials")
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(plots/"neuron_trial_distributions.png",dpi=140); plt.close(fig)
    regions=sorted(artifacts[0]["metrics"]["regions"])
    fig,ax=plt.subplots(figsize=(10,5))
    for region in regions:
        ax.plot([a["metrics"]["run"]["target_delay_bins"] for a in artifacts],
                [a["metrics"]["regions"][region]["single_bin"]["nll_improvement"] for a in artifacts],"o-",label=region)
    ax.set(xlabel="D (bins)",ylabel="Single-bin NLL gain",title="Matched target summaries by recorded region")
    ax.legend(fontsize=6,ncol=3); fig.tight_layout(); fig.savefig(plots/"regions.png",dpi=140); plt.close(fig)
    # Read stored predictions only, never inference. Same trial/order is manifest-bound.
    index=read(artifacts[0]["attempt"]/"predictions/index.json")
    selected = np.linspace(0,len(index)-1,min(cfg["representative_trials"],len(index)),dtype=int)
    for i in iter_progress(selected, "comparison: temporal plots", total=len(selected), unit="plots"):
        fig,axes=plt.subplots(len(windows),1,figsize=(12,2.8*len(windows)),squeeze=False)
        for a in artifacts:
            entry=read(a["attempt"]/"predictions/index.json")[i]
            if entry["sample_id"] != index[i]["sample_id"]:
                raise ValueError("Stored prediction trial identities differ")
            with np.load(a["attempt"]/"predictions"/entry["file"],allow_pickle=False) as row:
                endpoints=row["curve_endpoints"]
                for ax,w in zip(axes[:,0],windows):
                    if a is artifacts[0]:
                        ax.plot(row["time"][endpoints],rolling(row["observed"],w,endpoints).mean(1)*60,color="black",label="Observed")
                        ax.axhline(row["baseline"].mean()*60,color="gray",ls="--",label="Training mean")
                    ax.plot(row["time"][endpoints],rolling(row["expected"],w,endpoints).mean(1)*60,label=a["run_id"],alpha=.8)
                    ax.set(xlabel="Seconds from onset",ylabel="spikes/s/"+channel,title=f"Trial {entry['trial_id']}, W={w}")
        axes[0,0].legend(ncol=4,fontsize=8); fig.tight_layout()
        fig.savefig(plots/f"temporal_trial_{i:06d}.png",dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
