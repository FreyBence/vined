"""Compare two explicitly selected evaluation attempts without model inference."""

import argparse
import json
from copy import deepcopy
import math
from pathlib import Path
import webbrowser

from evaluation.artifacts import load_evaluation
from trainer.run_paths import attempt_name, scenario_directory
from utils.paths import output_dir


def neuron_key(neuron):
    identity = neuron["identity"]
    fields = ("eid", "pid", "collection", "revision", "source_unit_id")
    if any(identity.get(key) is None for key in ("eid", "pid", "source_unit_id")):
        raise ValueError("Neuron is missing source identity required for comparison")
    return json.dumps([identity.get(key) for key in fields], separators=(",", ":"))


def aggregate(records):
    values = [r["value"] for r in records if r.get("value") is not None and math.isfinite(r["value"])]
    return dict(value=sum(values) / len(values) if values else None,
                status="available" if values else "unavailable", partial=len(values) != len(records),
                available_count=len(values), total_count=len(records), unavailable_count=len(records)-len(values))


def compare_evaluations(left, right):
    """Compare verified artifacts on common observations and source neurons."""
    a, b = left.manifest, right.manifest
    direction = a["configuration"]["model_mode"]
    if direction != b["configuration"]["model_mode"]:
        raise ValueError("Cannot compare encoding and decoding evaluations")
    if direction not in ("encoding", "decoding"):
        raise ValueError("Only encoding or decoding evaluations are supported")
    for key in ("split", "target_coverage"):
        if a["configuration"].get(key) != b["configuration"].get(key):
            raise ValueError(f"Incompatible evaluation {key}")
    # Different context lengths are legitimate model comparisons. H=12 target
    # support and its boundary policy must stay fixed across those contexts.
    for key in ("target_support_bins", "boundary_policy"):
        if a["configuration"]["temporal_context"][key] != b["configuration"]["temporal_context"][key]:
            raise ValueError(f"Incompatible temporal {key}")
    if a["metric_configuration"] != b["metric_configuration"]:
        raise ValueError("Incompatible metric definitions or aggregation policies")
    if a["provenance"]["dataset_generation_id"] != b["provenance"]["dataset_generation_id"]:
        raise ValueError("Incompatible dataset generations")
    def samples(manifest):
        scalars = [deepcopy(entry["scalar"]) for entry in manifest["samples"]]
        for scalar in scalars:
            scalar["metadata"].pop("runtime_neural_selection", None)
        return {scalar["sample_id"]: scalar for scalar in scalars}
    if samples(a) != samples(b):
        raise ValueError("Incompatible test samples or physical temporal metadata")
    if set(a["units"]) != set(b["units"]):
        raise ValueError("Incompatible evaluated sessions")
    for session in a["units"]:
        for key in ("bin_size", "target_coverage"):
            if left.result["session_results"][session][key] != right.result["session_results"][session][key]:
                raise ValueError(f"Incompatible {key} for {session}")
    reports = []
    for artifact in (left, right):
        result = artifact.result
        reports.append(dict(path=str(artifact.path), evaluation_id=result["evaluation_id"],
                            context=artifact.manifest["configuration"]["temporal_context"],
                            neural_region_selection=artifact.manifest["configuration"].get("neural_region_selection"),
                            global_metrics=result["global_metrics"],
                            regions=result.get("bits_per_spike_by_region", {}),
                            sessions={sid: dict(metrics=row["metrics"], regions=row.get("bits_per_spike_by_region", {}),
                                neurons=[dict(column=n["column"], identity=n["identity"], metrics=n["metrics"])
                                         for n in row["neurons"]])
                                for sid, row in result["session_results"].items()}))
    population = {}
    if direction == "encoding":
        for session in reports[0]["sessions"]:
            rows = [report["sessions"][session] for report in reports]
            maps = [{neuron_key(n): n for n in row["neurons"]} for row in rows]
            if any(len(mapping) != len(row["neurons"]) for mapping, row in zip(maps, rows)):
                raise ValueError(f"Duplicate source neuron identity for {session}")
            shared = sorted(set(maps[0]) & set(maps[1]))
            if not shared:
                raise ValueError(f"No shared evaluated neurons for {session}")
            population[session] = dict(shared=len(shared), original=[len(row["neurons"]) for row in rows])
            for row, mapping in zip(rows, maps):
                row["neurons"] = [dict(mapping[key], key=key) for key in shared]
                row["metrics"] = {name: aggregate([n["metrics"][name] for n in row["neurons"]])
                                  for name in row["metrics"]}
                region_groups = {}
                for n in row["neurons"]:
                    identity = n["identity"]
                    region = identity.get("selected_region", identity.get("acronym"))
                    if region and str(region).lower() != "void":
                        region_groups.setdefault(str(region), []).append(n["metrics"]["bits_per_spike"])
                row["regions"] = {region: aggregate(records) for region, records in region_groups.items()}
        for report in reports:
            rows = list(report["sessions"].values())
            def pooled(name, region=None):
                if a["metric_configuration"]["aggregation"] == "neuron_weighted":
                    return [n["metrics"][name] for row in rows for n in row["neurons"]
                            if region is None or str(n["identity"].get("selected_region", n["identity"].get("acronym"))) == region]
                return [row["metrics"][name] if region is None else row["regions"][region]
                        for row in rows if region is None or region in row["regions"]]
            report["global_metrics"] = {name: aggregate(pooled(name)) for name in report["global_metrics"]}
            report["regions"] = {region: aggregate(pooled("bits_per_spike", region))
                                 for region in sorted({region for row in rows for region in row["regions"]})}
    return dict(direction=direction, runs=reports, population=population)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_a", type=int)
    parser.add_argument("attempt_a", type=attempt_name)
    parser.add_argument("scenario_b", type=int)
    parser.add_argument("attempt_b", type=attempt_name)
    parser.add_argument("--evaluation-dir", type=Path, default=output_dir() / "evaluation")
    parser.add_argument("--output", type=Path, help="HTML destination (must not already exist)")
    parser.add_argument("--no-open", action="store_true", help="Save without opening a browser")
    args = parser.parse_args()
    try:
        paths = [scenario_directory(args.evaluation_dir, scenario) / attempt
                 for scenario, attempt in ((args.scenario_a, args.attempt_a), (args.scenario_b, args.attempt_b))]
        # Reject direction mismatches before reading large prediction payloads.
        manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in paths]
        if manifests[0]["configuration"]["model_mode"] != manifests[1]["configuration"]["model_mode"]:
            raise ValueError("Cannot compare encoding and decoding evaluations")
        artifacts = [load_evaluation(path, load_predictions=False) for path in paths]
        data = compare_evaluations(*artifacts)
        data["labels"] = [f"A: scenario {args.scenario_a} / {args.attempt_a}",
                          f"B: scenario {args.scenario_b} / {args.attempt_b}"]
        destination = args.output or output_dir() / "comparisons" / (
            f"scenario-{args.scenario_a}-{args.attempt_a}_vs_scenario-{args.scenario_b}-{args.attempt_b}.html")
        destination = destination.resolve()
        # Reports must stay outside immutable evaluation artifacts.
        if any(destination.is_relative_to(path.resolve()) for path in paths):
            raise ValueError("Comparison report must be outside the source evaluation directories")
        template = Path(__file__).with_name("evaluation") / "comparison.html"
        encoded = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(template.read_text(encoding="utf-8").replace("__REPORT_DATA__", encoded))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(destination)
    if not args.no_open:
        webbrowser.open(destination.as_uri())


if __name__ == "__main__":
    main()
