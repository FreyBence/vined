"""Diagnose stored CLIP features without changing research pipeline artifacts.

Run from the checkout: .venv/Scripts/python.exe script/audit/analyze_clip_similarity.py
"""

import argparse
import json
from pathlib import Path
import sys
from uuid import UUID

import numpy as np

from audit_paths import AUDIT_ROOT
from utils.config_utils import load_config
from utils.paths import REPO_ROOT, visual_dir
from visual_features import EncodedObservation, FeatureArtifactReader


METHODS = {
    "raw_cosine": ("Raw cosine similarity", "Higher means closer directions"),
    "raw_l2": ("Raw Euclidean distance", "Lower means closer stored vectors"),
    "normalized_l2": ("Normalized Euclidean distance", "sqrt(2 - 2*cosine); same angular information"),
    "centered_cosine": ("Mean-centered cosine similarity", "Directions relative to the EID reference mean"),
    "standardized_l2": ("Standardized Euclidean distance", "Distance in reference standard-deviation units"),
    "whitened_l2": ("PCA-whitened Euclidean distance", "Regularized distance in retained covariance directions"),
}


def reservoir_add(sample, value, count, limit, rng):
    """Uniform sample of an ordered stream, bounded independently of its length."""
    if len(sample) < limit:
        sample.append(value)
    else:
        index = int(rng.integers(count))
        if index < limit:
            sample[index] = value


def collect(path, eid, args):
    reference, pairs = [], []
    reference_rng = np.random.default_rng(args.seed)
    pair_rng = np.random.default_rng(args.seed + 1)
    count = pair_count = 0
    previous = selected = None
    with FeatureArtifactReader(path, eid=eid) as reader:
        definition = reader.definition
        for item in reader:
            # An unavailable/unselected observation or trial outcome breaks adjacency.
            if not isinstance(item, EncodedObservation) or item.feature is None:
                previous = None
                continue
            vector = item.feature.astype(np.float64)
            if not np.isfinite(vector).all() or np.linalg.norm(vector) <= args.norm_floor:
                raise ValueError("Non-finite or zero-norm encoded observation")
            metadata = item.metadata
            count += 1
            reservoir_add(reference, vector, count, args.reference_samples, reference_rng)
            current = (metadata, vector)
            if (previous is not None and previous[0]["trial_id"] == metadata["trial_id"]
                    and previous[0]["schedule_index"] + 1 == metadata["schedule_index"]):
                pair_count += 1
                reservoir_add(pairs, (previous[1], vector), pair_count,
                              args.pair_samples, pair_rng)
                if selected is None and np.any(np.abs(vector - previous[1]) > args.tolerance):
                    selected = (previous, current)
            previous = current
        completion = reader.completion
        if completion is None:
            raise ValueError("Feature archive verification did not complete")
    return reference, pairs, selected, count, pair_count, definition, completion


def fit_reference(reference, args):
    values = np.asarray(reference, dtype=np.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    std = values.std(axis=0)  # Population standard deviation (ddof=0).
    active = std > args.std_floor
    scale = np.maximum(std, args.std_floor)
    components = np.empty((values.shape[1], 0))
    eigenvalues = np.empty(0)
    rank = 0
    ridge = args.variance_floor
    if len(values) >= 2:
        covariance = centered.T @ centered / (len(values) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        ridge = max(args.variance_floor, args.pca_ridge * float(eigenvalues[0]))
        rank = int(np.count_nonzero(eigenvalues > ridge))
        requested = args.pca_components or min(64, len(values) - 1, values.shape[1])
        retained = min(requested, rank, len(values) - 1)
        components = eigenvectors[:, order[:retained]]
        eigenvalues = eigenvalues[:retained]
    configuration = dict(
        reference_sample_count=len(values), original_dimension=values.shape[1],
        standardization_ddof=0, std_floor=args.std_floor,
        dimensions_above_std_floor=int(active.sum()),
        pca_requested_components=args.pca_components or "auto: min(64, samples-1, dimensions)",
        pca_fitted_components=components.shape[1], pca_rank_above_floor=rank,
        pca_relative_ridge=args.pca_ridge, pca_absolute_variance_floor=args.variance_floor,
        pca_effective_ridge=ridge, pca_covariance_ddof=1,
        whitening="retain eigenvalues > ridge; divide scores by sqrt(eigenvalue + ridge)",
    )
    return mean, scale, components, eigenvalues, ridge, configuration


def cosine(a, b, floor):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    result = np.full(na.shape, np.nan)
    np.divide(np.sum(a * b, axis=-1), na * nb, out=result,
              where=(na > floor) & (nb > floor))
    return np.clip(result, -1, 1)


def compare(a, b, fitted, args):
    mean, scale, components, eigenvalues, ridge, _ = fitted
    delta = b - a
    a_unit = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_unit = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return dict(
        raw_cosine=cosine(a, b, args.norm_floor),
        raw_l2=np.linalg.norm(delta, axis=-1),
        normalized_l2=np.linalg.norm(b_unit - a_unit, axis=-1),
        centered_cosine=cosine(a - mean, b - mean, args.norm_floor),
        # Subtracting the common mean cancels in both distances.
        standardized_l2=np.linalg.norm(delta / scale, axis=-1),
        whitened_l2=(np.linalg.norm((delta @ components) / np.sqrt(eigenvalues + ridge), axis=-1)
                     if components.shape[1] else np.full(a.shape[:-1], np.nan)),
    )


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def make_report(path, eid, collected, args):
    reference, pairs, selected, count, pair_count, definition, completion = collected
    report = dict(eid=eid, feature_source=str(path.resolve()),
                  generation_id=completion["generation_id"],
                  representation=definition["representation"],
                  reference_population_count=count, eligible_consecutive_pair_count=pair_count,
                  configuration=dict(tolerance=args.tolerance, norm_floor=args.norm_floor,
                                     reference_sample_limit=args.reference_samples,
                                     pair_sample_limit=args.pair_samples, seed=args.seed),
                  notes=["Diagnostic EID population includes all partitions. Fit on training partition before training use.",
                         "Larger numerical distances across different metrics do not imply better discrimination.",
                         "Pairs require adjacent source schedule indices in one trial; no invalid/unselected gaps are bridged.",
                         "Reference and pair reservoirs are uniform over observations and pairs, not balanced by trial.",
                         "The archive reader rejects malformed, nonfinite, zero-norm or non-unit stored vectors."])
    if selected is None:
        report["status"] = "no_differing_pair"
        report["message"] = "No valid consecutive same-trial pair differs above the absolute per-dimension tolerance."
        return report
    (meta_a, a), (meta_b, b) = selected
    fitted = fit_reference(reference, args)
    metrics = compare(a[None, :], b[None, :], fitted, args)
    distribution = compare(np.asarray([p[0] for p in pairs]),
                           np.asarray([p[1] for p in pairs]), fitted, args)
    delta = np.abs(b - a)
    top = np.argsort(-delta, kind="stable")[:args.top_dimensions]
    hashes = [m.get("rgb_sha256") for m in (meta_a, meta_b)]
    report.update(status="ok", selected_pair=dict(
        trial_id=meta_a["trial_id"], source_frame_indices=[m["schedule_index"] for m in (meta_a, meta_b)],
        observation_ids=[m["observation_id"] for m in (meta_a, meta_b)],
        session_times_seconds=[m.get("session_time") for m in (meta_a, meta_b)],
        timing=[m.get("timing") for m in (meta_a, meta_b)],
        original_embedding_shapes=[list(a.shape), list(b.shape)],
        image_comparison=dict(source="verified source RGB pixel hashes retained in feature archive",
                              rgb_sha256=hashes,
                              images_differ=hashes[0] != hashes[1] if all(hashes) else None),
        embedding_norms=[float(np.linalg.norm(a)), float(np.linalg.norm(b))],
        raw_difference_norm=float(np.linalg.norm(b - a)),
        meaningful_difference_dimensions=int(np.count_nonzero(delta > args.tolerance)),
        top_absolute_differences=[dict(dimension=int(i), absolute_difference=float(delta[i]),
                                       first=float(a[i]), second=float(b[i])) for i in top]),
        transformations=fitted[-1], pair_sample_count=len(pairs), metrics={})
    for key in METHODS:
        value = finite_number(metrics[key][0])
        population = distribution[key]
        population = population[np.isfinite(population)]
        report["metrics"][key] = dict(value=value, distribution=dict(
            finite_count=len(population), unavailable_count=len(pairs) - len(population),
            minimum=float(population.min()) if len(population) else None,
            median=float(np.median(population)) if len(population) else None,
            maximum=float(population.max()) if len(population) else None,
            selected_pair_percentile=(100 * float(np.mean(population <= value))
                                      if len(population) and value is not None else None)))
    raw_cosine = report["metrics"]["raw_cosine"]["value"]
    report["normalized_distance_from_cosine"] = float(np.sqrt(max(0, 2 - 2 * raw_cosine)))
    report["notes"].append("Percentiles use 100 * fraction(sample metric <= selected metric), including ties.")
    if len(reference) < 2 or fitted[-1]["pca_fitted_components"] == 0:
        report["notes"].append("Whitening unavailable: insufficient reference samples or no variance above the ridge floor.")
    report["notes"].append("Undefined centered cosine (zero centered norm) is reported as unavailable.")
    return report


def display(report):
    print(f"EID: {report['eid']}\nFeature source: {report['feature_source']}")
    print(f"Verified generation: {report['generation_id']}")
    print(f"Reference population: {report['reference_population_count']}; "
          f"eligible consecutive pairs: {report['eligible_consecutive_pair_count']}")
    if report["status"] != "ok":
        print(report["message"])
        return
    print("Selected pair:\n" + json.dumps(report["selected_pair"], indent=2))
    print("Transformations:\n" + json.dumps(report["transformations"], indent=2))
    print("\n| Method | Similarity / Distance | Interpretation |\n|---|---:|---|")
    for key, (label, interpretation) in METHODS.items():
        value = report["metrics"][key]["value"]
        formatted = f"{value:.10g}" if value is not None else "unavailable"
        print(f"| {label} | {formatted} | {interpretation} |")
    print(f"\nsqrt(2 - 2*cosine) = {report['normalized_distance_from_cosine']:.10g}")
    print(f"\nConsecutive-pair distribution ({report['pair_sample_count']} sampled pairs):")
    print("| Method | Min | Median | Max | Selected percentile |\n|---|---:|---:|---:|---:|")
    for key, (label, _) in METHODS.items():
        stats = report["metrics"][key]["distribution"]
        cells = ["unavailable" if stats[k] is None else f"{stats[k]:.8g}"
                 for k in ("minimum", "median", "maximum", "selected_pair_percentile")]
        print(f"| {label} | " + " | ".join(cells) + " |")
    for note in report["notes"]:
        print(f"- {note}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, default=REPO_ROOT / "src/configs/run.json")
    parser.add_argument("--visual-dir", type=Path, default=visual_dir())
    parser.add_argument("--tolerance", type=float, default=1e-7, help="Absolute per-dimension difference threshold")
    parser.add_argument("--norm-floor", type=float, default=1e-12)
    parser.add_argument("--reference-samples", type=int, default=4096, help="Uniform reservoir limit for fitted statistics")
    parser.add_argument("--pair-samples", type=int, default=1024, help="Uniform reservoir limit for consecutive pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--pca-components", type=int, default=None, help="Default min(64, samples-1, dimensions), capped by stable rank")
    parser.add_argument("--pca-ridge", type=float, default=1e-4, help="Ridge relative to largest covariance eigenvalue")
    parser.add_argument("--variance-floor", type=float, default=1e-12)
    parser.add_argument("--top-dimensions", type=int, default=10)
    parser.add_argument("--json-output", type=Path, default=AUDIT_ROOT / "clip-similarity.json",
                        help="New JSON report (default: output/audit/clip-similarity.json); existing paths are refused")
    args = parser.parse_args(argv)
    for name in ("tolerance", "norm_floor", "std_floor", "pca_ridge", "variance_floor"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("reference_samples", "pair_samples", "top_dimensions"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0 or (args.pca_components is not None and args.pca_components < 1):
        parser.error("seed must be nonnegative and PCA components must be positive")
    if args.json_output is not None and args.json_output.exists():
        raise FileExistsError(f"Report already exists: {args.json_output}")
    eid = load_config(args.run_config).get("eid")
    if not isinstance(eid, str):
        raise ValueError("run.json must select one explicit EID string; null/multiple EIDs are ambiguous")
    eid = str(UUID(eid))
    path = args.visual_dir / f"{eid}_visual_clip.npz"
    print(f"Reading and verifying {path}", flush=True)
    report = make_report(path, eid, collect(path, eid, args), args)
    display(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"JSON report: {args.json_output}")
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"CLIP similarity analysis failed: {exc}", file=sys.stderr)
        sys.exit(1)
