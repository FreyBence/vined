"""Frozen validation/test inference and explicitly named paired E2 metrics."""

import time

import numpy as np
from common import *

PAIRS = (("B", "A"), ("B", "G"), ("G", "A"))


def infer(arm, split):
    directory = OUT / "evaluation" / split / arm
    target = directory / "metrics.json"
    if target.exists():
        previous = read(target)
        current = read(OUT / "runs" / arm / "result.json")
        if (
            previous["training"] != current
            or previous["dataset"] != read(OUT / f"derived/{arm}.json")
            or file_hash(current["checkpoint"]) != current["checkpoint_sha256"]
        ):
            raise ValueError("Completed evaluation identity changed")
        predictions(arm, split)
        return previous
    if split == "test":
        frozen = read(OUT / "frozen.json")
        for relative, digest in frozen["provenance_files"].items():
            if file_hash(OUT / relative) != digest:
                raise ValueError(f"Frozen provenance changed: {relative}")
        if (
            file_hash(OUT / "val-comparison.json")
            != frozen["validation_comparison_sha256"]
        ):
            raise ValueError("Validation comparison changed after freezing")
        if frozen["sources"] != sources():
            raise ValueError("Implementation changed after freezing")
        for named in ARMS:
            result = read(OUT / "runs" / named / "result.json")
            if (
                result != frozen["training"][named]
                or file_hash(result["checkpoint"]) != result["checkpoint_sha256"]
            ):
                raise ValueError("Model changed after freezing")
    from evaluation.predictions import collect_predictions
    from evaluation.setup import resolve_setup

    derived = read(OUT / f"derived/{arm}.json")
    training = read(OUT / "runs" / arm / "result.json")
    resolved = read(OUT / "runs" / arm / "resolved.json")
    if (
        training["training_identity"] != fingerprint(resolved)
        or file_hash(training["checkpoint"]) != training["checkpoint_sha256"]
    ):
        raise ValueError("Training identity changed")
    start = time.perf_counter()
    setup = resolve_setup(
        checkpoint_path=training["checkpoint"],
        dataset_generation=derived["dataset_path"],
        expected_generation_id=derived["dataset_generation"],
        session_ids=[SESSION],
        context_mode="strict",
        context_bins=12,
        batch_size=1,
        device="cuda",
        seed=42,
        prediction_direction="encoding",
        neural_region_selection="all_recorded",
        split=split,
    )
    collection = collect_predictions(setup)
    support = resolved["support"]
    expected = support["compatibility"]["trials"][split]
    if [r.sample_id for r in collection.predictions] != [
        r["sample_id"] for r in expected
    ]:
        raise ValueError("Prediction membership differs from reference")
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    index = []
    for i, (row, reference) in enumerate(zip(collection.predictions, expected)):
        length, width = reference["sequence_length"], reference["neurons"]
        observed = row.observed_neural[:length, :width]
        if (
            not np.array_equal(
                np.flatnonzero(row.encoding_target_mask), np.arange(11, length)
            )
            or fingerprint(observed[11:].tolist()) != reference["target_counts_hash"]
            or fingerprint(row.physical_timestamps[:length].tolist())
            != reference["physical_time_hash"]
            or plain(row.neuron_identity.to_dict(orient="records"))
            != support["compatibility"]["populations"][row.session_id]
        ):
            raise ValueError("Prediction neural/bin/subregion support differs from E0")
        item = dict(
            sample_id=row.sample_id,
            session_id=row.session_id,
            trial_id=row.trial_id,
            observed=observed,
            log_expected=row.predicted_neural[:length, :width],
            original_bins=row.temporal_positions[:length],
            physical_timestamps=row.physical_timestamps[:length],
            bin_start_times=row.bin_start_times[:length],
            bin_end_times=row.bin_end_times[:length],
            eligible_mask=row.encoding_target_mask[:length],
            primary_mask=(row.temporal_positions[:length] >= 11)
            & (row.temporal_positions[:length] < 146),
        )
        records.append(item)
        filename = f"{i:06d}.npz"
        destination = directory / filename
        if destination.exists():
            raise FileExistsError(
                "Partial evaluation requires a new output root; no overwrite"
            )
        np.savez_compressed(
            destination, **{k: v for k, v in item.items() if isinstance(v, np.ndarray)}
        )
        index.append(
            {k: v for k, v in item.items() if not isinstance(v, np.ndarray)}
            | dict(
                file=filename,
                neuron_identity=plain(row.neuron_identity.to_dict(orient="records")),
                sha256=file_hash(destination),
            )
        )
    save(directory / "index.json", index)
    result = dict(
        native=metrics.evaluate(records, support),
        supplementary_parent=metrics.evaluate(records, support, parent=True),
        definitions=DEFINITIONS,
        split=split,
        training=training,
        dataset=derived,
        setup=setup.summary(),
        seconds=time.perf_counter() - start,
    )
    save(target, result)
    del setup, collection
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return result


def predictions(arm, split):
    directory = OUT / "evaluation" / split / arm
    index = read(directory / "index.json")
    records = []
    for row in index:
        if file_hash(directory / row["file"]) != row["sha256"]:
            raise ValueError("Saved prediction payload changed")
        with np.load(directory / row["file"], allow_pickle=False) as payload:
            records.append(dict(row, **{k: payload[k] for k in payload.files}))
    return records


def paired_named(first, second, first_name, second_name):
    # E0 metric primitive computes its second argument minus its first argument.
    result = metrics.paired(second, first)

    def rename(value):
        if isinstance(value, dict):
            return {
                k.replace("H12_minus_H6", f"{first_name}_minus_{second_name}")
                .replace("H12", first_name)
                .replace("H6", second_name): rename(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [rename(v) for v in value]
        return value

    return rename(result)


def compare(split):
    scores = {
        arm: read(OUT / "evaluation" / split / arm / "metrics.json") for arm in ARMS
    }
    records = {arm: predictions(arm, split) for arm in ARMS}
    for first, second in PAIRS:
        if len(records[first]) != len(records[second]):
            raise ValueError("Incomplete paired support")
        for a, b in zip(records[first], records[second]):
            for name in ("sample_id", "session_id", "trial_id", "neuron_identity"):
                if a[name] != b[name]:
                    raise ValueError(f"Unpaired {name}")
            for name in (
                "observed",
                "original_bins",
                "physical_timestamps",
                "bin_start_times",
                "bin_end_times",
                "eligible_mask",
                "primary_mask",
            ):
                if not np.array_equal(a[name], b[name]):
                    raise ValueError(f"Unpaired payload {name}")
    result = dict(
        split=split,
        definitions=DEFINITIONS,
        arms=scores,
        paired={
            f"{a}_minus_{b}": {
                kind: paired_named(scores[a][kind], scores[b][kind], a, b)
                for kind in ("native", "supplementary_parent")
            }
            for a, b in PAIRS
        },
        exact_trial_bin_subregion_pairing=True,
        automatic_superiority_claim=False,
    )
    save(OUT / f"{split}-comparison.json", result)
    report(result)
    return result


def freeze():
    validation = read(OUT / "val-comparison.json")
    training = {arm: read(OUT / "runs" / arm / "result.json") for arm in ARMS}
    primary = {
        arm: validation["arms"][arm]["native"]["primary"]["cell_weighted"][
            "poisson_nll"
        ]
        for arm in ARMS
    }
    best = min(primary, key=primary.get)
    decision_path = OUT / "validation-decision.json"
    decision = (
        read(decision_path)
        if decision_path.exists()
        else {
            "decision": "Retain all three predeclared arms for descriptive test comparison; no superiority declaration or tuning",
            "lowest_primary_nll_is_a_diagnostic_only": True,
        }
    )
    record = dict(
        validation_interpretation=decision,
        validation_decision_sha256=file_hash(decision_path)
        if decision_path.exists()
        else None,
        sources=sources(),
        provenance_files={
            name: file_hash(OUT / name)
            for name in [
                "qualification.json",
                "peripheral-qualification.json",
                "source-bundle.json",
                "preflight.json",
            ]
            + [f"derived/{arm}.json" for arm in ARMS]
        },
        training=training,
        validation_comparison_sha256=file_hash(OUT / "val-comparison.json"),
        validation_primary_nll=primary,
        lowest_validation_nll_arm=best,
        decision="Freeze all three specified arms, checkpoints, representations, settings, support, and metric definitions before the final descriptive test comparison. No further tuning.",
        test_history="E0 H6/H12 development previously considered this test membership; reused test results are exploratory, not independent confirmation.",
        no_test_based_choices=True,
    )
    save(OUT / "frozen.json", record)
    return record


def report(result):
    split = result["split"]
    lines = [
        f"# E2 controlled spatial CLIP comparison — {split}",
        "",
        "Fresh common-pass A: projected pooled CLS; B: normalized flattened 2×2 raw final-patch means; G: normalized repeated global patch mean.",
        "",
        "Strict H12, D0, S12; primary bins 11≤t<min(T,146). Optimization and checkpoint selection use full eligible support.",
        "",
        "| Arm | Primary NLL/cell | Gain/cell | Equal-trial NLL | Fraction beating baseline | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        score = result["arms"][arm]
        primary = score["native"]["primary"]
        lines.append(
            f"| {arm} | {primary['cell_weighted']['poisson_nll']} | {primary['cell_weighted']['nll_improvement']} | {primary['equal_trial']['poisson_nll']['mean']} | {primary['fraction_improved_trials']} | {score['training']['selected_epoch']} |"
        )
    lines += [
        "",
        "| Comparison | Cell-weighted NLL difference | Trial difference mean ± SD | Fraction first arm lower |",
        "|---|---:|---:|---:|",
    ]
    for a, b in PAIRS:
        primary = result["paired"][f"{a}_minus_{b}"]["native"]["primary"]
        d = primary["equal_trial_nll_difference"]
        lines.append(
            f"| {a}−{b} | {primary['cell_weighted_nll_difference']} | {d['mean']} ± {d['std']} | {primary[f'fraction_trials_{a}_lower_nll']} |"
        )
    lines += [
        "",
        "| Arm | W | Equal-trial correlation | Equal-trial variance ratio | Undefined correlations |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for w, row in result["arms"][arm]["native"]["primary"]["diagnostics"].items():
            lines.append(
                f"| {arm} | {w} | {row['correlation']['equal_trial']['mean']} | {row['variance_ratio']['equal_trial']['mean']} | {row['correlation']['cell_weighted']['undefined']} |"
            )
    lines += [
        "",
        "Machine-readable comparison includes complete per-trial/per-subregion results, low-activity channels, support counts, undefined reasons, full intervals, tails, and supplementary parent reconstructions. Parent predictions sum intensities before taking logs.",
        "",
        "B−G is the main matched-width spatial comparison. G has lower effective input rank. Both use contextualized CLIP states; matching parameter count does not establish capacity equivalence. A/B also differ in representation family and active capacity.",
        "",
        "Bin-center interpolation can access a source frame later than its bin center. This is an offline aligned experiment. The reused test membership influenced prior E0 development and is not fully independent confirmatory evidence.",
        "",
        "These three 50-epoch single-seed runs screen representations; they do not establish convergence. Tiny global NLL differences alone do not establish superiority.",
        "",
        "See qualification.json, preflight.json, runs/*/result.json, frozen.json, and verification.md for extraction, initialization, checkpoint identities, resources, and manual verification.",
    ]
    (OUT / f"{split}-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def curves():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in ARMS:
        result = read(OUT / "runs" / arm / "result.json")
        for section, key in [("training", "train_loss"), ("validation", "eval_loss")]:
            rows = [
                r
                for r in result["history"]
                if (r.get(section) or {}).get(key) is not None
            ]
            ax.plot(
                [r["epoch"] for r in rows],
                [r[section][key] for r in rows],
                label=f"{arm} {key}",
            )
        ax.axvline(result["selected_epoch"], linestyle="--", alpha=0.3)
    ax.set(xlabel="Epoch (production numbering)", ylabel="Full-support Poisson loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training-validation-curves.png")
    plt.close(fig)


def resources():
    def size(directory):
        return (
            sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            if directory.exists()
            else 0
        )

    result = {
        "extraction": {
            k: v for k, v in read(OUT / "source-bundle.json").items() if k != "arms"
        },
        "qualification": {
            k: read(OUT / "qualification.json")[k]
            for k in (
                "seconds",
                "peak_cuda_allocated_bytes",
                "peak_cuda_reserved_bytes",
            )
        },
        "arms": {
            arm: {
                "feature_archive_bytes": (OUT / f"features/{arm}.npz").stat().st_size,
                "alignment_files_bytes": size(OUT / f"alignment/{arm}"),
                "dataset_files_bytes": size(OUT / f"datasets/{arm}"),
                "training_files_bytes": size(OUT / f"runs/{arm}/training"),
                "derivation_seconds": read(OUT / f"derived/{arm}.json")["seconds"],
                "training": {
                    k: read(OUT / "runs" / arm / "result.json")[k]
                    for k in (
                        "seconds",
                        "peak_cuda_allocated_bytes",
                        "peak_cuda_reserved_bytes",
                        "registered_parameters",
                        "encoding_gradient_parameters",
                    )
                },
                "validation_inference_seconds": read(
                    OUT / "evaluation/val" / arm / "metrics.json"
                )["seconds"],
                "test_inference_seconds": read(
                    OUT / "evaluation/test" / arm / "metrics.json"
                )["seconds"],
            }
            for arm in ARMS
        },
        "size_semantics": "Measured logical file bytes; excludes temporary staging files and filesystem allocation overhead",
        "gpu_semantics": "Measured PyTorch peak allocated/reserved bytes per process stage, including loaded model; not inferred device-wide utilization",
    }
    save(OUT / "resources.json", result)
    return result
