"""Frozen E2 extraction and exact-reference derived generation production."""

import hashlib
import json
import time
import zipfile
from copy import deepcopy
from dataclasses import replace

import numpy as np
from common import *


def qualify():
    target = OUT / "qualification.json"
    if target.exists():
        return read(target)
    from visual_features import prepare_image

    reference = read(DATASET / "manifest.json")
    train = sorted(r["trial_id"] for r in reference["samples"] if r["split"] == "train")
    panel, categories = {}, {}
    peripheral = None
    for trial in train:
        rows = [
            json.loads(line)
            for line in (REPLAY_PATH / f"trials/{trial}/observations.jsonl")
            .read_text()
            .splitlines()
        ]
        valid = [r for r in rows if r["status"] == "valid"]
        if trial == train[0]:
            for r in valid[:65]:
                panel[r["observation_id"]] = r
        for r in valid:
            state = r.get("stimulus_state", {})
            category = "blank" if r.get("known_blank") else state.get("stage")
            if category not in categories:
                categories[category] = r["observation_id"]
                panel[r["observation_id"]] = r
            az = abs((r.get("render_parameters") or {}).get("azimuth_deg", 0))
            if peripheral is None or az > peripheral[0]:
                peripheral = (az, r)
    panel[peripheral[1]["observation_id"]] = peripheral[1]
    panel = list(panel.values())
    frames, prepared_hashes = [], []
    preview = OUT / "qualification-panel"
    preview.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(panel):
        with np.load(
            REPLAY_PATH / f"trials/{r['trial_id']}/images/{r['schedule_index']}.npz",
            allow_pickle=False,
        ) as payload:
            rgb = payload["rgb"]
        if hashlib.sha256(rgb.tobytes()).hexdigest() != r["rgb_sha256"]:
            raise ValueError("Panel source pixel hash mismatch")
        image = prepare_image(rgb, 224)
        prepared_hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
        image.save(preview / f"{i:03d}.png")
        frames.append(rgb)
    enc = encoder()
    import torch

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    values = np.concatenate(
        [enc.encode(frames[i : i + 64]) for i in range(0, len(frames), 64)]
    )
    enc._feature_cache.clear()  # Measure actual repeat inference, not a cache hit.
    repeated = np.concatenate(
        [enc.encode(frames[i : i + 64]) for i in range(0, len(frames), 64)]
    )
    with zipfile.ZipFile(OLD_FEATURES) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["generation_id"] != SOURCE:
            raise ValueError("Wrong archived source features")
        indices = {}
        wanted = {r["observation_id"] for r in panel}
        with archive.open("records.jsonl") as stream:
            for line in stream:
                row = json.loads(line)
                if (
                    row["kind"] == "observation"
                    and row["metadata"]["observation_id"] in wanted
                ):
                    indices[row["metadata"]["observation_id"]] = row["feature_index"]
        with archive.open("features.npy") as stream:
            archived = np.lib.format.read_array(stream, allow_pickle=False)
        old = np.stack([archived[indices[r["observation_id"]]] for r in panel])
        del archived
    a = values[:, :768].astype(np.float64)
    b = old.astype(np.float64)
    relative = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )
    result = dict(
        provenance=enc.provenance,
        representations={arm: enc.representation_provenance(arm) for arm in ARMS},
        batch_size=64,
        categories=categories,
        peripheral_azimuth=peripheral[0],
        panel=[
            dict(
                metadata=r,
                prepared_rgb_sha256=h,
                prepared_png=f"qualification-panel/{i:03d}.png",
                relative_l2=float(relative[i]),
                cosine=float(cosine[i]),
                max_absolute_error=float(abs(a[i] - b[i]).max()),
            )
            for i, (r, h) in enumerate(zip(panel, prepared_hashes))
        ],
        exact_prepared_rgb_shape=[224, 224, 3],
        source_rgb_shape=list(frames[0].shape),
        norm_max_deviations={
            arm: float(abs(np.linalg.norm(values[:, sl], axis=1) - 1).max())
            for arm, sl in [
                ("A", slice(0, 768)),
                ("B", slice(768, 4864)),
                ("G", slice(4864, 8960)),
            ]
        },
        relative_l2_max=float(relative.max()),
        cosine_min=float(cosine.min()),
        max_absolute_error=float(abs(a - b).max()),
        same_path_repeat_max_absolute=float(abs(values - repeated).max()),
        repeated_prepared_images=len(prepared_hashes) - len(set(prepared_hashes)),
        provisional_thresholds=dict(relative_l2_max=1e-3, cosine_min=0.999999),
        qualified=bool((relative <= 1e-3).all() and (cosine >= 0.999999).all()),
        common_pass_for_all_arms=True,
        seconds=time.perf_counter() - start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    np.savez_compressed(
        preview / "features.npz",
        fresh_bundle=values,
        repeat_bundle=repeated,
        archived_A=old,
    )
    save(target, result)
    del enc
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "qualified",
                    "relative_l2_max",
                    "cosine_min",
                    "same_path_repeat_max_absolute",
                )
            }
        )
    )
    return result


def extract():
    qualification = read(OUT / "qualification.json")
    if (
        not qualification["qualified"]
        and not (OUT / "qualification-disposition.json").exists()
    ):
        raise ValueError(
            "Inspect qualification failure and record its disposition before complete extraction"
        )
    target = OUT / "source-bundle.json"
    if target.exists():
        return read(target)
    from visual_features import ObservationSelection
    from visual_features.bundle import write_feature_bundle
    from visual_replay import ReplayArtifactReader

    enc = encoder()
    if enc.provenance != qualification["provenance"]:
        raise ValueError("Qualified extraction implementation/settings changed")
    import torch

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    reader = ReplayArtifactReader(REPLAY_PATH, workers=4)
    if reader.artifact_manifest["completion"]["generation_id"] != REPLAY:
        raise ValueError("Wrong replay generation")
    outputs = {arm: OUT / f"features/{arm}.npz" for arm in ARMS}
    manifests = write_feature_bundle(
        ObservationSelection(reader), enc, outputs, batch_size=64, workers=4
    )
    result = dict(
        arms={
            arm: dict(
                path=str(outputs[arm]),
                manifest=manifests[arm],
                bytes=outputs[arm].stat().st_size,
            )
            for arm in ARMS
        },
        seconds=time.perf_counter() - start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    save(target, result)
    del enc
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return result


def derive(arm):
    """Reuse saved interpolation brackets, neural bins, and split memberships."""
    target = OUT / f"derived/{arm}.json"
    if target.exists():
        return read(target)
    from alignment import load_alignment, publish_alignment
    from training_dataset import AlignmentSource, load_samples, publish_dataset
    from training_dataset.splits import DatasetSplits
    from visual_features import EncodedObservation, FeatureArtifactReader

    source = read(OUT / "source-bundle.json")["arms"][arm]
    reference = load_alignment(
        ROOT / "output/alignment" / SESSION / ALIGNMENT,
        expected_generation_id=ALIGNMENT,
    )
    trials = {r.trial_id: r for r in reference.trials}
    records, derived = [], []
    start = time.perf_counter()
    with FeatureArtifactReader(
        source["path"], eid=SESSION, replay_generation_id=REPLAY
    ) as reader:
        definition = reader.definition
        for row in reader:
            if isinstance(row, EncodedObservation):
                if row.metadata["trial_id"] in trials:
                    records.append(row)
                continue
            trial = trials.get(row.metadata["trial_id"])
            if trial is None:
                continue
            saved = trial.alignment_metadata["resampling"]
            encoded = [r for r in records if r.status == "encoded"]
            if (
                len(encoded) != len(saved["source_observation_ids"])
                or [r.metadata["observation_id"] for r in encoded]
                != list(saved["source_observation_ids"])
                or [r.metadata["schedule_index"] for r in encoded]
                != list(saved["source_schedule_indices"])
                or not np.array_equal(
                    [r.metadata["session_time"] for r in encoded], saved["source_times"]
                )
                or any(r.status != "encoded" for r in records)
                or row.metadata != trial.alignment_metadata["visual_outcome"]
            ):
                raise ValueError("Matched-generation source identities/support changed")
            values = np.asarray([r.feature for r in encoded], dtype=np.float64)
            left, right = saved["left_source_indices"], saved["right_source_indices"]
            w = np.asarray(saved["interpolation_weights"])[:, None]
            aligned = (1 - w) * values[left] + w * values[right]
            norms = np.linalg.norm(aligned, axis=1, keepdims=True)
            if (
                not np.isfinite(norms).all()
                or (norms <= np.finfo(np.float32).eps).any()
            ):
                raise ValueError("Representation invalidates a reference trial")
            aligned = (aligned / norms).astype(np.float32)
            metadata = deepcopy(trial.alignment_metadata)
            metadata.update(
                visual_definition=definition, visual_completion=source["manifest"]
            )
            derived.append(
                replace(trial, visual_features=aligned, alignment_metadata=metadata)
            )
            records = []
        if reader.completion != source["manifest"]:
            raise ValueError("Source completion mismatch")
    if [r.trial_id for r in derived] != list(trials) or len(derived) != 312:
        raise ValueError("All 312 reference trials must survive")
    generation = publish_alignment(
        derived,
        OUT / f"alignment/{arm}",
        requested_trial_ids=reference.manifest["requested_trial_ids"],
        excluded_trials=reference.manifest["excluded_trials"],
        source_provenance=dict(
            reference_alignment=ALIGNMENT,
            reference_dataset=REFERENCE,
            visual_definition=definition,
            visual_completion=source["manifest"],
        ),
    )
    generation_path, generation_id = generation.path, generation.generation_id
    del derived, reference, trials, generation
    samples = load_samples([AlignmentSource(generation_path, generation_id)])
    reference_manifest = read(DATASET / "manifest.json")
    ref_entries = {r["sample_id"]: r for r in reference_manifest["samples"]}
    splits = {name: [] for name in ("train", "val", "test")}
    for sample in samples:
        original = ref_entries[sample.sample_id]
        split = original["split"]
        # These frozen mechanical equality checks never make test-based choices.
        index = reference_manifest["samples"].index(original)
        with np.load(
            DATASET / f"samples/{index:06d}.npz", allow_pickle=False
        ) as payload:
            for name in (
                "neural",
                "physical_timestamps",
                "bin_start_times",
                "bin_end_times",
                "temporal_positions",
                "temporal_mask",
                "neuron_mask",
            ):
                if not np.array_equal(
                    getattr(sample, name), payload[name], equal_nan=True
                ):
                    raise ValueError(f"Reference payload changed: {name}")
        if (
            sample.neuron_count != original["neuron_count"]
            or sample.sequence_length != original["sequence_length"]
        ):
            raise ValueError("Reference shape changed")
        metadata = deepcopy(sample.metadata)
        metadata["split"] = deepcopy(original["metadata"]["split"])
        splits[split].append(replace(sample, split=split, metadata=metadata))
    dataset = DatasetSplits(
        *(tuple(splits[k]) for k in ("train", "val", "test")),
        deepcopy(reference_manifest["dataset_metadata"]),
    )
    published = publish_dataset(dataset, OUT / f"datasets/{arm}")
    result = dict(
        arm=arm,
        source_generation=source["manifest"]["generation_id"],
        alignment_path=str(generation_path),
        alignment_generation=generation_id,
        dataset_path=str(published.path),
        dataset_generation=published.generation_id,
        counts={k: len(v) for k, v in splits.items()},
        reference_payloads_equal=True,
        reference_brackets_equal=True,
        seconds=time.perf_counter() - start,
    )
    save(target, result)
    return result


def qualify_peripheral():
    """Fixed training-only visible peripheral supplement, using actual common-pass A."""
    from visual_features import prepare_image

    target = OUT / "peripheral-qualification.json"
    if target.exists():
        return read(target)
    trial, schedule = 83, 5
    membership = read(DATASET / "manifest.json")["dataset_metadata"]["memberships"][
        "train"
    ]
    if trial not in {r["trial_id"] for r in membership}:
        raise ValueError("Peripheral qualification must remain training-only")
    rows = [
        json.loads(line)
        for line in (REPLAY_PATH / f"trials/{trial}/observations.jsonl")
        .read_text()
        .splitlines()
    ]
    metadata = rows[schedule]
    if (
        metadata["known_blank"]
        or metadata["render_parameters"]["contrast"] != 1
        or abs(metadata["render_parameters"]["azimuth_deg"]) < 45
    ):
        raise ValueError("Expected visible peripheral frame")
    with np.load(
        REPLAY_PATH / f"trials/{trial}/images/{schedule}.npz", allow_pickle=False
    ) as payload:
        rgb = payload["rgb"]
    image = prepare_image(rgb, 224)
    image.save(OUT / "qualification-panel/peripheral-prepared.png")

    def locate(archive):
        with archive.open("records.jsonl") as stream:
            for line in stream:
                record = json.loads(line)
                if (
                    record["kind"] == "observation"
                    and record["metadata"]["observation_id"]
                    == metadata["observation_id"]
                ):
                    return record["feature_index"]
        raise ValueError("Peripheral observation missing")

    with zipfile.ZipFile(OLD_FEATURES) as archive:
        index = locate(archive)
        with archive.open("features.npy") as stream:
            np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            stream.read(index * 768 * 4)
            old = np.frombuffer(stream.read(768 * 4), dtype="<f4").copy()
    published = OUT / "features/A.npz"
    if published.exists():
        with zipfile.ZipFile(published) as archive:
            index = locate(archive)
            with archive.open("features.npy") as stream:
                np.lib.format.read_magic(stream)
                np.lib.format.read_array_header_2_0(stream)
                stream.read(index * 768 * 4)
                fresh = np.frombuffer(stream.read(768 * 4), dtype="<f4").copy()
    else:
        # A's raw staging file is uniquely identified by width/feature index.
        candidates = []
        for directory in (OUT / "features").glob(".features-*"):
            with (directory / "records.jsonl").open() as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if (
                        record["kind"] == "observation"
                        and record["metadata"]["observation_id"]
                        == metadata["observation_id"]
                    ):
                        candidates.append((directory, record["feature_index"]))
                        break
        if len(candidates) != 3:
            raise ValueError(
                "Wait for common-pass extraction to reach the fixed peripheral frame"
            )
        directory, index = min(
            candidates,
            key=lambda item: (item[0] / "pixels-free-features").stat().st_size,
        )
        with (directory / "pixels-free-features").open("rb") as stream:
            stream.seek(index * 768 * 4)
            fresh = np.frombuffer(stream.read(768 * 4), dtype="<f4").copy()
    a, b = fresh.astype(np.float64), old.astype(np.float64)
    relative = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    result = dict(
        metadata=metadata,
        prepared_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        fresh_A_sha256=hashlib.sha256(fresh.tobytes()).hexdigest(),
        relative_l2=relative,
        cosine=cosine,
        max_absolute_error=float(abs(a - b).max()),
        norm_deviation=float(abs(np.linalg.norm(a) - 1)),
        qualified=bool(relative <= 1e-3 and cosine >= 0.999999),
        batch_size=64,
        interpretation="Visible contrast-1 peripheral content; initial largest-azimuth panel member was zero contrast. This supplement uses the frozen actual common-pass extraction, without a neural outcome.",
        nonneutral_source_values=int(np.count_nonzero(rgb != 128)),
    )
    np.savez_compressed(
        OUT / "qualification-panel/peripheral-features.npz",
        fresh_A=fresh,
        archived_A=old,
    )
    save(target, result)
    if not result["qualified"]:
        raise ValueError(
            "Investigate peripheral numerical qualification before model training"
        )
    return result
