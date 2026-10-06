"""Complete immutable dataset generations with verified standalone loading."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from alignment import AlignedTrial
from utils.provenance import file_hash, fingerprint, source_hashes, write_json
from .samples import SampleConfig, TrainingSample, build_samples, load_samples, _integer
from .splits import DatasetSplits, SPLITS, SplitConfig, assign_splits, _record, _session

ARRAY_FIELDS = ("neural", "visual", "physical_timestamps", "bin_start_times", "bin_end_times",
                "temporal_positions", "temporal_mask", "neuron_mask")
SCALAR_FIELDS = tuple(field.name for field in fields(TrainingSample)
                      if field.name not in ARRAY_FIELDS + ("neuron_identity", "metadata"))


@dataclass(frozen=True)
class DatasetGeneration:
    generation_id: str
    path: Path
    dataset: DatasetSplits
    manifest: dict


def _json(value):
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _implementation():
    return dict(sources=source_hashes("src/training_dataset/__init__.py",
        "src/training_dataset/samples.py", "src/training_dataset/splits.py",
        "src/training_dataset/artifacts.py", "src/training_dataset/handoff.py",
        "src/create_dataset.py", "src/utils/provenance.py"),
        packages={name: version(name) for name in ("numpy", "pandas", "pyarrow")})


def _validate_sample(sample):
    length = _integer(sample.sequence_length, "sequence_length", minimum=1)
    neurons = _integer(sample.neuron_count, "neuron_count", minimum=1)
    trial = AlignedTrial(sample.session_id, sample.trial_id, sample.stim_on, sample.stim_off,
        sample.aligned_start, sample.aligned_end, sample.bin_size, length,
        sample.discarded_tail_duration, sample.bin_start_times[:length],
        sample.physical_timestamps[:length], sample.bin_end_times[:length],
        sample.neural[:length, :neurons], sample.neuron_identity,
        sample.visual[:length], sample.metadata["alignment"])
    rebuilt = build_samples([trial], config=SampleConfig(**sample.metadata["construction"]["padding"]))[0]
    if (sample.sample_id != rebuilt.sample_id or sample.neuron_count != len(sample.neuron_identity)
            or sample.metadata["construction"]["representation"] != "aligned-trial-v1"):
        raise ValueError("Dataset sample identity or construction metadata mismatch")
    for name in ARRAY_FIELDS:
        actual, expected = getattr(sample, name), getattr(rebuilt, name)
        if (not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype
                or actual.shape != expected.shape or not np.array_equal(actual, expected, equal_nan=True)):
            raise ValueError(f"Invalid dataset array or padding: {name}")
    source = sample.metadata["source_alignment"]
    if source is not None and (not isinstance(source, dict) or source.get("schema_version") != 1
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("generation_id"), str) or len(source["generation_id"]) != 64):
        raise ValueError("Invalid source alignment generation provenance")


def _validate_dataset(dataset):
    if not isinstance(dataset, DatasetSplits):
        raise TypeError("Publication requires DatasetSplits")
    metadata = dataset.metadata
    conf = metadata["configuration"]
    exclusions = {}
    for record in conf["exclusions"]:
        identity = (_session(record["session_id"]), _integer(record["trial_id"], "trial_id"))
        if identity in exclusions:
            raise ValueError("Duplicate configured exclusion")
        exclusions[identity] = record["reason"]
    config = SplitConfig(conf["strategy"], conf["ratios"], conf["seed"], conf["session_assignments"], exclusions)
    config_id = fingerprint(conf)
    if metadata["configuration_id"] != config_id:
        raise ValueError("Split configuration fingerprint mismatch")
    ids, trial_splits, populations, sources, all_samples = set(), {}, {}, {}, []
    for split in SPLITS:
        samples = getattr(dataset, split)
        order = [(sample.session_id, sample.trial_id, sample.sample_id) for sample in samples]
        if order != sorted(order) or metadata["memberships"][split] != [_record(s) for s in samples]:
            raise ValueError("Split membership or ordering mismatch")
        for sample in samples:
            _validate_sample(sample)
            identity = (sample.session_id, sample.trial_id)
            if (sample.sample_id in ids or sample.split != split or identity in exclusions
                    or sample.metadata["split"] != dict(name=split, configuration_id=config_id)
                    or identity in trial_splits and trial_splits[identity] != split):
                raise ValueError("Duplicate, excluded, or conflicting split membership")
            ids.add(sample.sample_id)
            trial_splits[identity] = split
            if sample.session_id in populations:
                units, bin_size = populations[sample.session_id]
                if (not units.reset_index(drop=True).equals(sample.neuron_identity.reset_index(drop=True))
                        or bin_size != sample.bin_size
                        or fingerprint(_json(sources[sample.session_id])) !=
                           fingerprint(_json(sample.metadata["source_alignment"]))):
                    raise ValueError("Conflicting session population, bin size, or source generation")
            populations[sample.session_id] = (sample.neuron_identity, sample.bin_size)
            sources[sample.session_id] = sample.metadata["source_alignment"]
            all_samples.append(replace(sample, split=None))
    expected = assign_splits(all_samples, config=SplitConfig(config.strategy, config.ratios,
                            config.seed, config.session_assignments))
    if (expected.metadata["memberships"] != metadata["memberships"]
            or expected.metadata["algorithm"] != metadata["algorithm"]
            or expected.metadata["grouping"] != metadata["grouping"]):
        raise ValueError("Dataset memberships do not follow the declared split strategy")
    excluded_ids, excluded_trials = set(), set()
    for record in metadata["exclusions"]:
        identity = (_session(record["session_id"]), _integer(record["trial_id"], "trial_id"))
        if (identity not in exclusions or record["reason"] != exclusions[identity]
                or record["sample_id"] in ids or record["sample_id"] in excluded_ids):
            raise ValueError("Excluded sample accounting mismatch")
        excluded_ids.add(record["sample_id"])
        excluded_trials.add(identity)
    if excluded_trials != set(exclusions):
        raise ValueError("Missing configured exclusion accounting")
    return populations


def _restore_metadata(metadata):
    metadata = deepcopy(metadata)
    resampling = metadata["alignment"]["resampling"]
    for name in ("source_times", "interpolation_weights"):
        resampling[name] = np.asarray(resampling[name], dtype=np.float64)
    for name in ("left_source_indices", "right_source_indices"):
        resampling[name] = np.asarray(resampling[name], dtype=np.int64)
    return metadata


def _load_dataset(path, expected_generation_id):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "generation_id"}
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "training_dataset"
            or manifest.get("complete") is not True
            or fingerprint(payload) != manifest.get("generation_id")
            or expected_generation_id is not None and expected_generation_id != manifest["generation_id"]):
        raise ValueError("Incomplete, incompatible, or mismatched dataset generation")
    entries = manifest["samples"]
    unit_paths = {eid: f"units/{_session(eid)}.parquet" for eid in manifest["sessions"]}
    expected_files = set(unit_paths.values()) | {f"samples/{i:06d}.npz" for i in range(len(entries))}
    if set(manifest["files"]) != expected_files or not entries:
        raise ValueError("Dataset file accounting mismatch")
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or file_hash(target) != digest:
            raise ValueError(f"Dataset payload hash mismatch: {name}")
    units = {eid: pd.read_parquet(path / name) for eid, name in unit_paths.items()}
    splits = {name: [] for name in SPLITS}
    for index, entry in enumerate(entries):
        with np.load(path / f"samples/{index:06d}.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(ARRAY_FIELDS):
                raise ValueError("Unsupported dataset array schema")
            sample = TrainingSample(**{name: entry[name] for name in SCALAR_FIELDS},
                **{name: arrays[name] for name in ARRAY_FIELDS},
                neuron_identity=units[entry["session_id"]].copy(deep=True),
                metadata=_restore_metadata(entry["metadata"]))
        splits[sample.split].append(sample)
    dataset = DatasetSplits(*(tuple(splits[name]) for name in SPLITS), manifest["dataset_metadata"])
    populations = _validate_dataset(dataset)
    if manifest["sessions"] != sorted(populations):
        raise ValueError("Dataset session accounting mismatch")
    return DatasetGeneration(manifest["generation_id"], path, dataset, manifest)


def load_dataset(path, *, expected_generation_id=None):
    """Load one verified standalone generation without reopening alignment inputs."""
    try:
        return _load_dataset(path, expected_generation_id)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f"Malformed dataset generation: {error}") from error


def publish_dataset(dataset, output_dir):
    """Publish only after payload verification and a complete consumer readback."""
    populations = _validate_dataset(dataset)
    implementation = _implementation()
    root = Path(output_dir).resolve() / "generations"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=root) as temporary:
        staging = Path(temporary)
        (staging / "units").mkdir()
        (staging / "samples").mkdir()
        for eid, (units, _) in sorted(populations.items()):
            units.to_parquet(staging / "units" / f"{eid}.parquet", index=True)
        entries = []
        for split in SPLITS:
            for sample in getattr(dataset, split):
                np.savez(staging / "samples" / f"{len(entries):06d}.npz",
                         **{name: getattr(sample, name) for name in ARRAY_FIELDS})
                entries.append(dict(**{name: getattr(sample, name) for name in SCALAR_FIELDS},
                                    metadata=_json(sample.metadata)))
        manifest = dict(schema_version=1, kind="training_dataset", complete=True,
                        sessions=sorted(populations), samples=entries,
                        dataset_metadata=_json(dataset.metadata), implementation=implementation,
                        files={file.relative_to(staging).as_posix(): file_hash(file)
                               for file in sorted(staging.rglob("*")) if file.is_file()})
        manifest["generation_id"] = fingerprint(manifest)
        write_json(staging / "manifest.json", manifest)
        loaded = load_dataset(staging, expected_generation_id=manifest["generation_id"])
        if _implementation() != implementation:
            raise ValueError("Dataset implementation changed during publication")
        destination = root / loaded.generation_id
        if destination.exists():
            raise FileExistsError(f"Dataset generation already exists: {destination}; load it explicitly")
        staging.rename(destination)
    return replace(loaded, path=destination)


def generate_dataset(sources, output_dir, *, split_config, sample_config=None):
    """Load selected alignment inputs, package and split samples, and publish."""
    samples = load_samples(sources, config=sample_config)
    return publish_dataset(assign_splits(samples, config=split_config), output_dir)
