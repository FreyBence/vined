"""Immutable cache generations, selected by an atomically published EID manifest."""
import json
from importlib.metadata import version
from pathlib import Path

from utils.provenance import file_hash, fingerprint, source_hashes

CACHE_SCHEMA = 1


def preprocessing_sources():
    return source_hashes("src/loader/base.py", "src/utils/dataset_utils.py",
                         "src/utils/cache_manifest.py", "src/create_dataset.py",
                         "src/utils/provenance.py")


def aligned_hashes(directory):
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): file_hash(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def preprocessing_packages():
    return {name: version(name) for name in ("numpy", "scipy", "torch", "datasets", "pyarrow")}


def cache_records(root, mode, eids, options=None):
    root = Path(root).resolve()
    if mode not in {"train", "val", "test"} or not eids:
        raise ValueError("Cache loading requires a split and explicit session IDs")
    records = []
    for eid in sorted(set(eids)):
        path = root / f"{eid}.manifest.json"
        if not path.exists():
            raise ValueError(f"Missing versioned cache manifest for {eid}; rebuild caches")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        stored_hash = manifest.pop("fingerprint", None)
        if (manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("eid") != eid
                or fingerprint(manifest) != stored_hash):
            raise ValueError(f"Invalid cache manifest: {path}")
        if manifest["sources"] != preprocessing_sources():
            raise ValueError("Cache preprocessing code changed; rebuild caches")
        if manifest["preprocessing_packages"] != preprocessing_packages():
            raise ValueError("Cache preprocessing package versions changed; rebuild caches")
        if options is not None and manifest["options"] != options:
            raise ValueError(f"Cache preprocessing options differ for {eid}; rebuild with requested settings")
        aligned = root.parent / f"{eid}_aligned" / "provenance.json"
        if not aligned.exists() or file_hash(aligned) != manifest["aligned_provenance_sha256"]:
            raise ValueError(f"Aligned provenance changed or is missing for {eid}; rebuild caches")
        if aligned_hashes(aligned.parent) != manifest["aligned_files"]:
            raise ValueError(f"Aligned dataset files changed for {eid}; rebuild caches")
        seen = set()
        for record in manifest["samples"]:
            identity = record["trial_id"]
            if identity in seen:
                raise ValueError("Duplicate sample identity in cache manifest")
            seen.add(identity)
            if record["split"] != mode:
                continue
            sample = (root / record["file"]).resolve()
            if root not in sample.parents or not sample.is_file():
                raise ValueError("Missing or out-of-root cache file")
            records.append(dict(record, path=str(sample), eid=eid,
                                provenance_id=manifest["provenance_id"]))
    return records
