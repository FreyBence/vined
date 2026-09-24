"""Inventory cached session evidence without downloading or changing research data.

Run from the checkout: python src/inventory_visual_evidence.py --output PATH
This records local observations, not current remote availability or renderer fidelity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from utils.paths import REPO_ROOT, dataset_dir


RAW_FIELDS = (
    "trial_num", "stim_pos_init", "stim_contrast", "stim_freq", "stim_angle",
    "stim_gain", "stim_sigma", "stim_phase", "bns_ts",
)


def category(name):
    name = name.lower()
    for label, words in (
        ("raw_task", ("taskdata", "tasksettings", "encoder", "stimposition", "syncsquare")),
        ("display_calibration", ("calibration", "gamma", "screen", "display")),
        ("wheel", ("wheel.",)),
        ("trials", ("trials.",)),
        ("synchronization", ("sync.",)),
    ):
        if any(word in name for word in words):
            return label
    return None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(path, session_path):
    relative = path.relative_to(session_path)
    revision_parts = [part for part in relative.parts[:-1] if part.startswith("#") and part.endswith("#")]
    record = {
        "path": relative.as_posix(), "category": category(path.name),
        "collection": "/".join(part for part in relative.parts[:-1] if part not in revision_parts),
        "revision_directory": revision_parts, "size_bytes": path.stat().st_size,
        "sha256": sha256(path), "availability": "available",
        "content_status": "uninspected",
    }
    try:
        if path.suffix == ".npy":
            values = np.load(path, allow_pickle=False)
            record.update(shape=list(values.shape), dtype=str(values.dtype), content_status="inspected")
            if np.issubdtype(values.dtype, np.number):
                finite = np.isfinite(values)
                record["nonfinite_count"] = int((~finite).sum())
                if finite.any():
                    record["finite_range"] = [float(values[finite].min()), float(values[finite].max())]
                if "wheel.timestamps" in path.name and values.ndim == 1:
                    delta = np.diff(values)
                    record.update(duplicate_steps=int((delta == 0).sum()), backward_steps=int((delta < 0).sum()))
        elif path.name.endswith("trials.table.pqt"):
            frame = pd.read_parquet(path)
            record.update(rows=len(frame), columns=list(frame.columns), content_status="inspected")
            record["nonfinite_by_column"] = {
                name: int((~np.isfinite(frame[name].to_numpy())).sum())
                for name in frame.select_dtypes(include="number").columns
            }
            record["values"] = {
                name: sorted(float(value) for value in frame[name].dropna().unique())
                for name in ("contrastLeft", "contrastRight", "feedbackType") if name in frame
            }
    except (OSError, ValueError, ImportError, TypeError) as exc:
        record["inspection_error"] = str(exc)
    return record


def candidate_summary(session_path, local):
    """Summarize observed timing coverage; this is not rendering validation."""
    tables = [item for item in local if item["path"].endswith("trials.table.pqt")]
    wheel = session_path / "alf" / "_ibl_wheel.timestamps.npy"
    position = session_path / "alf" / "_ibl_wheel.position.npy"
    if len(tables) != 1 or not wheel.is_file() or not position.is_file():
        return {"availability": "uninspected", "reason": "Requires one trial table and both local wheel arrays"}
    table = session_path / tables[0]["path"]
    offset = table.parent / "_ibl_trials.stimOff_times.npy"
    if not offset.is_file():
        return {"availability": "uninspected", "reason": "No offsets beside the selected trial table"}
    try:
        frame = pd.read_parquet(table)
        times = np.load(wheel, allow_pickle=False)
        positions = np.load(position, allow_pickle=False)
        off = np.load(offset, allow_pickle=False)
        on = frame.stimOn_times.to_numpy()
        if times.ndim != 1 or len(times) < 2 or positions.shape != times.shape or off.shape != on.shape:
            raise ValueError("Incompatible trial/offset/wheel shapes")
        covered = np.isfinite(on) & np.isfinite(off) & (off > on) & (on >= times[0]) & (off <= times[-1])
        cases = {"left": frame.contrastLeft.notna(), "right": frame.contrastRight.notna(),
                 "zero_contrast": (frame.contrastLeft == 0) | (frame.contrastRight == 0),
                 "correct": frame.feedbackType == 1, "error": frame.feedbackType == -1}
        return {
            "availability": "available", "trial_count": len(frame),
            "onset_offset_within_wheel_range": int(covered.sum()),
            "uncovered_original_row_ids": np.flatnonzero(~covered).tolist(),
            "wheel_samples": len(times), "wheel_all_finite": bool(np.isfinite(times).all() and np.isfinite(positions).all()),
            "duplicate_wheel_steps": int((np.diff(times) == 0).sum()),
            "backward_wheel_steps": int((np.diff(times) < 0).sum()),
            "example_original_row_ids": {name: np.flatnonzero(covered & mask)[:3].tolist() for name, mask in cases.items()},
            "limitations": "Range coverage only; raw trial numbering, clock mapping, movement direction, freeze and renderer fidelity unverified",
        }
    except (OSError, ValueError, KeyError, AttributeError, ImportError) as exc:
        return {"availability": "uninspected", "inspection_error": str(exc)}


def inventory(cache_root):
    lists = {name: (REPO_ROOT / "data" / name).read_text().split()
             for name in ("eids.txt", "train_eids.txt", "test_eids.txt")}
    eids = list(dict.fromkeys(eid for entries in lists.values() for eid in entries))
    snapshots = {eid: [] for eid in eids}
    errors = []
    for path in sorted((cache_root / ".rest").glob("*")):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append({"path": path.name, "error": str(exc)})
            continue
        if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
            continue
        session = payload[0]
        eid = session.get("id")
        if eid in snapshots and "data_dataset_session_related" in session:
            snapshots[eid].append((path, session, payload[1] if len(payload) > 1 else None))

    sessions = []
    for eid in eids:
        record = {"eid": eid, "lists": [name for name, ids in lists.items() if eid in ids],
                  "remote_current_availability": "uninspected"}
        candidates = snapshots[eid]
        if not candidates:
            record["metadata_availability"] = "uninspected"
            record["reason"] = "No cached session response found; session path not resolved."
            sessions.append(record)
            continue
        # ONE's second cache element is an expiry, not a retrieval timestamp.
        path, session, expiry = max(candidates, key=lambda item: (str(item[2] or ""), item[0].name))
        session_relative = Path(session["lab"]) / "Subjects" / session["subject"] / session["start_time"][:10] / f'{session["number"]:03d}'
        session_path = cache_root / session_relative
        record.update(
            metadata_availability="available", subject=session["subject"],
            task_protocol=session.get("task_protocol"), session_path=session_relative.as_posix(),
            session_url=session["url"], metadata_path=path.relative_to(cache_root).as_posix(),
            metadata_sha256=sha256(path), cache_expiry=expiry,
            metadata_snapshot_count=len(candidates),
            metadata_selection="greatest expiry, then filename; expiry is not acquisition time",
        )
        record["catalog_datasets"] = [
            {**{key: data.get(key) for key in (
                "id", "name", "collection", "revision", "default_revision", "data_url", "url", "hash", "version", "qc")},
             "category": category(data["name"]), "payload_availability": "uninspected"}
            for data in session["data_dataset_session_related"] if category(data["name"])
        ]
        local = [inspect_file(file, session_path) for file in sorted(session_path.rglob("*"))
                 if file.is_file() and category(file.name)]
        record["local_datasets"] = local
        record["candidate_summary"] = candidate_summary(session_path, local)
        record["local_category_availability"] = {
            label: ("available" if any(item["category"] == label for item in local)
                    else "unavailable" if session_path.is_dir() else "uninspected")
            for label in ("raw_task", "display_calibration", "wheel", "trials", "synchronization")
        }
        record["raw_fields"] = {
            field: {"availability": "uninspected" if any(item["category"] == "raw_task" for item in local)
                    else "unavailable" if session_path.is_dir() else "uninspected",
                    "scope": "local raw task files only; remote sources uninspected",
                    "units_and_clock": "unverified"}
            for field in RAW_FIELDS
        }
        record["session_renderer_revision"] = "unverified; protocol string does not identify workflow/package commit"
        record["clock_and_units"] = {
            "alf_trials_and_wheel": "IBL convention: synchronized session seconds; wheel position radians. Session extraction mapping not revalidated.",
            "raw_bonsai_to_alf": "uninspected; no verified clock transform",
            "display_frame_times": "uninspected; ALF stimulus events do not establish a complete frame sequence",
            "stimulus_parameters": "raw units unavailable; do not infer from protocol defaults",
        }
        sessions.append(record)
    environment = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for package in ("pyarrow", "fastparquet"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    return {
        "schema_version": 1, "task": "VR04", "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Local files and cached Alyx session catalogs only; no remote requests",
        "environment": environment,
        "cache_root": str(cache_root), "configured_lists": lists,
        "list_sha256": {name: sha256(REPO_ROOT / "data" / name) for name in lists},
        "script_sha256": sha256(Path(__file__)), "cache_read_errors": errors, "sessions": sessions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=dataset_dir())
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing output is preserved")
    args = parser.parse_args()
    result = inventory(args.cache_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Inventoried {len(result['sessions'])} sessions: {args.output}")


if __name__ == "__main__":
    main()
