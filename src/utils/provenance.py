"""Stable identities and provenance shared by preparation and cache consumers."""
import hashlib
import json
from importlib.metadata import distributions
from pathlib import Path

from utils.paths import REPO_ROOT


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes(*paths):
    return {path: file_hash(REPO_ROOT / path) for path in paths}


def package_versions():
    return {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def split_trials(eid, trial_ids, intervals, seed):
    """Assign overlapping half-open intervals together, independent of session order.

    Randomize connected interval groups; ratios are approximate when groups have
    multiple trials. This prevents shared physical observations crossing splits.
    """
    import numpy as np
    intervals = np.asarray(intervals, dtype=float)
    trial_ids = np.asarray(trial_ids)
    if (intervals.shape != (len(trial_ids), 2) or not np.isfinite(intervals).all()
            or np.any(intervals[:, 1] <= intervals[:, 0]) or not len(trial_ids)
            or len(np.unique(trial_ids)) != len(trial_ids)):
        raise ValueError("Splitting requires unique trials with finite positive intervals")
    groups, end = [], -np.inf
    for row in sorted(range(len(trial_ids)), key=lambda i: (intervals[i, 0], int(trial_ids[i]))):
        if intervals[row, 0] >= end:
            groups.append([])
        groups[-1].append(row)
        end = max(end, intervals[row, 1])
    if len(groups) < 3:
        raise ValueError("Need at least three non-overlapping interval groups for train/val/test")
    session_seed = int(fingerprint(dict(eid=eid, seed=int(seed)))[:16], 16)
    rng = np.random.default_rng(session_seed)
    order = rng.permutation(len(groups))
    a = min(max(1, int(.7 * len(groups))), len(groups)-2)
    b = min(max(a+1, int(.8 * len(groups))), len(groups)-1)
    splits = {name: np.asarray([row for g in indices for row in groups[g]], dtype=np.int64)
              for name, indices in zip(("train", "val", "test"), (order[:a], order[a:b], order[b:]))}
    metadata = dict(seed=int(seed), session_seed=session_seed,
                    algorithm="overlap-components-pcg64-v1", requested_ratios=[.7, .1, .2],
                    interval_groups=len(groups), cross_split_interval_overlap=False,
                    block_independence="not enforced", subject_independence="not implied by trial splits",
                    memberships={name: trial_ids[rows].tolist() for name, rows in splits.items()})
    return splits, metadata
