"""Shared ordered session selection and preparation-stage reporting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import UUID

from utils.paths import REPO_ROOT


def add_session_arguments(parser):
    parser.add_argument("--eid", help="Process this EID instead of a manifest")
    parser.add_argument("--eids-file", "--eids_file", type=Path,
                        default=Path(os.environ.get("VINED_EIDS_FILE", REPO_ROOT / "data/eids.txt")))
    parser.add_argument("--n-sessions", "--n_sessions", type=int, default=None,
                        help="Process the first N manifest entries (default: all)")


def select_sessions(eid=None, eids_file=None, n_sessions=None):
    if n_sessions is not None and n_sessions <= 0:
        raise ValueError("n_sessions must be positive")
    if eid is not None:
        if n_sessions not in (None, 1):
            raise ValueError("--eid cannot be combined with n_sessions other than 1")
        values = [eid]
    else:
        path = Path(eids_file or os.environ.get("VINED_EIDS_FILE", REPO_ROOT / "data/eids.txt"))
        if not path.is_absolute():
            path = REPO_ROOT / path
        values = [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
        values = [value for value in values if value]
    values = [str(UUID(value)) for value in values]
    if not values or len(set(values)) != len(values):
        raise ValueError("Session selection must be nonempty and contain unique EIDs")
    if n_sessions is not None and n_sessions > len(values):
        raise ValueError(f"Requested {n_sessions} sessions but manifest contains {len(values)}")
    return values[:n_sessions]


class SkipSession(Exception):
    """An explicitly unavailable session, not malformed data or an IO failure."""


def run_sessions(eids, operation, stage):
    """Report every requested EID; continue ordinary failures, propagate interrupts."""
    eids = list(eids)
    report = dict(stage=stage, requested=list(eids), completed=[], skipped={}, failed={})
    print(f"{stage}: requested {len(eids)} sessions: {', '.join(eids)}", flush=True)
    try:
        for eid in eids:
            try:
                operation(eid)
            except SkipSession as exc:
                report["skipped"][eid] = str(exc)
                print(f"{stage}: skipped {eid}: {exc}", flush=True)
            except Exception as exc:
                report["failed"][eid] = f"{type(exc).__name__}: {exc}"
                print(f"{stage}: failed {eid}: {exc}", file=sys.stderr, flush=True)
            else:
                report["completed"].append(eid)
                print(f"{stage}: completed {eid}", flush=True)
    finally:
        finished = set(report["completed"]) | report["skipped"].keys() | report["failed"].keys()
        report["not_processed"] = [eid for eid in eids if eid not in finished]
        print(json.dumps(report, indent=2), flush=True)
    if report["failed"]:
        raise RuntimeError(f"{stage}: {len(report['failed'])} session(s) failed; see report above")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    parser.add_argument("--cache", action="store_true", help="Run the cache wrapper for each selected EID")
    args = parser.parse_args()
    eids = select_sessions(args.eid, args.eids_file, args.n_sessions)
    if args.cache:
        def cache(eid):
            subprocess.run(["bash", str(REPO_ROOT / "script/create_dataset.sh"), "1", eid],
                           cwd=REPO_ROOT, check=True)
        run_sessions(eids, cache, "cache")
    else:
        print("\n".join(eids))
