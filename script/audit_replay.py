"""Report missing/incomplete trials using the newest saved definition per trial.

Run from any directory: python script/audit_replay.py [--replay-dir DIR]
Only trials requested by saved definitions are audited; no downloads or renders.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.paths import replay_dir


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_fixable(status, reason):
    """Identify potentially repairable artifacts, excluding known source limits."""
    if "image files:" in reason or "missing observations.jsonl" in reason:
        return True
    if status in {"unavailable", "invalid"}:
        return False
    return not (status == "partial"
                and reason == "Requested domain contains unavailable reconstruction")


def audit(root):
    latest = {}
    errors = []
    for directory, children, files in os.walk(root):
        # Frame directories can contain millions of files; discover sessions only.
        children[:] = [name for name in children if name != "trials"]
        if "definition.json" not in files:
            continue
        path = Path(directory) / "definition.json"
        try:
            inputs = read_json(path)["inputs"]
            eid, ids = inputs["eid"], inputs["requested_trial_ids"]
            if not isinstance(eid, str) or not isinstance(ids, list) or any(
                type(i) is not int or i < 0 for i in ids
            ):
                raise ValueError("Invalid EID or requested trial IDs")
            key = (path.stat().st_mtime_ns, str(path))
            for trial_id in ids:
                identity = (eid, trial_id)
                if identity not in latest or key > latest[identity][0]:
                    latest[identity] = (key, path.parent)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path}: {exc}")

    rows = []
    checked = 0
    sessions = {}
    for (eid, trial_id), (_, session) in latest.items():
        sessions.setdefault((eid, session), []).append(trial_id)
    for (eid, session), ids in sorted(sessions.items()):
        for trial_id in sorted(ids):
            checked += 1
            trial = session / "trials" / str(trial_id)
            try:
                outcome = read_json(trial / "outcome.json")
                if outcome["eid"] != eid or outcome["trial_id"] != trial_id:
                    raise ValueError("Outcome identity does not match definition")
                status = outcome["status"]
                expected = outcome["image_count"]
                images = trial / "images"
                actual = 0
                if images.is_dir():
                    with os.scandir(images) as entries:
                        actual = sum(entry.name.endswith(".npz") and entry.is_file()
                                     for entry in entries)
                reason = outcome.get("reason") or ""
                if actual != expected:
                    reason += f"; image files: {actual}, recorded: {expected}"
                if not (trial / "observations.jsonl").is_file():
                    reason += "; missing observations.jsonl"
                if status == "complete" and actual == expected and actual > 0 and not reason:
                    continue
                category = "missing" if actual == 0 else "incomplete"
                rows.append((eid, trial_id, category, actual, status, reason.strip("; "), session))
            except FileNotFoundError:
                rows.append((eid, trial_id, "missing", "?", "no_outcome",
                             "No published outcome; trial may be unattempted or still generating", session))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                rows.append((eid, trial_id, "incomplete", "?", "unreadable_outcome", str(exc), session))
    rows = [(*row, is_fixable(row[4], row[5])) for row in rows]
    return latest, checked, rows, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=replay_dir())
    args = parser.parse_args()
    root = args.replay_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Replay directory does not exist: {root}")
    latest, checked, rows, errors = audit(root)
    counts = Counter(row[2] for row in rows)
    report = [
        "Replay trial audit",
        f"Source: {root}",
        "Selection: newest definition.json modification time per (EID, trial ID); requested trials only.",
        "Incomplete includes partial reconstruction and missing image/metadata files.",
        "Snapshot only: active generations may still be writing. Image contents/hashes are not verified.",
        f"Sessions: {len({eid for eid, trial_id in latest})} | Trials checked: {checked} | Missing: {counts['missing']} | Incomplete: {counts['incomplete']}",
        "",
        "eid\ttrial_id\tcategory\timage_files\toutcome_status\treason\tgeneration\tFixable",
    ]
    report.extend("\t".join(str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")
                            for value in row) for row in rows)
    if not latest:
        report.append("No replay definitions found; no trials audited.")
    if errors:
        report.extend(["", "Discovery errors (audit coverage is incomplete):", *errors])
    destination = ROOT / "output" / "audit-replay.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report[5])
    print(f"Report: {destination}")
    return 1 if errors or not latest else 0


if __name__ == "__main__":
    raise SystemExit(main())
