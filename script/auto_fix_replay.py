"""Retry audited trials in place under the replay root's EID folders."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_replay import main as prepare_session
from utils.paths import replay_dir


def non_retryable_reason(status, reason):
    """Identify source reconstruction limits, keeping artifact repairs retryable."""
    if "image files:" in reason or "missing observations.jsonl" in reason:
        return None
    if status in {"unavailable", "invalid"}:
        return reason or f"Source reconstruction is {status}"
    if status == "partial" and reason == "Requested domain contains unavailable reconstruction":
        return reason
    # Missing outcomes, rendering/storage failures, and unknown reasons may be
    # recoverable, so a source-gap filter must not discard them speculatively.
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_file", nargs="?", type=Path, default=ROOT / "output/audit-replay.txt")
    parser.add_argument("--config", type=Path, default=ROOT / "data/replay-config.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--projection", choices=("on", "off"), default="on")
    parser.add_argument("--access-policy", choices=("local-only", "remote-allowed"), default="remote-allowed")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--retry-all", action="store_true", help="Also retry source gaps/invalid inputs after changing data, configuration, or replay code")
    parser.add_argument("--dry-run", action="store_true", help="List selected trials without generating")
    args = parser.parse_args()
    if args.force_reload and args.access_policy != "remote-allowed":
        parser.error("--force-reload requires remote-allowed access")
    selected = defaultdict(set)
    skipped = {}
    try:
        lines = args.audit_file.read_text(encoding="utf-8").splitlines()
        header = "eid\ttrial_id\tcategory\timage_files\toutcome_status\treason\tgeneration"
        start = lines.index(header)
        for line in lines[start + 1:]:
            if not line.strip():
                break
            if line.startswith("No replay definitions"):
                break
            fields = line.split("\t")
            if len(fields) != 7 or fields[2] not in {"missing", "incomplete"}:
                raise ValueError(f"Invalid audit row: {line}")
            eid, trial_id = str(uuid.UUID(fields[0])), int(fields[1])
            if trial_id < 0:
                raise ValueError("Trial numbers must be nonnegative")
            identity = (eid, trial_id)
            reason = None if args.retry_all else non_retryable_reason(fields[4], fields[5])
            if reason is not None:
                if trial_id not in selected.get(eid, ()):
                    skipped[identity] = reason
                continue
            selected[eid].add(trial_id)
            skipped.pop(identity, None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Retry selection: {sum(map(len, selected.values()))} trials in {len(selected)} sessions")
    if skipped:
        print(f"Skipped {len(skipped)} trials with source-data/coverage limits (use --retry-all to include them):")
        for reason, count in sorted(Counter(skipped.values()).items()):
            print(f"  {count}: {reason}")
    if args.dry_run:
        for eid, ids in sorted(selected.items()):
            print(f"{eid}: {','.join(map(str, sorted(ids)))}")
        return 0
    if not selected:
        return 0
    output = args.output or replay_dir()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Retry output: {output}", flush=True)
    status = 0
    for eid, ids in sorted(selected.items()):
        command = ["--config", str(args.config), "--eid", eid, "--generate", "--overwrite",
                   "--output", str(output / eid), "--access-policy", args.access_policy,
                   "--image-space", "mouse_view" if args.projection == "on" else "display"]
        for trial_id in sorted(ids):
            command.extend(["--trial-id", str(trial_id)])
        if args.force_reload:
            command.append("--force-reload")
        try:
            code = prepare_session(command, progress_position=0)
        except Exception as exc:
            print(f"{eid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            code = 1
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code in (130, 143):
                raise
        status = max(status, code)
        print(f"{eid}: retry status {code}", flush=True)
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
