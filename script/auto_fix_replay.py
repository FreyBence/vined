"""Retry audited trials in place under the replay root's EID folders."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_replay import main as prepare_session
from utils.paths import replay_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_file", nargs="?", type=Path, default=ROOT / "output/audit-replay.txt")
    parser.add_argument("--config", type=Path, default=ROOT / "data/replay-config.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--projection", choices=("on", "off"), default="on")
    parser.add_argument("--access-policy", choices=("local-only", "remote-allowed"), default="remote-allowed")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--retry-all", action="store_true", help="Also retry Fixable=False rows after changing data, configuration, or replay code")
    parser.add_argument("--dry-run", action="store_true", help="List selected trials without generating")
    args = parser.parse_args()
    if args.force_reload and args.access_policy != "remote-allowed":
        parser.error("--force-reload requires remote-allowed access")
    selected = defaultdict(set)
    skipped = {}
    recovered = {}
    output = args.output or replay_dir()
    try:
        lines = args.audit_file.read_text(encoding="utf-8").splitlines()
        header = "eid\ttrial_id\tcategory\timage_files\toutcome_status\treason\tgeneration\tFixable"
        if header not in lines:
            raise ValueError("Audit must include the boolean Fixable column; rerun script/audit_replay.py")
        start = lines.index(header)
        for line in lines[start + 1:]:
            if not line.strip():
                break
            if line.startswith("No replay definitions"):
                break
            fields = line.split("\t")
            if (len(fields) != 8 or fields[2] not in {"missing", "incomplete"}
                    or fields[7].lower() not in {"true", "false"}):
                raise ValueError(f"Invalid audit row: {line}")
            eid, trial_id = str(uuid.UUID(fields[0])), int(fields[1])
            if trial_id < 0:
                raise ValueError("Trial numbers must be nonnegative")
            identity = (eid, trial_id)
            if fields[7].lower() == "false" and not args.retry_all:
                if trial_id not in selected.get(eid, ()):
                    skipped[identity] = fields[5] or f"Outcome is {fields[4]}"
                continue
            selected[eid].add(trial_id)
            skipped.pop(identity, None)
        for eid, ids in selected.items():
            generation = output / eid
            if (generation / "manifest.json").exists() or not (generation / "definition.json").is_file():
                continue
            definition = json.loads((generation / "definition.json").read_text(encoding="utf-8"))
            inputs = definition.get("inputs", {})
            requested = inputs.get("requested_trial_ids")
            if (definition.get("kind") != "replay_stream_definition"
                    or inputs.get("eid") != eid or not isinstance(requested, list)
                    or not requested or any(type(i) is not int or i < 0 for i in requested)):
                raise ValueError(f"Invalid unfinished replay definition: {generation}")
            # Without a completion manifest, unselected trials cannot be verified
            # for retention. Regenerate the saved request before replacing it.
            ids.update(requested)
            recovered[eid] = len(ids)
            for trial_id in ids:
                skipped.pop((eid, trial_id), None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Retry selection: {sum(map(len, selected.values()))} trials in {len(selected)} sessions")
    for eid, count in sorted(recovered.items()):
        print(f"{eid}: missing manifest; regenerating saved request ({count} trials)")
    if skipped:
        print(f"Skipped {len(skipped)} Fixable=False trials (use --retry-all to include them):")
        for reason, count in sorted(Counter(skipped.values()).items()):
            print(f"  {count}: {reason}")
    if args.dry_run:
        for eid, ids in sorted(selected.items()):
            print(f"{eid}: {','.join(map(str, sorted(ids)))}")
        return 0
    if not selected:
        return 0
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
