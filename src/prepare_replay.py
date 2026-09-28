"""Resolve replay inputs or generate lossless replay artifacts from their stream."""
import argparse
import json
from pathlib import Path
import sys
from contextlib import nullcontext
from tqdm import tqdm

from session_data import SessionAccess
from utils.stimulus_parameters import _unique_object
from utils.sessions import select_sessions
from visual_replay import ReplayStream, resolve_reconstruction_plan, write_replay


def main(argv=None, *, progress_position=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Session-specific reconstruction configuration JSON")
    parser.add_argument("--output", type=Path, required=True, help="New definition JSON file, or new generation directory with --generate")
    parser.add_argument("--generate", action="store_true", help="Generate lossless images, observation records, and completion manifest")
    parser.add_argument("--image-space", choices=("mouse_view", "display"), default="mouse_view")
    parser.add_argument("--video", action="store_true", help="Also encode complete trials as MP4 with source-observation mappings")
    parser.add_argument("--access-policy", choices=("local-only", "remote-allowed"), default="local-only")
    parser.add_argument("--force-reload", action="store_true", help="Refresh remote metadata and re-download requested source datasets")
    parser.add_argument("--stimulus-parameters", type=Path)
    parser.add_argument("--eid", help="Session for a reusable configuration whose eid is null")
    parser.add_argument("--eids-file", type=Path, help="Select the first EID when configuration eid is null")
    trials = parser.add_mutually_exclusive_group()
    trials.add_argument("--trial-id", type=int, action="append")
    trials.add_argument("--n-trials", type=int, help="First N original trials; omit for all trials")
    parser.add_argument("--revision")
    parser.add_argument("--wheel-revision")
    parser.add_argument("--settings-collection", default="raw_behavior_data")
    parser.add_argument("--settings-revision")
    args = parser.parse_args(argv)
    report = print if progress_position is None else tqdm.write
    if not args.generate and (args.video or args.image_space != "mouse_view"):
        parser.error("--video and --image-space require --generate")
    if args.force_reload and args.access_policy != "remote-allowed":
        parser.error("--force-reload requires --access-policy remote-allowed")
    config = json.loads(args.config.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if args.eid and args.eids_file:
        parser.error("Choose --eid or --eids-file")
    if args.n_trials is not None and args.n_trials <= 0:
        parser.error("--n-trials must be positive")
    if config.get("eid") is None:
        config["eid"] = select_sessions(args.eid, args.eids_file, n_sessions=1)[0]
    elif args.eid or args.eids_file:
        selected = select_sessions(args.eid, args.eids_file, n_sessions=1)[0]
        if selected != config["eid"]:
            parser.error("Configuration belongs to another EID; use a reusable configuration with eid: null")
    trial_ids = list(range(args.n_trials)) if args.n_trials is not None else args.trial_id
    report(f"Replay session: {config['eid']}; trials: {trial_ids if trial_ids is not None else 'all'}")
    plan = resolve_reconstruction_plan(
        SessionAccess(policy=args.access_policy, force_reload=args.force_reload), config["eid"], config,
        stimulus_parameters=args.stimulus_parameters, trial_ids=trial_ids,
        revision=args.revision, wheel_revision=args.wheel_revision,
        settings_collection=args.settings_collection, settings_revision=args.settings_revision)
    if args.generate:
        progress = (nullcontext(None) if progress_position is None else
                    tqdm(total=len(plan.definition["requested_trial_ids"]),
                         desc=f"Trials {config['eid'][:8]}", unit="trial",
                         position=progress_position, leave=False, dynamic_ncols=True))
        with progress as bar:
            manifest = write_replay(
                ReplayStream(plan, image_space=args.image_space), args.output, video=args.video,
                on_trial_published=None if bar is None else lambda outcome: bar.update(1))
        completion = manifest["completion"]
        video_failed = any(item["video"]["status"] == "failed" for item in manifest["trials"])
        report(json.dumps(dict(output=str(args.output), generation_id=completion["generation_id"],
                              reconstruction_status=completion["reconstruction_status"],
                              accounting_complete=completion["accounting_complete"],
                              image_count=completion["image_count"], video_failed=video_failed)))
        return 3 if video_failed else 0 if completion["reconstruction_status"] == "success" else 2
    encoded = json.dumps(plan.definition, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
    counts = {status: sum(row["status"] == status for row in plan.definition["trials"])
              for status in ("prepared", "unavailable", "invalid")}
    print(json.dumps(dict(output=str(args.output), trial_inputs=counts)))


if __name__ == "__main__":
    sys.exit(main())
