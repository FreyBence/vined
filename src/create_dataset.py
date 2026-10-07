"""Create one scientific dataset from EID-selected or explicit alignment generations."""

import argparse
import json
from pathlib import Path
from uuid import UUID

from training_dataset import AlignmentSource, SampleConfig, SplitConfig, generate_dataset
from utils.paths import dataset_dir, output_dir
from utils.sessions import add_session_arguments, select_sessions


def resolve_alignment(root, eid):
    """Select one published alignment; never choose between generation versions."""
    root = Path(root)
    if (root / "manifest.json").is_file():
        candidates = [root]
    else:
        session = root if root.name == eid else root / eid
        candidates = ([session] if (session / "manifest.json").is_file() else [])
        candidates.extend(sorted(path.parent for path in session.glob("*/manifest.json")
                                 if not path.parent.name.startswith(".")))
    if not candidates:
        raise FileNotFoundError(f"No published alignment for {eid} under {root}")
    if len(candidates) != 1:
        raise ValueError(f"Multiple alignment generations for {eid}; select an explicit "
                         "--alignment-generation instead: " + ", ".join(map(str, candidates)))
    path = candidates[0]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != "aligned_trials" or manifest.get("complete") is not True
            or str(UUID(manifest["eid"])) != eid):
        raise ValueError(f"Not a completed alignment for {eid}: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    parser.set_defaults(eids_file=None)
    parser.add_argument("--alignment-root", type=Path,
                        help="Prepared alignment root (default: VINED_OUTPUT_DIR/alignment); requires one generation per selected EID")
    parser.add_argument("--alignment-generation", type=Path, action="append",
                        help="One explicit alignment generation per session; repeat for multiple sessions")
    parser.add_argument("--expected-alignment-generation-id", action="append",
                        help="Expected IDs in explicit-path or selected-EID order")
    parser.add_argument("--output-dir", type=Path, default=dataset_dir() / "training-dataset")
    parser.add_argument("--split-strategy", required=True, choices=("within_session", "session_held_out"))
    parser.add_argument("--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--session-assignments", type=Path, help="JSON object mapping sessions to train/val/test")
    parser.add_argument("--exclusions", type=Path, help="JSON list of session_id/trial_id/reason objects")
    parser.add_argument("--max-time-length", type=int)
    parser.add_argument("--max-neuron-count", type=int)
    args = parser.parse_args(argv)
    if args.alignment_generation and any(value is not None for value in
            (args.eid, args.eids_file, args.n_sessions, args.alignment_root)):
        parser.error("Choose explicit --alignment-generation paths or EID selection with --alignment-root")
    if args.session_assignments is not None:
        if args.split_strategy != "session_held_out" or args.split_ratios is not None or args.split_seed is not None:
            parser.error("Session assignments require session_held_out without ratios or seed")
    elif args.split_ratios is None or args.split_seed is None:
        parser.error("Ratio-based splitting requires --split-ratios and --split-seed")
    try:
        paths = args.alignment_generation
        if paths is None:
            eids = select_sessions(args.eid, args.eids_file, args.n_sessions)
            root = args.alignment_root or output_dir() / "alignment"
            paths = [resolve_alignment(root, eid) for eid in eids]
            for eid, path in zip(eids, paths):
                print(f"Alignment source: {eid}: {path}", flush=True)
        if args.expected_alignment_generation_id is not None and len(args.expected_alignment_generation_id) != len(paths):
            parser.error("Provide one expected ID per selected alignment generation, or omit all expected IDs")
        assignments = json.loads(args.session_assignments.read_text(encoding="utf-8")) if args.session_assignments else None
        exclusions = {}
        if args.exclusions:
            records = json.loads(args.exclusions.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError("Exclusions must be a JSON list")
            for record in records:
                identity = (record["session_id"], record["trial_id"])
                if identity in exclusions:
                    raise ValueError("Duplicate exclusion identity")
                exclusions[identity] = record["reason"]
        split_config = SplitConfig(args.split_strategy, args.split_ratios, args.split_seed, assignments, exclusions)
        sample_config = SampleConfig(args.max_time_length, args.max_neuron_count)
        expected = args.expected_alignment_generation_id or [None] * len(paths)
        generation = generate_dataset([AlignmentSource(path, identity) for path, identity in
                                      zip(paths, expected)], args.output_dir,
                                      split_config=split_config, sample_config=sample_config)
    except Exception as error:
        parser.exit(1, f"Dataset creation failed: {error}\n")
    print(json.dumps(dict(generation_id=generation.generation_id, path=str(generation.path),
                         sessions=generation.manifest["sessions"],
                         samples={name: len(getattr(generation.dataset, name)) for name in ("train", "val", "test")})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
