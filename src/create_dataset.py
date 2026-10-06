"""Publish scientific training datasets from explicitly selected alignment generations."""

import argparse
import json
from pathlib import Path

from training_dataset import AlignmentSource, SampleConfig, SplitConfig, generate_dataset
from utils.paths import dataset_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-generation", required=True, type=Path, action="append",
                        help="One explicit alignment generation per session; repeat for multiple sessions")
    parser.add_argument("--expected-alignment-generation-id", action="append",
                        help="Expected IDs in the same order as alignment-generation arguments")
    parser.add_argument("--output-dir", type=Path, default=dataset_dir() / "training-dataset")
    parser.add_argument("--split-strategy", required=True, choices=("within_session", "session_held_out"))
    parser.add_argument("--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--session-assignments", type=Path, help="JSON object mapping sessions to train/val/test")
    parser.add_argument("--exclusions", type=Path, help="JSON list of session_id/trial_id/reason objects")
    parser.add_argument("--max-time-length", type=int)
    parser.add_argument("--max-neuron-count", type=int)
    args = parser.parse_args(argv)
    if args.expected_alignment_generation_id is not None and len(args.expected_alignment_generation_id) != len(args.alignment_generation):
        parser.error("Provide one expected ID per alignment-generation argument, or omit all expected IDs")
    if args.session_assignments is not None:
        if args.split_strategy != "session_held_out" or args.split_ratios is not None or args.split_seed is not None:
            parser.error("Session assignments require session_held_out without ratios or seed")
    elif args.split_ratios is None or args.split_seed is None:
        parser.error("Ratio-based splitting requires --split-ratios and --split-seed")
    try:
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
        expected = args.expected_alignment_generation_id or [None] * len(args.alignment_generation)
        generation = generate_dataset([AlignmentSource(path, identity) for path, identity in
                                      zip(args.alignment_generation, expected)], args.output_dir,
                                      split_config=split_config, sample_config=sample_config)
    except Exception as error:
        parser.exit(1, f"Dataset creation failed: {error}\n")
    print(json.dumps(dict(generation_id=generation.generation_id, path=str(generation.path),
                         sessions=generation.manifest["sessions"],
                         samples={name: len(getattr(generation.dataset, name)) for name in ("train", "val", "test")})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
