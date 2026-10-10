"""Publish unsplit alignment from explicit inputs or session-selected preparation outputs."""

import argparse
import json
import sys
from pathlib import Path

from alignment import generate_alignment
from visual_features import FeatureArtifactReader
from utils.paths import output_dir, visual_dir
from utils.progress import configure_progress
from utils.sessions import add_session_arguments, select_sessions, run_sessions


def resolve_neural_generation(root, eid, expected_id=None):
    """Resolve exactly one completed neural generation for the selected session."""
    root = Path(root)
    session = root if root.name == eid else root / eid
    candidates = ([root] if (root / "manifest.json").is_file() else
                  sorted(path.parent for path in session.glob("*/manifest.json")
                         if not path.parent.name.startswith(".")))
    completed = []
    for path in candidates:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("complete") is True and manifest.get("eid") == eid:
            if expected_id is None or manifest.get("generation_id") == expected_id:
                completed.append(path)
    if not completed:
        raise FileNotFoundError(f"No matching completed neural generation for {eid} under {root}")
    if len(completed) != 1:
        raise ValueError(f"Multiple neural generations for {eid}; select --neural-generation "
                         "or --expected-neural-generation-id: " + ", ".join(map(str, completed)))
    return completed[0]


def main(argv=None):
    configure_progress()
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    parser.set_defaults(eids_file=None)
    parser.add_argument("--neural-root", type=Path, default=output_dir() / "neural",
                        help="Neural output root (default: VINED_OUTPUT_DIR/neural)")
    parser.add_argument("--visual-root", type=Path, default=visual_dir(),
                        help="Visual feature output folder (default: VINED_VISUAL_DIR)")
    parser.add_argument("--neural-generation", type=Path)
    parser.add_argument("--visual-features", type=Path)
    parser.add_argument("--output-dir", type=Path, default=output_dir() / "alignment")
    parser.add_argument("--request-trial-ids", type=Path, help="JSON object mapping absolute request IDs to original trial IDs")
    parser.add_argument("--expected-neural-generation-id")
    parser.add_argument("--expected-visual-generation-id")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel trial resampling/compression threads (default: 4; 1 is sequential)")
    parser.add_argument("--strict-trials", action="store_true",
                        help="Fail on the first unusable trial instead of recording exclusions")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.eid is not None and args.eids_file is not None:
        parser.error("Choose --eid or --eids-file")

    def operation(eid):
        neural = args.neural_generation
        if neural is None:
            neural = resolve_neural_generation(args.neural_root, eid, args.expected_neural_generation_id)
        visual = args.visual_features if args.visual_features is not None else args.visual_root / f"{eid}_visual_clip.npz"
        if eid is not None:
            manifest = json.loads((neural / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("eid") != eid:
                raise ValueError(f"Neural generation does not belong to selected session {eid}")
            with FeatureArtifactReader(visual, eid=eid):
                pass
        print(f"Neural source: {neural}", file=sys.stderr, flush=True)
        print(f"Visual source: {visual}", file=sys.stderr, flush=True)
        mapping = None
        if args.request_trial_ids:
            mapping = json.loads(args.request_trial_ids.read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                raise ValueError("Request/trial mapping must be a JSON object")
            if any(str(int(key)) != key for key in mapping):
                raise ValueError("Request ID keys must be canonical decimal integers")
            mapping = {int(key): value for key, value in mapping.items()}
        generation = generate_alignment(neural, visual, args.output_dir,
            workers=args.workers, skip_invalid=not args.strict_trials, request_trial_ids=mapping, expected_neural_generation_id=args.expected_neural_generation_id,
            expected_visual_generation_id=args.expected_visual_generation_id)
        print(json.dumps(dict(generation_id=generation.generation_id, path=str(generation.path),
                             session_id=generation.manifest["eid"],
                             trial_ids=[trial.trial_id for trial in generation.trials],
                             requested_trial_count=len(generation.manifest["requested_trial_ids"]),
                             retained_trial_count=len(generation.trials),
                             excluded_trials=generation.manifest.get("excluded_trials", []))), flush=True)

    try:
        explicit_pair = args.neural_generation is not None and args.visual_features is not None
        if explicit_pair and args.eid is None and args.eids_file is None and args.n_sessions is None:
            operation(None)
        else:
            eids = select_sessions(args.eid, args.eids_file, args.n_sessions)
            if len(eids) > 1 and any(value is not None for value in (
                    args.neural_generation, args.visual_features, args.request_trial_ids,
                    args.expected_neural_generation_id, args.expected_visual_generation_id)):
                parser.error("Explicit input paths, generation IDs, and request mappings require one selected session")
            run_sessions(eids, operation, "alignment")
    except Exception as exc:
        parser.exit(1, f"Alignment failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
