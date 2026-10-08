"""Publish unsplit alignment from explicit prepared neural and visual generations."""

import argparse
import json
from pathlib import Path

from alignment import generate_alignment
from utils.paths import output_dir
from utils.progress import configure_progress


def main(argv=None):
    configure_progress()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-generation", required=True, type=Path)
    parser.add_argument("--visual-features", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=output_dir() / "alignment")
    parser.add_argument("--request-trial-ids", type=Path, help="JSON object mapping absolute request IDs to original trial IDs")
    parser.add_argument("--expected-neural-generation-id")
    parser.add_argument("--expected-visual-generation-id")
    args = parser.parse_args(argv)
    try:
        mapping = None
        if args.request_trial_ids:
            mapping = json.loads(args.request_trial_ids.read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                raise ValueError("Request/trial mapping must be a JSON object")
            if any(str(int(key)) != key for key in mapping):
                raise ValueError("Request ID keys must be canonical decimal integers")
            mapping = {int(key): value for key, value in mapping.items()}
        generation = generate_alignment(args.neural_generation, args.visual_features, args.output_dir,
            request_trial_ids=mapping, expected_neural_generation_id=args.expected_neural_generation_id,
            expected_visual_generation_id=args.expected_visual_generation_id)
    except Exception as exc:
        parser.exit(1, f"Alignment failed: {exc}\n")
    print(json.dumps(dict(generation_id=generation.generation_id, path=str(generation.path),
                         session_id=generation.manifest["eid"],
                         trial_ids=[trial.trial_id for trial in generation.trials])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
