"""Generate standalone neural counts without visual artifacts or dataset splits."""

import argparse
import json
from pathlib import Path

from neural_data import (Coverage, QualitySelection, RecordingRequest, RegionSelection,
                         generate_neural)
from session_data import SessionAccess
from utils.paths import dataset_dir, output_dir
from utils.sessions import add_session_arguments, run_sessions, select_sessions
from utils.progress import configure_progress


def main():
    configure_progress()
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_arguments(parser)
    parser.add_argument("--config", type=Path, required=True, help="Neural request JSON; see neural-data/interface.md")
    parser.add_argument("--output-dir", type=Path, default=output_dir() / "neural")
    parser.add_argument("--cache-dir", type=Path, default=dataset_dir())
    parser.add_argument("--access-policy", choices=("local-only", "remote-allowed"), default="local-only")
    parser.add_argument("--workers", type=int, default=1,
                        help="Shared-memory trial/interval counting and compression workers (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    allowed = {"recordings", "quality", "anatomy", "intervals", "request_ids", "event",
               "offsets", "bin_size", "unit_coverage", "trial_collection", "trial_revision"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown neural request configuration fields")
    requests = []
    for recording in config.pop("recordings"):
        recording = dict(recording)
        if "coverage" in recording:
            recording["coverage"] = Coverage(**recording["coverage"])
        requests.append(RecordingRequest(**recording))
    for key, cls in (("quality", QualitySelection), ("anatomy", RegionSelection)):
        if config.get(key) is not None:
            config[key] = cls(**config[key])
    if config.get("unit_coverage") is not None:
        config["unit_coverage"] = {int(key): Coverage(**value)
                                   for key, value in config["unit_coverage"].items()}
    access = SessionAccess(policy=args.access_policy, cache_dir=args.cache_dir)
    eids = select_sessions(args.eid, args.eids_file, args.n_sessions)

    def prepare(eid):
        generation = generate_neural(access, eid, requests, args.output_dir, workers=args.workers, **config)
        print(json.dumps(dict(eid=eid, generation_id=generation.generation_id,
                              path=str(generation.path), outcomes=generation.manifest["outcomes"])), flush=True)

    run_sessions(eids, prepare, "neural-data")


if __name__ == "__main__":
    main()
