"""Extract complete CLIP feature generations from saved mouse-view replays."""

import argparse
import json
import math
from pathlib import Path
import sys
from uuid import UUID

from utils.paths import replay_dir, visual_dir
from utils.sessions import run_sessions
from utils.progress import configure_progress
from visual_replay import ReplayArtifactReader
from visual_features import ClipEncoder, ObservationSelection, write_features


def resolve_replay(root, eid):
    """Accept an explicit generation, a run directory, or an unambiguous root."""
    root = Path(root)
    if (root / "manifest.json").is_file():
        return root
    candidates = [path for path in [root / eid, *root.glob(f"*/{eid}")] if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No saved replay for {eid} under {root}")
    if len(candidates) != 1:
        raise ValueError(f"Multiple replay generations for {eid}; select --replay-dir explicitly: "
                         + ", ".join(str(path) for path in candidates))
    if not (candidates[0] / "manifest.json").is_file():
        raise ValueError(f"Unfinished replay generation: {candidates[0]}")
    return candidates[0]


def discover_replays(root):
    """Find published session generations without consulting an EID list."""
    root = Path(root)
    manifests = ([root / "manifest.json"] if (root / "manifest.json").is_file() else
                 sorted([*root.glob("*/manifest.json"), *root.glob("*/*/manifest.json")]))
    sources = {}
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "replay_artifacts":
            raise ValueError(f"Not a replay artifact manifest: {path}")
        eid = str(UUID(manifest["completion"]["eid"]))
        if eid in sources:
            raise ValueError(f"Multiple replay generations for {eid}: {sources[eid]}, {path.parent}; "
                             "select a run with --replay-dir")
        sources[eid] = path.parent
    if not sources:
        raise FileNotFoundError(f"No published replays found under {root}")
    return sources


def main(argv=None):
    configure_progress()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eid", type=UUID,
                        help="Process this session; omit to process all published replays")
    parser.add_argument("--replay-dir", type=Path, default=replay_dir(),
                        help="Explicit session generation, run directory, or unambiguous replay root")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=visual_dir())
    parser.add_argument("--clip-model", "--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip-revision", default="main")
    parser.add_argument("--sample-fps", "--sample_fps", type=float, default=None,
                        help="Explicit source-time subsampling rate; default selects all observations")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel replay image loading and image preparation; above one overlaps reads with inference")
    parser.add_argument("--device", help="Torch device; default auto-selects CUDA/CPU")
    parser.add_argument("--precision", choices=("auto", "float32", "float16"), default="auto",
                        help="Inference precision; auto uses CUDA float16 on compute capability >= 7, otherwise float32")
    parser.add_argument("--gpu-duty-cycle", type=float, default=1.0,
                        help="CUDA batch duty cycle in (0,1]; e.g. 0.6 adds 40%% idle time, default 1 disables pauses")
    parser.add_argument("--feature-cache-size", type=int, default=512,
                        help="Bounded cache of identical prepared-image features; default 512, 0 disables reuse")
    parser.add_argument("--attention-backend", choices=("auto", "eager", "sdpa"), default="auto",
                        help="CLIP vision attention; auto uses PyTorch SDPA on CUDA, eager on CPU")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.feature_cache_size < 0:
        parser.error("--feature-cache-size must be nonnegative")
    if not math.isfinite(args.gpu_duty_cycle) or not 0 < args.gpu_duty_cycle <= 1:
        parser.error("--gpu-duty-cycle must be finite and in (0, 1]")
    if args.batch_size <= 0 or (args.sample_fps is not None
                              and (not math.isfinite(args.sample_fps) or args.sample_fps <= 0)):
        parser.error("batch size and sample FPS must be positive and finite")
    sources = ({str(args.eid): resolve_replay(args.replay_dir, str(args.eid))}
               if args.eid is not None else discover_replays(args.replay_dir))
    encoder = None
    partial = []

    def extract(eid):
        nonlocal encoder
        source = sources[eid]
        destination = args.output_dir / f"{eid}_visual_clip.npz"
        if destination.exists():
            raise FileExistsError(f"Output exists: {destination}; use a new --output-dir")
        print(f"Replay source: {source}", flush=True)
        with ReplayArtifactReader(source, workers=args.workers) as replay:
            if replay.definition["inputs"]["eid"] != eid:
                raise ValueError("Selected replay belongs to another EID")
            with ObservationSelection(replay, sample_fps=args.sample_fps) as observations:
                if encoder is None:
                    encoder = ClipEncoder(args.clip_model, revision=args.clip_revision, device=args.device,
                                          workers=args.workers, precision=args.precision,
                                          gpu_duty_cycle=args.gpu_duty_cycle,
                                          feature_cache_size=args.feature_cache_size,
                                          attention_backend=args.attention_backend)
                manifest = write_features(observations, encoder, destination, batch_size=args.batch_size,
                                          workers=args.workers)
        status = manifest["replay_completion"]["reconstruction_status"]
        if status != "success":
            partial.append(eid)
        print(json.dumps(dict(eid=eid, output=str(destination), generation_id=manifest["generation_id"],
                              feature_count=manifest["feature_count"], replay_status=status)), flush=True)

    run_sessions(list(sources), extract, "visual-features")
    return 2 if partial else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Visual feature extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)
