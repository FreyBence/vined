"""Create inspection MP4s for one or all trials of a saved replay session."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from uuid import UUID

from utils.paths import output_dir, replay_dir
from visual_replay import ReplayArtifactReader, ReplayObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eid", type=UUID)
    parser.add_argument("trial_number", type=int, nargs="?",
                        help="Original trial ID; omit to process all trials")
    args = parser.parse_args()
    if args.trial_number is not None and args.trial_number < 0:
        parser.error("trial_number must be nonnegative")
    eid = str(args.eid)
    root = replay_dir()
    candidates = [path for path in [root / eid, *root.glob(f"*/{eid}")]
                  if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No trial images found for {eid} under {root}")
    source = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    print(f"Replay source: {source}", flush=True)
    with ReplayArtifactReader(source) as reader:
        manifest = reader.artifact_manifest
        if reader.definition["inputs"]["eid"] != eid:
            raise ValueError("Replay source belongs to another EID")
        outcomes = {trial["trial_id"]: trial for trial in manifest["completion"]["trials"]}
        if args.trial_number is not None:
            if args.trial_number not in outcomes:
                raise FileNotFoundError(f"Trial {args.trial_number} not found in {source}")
            selected = {args.trial_number}
            outcome = outcomes[args.trial_number]
            if outcome["status"] != "complete" or not outcome["image_count"]:
                raise ValueError(f"Trial {args.trial_number} has no complete image sequence: {outcome['status']}")
        else:
            selected = set(outcomes)
        extension = ".npz" if manifest["format"] == "npz_rgb8" else ".npy"
        # Check all declared images before starting any encoding.
        image_count = 0
        for trial_id in outcomes:
            directory = source / "trials" / str(trial_id)
            with (directory / "observations.jsonl").open(encoding="utf-8") as records:
                for line in records:
                    metadata = json.loads(line)
                    if metadata["status"] == "valid":
                        index = metadata["schedule_index"]
                        if type(index) is not int or index < 0:
                            raise ValueError("Invalid image schedule index")
                        path = directory / "images" / f"{index}{extension}"
                        if not path.is_file():
                            raise FileNotFoundError(f"Trial image not found: {path}")
                        image_count += 1
        if not image_count:
            raise FileNotFoundError(f"No trial images found in {source}")

        import cv2

        destination = output_dir() / "trial_preivew" / eid
        destination.mkdir(parents=True, exist_ok=True)
        generated = []
        skipped = 0
        writer = None
        mapping = None
        with tempfile.TemporaryDirectory(prefix=".preview-", dir=destination) as temporary:
            staging = Path(temporary)
            try:
                for item in reader:
                    metadata = item.metadata
                    trial_id = metadata["trial_id"]
                    if trial_id not in selected:
                        continue
                    outcome = outcomes[trial_id]
                    if outcome["status"] != "complete":
                        if not isinstance(item, ReplayObservation):
                            print(f"Skipping trial {trial_id}: {outcome['status']} (incomplete images/timing)")
                            skipped += 1
                        continue
                    if isinstance(item, ReplayObservation):
                        rgb = item.rgb
                        if writer is None:
                            height, width = rgb.shape[:2]
                            fps = outcome["timing"]["cadence_hz"]
                            if width % 2 or height % 2:
                                raise ValueError(f"Trial {trial_id}: MP4 requires even image dimensions")
                            writer = cv2.VideoWriter(str(staging / f"{trial_id}.mp4"),
                                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                            if not writer.isOpened():
                                raise RuntimeError("MP4 encoder could not be opened")
                            mapping = (staging / f"{trial_id}.jsonl").open("w", encoding="utf-8")
                        if rgb.shape != (height, width, 3):
                            raise ValueError(f"Trial {trial_id}: image dimensions changed")
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        index = metadata["schedule_index"]
                        mapping.write(json.dumps(dict(observation_id=metadata["observation_id"],
                            session_time=metadata["session_time"], encoded_index=index,
                            presentation_time=index / fps,
                            generation_id=manifest["completion"]["generation_id"])) + "\n")
                    else:
                        if writer is None:
                            raise ValueError(f"Trial {trial_id}: no images to encode")
                        writer.release()
                        writer = None
                        mapping.close()
                        mapping = None
                        capture = cv2.VideoCapture(str(staging / f"{trial_id}.mp4"))
                        count = 0
                        try:
                            while True:
                                ok, frame = capture.read()
                                if not ok:
                                    break
                                if frame.shape != (height, width, 3):
                                    raise RuntimeError("Encoded video dimensions changed")
                                count += 1
                        finally:
                            capture.release()
                        if count != outcome["image_count"]:
                            raise RuntimeError(f"Trial {trial_id}: incomplete encoded video")
                        generated.append(trial_id)
                        print(f"Encoded trial {trial_id}: {count} frames", flush=True)
                if reader.completion is None:
                    raise ValueError("Replay verification did not complete")
                for trial_id in generated:
                    for suffix in (".mp4", ".jsonl"):
                        os.replace(staging / f"{trial_id}{suffix}", destination / f"{trial_id}{suffix}")
            finally:
                if writer is not None:
                    writer.release()
                if mapping is not None:
                    mapping.close()
        print(f"Created {len(generated)} previews in {destination}; skipped {skipped} incomplete trials.")
        return 2 if skipped else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Preview failed: {exc}", file=sys.stderr)
        sys.exit(1)
