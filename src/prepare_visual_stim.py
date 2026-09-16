import argparse
import os
import re
import json
from utils.visual_data import SCHEMA_VERSION, FEATURE_WIDTH, validate_ids
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def sample_video_frames(
    video_path,
    sample_fps=5,
    *, eid=None, trial_id=None
):
    """
    Sample frames from video at fixed FPS.
    """

    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    with np.load(Path(video_path).with_suffix(".npz"), allow_pickle=False) as data:
        times = data["frame_times"].copy()
        valid = data["valid"].copy()
        if data["eid"].item() != eid or data["trial_id"].item() != trial_id:
            raise ValueError(f"Replay sidecar identity mismatch: {video_path}")
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or valid.shape != times.shape
            or valid.dtype.kind != "b"):
        raise ValueError(f"Invalid replay timestamp/validity sidecar: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    frames, sampled_times, sampled_valid = [], [], []
    frame_idx, next_sample = 0, 0.0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(native_fps) or native_fps <= 0 or sample_fps > native_fps:
            raise ValueError("Require 0 < sample_fps <= native video FPS")
        if not np.allclose(times-times[0], np.arange(len(times))/native_fps, rtol=0, atol=1e-7):
            raise ValueError("Replay timestamps disagree with video FPS")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= len(times):
                raise ValueError("Video has more frames than its timestamp sidecar")
            if frame_idx/native_fps + 1e-9 >= next_sample:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                sampled_times.append(times[frame_idx])
                sampled_valid.append(valid[frame_idx])
                next_sample = len(frames)/sample_fps
            frame_idx += 1
        if frame_idx != len(times):
            raise ValueError(f"Incomplete video: decoded {frame_idx} of {len(times)} frames")
    finally:
        cap.release()
    return frames, np.asarray(sampled_times, dtype=np.float64), np.asarray(sampled_valid, dtype=bool)


@torch.no_grad()
def extract_clip_features(
    frames,
    model,
    processor,
    device,
    batch_size=32
):
    """
    Extract CLIP embeddings from frames.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    all_features = []

    for start_idx in range(0, len(frames), batch_size):

        batch_frames = frames[
            start_idx:start_idx + batch_size
        ]

        inputs = processor(
            images=batch_frames,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        features = model.get_image_features(**inputs)

        features = torch.nn.functional.normalize(
            features,
            dim=-1
        )

        all_features.append(
            features.cpu().numpy()
        )

    return np.concatenate(all_features, axis=0)


def main(args):

    if args.batch_size <= 0 or args.sample_fps <= 0:
        raise ValueError("batch_size and sample_fps must be positive")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = CLIPModel.from_pretrained(
        args.clip_model
    ).to(device)

    processor = CLIPProcessor.from_pretrained(
        args.clip_model
    )

    model.eval()

    video_dir = Path(args.video_dir) / args.eid

    video_paths = sorted(
        video_dir.glob("*.mp4")
    )

    if len(video_paths) == 0:
        raise RuntimeError(
            f"No mp4 files found in {video_dir}"
        )

    manifest = json.loads((video_dir / "replay_metadata.json").read_text(encoding="utf-8"))
    if manifest["eid"] != args.eid:
        raise ValueError("Replay manifest EID mismatch")
    expected_ids = validate_ids(np.asarray([r["trial_id"] for r in manifest["trials"] if r["valid"]], dtype=np.int64))
    parsed_ids = []
    for path in video_paths:
        match = re.fullmatch(r"trial_(\d+)\.mp4", path.name)
        if match is None:
            raise ValueError(f"Unexpected replay filename: {path.name}")
        parsed_ids.append(int(match.group(1)))
    validate_ids(np.asarray(parsed_ids, dtype=np.int64))
    if set(parsed_ids) != set(expected_ids.tolist()):
        raise ValueError("Replay videos do not match completed manifest")
    all_valid = []
    all_trial_ids = []
    all_times = []
    all_features = []

    for trial_idx, video_path in tqdm(list(zip(parsed_ids, video_paths))):

        frames, frame_times, frame_valid = sample_video_frames(
            video_path,
            sample_fps=args.sample_fps, eid=args.eid, trial_id=trial_idx
        )

        if len(frames) == 0:
            print(f"Skipping empty video: {video_path}")
            continue

        features = extract_clip_features(
            frames,
            model,
            processor,
            device,
            batch_size=args.batch_size
        )

        if features.shape != (len(frame_times), FEATURE_WIDTH) or not np.isfinite(features).all():
            raise ValueError("CLIP feature shape/values do not match the pipeline contract")
        all_valid.append(frame_valid)
        all_trial_ids.append(trial_idx)

        all_times.append(frame_times.astype(np.float64))

        all_features.append(
            features.astype(np.float32)
        )

    output_path = os.path.join(
        args.output_dir,
        f"{args.eid}_visual_clip.npz"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    temporary = output_path + ".tmp"
    with open(temporary, "wb") as output:
        np.savez_compressed(
            output, schema_version=SCHEMA_VERSION, eid=args.eid, clock="session_seconds",
            trial_ids=np.asarray(all_trial_ids, dtype=np.int64),
            offsets=np.concatenate(([0], np.cumsum([len(t) for t in all_times]))),
            times=np.concatenate(all_times), features=np.concatenate(all_features),
            valid=np.concatenate(all_valid), clip_model=args.clip_model,
        )
    from utils.visual_data import load_archive
    load_archive(temporary, args.eid)
    os.replace(temporary, output_path)



if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--eid",
        type=str,
        default=None
    )

    ap.add_argument(
        "--video_dir",
        type=str,
        required=True
    )

    ap.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    ap.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14"
    )

    ap.add_argument(
        "--sample_fps",
        type=int,
        default=5
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    ap.add_argument(
        "--n_sessions",
        type=int,
        default=20
    )

    args = ap.parse_args()

    if args.n_sessions == 1:
        if args.eid is None:
            raise ValueError("Session EID is required.")
        else:
            eids = [args.eid]
    else:
        PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(f"{PROJ_DIR}/data/eids.txt") as file:
            eids = [line.rstrip() for line in file][9:args.n_sessions]

    func_args = args

    for eid in eids:
        func_args.eid = eid
        main(func_args)