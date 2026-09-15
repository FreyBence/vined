import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def sample_video_frames(
    video_path,
    sample_fps=5
):
    """
    Sample frames from video at fixed FPS.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_stride = max(
        int(native_fps / sample_fps),
        1
    )

    frames = []
    frame_times = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % sample_stride == 0:

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil_img = Image.fromarray(frame_rgb)

            timestamp = frame_idx / native_fps

            frames.append(pil_img)
            frame_times.append(timestamp)

        frame_idx += 1

    cap.release()

    return frames, np.array(frame_times)


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

    all_trial_ids = []
    all_times = []
    all_features = []

    for trial_idx, video_path in enumerate(
        tqdm(video_paths)
    ):

        frames, frame_times = sample_video_frames(
            video_path,
            sample_fps=args.sample_fps
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

        all_trial_ids.append(trial_idx)

        all_times.append(frame_times.astype(np.float32))

        all_features.append(
            features.astype(np.float32)
        )

    output_path = os.path.join(
        args.output_dir,
        f"{args.eid}_visual_clip.npz"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    np.savez_compressed(
        output_path,
        trial_ids=np.array(all_trial_ids, dtype=object),
        times=np.array(all_times, dtype=object),
        features=np.array(all_features, dtype=object),
    )


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