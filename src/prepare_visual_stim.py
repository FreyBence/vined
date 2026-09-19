import argparse
import os
import re
import json
import tempfile
import shutil
import zipfile
from contextlib import closing
from time import perf_counter
from utils.sessions import add_session_arguments, select_sessions, run_sessions, SkipSession
from utils.visual_data import SCHEMA_VERSION, FEATURE_WIDTH, validate_ids
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import HfApi, snapshot_download
from utils.provenance import file_hash, source_hashes, package_versions


def iter_frame_batches(
    video_path,
    sample_fps=5,
    *, eid=None, trial_id=None, expected_frames=None, expected_fps=None,
    expected_canvas=None, stim_on=None, stim_off=None, batch_size=32, render_record=None
):
    """
    Yield bounded image batches with their original timestamps and validity.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    with np.load(Path(video_path).with_suffix(".npz"), allow_pickle=False) as data:
        times = data["frame_times"].copy()
        valid = data["valid"].copy()
        wheel_delta = data["wheel_delta"].copy() if render_record is not None else None
        if data["eid"].item() != eid or data["trial_id"].item() != trial_id:
            raise ValueError(f"Replay sidecar identity mismatch: {video_path}")
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or valid.shape != times.shape
            or valid.dtype.kind != "b"):
        raise ValueError(f"Invalid replay timestamp/validity sidecar: {video_path}")
    if expected_frames is not None and len(times) != expected_frames:
        raise ValueError("Replay sidecar frame count disagrees with the manifest")
    if stim_on is not None and (not np.isfinite(stim_on)
                                or not np.isclose(times[0], stim_on, rtol=0, atol=1e-7)):
        raise ValueError("Replay timestamps do not start at manifest stimulus onset")
    if stim_off is not None and (not np.isfinite(stim_off) or np.any(times >= stim_off)):
        raise ValueError("Replay timestamps extend beyond the manifest visible interval")
    render = None
    if render_record is not None:
        from visual_stim_gen import ReplayConfig, create_grating_patch, render_trial_frame
        config = ReplayConfig(**render_record["effective_config"])
        if wheel_delta.shape != times.shape or not np.isfinite(wheel_delta).all():
            raise ValueError("Invalid direct-render wheel trajectory")
        patch = create_grating_patch(
            render_record["patch_size_px"], config.spatial_frequency_cpd,
            contrast=render_record["contrast"], phase=render_record["phase_rad"],
            pixels_per_degree=config.pixels_per_degree, sigma_px=config.sigma_px,
            orientation_deg=config.orientation_deg)
        def render(index):
            gray = render_trial_frame(wheel_delta[index], config.movement_gain, patch,
                render_record["side"], initial_offset_px=config.initial_azimuth_deg * config.pixels_per_degree)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cap = None if render else cv2.VideoCapture(str(video_path))
    frames, sampled_times, sampled_valid = [], [], []
    frame_idx, sample_count, next_sample = 0, 0, 0.0
    try:
        if cap is not None and not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        native_fps = expected_fps if render else cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(native_fps) or native_fps <= 0 or sample_fps > native_fps:
            raise ValueError("Require 0 < sample_fps <= native video FPS")
        if expected_fps is not None and not np.isclose(native_fps, expected_fps):
            raise ValueError("Video FPS disagrees with the replay manifest")
        if not np.allclose(times-times[0], np.arange(len(times))/native_fps, rtol=0, atol=1e-7):
            raise ValueError("Replay timestamps disagree with video FPS")
        while True:
            selected = frame_idx/native_fps + 1e-9 >= next_sample
            if render:
                if frame_idx == len(times):
                    break
                frame = render(frame_idx) if selected else None
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            if frame_idx >= len(times):
                raise ValueError("Video has more frames than its timestamp sidecar")
            if frame is not None and expected_canvas is not None and frame.shape != (*reversed(expected_canvas), 3):
                raise ValueError("Decoded frame dimensions disagree with the replay manifest")
            if selected:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                sampled_times.append(times[frame_idx])
                sampled_valid.append(valid[frame_idx])
                sample_count += 1
                next_sample = sample_count/sample_fps
                if len(frames) == batch_size:
                    yield frames, np.asarray(sampled_times, dtype=np.float64), np.asarray(sampled_valid, dtype=bool)
                    frames, sampled_times, sampled_valid = [], [], []
            frame_idx += 1
        if frame_idx != len(times):
            raise ValueError(f"Incomplete video: decoded {frame_idx} of {len(times)} frames")
    finally:
        if cap is not None:
            cap.release()
    if frames:
        yield frames, np.asarray(sampled_times, dtype=np.float64), np.asarray(sampled_valid, dtype=bool)


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

    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not len(frames):
        raise ValueError("Cannot extract features from an empty frame sequence")
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


def load_encoder(args):
    """Load and fingerprint one immutable encoder for the entire session run."""
    if (isinstance(args.batch_size, bool) or not isinstance(args.batch_size, (int, np.integer))
            or args.batch_size <= 0 or not np.isfinite(args.sample_fps) or args.sample_fps <= 0):
        raise ValueError("batch_size and sample_fps must be positive")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Resolve a moving name once, then load model AND processor from that exact
    # snapshot. Supply --clip-revision with the recorded SHA to reproduce a run.
    revision = args.clip_revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().model_info(args.clip_model, revision=revision).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("CLIP revision must resolve to an immutable commit SHA")
    snapshot = Path(snapshot_download(
        args.clip_model, revision=revision,
        allow_patterns=["*.json", "*.txt", "*.model", "pytorch_model.bin", "model.safetensors"]))
    model = CLIPModel.from_pretrained(str(snapshot), local_files_only=True).to(device)
    if model.config.projection_dim != FEATURE_WIDTH:
        raise ValueError(f"CLIP projection must have {FEATURE_WIDTH} features")
    processor = CLIPProcessor.from_pretrained(str(snapshot), local_files_only=True)
    provenance = dict(
        schema_version=1, clip_model=args.clip_model, clip_revision=revision,
        artifact_hashes={p.name: file_hash(p) for p in snapshot.iterdir() if p.is_file()},
        image_processor=processor.image_processor.to_dict(),
        packages=package_versions(), sample_fps=args.sample_fps,
        sampling="first frame at or after each requested sample time",
        dtype="float32", feature_width=FEATURE_WIDTH, normalization="L2 per frame",
        sources=source_hashes("src/prepare_visual_stim.py", "src/utils/visual_data.py"))

    model.eval()

    return model, processor, device, provenance


def main(args, encoder=None):

    if (isinstance(args.batch_size, bool) or not isinstance(args.batch_size, (int, np.integer))
            or args.batch_size <= 0 or not np.isfinite(args.sample_fps) or args.sample_fps <= 0):
        raise ValueError("batch_size and sample_fps must be positive")
    video_dir = Path(args.video_dir) / args.eid

    frame_source = getattr(args, "frame_source", "video")
    if frame_source not in ("video", "render"):
        raise ValueError("frame_source must be video or render")
    video_paths = sorted(video_dir.glob("*.mp4" if frame_source == "video" else "trial_*.npz"))

    manifest = json.loads((video_dir / "replay_metadata.json").read_text(encoding="utf-8"))
    if manifest["eid"] != args.eid:
        raise ValueError("Replay manifest EID mismatch")
    if not np.isfinite(manifest["fps"]) or not 0 < args.sample_fps <= manifest["fps"]:
        raise ValueError("Require 0 < sample_fps <= replay manifest FPS")
    canvas = manifest["canvas"]
    if (not isinstance(canvas, list) or len(canvas) != 2
            or any(type(value) is not int or value <= 0 for value in canvas)):
        raise ValueError("Replay canvas must contain positive integer width and height")
    expected_ids = validate_ids(np.asarray([r["trial_id"] for r in manifest["trials"] if r["valid"]], dtype=np.int64))
    parsed_ids = []
    for path in video_paths:
        match = re.fullmatch(r"trial_(\d+)\.(?:mp4|npz)", path.name)
        if match is None:
            raise ValueError(f"Unexpected replay filename: {path.name}")
        parsed_ids.append(int(match.group(1)))
    validate_ids(np.asarray(parsed_ids, dtype=np.int64))
    if set(parsed_ids) != set(expected_ids.tolist()):
        raise ValueError("Replay inputs do not match completed manifest")
    if not len(expected_ids):
        raise SkipSession("Replay manifest contains no valid trials")
    if frame_source == "render":
        if manifest.get("renderer_sha256") != file_hash(Path(__file__).with_name("visual_stim_gen.py")):
            raise ValueError("Direct rendering requires replay metadata from the current renderer")
    trial_records = {r["trial_id"]: r for r in manifest["trials"] if r["valid"]}
    if encoder is None:
        encoder = load_encoder(args)
    model, processor, device, encoder_provenance = encoder
    provenance = dict(encoder_provenance,
        replay_manifest_sha256=file_hash(video_dir / "replay_metadata.json"),
        frame_source=frame_source,
        renderer_sha256=manifest.get("renderer_sha256"))
    started = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.eid}_visual_clip.npz"
    # Raw numeric spools bound RAM use while retaining the schema-v2 NPZ interface.
    with tempfile.TemporaryDirectory(prefix=".clip-", dir=output_dir) as directory:
        staging = Path(directory)
        offsets = [0]
        with (staging / "times").open("wb") as time_file, (staging / "features").open("wb") as feature_file, (staging / "valid").open("wb") as valid_file:
            for trial_idx, video_path in tqdm(list(zip(parsed_ids, video_paths))):
                record = trial_records[trial_idx]
                batches = iter_frame_batches(
                    video_path, args.sample_fps, eid=args.eid, trial_id=trial_idx,
                    expected_frames=record["frame_count"], expected_fps=manifest["fps"],
                    expected_canvas=canvas, stim_on=record["stim_on"], stim_off=record["stim_off"],
                    batch_size=args.batch_size,
                    render_record=record if frame_source == "render" else None)
                count = 0
                with closing(batches):
                    for frames, times, valid in batches:
                        features = extract_clip_features(frames, model, processor, device, args.batch_size)
                        if (features.shape != (len(times), FEATURE_WIDTH) or not np.isfinite(features).all()
                                or np.any(np.linalg.norm(features[valid], axis=-1) == 0)):
                            raise ValueError("Invalid CLIP features")
                        times.astype(np.float64).tofile(time_file)
                        features.astype(np.float32).tofile(feature_file)
                        valid.tofile(valid_file)
                        count += len(times)
                if count == 0:
                    raise ValueError(f"No sampled frames: {video_path}")
                offsets.append(offsets[-1] + count)
        provenance["extraction_seconds"] = perf_counter() - started
        temporary = staging / "features.npz"
        metadata = dict(schema_version=SCHEMA_VERSION, eid=args.eid, clock="session_seconds",
            trial_ids=np.asarray(parsed_ids, dtype=np.int64), offsets=np.asarray(offsets, dtype=np.int64),
            clip_model=args.clip_model, clip_revision=provenance["clip_revision"],
            provenance=json.dumps(provenance, sort_keys=True, allow_nan=False))
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for name, value in metadata.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
            for name, dtype, shape in (("times", np.dtype("float64"), (offsets[-1],)),
                    ("features", np.dtype("float32"), (offsets[-1], FEATURE_WIDTH)),
                    ("valid", np.dtype("bool"), (offsets[-1],))):
                if (staging / name).stat().st_size != int(np.prod(shape)) * dtype.itemsize:
                    raise ValueError(f"Incomplete {name} spool")
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array_header_2_0(member, dict(
                        descr=np.lib.format.dtype_to_descr(dtype), fortran_order=False, shape=shape))
                    with (staging / name).open("rb") as source:
                        shutil.copyfileobj(source, member, length=1024*1024)
        # Check compressed members without loading the full session back into RAM.
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Corrupt feature archive")
        os.replace(temporary, output_path)
    print(f"{args.eid}: {offsets[-1]} frames via {frame_source} in {perf_counter()-started:.2f}s")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    add_session_arguments(ap)

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
        type=float,
        default=5
    )
    ap.add_argument("--clip-revision", default="main",
                    help="Hub revision; resolved to a recorded immutable SHA before downloading")

    ap.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    ap.add_argument("--frame-source", choices=("video", "render"), default="video",
                    help="render bypasses MP4 using current-renderer sidecars; video preserves decoded input")
    args = ap.parse_args()
    eids = select_sessions(args.eid, args.eids_file, args.n_sessions)
    if not re.fullmatch(r"[0-9a-f]{40}", args.clip_revision):
        args.clip_revision = HfApi().model_info(args.clip_model, revision=args.clip_revision).sha

    encoder = None

    def extract(eid):
        global encoder
        if encoder is None:
            encoder = load_encoder(args)
        session_args = argparse.Namespace(**vars(args))
        session_args.eid = eid
        main(session_args, encoder)

    run_sessions(eids, extract, "visual-features")
