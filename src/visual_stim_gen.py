"""Approximate IBL task replay; see docs/ibl-visual-data-specs.md.

ALF events and wheel samples must share the synchronized session clock. Physical
defaults are protocol references, not recovered session calibration. Raw Bonsai
fields are deliberately not interpreted without verified units/synchronization.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from utils.paths import REPO_ROOT, dataset_dir, replay_dir

FPS = 30
VIDEO_WIDTH = VIDEO_HEIGHT = 720
STIM_RADIUS = 120
BACKGROUND = 128


@dataclass(frozen=True)
class ReplayConfig:
    wheel_radius_mm: float = 31.0
    gain_deg_per_mm: float = 4.0
    horizontal_fov_deg: float = 102.0
    initial_azimuth_deg: float = 35.0
    spatial_frequency_cpd: float = 0.1
    sigma_px: float = 30.0  # Project approximation; NOT an IBL sigma measurement.
    orientation_deg: float = 0.0  # Zero means vertical bars.
    phase_seed: int = 0

    def __post_init__(self):
        for name in ("wheel_radius_mm", "gain_deg_per_mm", "horizontal_fov_deg",
                     "initial_azimuth_deg", "spatial_frequency_cpd", "sigma_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.orientation_deg):
            raise ValueError("orientation_deg must be finite")

    @property
    def pixels_per_degree(self):
        return VIDEO_WIDTH / self.horizontal_fov_deg

    @property
    def movement_gain(self):
        return self.wheel_radius_mm * self.gain_deg_per_mm * self.pixels_per_degree


def validate_wheel(timestamps, positions):
    timestamps = np.asarray(timestamps, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if (timestamps.ndim != 1 or positions.shape != timestamps.shape
            or len(timestamps) < 2 or not np.all(np.isfinite(timestamps))
            or not np.all(np.isfinite(positions))
            or not np.all(np.diff(timestamps) > 0)):
        raise ValueError("Wheel arrays must be finite, matching 1-D arrays with increasing timestamps")
    return timestamps, positions


def interpolate_wheel(timestamps, positions, query_time):
    """Interpolate without silently extrapolating missing wheel observations."""
    query = np.asarray(query_time)
    if (not np.all(np.isfinite(query)) or np.any(query < timestamps[0])
            or np.any(query > timestamps[-1])):
        raise ValueError("Wheel timestamps do not cover the requested interval")
    return np.interp(query, timestamps, positions)


def create_grating_patch(size=240, spatial_frequency=0.1, *, contrast=1.0,
                         phase=0.0, pixels_per_degree=VIDEO_WIDTH / 102.0,
                         sigma_px=30.0, orientation_deg=0.0):
    """Gabor with frequency in cycles/degree and phase in radians."""
    if not np.isfinite(contrast) or not 0 <= contrast <= 1:
        raise ValueError("Contrast must be a finite fraction in [0, 1]")
    xy = np.arange(size) - (size - 1) / 2
    x, y = np.meshgrid(xy, xy)
    angle = np.deg2rad(orientation_deg)
    carrier = (x * np.cos(angle) + y * np.sin(angle)) / pixels_per_degree
    stimulus = np.sin(2 * np.pi * spatial_frequency * carrier + phase)
    stimulus *= np.exp(-(x*x + y*y) / (2 * sigma_px**2))
    return np.rint(BACKGROUND + 127 * contrast * stimulus).clip(0, 255).astype(np.uint8)


def render_trial_frame(wheel_value, movement_gain, grating_patch, stim_side="left",
                       *, initial_offset_px=35 * VIDEO_WIDTH / 102.0):
    """Positive wheel rotation moves both sides left; outward motion is allowed."""
    if stim_side not in ("left", "right"):
        raise ValueError("Stimulus side must be left or right")
    frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH), BACKGROUND, dtype=np.uint8)
    sign = -1 if stim_side == "left" else 1
    center_x = VIDEO_WIDTH / 2 + sign * initial_offset_px - wheel_value * movement_gain
    height, width = grating_patch.shape
    x0 = int(round(center_x)) - width // 2
    y0 = VIDEO_HEIGHT // 2 - height // 2
    left, right = max(0, x0), min(VIDEO_WIDTH, x0 + width)
    top, bottom = max(0, y0), min(VIDEO_HEIGHT, y0 + height)
    if left < right and top < bottom:
        frame[top:bottom, left:right] = grating_patch[
            top-y0:bottom-y0, left-x0:right-x0]
    return frame


def trial_parameters(trial, stim_off):
    contrasts = np.array([trial["contrastLeft"], trial["contrastRight"]], dtype=float)
    if np.count_nonzero(np.isfinite(contrasts)) != 1 or np.any(np.isinf(contrasts)):
        raise ValueError("Exactly one contrast side must be finite and the other NaN")
    side = "left" if np.isfinite(contrasts[0]) else "right"
    contrast = float(contrasts[0 if side == "left" else 1])
    if not 0 <= contrast <= 1:
        raise ValueError("Contrast must be in [0, 1]")
    stim_on = float(trial["stimOn_times"])
    if not np.isfinite(stim_on) or not np.isfinite(stim_off) or stim_off <= stim_on:
        raise ValueError("Valid stimulus onset and later offset are required")
    freeze = trial.get("stimFreeze_times", np.nan)
    freeze_source = "stimFreeze_times"
    if not np.isfinite(freeze):
        freeze = trial.get("response_times", np.nan)
        freeze_source = "response_times"
    if not np.isfinite(freeze) or not stim_on <= freeze <= stim_off:
        raise ValueError("A response/display-freeze event within the visible interval is required")
    return side, contrast, stim_on, float(freeze), freeze_source


def generate_trial_video(out_dir, trial_index, trial, wheel_timestamps,
                         wheel_position, stim_off, *, config=ReplayConfig(), eid=""):
    """Write onset-to-offset video and exact session timestamps (offset excluded).

    Invalid input raises ValueError before any video is written. Phase is a
    deterministic synthetic draw per (seed, EID, original row), not observed phase.
    """
    side, contrast, stim_on, freeze, freeze_source = trial_parameters(trial, stim_off)
    wheel_timestamps, wheel_position = validate_wheel(wheel_timestamps, wheel_position)
    endpoints = interpolate_wheel(wheel_timestamps, wheel_position, [stim_on, freeze])
    relative_times = np.arange(int(np.ceil((stim_off - stim_on) * FPS))) / FPS
    relative_times = relative_times[relative_times < stim_off - stim_on]
    frame_times = stim_on + relative_times
    wheel_delta = interpolate_wheel(wheel_timestamps, wheel_position,
                                    np.minimum(frame_times, freeze)) - endpoints[0]
    seed = hashlib.sha256(f"{config.phase_seed}:{eid}:{trial_index}".encode()).digest()
    phase = float(np.random.default_rng(int.from_bytes(seed[:8], "little")).uniform(0, 2*np.pi))
    patch = create_grating_patch(
        STIM_RADIUS * 2, config.spatial_frequency_cpd, contrast=contrast,
        phase=phase, pixels_per_degree=config.pixels_per_degree,
        sigma_px=config.sigma_px, orientation_deg=config.orientation_deg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"trial_{trial_index:04d}.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
        for delta in wheel_delta:
            frame = render_trial_frame(
                delta, config.movement_gain, patch, side,
                initial_offset_px=config.initial_azimuth_deg * config.pixels_per_degree)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()
    np.savez_compressed(output_path.with_suffix(".npz"),
                        frame_times=frame_times, relative_times=relative_times,
                        valid=np.ones(len(frame_times), dtype=bool),
                        trial_id=trial_index, eid=eid)
    return dict(trial_id=trial_index, valid=True, side=side, contrast=contrast,
                stim_on=stim_on, stim_off=float(stim_off), freeze=freeze,
                freeze_source=freeze_source, phase_rad=phase, frame_count=len(frame_times))


def generate_visual_stimulus_for_session(one, eids, *, config=ReplayConfig()):
    for eid in eids:
        out_dir = replay_dir() / eid
        out_dir.mkdir(parents=True, exist_ok=True)
        # Refuse mixed old/new artifacts, including stale videos of invalid trials.
        if any(out_dir.glob("trial_*")) or (out_dir / "replay_metadata.json").exists():
            raise FileExistsError(f"Use a fresh VINED_REPLAY_DIR; replay already exists: {out_dir}")
        def load(name):
            return one.load_dataset(eid, dataset=name, collection="alf")
        trials = load("_ibl_trials.table.pqt")
        stim_off_times = (trials["stimOff_times"].to_numpy() if "stimOff_times" in trials
                          else load("_ibl_trials.stimOff_times.npy"))
        if np.asarray(stim_off_times).shape != (len(trials),):
            raise ValueError("Stimulus offsets must match original trial rows")
        wheel_timestamps, wheel_position = validate_wheel(
            load("_ibl_wheel.timestamps.npy"), load("_ibl_wheel.position.npy"))
        records = []
        for row, (_, trial) in enumerate(trials.iterrows()):
            try:
                records.append(generate_trial_video(
                    out_dir, row, trial, wheel_timestamps, wheel_position,
                    stim_off_times[row], config=config, eid=eid))
            except ValueError as exc:
                records.append(dict(trial_id=row, valid=False, reason=str(exc)))
                print(f"{eid} trial {row}: skipped ({exc})")
        metadata = dict(
            eid=eid, collection="alf", clock="ALF synchronized session seconds (assumed)",
            dataset_revision=None, session_calibration_verified=False,
            renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            config=asdict(config), fps=FPS, canvas=[VIDEO_WIDTH, VIDEO_HEIGHT],
            codec="mp4v", background=BACKGROUND,
            assumptions=["Linear angular-to-pixel projection using reference field of view",
                         "Project Gaussian sigma and synthetic seeded random phase",
                         "Wheel coupling starts at stimulus onset",
                         "No pre-onset or post-offset observations generated"],
            trials=records)
        (out_dir / "replay_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    from one.api import ONE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eids-file", type=Path, default=REPO_ROOT / "data/eids.txt")
    for name, default in asdict(ReplayConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args())
    eids_file = args.pop("eids_file")
    eids = [line.strip() for line in eids_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    one = ONE(base_url="https://openalyx.internationalbrainlab.org",
              password="international", silent=True, cache_dir=str(dataset_dir()))
    generate_visual_stimulus_for_session(one, eids, config=ReplayConfig(**args))
