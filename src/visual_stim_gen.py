from pathlib import Path
from utils.paths import dataset_dir, replay_dir

import cv2
import numpy as np
import pandas as pd
from one.api import ONE
from tqdm import tqdm

FPS = 30

VIDEO_WIDTH = 720
VIDEO_HEIGHT = 720

STIM_RADIUS = VIDEO_WIDTH // 6


def interpolate_wheel(
    timestamps,
    positions,
    query_time
):
    """
    Linear interpolation of wheel position.
    """

    return np.interp(
        query_time,
        timestamps,
        positions
    )


def compute_session_movement_gain(
    trials,
    wheel_position,
    wheel_timestamps,
    go_cue_times
):
    """
    Compute wheel->screen gain from the
    first successful trial.
    """

    canvas_center = VIDEO_WIDTH // 2

    margin = STIM_RADIUS

    for idx, trial in trials.iterrows():

        # ---------------------------------------------------
        # Only successful trials
        # ---------------------------------------------------

        if trial["feedbackType"] <= 0:
            continue

        # ---------------------------------------------------
        # Determine side
        # ---------------------------------------------------

        if not np.isnan(trial["contrastLeft"]):
            stim_side = "left"
            start_x = margin
        else:
            stim_side = "right"
            start_x = VIDEO_WIDTH - margin

        # ---------------------------------------------------
        # Timing
        # ---------------------------------------------------

        stim_on = go_cue_times[idx]

        feedback_time = trial["feedback_times"]

        if np.isnan(feedback_time):
            continue

        # ---------------------------------------------------
        # Wheel values
        # ---------------------------------------------------

        start_wheel = interpolate_wheel(
            wheel_timestamps,
            wheel_position,
            stim_on
        )

        feedback_wheel = interpolate_wheel(
            wheel_timestamps,
            wheel_position,
            feedback_time
        )

        wheel_delta = (
            feedback_wheel - start_wheel
        )

        # ---------------------------------------------------
        # Required displacement
        # ---------------------------------------------------

        required_displacement = abs(
            canvas_center - start_x
        )

        if abs(wheel_delta) < 1e-6:
            continue

        # ---------------------------------------------------
        # Compute gain
        # ---------------------------------------------------

        gain = (
            required_displacement
            / abs(wheel_delta)
        )

        return gain

    # fallback
    return 500


def create_grating_patch(
    size=240,
    spatial_frequency=3
):
    """
    Create static grating patch.
    """

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)

    xv, yv = np.meshgrid(x, y)

    # ---------------------------------------------------
    # Vertical bars
    # ---------------------------------------------------

    grating = np.sin(
        2 * np.pi * spatial_frequency * xv
    )

    # ---------------------------------------------------
    # Gaussian envelope
    # ---------------------------------------------------

    gaussian = np.exp(
        -(xv**2 + yv**2) / (2 * 0.25**2)
    )

    stimulus = grating * gaussian

    stimulus = (
        (stimulus + 1) / 2
    )

    stimulus = np.clip(stimulus, 0, 1)

    stimulus = (
        stimulus * 255
    ).astype(np.uint8)

    return stimulus


def render_trial_frame(
    wheel_value,
    movement_gain,
    grating_patch,
    stim_side="left"
):
    """
    Render one behavioral task frame.
    """

    # ---------------------------------------------------
    # Gray background
    # ---------------------------------------------------

    frame = np.ones(
        (VIDEO_HEIGHT, VIDEO_WIDTH),
        dtype=np.uint8
    ) * 128

    # ---------------------------------------------------
    # Base stimulus position
    # ---------------------------------------------------

    center_y = VIDEO_HEIGHT // 2
    margin = STIM_RADIUS

    if stim_side == "left":

        start_x = margin

        # target is center
        target_x = VIDEO_WIDTH // 2

        # negative wheel moves RIGHT
        normalized = -wheel_value

    else:

        start_x = VIDEO_WIDTH - margin

        target_x = VIDEO_WIDTH // 2

        # positive wheel moves LEFT
        normalized = wheel_value

    # ---------------------------------------------------
    # Normalize wheel movement
    # ---------------------------------------------------

    movement = normalized * movement_gain

    # clamp movement
    movement = np.clip(
        movement,
        0,
        abs(target_x - start_x)
    )

    # ---------------------------------------------------
    # Final stimulus position
    # ---------------------------------------------------

    if stim_side == "left":

        stim_x = int(start_x + movement)

    else:

        stim_x = int(start_x - movement)
    # ---------------------------------------------------
    # Stimulus bounds
    # ---------------------------------------------------

    x0 = stim_x - STIM_RADIUS
    y0 = center_y - STIM_RADIUS

    x1 = x0 + STIM_RADIUS * 2
    y1 = y0 + STIM_RADIUS * 2

    # ---------------------------------------------------
    # Paste stimulus
    # ---------------------------------------------------

    if (
        x0 >= 0 and
        y0 >= 0 and
        x1 < VIDEO_WIDTH and
        y1 < VIDEO_HEIGHT
    ):

        frame[y0:y1, x0:x1] = grating_patch

    return frame


# =====================================================
# TRIAL VIDEO GENERATION
# =====================================================

def generate_trial_video(
    out_dir,
    trial_index,
    trial,
    wheel_timestamps,
    wheel_position,
    stim_off_times,
    movement_gain,
    grating_patch,
):
    # ---------------------------------------------------
    # Determine stimulus side
    # ---------------------------------------------------

    if not np.isnan(trial["contrastLeft"]):

        stim_side = "left"

    else:

        stim_side = "right"

    # ---------------------------------------------------
    # Timing
    # ---------------------------------------------------
    stim_on = trial["stimOn_times"]

    stim_off = stim_off_times[trial_index]


    if np.isnan(stim_off):

        stim_off = stim_on + 2.0

    feedback_time = trial["feedback_times"]

    if np.isnan(feedback_time):
        feedback_time = stim_off

    duration = stim_off - stim_on

    duration = max(duration, 1.0)

    if np.isnan(duration):
        print()
        print("Broken trial:", trial_index)
        print("Len trial:", len(stim_off_times))
        print(trial)

    # ---------------------------------------------------
    # Output video
    # ---------------------------------------------------

    output_path = out_dir / (
        f"trial_{trial_index:04d}.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (VIDEO_WIDTH, VIDEO_HEIGHT),
        isColor=True
    )

    # ---------------------------------------------------
    # Render frames
    # ---------------------------------------------------

    n_frames = int(duration * FPS)

    frame_times = np.linspace(
        stim_on,
        stim_off,
        n_frames
    )

    initial_value = wheel_value = interpolate_wheel(
                wheel_timestamps,
                wheel_position,
                frame_times[0]
            )
    
    for t in frame_times:
        # ---------------------------------------------------
        # Before feedback -> live wheel movement
        # After feedback  -> freeze at final value
        # ---------------------------------------------------

        if t <= feedback_time:

            wheel_value = interpolate_wheel(
                wheel_timestamps,
                wheel_position,
                t
            )

        else:
            # freeze at feedback position
            wheel_value = interpolate_wheel(
                wheel_timestamps,
                wheel_position,
                feedback_time
            )

        wheel_value = wheel_value - initial_value

        frame = render_trial_frame(
            wheel_value,
            movement_gain, 
            grating_patch,
            stim_side
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_GRAY2BGR
        )

        writer.write(frame)

    writer.release()


def generate_visual_stimulus_for_session(one: ONE, eids: list[str]):
    grating_patch = create_grating_patch(
        size=STIM_RADIUS * 2
    )


    for eid in tqdm(eids):
        out_dir = replay_dir() / eid
        out_dir.mkdir(exist_ok=True)

        trials = one.load_dataset(
            eid,
            dataset="_ibl_trials.table.pqt",
            collection="alf"
        )

        stim_off_times = one.load_dataset(
            eid,
            dataset="_ibl_trials.stimOff_times.npy",
            collection="alf"
        )

        go_cue_times = one.load_dataset(
            eid,
            dataset="_ibl_trials.goCueTrigger_times.npy",
            collection="alf"
        )

        wheel_position = one.load_dataset(
            eid,
            dataset="_ibl_wheel.position.npy",
            collection="alf"
        )

        wheel_timestamps = one.load_dataset(
            eid,
            dataset="_ibl_wheel.timestamps.npy",
            collection="alf"
        )

        movement_gain = (
            compute_session_movement_gain(
                trials,
                wheel_position,
                wheel_timestamps,
                go_cue_times
            )
        )  

        for idx, trial in trials.iterrows(): # list(trials.iterrows())[401:]:
            generate_trial_video(
                        out_dir,
                        idx,
                        trial,
                        wheel_timestamps,
                        wheel_position,
                        stim_off_times,
                        movement_gain,
                        grating_patch,
                    )
            

if __name__ == "__main__":
    eids = [
        #"781b35fd-e1f0-4d14-b2bb-95b7263082bb",
        # "754b74d5-7a06-4004-ae0c-72a10b6ed2e6",
        # "4b00df29-3769-43be-bb40-128b1cba6d35",
        # "7cb81727-2097-4b52-b480-c89867b5b34c",
        # "0c828385-6dd6-4842-a702-c5075f5f5e81",
        # "b196a2ad-511b-4e90-ac99-b5a29ad25c22",
        # "aad23144-0e52-4eac-80c5-c4ee2decb198",
        # "61e11a11-ab65-48fb-ae08-3cb80662e5d6",
        # "7af49c00-63dd-4fed-b2e0-1b3bd945b20b",
        # "b03fbc44-3d8e-4a6c-8a50-5ea3498568e0",
        "d0ea3148-948d-4817-94f8-dcaf2342bbbe",
        "8928f98a-b411-497e-aa4b-aa752434686d",
        "ecb5520d-1358-434c-95ec-93687ecd1396",
        "746d1902-fa59-4cab-b0aa-013be36060d5",
        "91bac580-76ed-41ab-ac07-89051f8d7f6e",
        "0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63",
        "6899a67d-2e53-4215-a52a-c7021b5da5d4",
        "6f09ba7e-e3ce-44b0-932b-c003fb44fb89",
        "54238fd6-d2d0-4408-b1a9-d19d24fd29ce",
        "73918ae1-e4fd-4c18-b132-00cb555b1ad2",
    ]

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international", 
        silent=True,
        cache_dir=str(dataset_dir())
    )

    generate_visual_stimulus_for_session(one, eids)
