"""Repository defaults shared by preparation and dataset loading.

Absolute environment overrides also work when launched outside the checkout.
Relative overrides are resolved against the checkout, never the current directory.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def project_path(variable, default):
    path = Path(os.environ.get(variable, str(default))).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def dataset_dir():
    return project_path("VINED_DATA_DIR", REPO_ROOT / "datasets")


def visual_dir():
    return project_path("VINED_VISUAL_DIR", dataset_dir() / "vis_stim")


def replay_dir():
    return project_path("VINED_REPLAY_DIR", REPO_ROOT / "ibl_task_replay")
