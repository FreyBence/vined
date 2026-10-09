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


def output_dir():
    return project_path("VINED_OUTPUT_DIR", REPO_ROOT / "output")


def dataset_dir():
    return project_path("VINED_DATA_DIR", output_dir() / "datasets")


def visual_dir():
    return project_path("VINED_VISUAL_DIR", output_dir() / "visual_features")


def replay_dir():
    return project_path("VINED_REPLAY_DIR", output_dir() / "visual_replays")
