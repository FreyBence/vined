"""E2 fixed identities and small artifact helpers."""

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
SESSION = "0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63"
REFERENCE = "01e5d3830786d45e1aa6f074d3e6c08fb0e476165f93b11a94d605edefb7995c"
ALIGNMENT = "7e635484272fdd3e209ab442db97a391b6fc9352f31101fa9e08dc2a3145b229"
REPLAY = "4f1b31578f8e28ec1b926b1cab7d36e9f0124cd6eb31728274cb7b16c3ca8184"
SOURCE = "8605a05d5fbf67eb01d4155fb3d552e3cbae666509dfd9148747869718714cb7"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
ARMS = ("A", "B", "G")
OUT = Path(
    os.environ.get("VINED_E2_OUTPUT_DIR", ROOT / "output/experiment/spatial-clip/e2")
).resolve()
if not OUT.is_relative_to(ROOT / "output/experiment/spatial-clip"):
    raise ValueError("E2 outputs must remain under output/experiment/spatial-clip")
DATASET = ROOT / "output/datasets/training-dataset/generations" / REFERENCE
REPLAY_PATH = ROOT / "output/visual_replays" / SESSION
OLD_FEATURES = ROOT / "output/visual_features" / (SESSION + "_visual_clip.npz")
from trainer.artifacts import plain
from utils.provenance import file_hash, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = plain(value)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if read(path) != value:
            raise FileExistsError(f"Immutable E2 record differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load_module(
    "e2_e0_metric_definitions", HERE.parent / "temporal-horizon/analysis.py"
)
DEFINITIONS = dict(
    metrics.DEFINITIONS,
    paired_difference="first named arm minus second named arm; negative NLL favors first",
)


def sources():
    files = list((ROOT / "src").rglob("*.py")) + list(HERE.glob("*.py"))
    files += [
        HERE.parent / "temporal-horizon/analysis.py",
        HERE.parent / "temporal-horizon/configs/training.json",
    ]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)}


def encoder():
    from visual_features import ClipEncoder

    return ClipEncoder(
        revision=REVISION,
        device="cuda",
        precision="float16",
        attention_backend="sdpa",
        workers=4,
        feature_cache_size=512,
        spatial_bundle=True,
    )


__all__ = [
    "HERE",
    "ROOT",
    "SESSION",
    "REFERENCE",
    "ALIGNMENT",
    "REPLAY",
    "SOURCE",
    "REVISION",
    "ARMS",
    "OUT",
    "DATASET",
    "REPLAY_PATH",
    "OLD_FEATURES",
    "fingerprint",
    "file_hash",
    "plain",
    "read",
    "save",
    "metrics",
    "DEFINITIONS",
    "sources",
    "encoder",
]
