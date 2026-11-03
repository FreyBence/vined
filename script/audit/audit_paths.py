"""Checkout imports and output locations shared by standalone audit commands."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.paths import output_dir

AUDIT_ROOT = output_dir() / "audit"


def next_output(checkpoint, kind):
    """Choose an unused diagnostic generation without creating or replacing it."""
    run = Path(checkpoint).expanduser().resolve().parent
    base = AUDIT_ROOT / run.parent.name / run.name if run.parent.name.startswith("scenario-") else AUDIT_ROOT / run.name
    number = 1
    while (base / f"{kind}-{number:03d}").exists():
        number += 1
    return base / f"{kind}-{number:03d}"
