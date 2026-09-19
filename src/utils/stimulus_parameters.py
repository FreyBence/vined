"""Explicit, unit-normalized session stimulus parameters; no raw-clock inference."""
import hashlib
import json
import math
from pathlib import Path


SESSION_FIELDS = {"wheel_radius_mm", "gain_deg_per_mm", "horizontal_fov_deg"}
TRIAL_FIELDS = {"initial_azimuth_deg", "spatial_frequency_cpd", "sigma_deg",
                "orientation_deg", "phase_rad", "contrast"}


def _numbers(values, allowed, label):
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError(f"Unknown or malformed {label} parameters")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label}.{key} must be a finite number")
        if key not in {"initial_azimuth_deg", "orientation_deg", "phase_rad", "contrast"} and value <= 0:
            raise ValueError(f"{label}.{key} must be positive")
        if key == "initial_azimuth_deg" and value == 0:
            raise ValueError("Initial azimuth must identify a nonzero stimulus side")
        if key == "contrast" and not 0 <= value <= 1:
            raise ValueError("Contrast must be in [0, 1]")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def load_stimulus_parameters(path):
    """Load normalized parameters with explicit source and conversion evidence.

    Missing fields may use renderer approximations unless strict mode is enabled.
    A supplied manifest is never silently ignored for an absent session or trial.
    """
    raw = Path(path).read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_object)
    if set(document) != {"schema_version", "sessions"} or document["schema_version"] != 1:
        raise ValueError("Expected stimulus parameter schema_version 1 and sessions")
    if not isinstance(document["sessions"], dict) or not document["sessions"]:
        raise ValueError("Manifest sessions must be a nonempty mapping")
    for eid, session in document["sessions"].items():
        if not isinstance(session, dict) or set(session) != {"source", "conversion_notes", "session", "defaults", "trials"}:
            raise ValueError(f"Invalid parameter entry for {eid}")
        for field in ("source", "conversion_notes"):
            if not isinstance(session[field], str) or not session[field].strip():
                raise ValueError(f"{eid}: {field} must identify evidence/units and trial mapping")
        _numbers(session["session"], SESSION_FIELDS, "session")
        _numbers(session["defaults"], TRIAL_FIELDS, "defaults")
        if not isinstance(session["trials"], dict):
            raise ValueError("trials must map original zero-based row IDs to parameters")
        for trial_id, values in session["trials"].items():
            if not trial_id.isdecimal() or str(int(trial_id)) != trial_id:
                raise ValueError(f"Invalid original trial ID: {trial_id}")
            _numbers(values, TRIAL_FIELDS, f"trial {trial_id}")
    return document["sessions"], hashlib.sha256(raw).hexdigest()


def resolve_parameters(session, trial_id, strict=False):
    if session is None:
        if strict:
            raise ValueError("Strict parameters require a session parameter manifest")
        return {}, {}
    key = str(trial_id)
    if key not in session["trials"]:
        raise ValueError(f"Parameter manifest has no original trial {trial_id}")
    values = session["defaults"] | session["trials"][key]
    if strict and (SESSION_FIELDS - session["session"].keys() or TRIAL_FIELDS - values.keys()):
        raise ValueError(f"Incomplete parameters for original trial {trial_id}")
    return session["session"], values
