"""Explicit, unit-normalized session stimulus parameters; no raw-clock inference."""
import hashlib
import json
import math
from pathlib import Path


SESSION_FIELDS = {"wheel_radius_mm", "gain_deg_per_mm", "horizontal_fov_deg"}
TRIAL_FIELDS = {"initial_azimuth_deg", "spatial_frequency_cpd", "sigma_deg",
                "orientation_deg", "phase_rad", "contrast"}
REQUIRED_TRIAL_FIELDS = TRIAL_FIELDS.copy()
TRIAL_FIELDS = TRIAL_FIELDS | {"gain_deg_per_mm"}
UNITS = {"wheel_radius_mm": "mm", "gain_deg_per_mm": "deg/mm",
         "horizontal_fov_deg": "deg", "initial_azimuth_deg": "deg",
         "spatial_frequency_cpd": "cycles/deg", "sigma_deg": "deg",
         "orientation_deg": "deg", "phase_rad": "rad", "contrast": "fraction"}
SOURCE_ORDER = {name: rank for rank, name in enumerate((
    "session", "historical_source", "task_configuration", "publication",
    "manufacturer", "retrospective", "project_assumption", "synthetic"))}


def alf_table_fingerprint(frame):
    """Bind recovered row IDs to table content/order without relying on file encoding."""
    from pandas.util import hash_pandas_object
    schema = json.dumps([(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]).encode()
    rows = hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes()
    return hashlib.sha256(schema + rows).hexdigest()


def validate_evidence(field, evidence):
    required = {"source_kind", "source", "original_value", "original_unit",
                "normalized_unit", "conversion", "fallback_reason", "applicability"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ValueError(f"{field}: malformed parameter evidence")
    if evidence["source_kind"] not in SOURCE_ORDER or evidence["normalized_unit"] != UNITS[field]:
        raise ValueError(f"{field}: unsupported evidence source or normalized unit")
    for key in ("source", "original_unit", "conversion", "applicability"):
        if not isinstance(evidence[key], str) or not evidence[key].strip():
            raise ValueError(f"{field}: {key} requires explicit evidence")
    if evidence["fallback_reason"] is not None and not isinstance(evidence["fallback_reason"], str):
        raise ValueError(f"{field}: invalid fallback reason")
    if evidence["source_kind"] != "session" and not evidence["fallback_reason"]:
        raise ValueError(f"{field}: lower-priority evidence requires a fallback reason")
    if (isinstance(evidence["original_value"], bool)
            or not isinstance(evidence["original_value"], (float, int))
            or not math.isfinite(evidence["original_value"])):
        raise ValueError(f"{field}: original value must be a finite number")


def _numbers(values, allowed, label):
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError(f"Unknown or malformed {label} parameters")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label}.{key} must be a finite number")
        if key == "gain_deg_per_mm" and value == 0:
            raise ValueError("Gain must be nonzero; its sign encodes contingency")
        if key not in {"initial_azimuth_deg", "orientation_deg", "phase_rad", "contrast", "gain_deg_per_mm"} and value <= 0:
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
    validate_document(document)
    return document["sessions"], hashlib.sha256(raw).hexdigest()


def validate_document(document):
    if (not isinstance(document, dict) or set(document) != {"schema_version", "sessions"}
            or type(document["schema_version"]) is not int or document["schema_version"] not in (1, 2)):
        raise ValueError("Expected stimulus parameter schema_version 1 or 2 and sessions")
    if not isinstance(document["sessions"], dict) or not document["sessions"]:
        raise ValueError("Manifest sessions must be a nonempty mapping")
    for eid, session in document["sessions"].items():
        fields = {"source", "conversion_notes", "session", "defaults", "trials"}
        if document["schema_version"] == 2:
            fields |= {"provenance", "recovery"}
        if not isinstance(session, dict) or set(session) != fields:
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
        if document["schema_version"] == 2:
            provenance = session["provenance"]
            if not isinstance(provenance, dict) or set(provenance) != {"session", "defaults", "trials"}:
                raise ValueError("Expected session/defaults/trials provenance")
            if not isinstance(provenance["trials"], dict) or set(provenance["trials"]) != set(session["trials"]):
                raise ValueError("Provenance must preserve every original trial ID")
            for scope, values in (("session", session["session"]), ("defaults", session["defaults"])):
                _validate_provenance(values, provenance[scope])
            for key, values in session["trials"].items():
                _validate_provenance(values, provenance["trials"][key])
            if not isinstance(session["recovery"], dict):
                raise ValueError("Recovery evidence must be a mapping")


def _validate_provenance(values, provenance):
    if not isinstance(provenance, dict) or set(values) != set(provenance):
        raise ValueError("Each normalized value requires exactly one provenance entry")
    for field, evidence in provenance.items():
        validate_evidence(field, evidence)


def resolved_evidence(session, trial_id):
    """Select source hierarchy first, then trial specificity within the same tier."""
    if session is None:
        return {}, {}
    key = str(trial_id)
    if key not in session["trials"]:
        raise ValueError(f"Parameter manifest has no original trial {trial_id}")
    values, evidence = {}, {}
    for scope, supplied in (("session", session["session"]), ("defaults", session["defaults"]),
                            ("trials", session["trials"][key])):
        for field, value in supplied.items():
            if "provenance" in session:
                origin = session["provenance"][scope]
                origin = origin[key][field] if scope == "trials" else origin[field]
            else:
                origin = dict(source_kind="session", source=session["source"], original_value=value,
                              original_unit=UNITS[field], normalized_unit=UNITS[field],
                              conversion=session["conversion_notes"], fallback_reason=None,
                              applicability="Legacy v1 user declaration; field-level evidence not verified")
            if field not in evidence or SOURCE_ORDER[origin["source_kind"]] <= SOURCE_ORDER[evidence[field]["source_kind"]]:
                values[field], evidence[field] = value, dict(origin)
    return values, evidence


def resolve_parameters(session, trial_id, strict=False):
    if session is None:
        if strict:
            raise ValueError("Strict parameters require a session parameter manifest")
        return {}, {}
    values, evidence = resolved_evidence(session, trial_id)
    if strict and (SESSION_FIELDS | REQUIRED_TRIAL_FIELDS) - values.keys():
        raise ValueError(f"Incomplete parameters for original trial {trial_id}")
    if strict and "provenance" in session and any(
            item["source_kind"] in {"project_assumption", "synthetic"} for item in evidence.values()):
        raise ValueError(f"Strict parameters reject project/synthetic fallbacks for trial {trial_id}")
    return ({key: value for key, value in values.items() if key in SESSION_FIELDS},
            {key: value for key, value in values.items() if key in TRIAL_FIELDS})
