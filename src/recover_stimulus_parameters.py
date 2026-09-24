"""Recover schema-v2 parameters from ALF and explicitly described raw evidence.

No downloads, guessed trial offsets, inferred units, or implicit clock conversion.
See docs/visual-extraction.md for the recovery evidence contract.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from utils.stimulus_parameters import (
    SESSION_FIELDS, TRIAL_FIELDS, UNITS, SOURCE_ORDER, _numbers, _unique_object,
    validate_document, validate_evidence, resolve_parameters, alf_table_fingerprint,
)

RAW_NAMES = ("trial_num", "stim_pos_init", "stim_contrast", "stim_freq", "stim_angle",
             "stim_gain", "stim_sigma", "stim_phase", "bns_ts")
RAW_FIELDS = dict(zip(RAW_NAMES[1:-1], (
    "initial_azimuth_deg", "contrast", "spatial_frequency_cpd", "orientation_deg",
    "gain_deg_per_mm", "sigma_deg", "phase_rad")))
REFERENCE = "https://github.com/int-brain-lab/iblrig/blob/a0a031e2f9969c192767a255781920028034b7b0/iblrig/base_choice_world_params.yaml"


def file_evidence(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def convert(field, value, unit):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{field}: expected finite numeric value")
    target = UNITS[field]
    conversions = {("rad", "deg"): (180 / math.pi, "multiply by 180/pi"),
                   ("deg", "rad"): (math.pi / 180, "multiply by pi/180"),
                   ("cm", "mm"): (10, "multiply by 10"),
                   ("m", "mm"): (1000, "multiply by 1000"),
                   ("percent", "fraction"): (0.01, "divide by 100")}
    if unit == target:
        normalized, rule = float(value), "identity"
    elif field == "gain_deg_per_mm" and unit == "mm/deg":
        if value == 0:
            raise ValueError("Cannot invert zero gain")
        normalized, rule = 1 / value, "reciprocal; preserve transmitted sign"
    elif (unit, target) in conversions:
        factor, rule = conversions[unit, target]
        normalized = value * factor
    else:
        raise ValueError(f"Unsupported {field} unit {unit!r}; expected {target}")
    _numbers({field: normalized}, SESSION_FIELDS | TRIAL_FIELDS, "recovered")
    return normalized, rule


def candidate(field, value, unit, *, source_kind, source, applicability, fallback_reason=None):
    value, rule = convert(field, value, unit)
    # The caller's original value is retained even when the conversion changes units.
    return value, dict(source_kind=source_kind, source=source, original_unit=unit,
                       normalized_unit=UNITS[field], conversion=rule,
                       applicability=applicability, fallback_reason=fallback_reason)


def add(values, provenance, field, raw_value, unit, **origin):
    value, evidence = candidate(field, raw_value, unit, **origin)
    evidence["original_value"] = raw_value
    validate_evidence(field, evidence)
    if field in provenance:
        old_rank = SOURCE_ORDER[provenance[field]["source_kind"]]
        rank = SOURCE_ORDER[evidence["source_kind"]]
        if rank == old_rank and not math.isclose(values[field], value, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Conflicting equally ranked evidence for {field}")
        if rank >= old_rank:
            return
    values[field], provenance[field] = value, evidence


def timestamp(value, unit):
    if unit == "iso8601":
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Bonsai ISO timestamp requires an explicit UTC offset")
        return parsed.timestamp()
    if unit not in {"s", "ms", "us"}:
        raise ValueError(f"Unsupported raw clock unit: {unit}")
    if isinstance(value, bool):
        raise ValueError("Boolean is not a raw timestamp")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite raw timestamp")
    return number * {"s": 1, "ms": .001, "us": .000001}[unit]


def synchronize(raw, clock):
    if not isinstance(clock, dict) or set(clock) != {"unit", "anchors", "evidence"}:
        raise ValueError("Clock requires unit, anchors and synchronization evidence")
    if not isinstance(clock["evidence"], str) or not clock["evidence"].strip():
        raise ValueError("Clock anchors require evidence identifying matched synchronization events")
    anchors = clock["anchors"]
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise ValueError("At least two measured clock anchors required")
    if any(not isinstance(item, dict) or set(item) != {"raw", "alf_seconds"} for item in anchors):
        raise ValueError("Each clock anchor requires raw and alf_seconds")
    if any(isinstance(item["alf_seconds"], bool) or not isinstance(item["alf_seconds"], (int, float)) for item in anchors):
        raise ValueError("ALF clock anchors must be numeric seconds")
    source = np.array([timestamp(item["raw"], clock["unit"]) for item in anchors])
    target = np.array([item["alf_seconds"] for item in anchors], dtype=float)
    if (not np.isfinite(source).all() or not np.isfinite(target).all()
            or np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0)):
        raise ValueError("Clock anchors must be finite and strictly increasing in both clocks")
    queries = np.array([timestamp(value, clock["unit"]) for value in raw])
    if np.any(np.diff(queries) < 0) or np.any(queries < source[0]) or np.any(queries > source[-1]):
        raise ValueError("Raw clock goes backward or requires unsupported extrapolation")
    return np.interp(queries, source, target)


def read_raw(path):
    if path.suffix.lower() == ".ssv":
        # Explicit original nine-column format; trailing whitespace is harmless.
        records = [line.split() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if any(len(row) != len(RAW_NAMES) for row in records):
            raise ValueError("Expected nine fields per encoderTrialInfo SSV record")
        return pd.DataFrame(records, columns=RAW_NAMES)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(data.columns) != list(RAW_NAMES):
            raise ValueError("CSV must have the nine raw trial-info column names in documented order")
        return data
    raise ValueError("Raw trial info must be a nine-column .ssv or headered .csv")


def recover(eid, trials_path, evidence=None, raw_path=None):
    frame = pd.read_parquet(trials_path)
    config = {} if evidence is None else evidence
    if (not isinstance(config, dict) or set(config) - {"eid", "session", "defaults", "reference_profile", "raw"}
            or (evidence is not None and config.get("eid") != eid)):
        raise ValueError("Evidence must identify this EID and contain only documented sections")
    source = file_evidence(trials_path)
    entry = dict(source=json.dumps(source), conversion_notes="Original zero-based ALF row IDs; per-field units and evidence in provenance",
                 session={}, defaults={}, trials={str(row): {} for row in range(len(frame))},
                 provenance=dict(session={}, defaults={}, trials={str(row): {} for row in range(len(frame))}),
                 recovery=dict(alf=source, trial_count=len(frame), raw_available=raw_path is not None,
                               alf_table_fingerprint=alf_table_fingerprint(frame),
                               fingerprint_algorithm="SHA256(column names/dtypes + pandas row hashes excluding index; little-endian uint64)",
                               pandas_version=pd.__version__,
                               parameter_semantics="Task inputs, not effective shader phase/aperture/angle",
                               wheel_mapping="unverified: ALF sign, encoder reset and coupling events not recovered"))
    for scope, allowed in (("session", SESSION_FIELDS), ("defaults", TRIAL_FIELDS)):
        supplied = config.get(scope, {})
        if not isinstance(supplied, dict) or set(supplied) - allowed:
            raise ValueError(f"Unknown {scope} evidence fields")
        for field, options in supplied.items():
            if not isinstance(options, list) or not options:
                raise ValueError(f"{field}: supply a nonempty list of evidence candidates")
            for option in options:
                if not isinstance(option, dict) or set(option) != {"value", "unit", "source_kind", "source", "applicability", "fallback_reason"}:
                    raise ValueError(f"{field}: malformed evidence candidate")
                add(entry[scope], entry["provenance"][scope], field, option["value"], option["unit"],
                    **{key: option[key] for key in ("source_kind", "source", "applicability", "fallback_reason")})
    profile = config.get("reference_profile")
    if profile is not None:
        if (not isinstance(profile, dict) or set(profile) != {"id", "applicability_evidence", "stim_reverse"}
                or profile["id"] != "iblrig-a0a031e2"
                or not isinstance(profile["applicability_evidence"], str) or not profile["applicability_evidence"].strip()
                or type(profile["stim_reverse"]) is not bool):
            raise ValueError("Reference profile requires known ID, session applicability evidence and explicit reversal")
        for field, value in {"spatial_frequency_cpd": .1, "sigma_deg": 7., "orientation_deg": 0.,
                             "gain_deg_per_mm": -4. if profile["stim_reverse"] else 4.}.items():
            add(entry["defaults"], entry["provenance"]["defaults"], field, value, UNITS[field],
                source_kind="task_configuration", source=REFERENCE,
                applicability=profile["applicability_evidence"], fallback_reason="No higher-ranked value supplied")
        entry["recovery"]["reference_profile"] = profile
        gain_origin = entry["provenance"]["defaults"].get("gain_deg_per_mm", {})
        if profile["stim_reverse"] and gain_origin.get("source") == REFERENCE:
            gain_origin.update(original_value=4., conversion="Negate configured gain magnitude for declared stim_reverse",
                               source=REFERENCE + " ; controller: " + REFERENCE.replace("base_choice_world_params.yaml", "base_tasks.py"))
    invalid_contrasts = []
    for row, (_, trial) in enumerate(frame.iterrows()):
        key = str(row)
        contrasts = np.array([trial["contrastLeft"], trial["contrastRight"]], dtype=float)
        if (np.isinf(contrasts).any() or np.isfinite(contrasts).sum() != 1
                or np.any(contrasts < 0) or np.any(contrasts > 1)):
            invalid_contrasts.append(row)
            continue
        side = 0 if np.isfinite(contrasts[0]) else 1
        add(entry["trials"][key], entry["provenance"]["trials"][key], "contrast", float(contrasts[side]), "fraction",
            source_kind="session", source=f"{source['path']} sha256:{source['sha256']} row:{row} contrast{'Left' if side == 0 else 'Right'}",
            applicability="Observed ALF contrast fraction for original row")
        if profile is not None:
            add(entry["trials"][key], entry["provenance"]["trials"][key], "initial_azimuth_deg", -35. if side == 0 else 35., "deg",
                source_kind="task_configuration", source=REFERENCE, applicability=profile["applicability_evidence"],
                fallback_reason="No measured initial position; protocol magnitude with observed ALF side")
    entry["recovery"]["invalid_contrast_row_ids"] = invalid_contrasts
    if (raw_path is None) != ("raw" not in config):
        raise ValueError("Raw file and raw evidence section must be supplied together")
    if raw_path is not None:
        raw = read_raw(Path(raw_path))
        contract = config["raw"]
        if not isinstance(contract, dict) or set(contract) != {"units", "unit_evidence", "trial_map", "mapping_evidence", "clock", "excluded_trials"}:
            raise ValueError("Raw evidence requires units, unit_evidence, trial_map, mapping_evidence, clock, excluded_trials")
        for name in ("unit_evidence", "mapping_evidence"):
            if not isinstance(contract[name], str) or not contract[name].strip():
                raise ValueError(f"Raw {name} is required")
        if not isinstance(contract["units"], dict) or set(contract["units"]) != set(RAW_FIELDS):
            raise ValueError("Explicit units required for all seven raw stimulus fields")
        for name, field in RAW_FIELDS.items():
            convert(field, 1., contract["units"][name])
        mapping, excluded = contract["trial_map"], contract["excluded_trials"]
        if not isinstance(mapping, dict) or not isinstance(excluded, dict):
            raise ValueError("Trial map and exclusions must be mappings")
        raw_ids = raw.trial_num.tolist()
        if not raw_ids or any(not value.isdecimal() or str(int(value)) != value for value in raw_ids) or len(set(raw_ids)) != len(raw_ids):
            raise ValueError("Raw trial numbers must be unique canonical nonnegative integers")
        if (set(mapping) & set(excluded) or set(mapping) | set(excluded) != set(raw_ids)
                or any(not isinstance(reason, str) or not reason.strip() for reason in excluded.values())):
            raise ValueError("Map or explicitly exclude every raw trial; no silent final-row removal")
        if (any(type(row) is not int or not 0 <= row < len(frame) for row in mapping.values())
                or len(set(mapping.values())) != len(mapping)):
            raise ValueError("Trial mapping must be one-to-one into original ALF row indices")
        synchronized = synchronize(raw.bns_ts.tolist(), contract["clock"])
        raw_source = file_evidence(raw_path)
        entry["recovery"].update(raw=raw_source, raw_contract=contract, mapped_trials={})
        for index, raw_trial in raw.iterrows():
            raw_id = raw_trial.trial_num
            if raw_id in excluded:
                continue
            row = mapping[raw_id]
            key = str(row)
            entry["recovery"]["mapped_trials"][key] = dict(raw_trial_num=int(raw_id), bns_ts=raw_trial.bns_ts,
                original_fields={name: raw_trial[name] for name in RAW_FIELDS},
                missing_fields=[name for name in RAW_FIELDS if raw_trial[name] in ("", "nan", "NaN")],
                parameter_log_time_alf_seconds=float(synchronized[index]), clock_conversion="piecewise linear measured anchors; no extrapolation")
            for name, field in RAW_FIELDS.items():
                cell = raw_trial[name]
                if cell in ("", "nan", "NaN"):
                    continue
                add(entry["trials"][key], entry["provenance"]["trials"][key], field, float(cell), contract["units"][name],
                    source_kind="session", source=f"{raw_source['path']} sha256:{raw_source['sha256']} raw_trial:{raw_id} field:{name}",
                    applicability=contract["unit_evidence"])
            position = entry["trials"][key].get("initial_azimuth_deg")
            if position is not None and row not in invalid_contrasts:
                left = math.isfinite(float(frame.iloc[row].contrastLeft))
                if (position < 0) != left:
                    raise ValueError(f"Raw stimulus side disagrees with ALF row {row}")
    # Check the fully resolved candidates, not only the values in a single scope.
    for row in range(len(frame)):
        _, values = resolve_parameters(entry, row)
        if row not in invalid_contrasts:
            observed = entry["trials"][str(row)]["contrast"]
            if not math.isclose(values["contrast"], observed, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Resolved contrast disagrees with ALF row {row}")
            if "initial_azimuth_deg" in values:
                left = math.isfinite(float(frame.iloc[row].contrastLeft))
                if (values["initial_azimuth_deg"] < 0) != left:
                    raise ValueError(f"Resolved initial position disagrees with ALF row {row}")
    document = {"schema_version": 2, "sessions": {eid: entry}}
    validate_document(document)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eid", required=True)
    parser.add_argument("--alf-trials", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--raw-trial-info", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-parameters", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.evidence.read_text(encoding="utf-8"), object_pairs_hook=_unique_object) if args.evidence else None
    result = recover(args.eid, args.alf_trials, config, args.raw_trial_info)
    entry = result["sessions"][args.eid]
    if args.evidence:
        entry["recovery"]["evidence_file"] = file_evidence(args.evidence)
    entry["recovery"]["source_sha256"] = file_evidence(__file__)["sha256"]
    if args.require_parameters:
        for row in range(entry["recovery"]["trial_count"]):
            resolve_parameters(entry, row, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Recovered {entry['recovery']['trial_count']} original rows into {args.output}; missing fields remain explicit fallbacks")


if __name__ == "__main__":
    main()
