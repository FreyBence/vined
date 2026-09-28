"""Resolve evidence once, before task evaluation or image generation.

Only SessionAccess public methods acquire source evidence. The returned object
is preparation input, never a claim that any images have been generated.
"""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from session_data import SessionAccessError
from utils.stimulus_parameters import (
    SOURCE_ORDER,
    UNITS,
    _unique_object,
    alf_table_fingerprint,
    resolved_evidence,
    validate_document,
)

from .profiles import PARAMETERS, WORKFLOW, behavior_profile


@dataclass
class ReconstructionPlan:
    definition: dict
    wheel_timestamps: np.ndarray
    wheel_positions: np.ndarray


class UnavailableInput(ValueError):
    """Essential trial evidence is absent, rather than contradictory."""


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid {name}")
    return value


def _evidence(value, unit, source, kind, rationale):
    return dict(
        effective_value=value,
        normalized_unit=unit,
        original_value=value,
        original_unit=unit,
        conversion="identity",
        source=source,
        source_kind=kind,
        applicability=rationale,
        fallback_reason=None if kind == "session" else rationale,
    )


def _declaration(item, label):
    if (
        not isinstance(item, dict)
        or set(item) != {"kind", "evidence"}
        or item["kind"] not in {"session", "project_assumption"}
        or not isinstance(item["evidence"], str)
        or not item["evidence"].strip()
    ):
        raise ValueError(
            f"{label} requires kind (session/project_assumption) and evidence"
        )


def _configuration(config, eid):
    config = deepcopy(config)
    required = {
        "eid",
        "profile",
        "applicability",
        "scene",
        "observation",
        "movement",
        "events",
        "phase_seed",
    }
    if not isinstance(config, dict) or set(config) != required or config["eid"] != eid:
        raise ValueError(
            "Configuration must identify this canonical EID and all preparation sections"
        )
    _declaration(config["applicability"], "Profile applicability")
    if type(config["phase_seed"]) is not int:
        raise ValueError("phase_seed must be an integer")
    scene = config["scene"]
    if not isinstance(scene, dict) or set(scene) != {
        "screen_width_mm",
        "screen_height_mm",
        "distance_mm",
        "display_size",
        "image_size",
        "horizontal_fov_deg",
        "provenance",
    }:
        raise ValueError(
            "Scene requires screen dimensions, distance, display/image sizes, horizontal FOV and provenance"
        )
    _declaration(scene["provenance"], "Scene")
    for key in (
        "screen_width_mm",
        "screen_height_mm",
        "distance_mm",
        "horizontal_fov_deg",
    ):
        scene[key] = _number(scene[key], key, positive=True)
    if scene["horizontal_fov_deg"] >= 180:
        raise ValueError("Perspective field of view must be below 180 degrees")
    for key in ("display_size", "image_size"):
        size = scene[key]
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(type(x) is not int or x <= 0 for x in size)
        ):
            raise ValueError(f"{key} requires [width, height] positive integers")
    ratio = scene["screen_width_mm"] / scene["screen_height_mm"]
    if not math.isclose(
        ratio, scene["display_size"][0] / scene["display_size"][1], rel_tol=0.01
    ):
        raise ValueError("Display raster must preserve physical screen aspect ratio")
    half_width = scene["distance_mm"] * math.tan(
        math.radians(scene["horizontal_fov_deg"] / 2)
    )
    half_height = half_width * scene["image_size"][1] / scene["image_size"][0]
    if (
        half_width < scene["screen_width_mm"] / 2
        or half_height < scene["screen_height_mm"] / 2
    ):
        raise ValueError("Mouse camera must contain the complete screen")
    scene.update(
        camera_position_mm=[0.0, 0.0, 0.0],
        camera_direction=[0.0, 0.0, 1.0],
        camera_up=[0.0, 1.0, 0.0],
        screen_center_mm=[0.0, 0.0, scene["distance_mm"]],
        screen_normal=[0.0, 0.0, -1.0],
        camera_provenance="Fixed frontal project viewpoint; not measured eye/head pose",
        surround_rgb=[128, 128, 128],
        surround_provenance="Neutral project backdrop; not measured room illumination",
    )
    observation = config["observation"]
    if not isinstance(observation, dict) or set(observation) != {
        "cadence_hz",
        "domain",
        "evidence",
    }:
        raise ValueError("Observation requires cadence_hz, domain and evidence")
    observation["cadence_hz"] = _number(
        observation["cadence_hz"], "cadence_hz", positive=True
    )
    if (
        observation["domain"] != "visible_interval"
        or not isinstance(observation["evidence"], str)
        or not observation["evidence"].strip()
    ):
        raise ValueError(
            "Only explicit visible_interval reconstruction schedules are supported"
        )
    observation.update(
        time_kind="reconstructed",
        clock="session_seconds",
        boundary="[onset, offset)",
        timing_precision="float64 storage; source measurement precision unknown",
        support_intervals=None,
    )
    movement = config["movement"]
    if not isinstance(movement, dict) or set(movement) != {
        "wheel_sign",
        "max_gap_seconds",
        "provenance",
    }:
        raise ValueError("Movement requires wheel_sign, max_gap_seconds and provenance")
    _declaration(movement["provenance"], "Wheel interpretation")
    if type(movement["wheel_sign"]) is not int or movement["wheel_sign"] not in {-1, 1}:
        raise ValueError(
            "wheel_sign must be -1 or 1, mapping ALF wheel to task azimuth"
        )
    movement["max_gap_seconds"] = _number(
        movement["max_gap_seconds"], "max_gap_seconds", positive=True
    )
    movement.update(
        interpolation="linear",
        extrapolation="unavailable",
        baseline="closed_loop_event",
        gap_policy="intervals wider than max_gap_seconds are unavailable",
    )
    events = config["events"]
    if not isinstance(events, dict) or set(events) != {"closed_loop", "freeze"}:
        raise ValueError("Explicit closed_loop and freeze event bindings are required")
    for name, binding in events.items():
        if not isinstance(binding, dict) or set(binding) != {"field", "provenance"}:
            raise ValueError(f"{name} requires field and provenance")
        _declaration(binding["provenance"], name)
        allowed = {
            "closed_loop": {"closedLoop_times", "goCue_times"},
            "freeze": {"stimFreeze_times", "response_times"},
        }
        if binding["field"] not in allowed[name]:
            raise ValueError(f"Unsupported {name} event binding")
        if (
            binding["field"] in {"goCue_times", "response_times"}
            and binding["provenance"]["kind"] != "project_assumption"
        ):
            raise ValueError(
                f"{binding['field']} is a proxy, not a measured {name} event"
            )
    return config


def _source(loaded):
    source = asdict(loaded.source)
    source["path"] = str(source["path"])
    return source


def _optional(access, eid, name, collection, revision=None):
    try:
        return access.load_dataset(
            eid, name, collection=collection, revision=revision
        ), None
    except SessionAccessError as exc:
        # Auxiliary events must match the trial-table revision. Absence at that
        # exact revision is missing optional evidence, not a request to load an
        # older revision or abort before applying the declared event proxy.
        if exc.reason not in {"source_unavailable", "local_source_unavailable", "revision_unavailable"}:
            raise
        return None, dict(reason=exc.reason, message=str(exc))


def _manifest(path, eid, table):
    if path is None:
        return None, None
    raw = Path(path).read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "sessions",
    }:
        raise ValueError("Malformed parameter manifest")
    entry = document["sessions"][eid]
    # Validate shared structure first; trial values are validated inside isolation.
    shared = deepcopy(entry)
    if not isinstance(shared.get("trials"), dict):
        raise ValueError("Parameter trials must map original row IDs")
    if set(shared["trials"]) - {str(i) for i in range(len(table))}:
        raise ValueError("Parameter manifest contains unknown original trial IDs")
    shared["trials"] = {}
    if document["schema_version"] == 2:
        if set(entry["provenance"]["trials"]) != set(entry["trials"]):
            raise ValueError(
                "Parameter provenance must account for each original trial"
            )
        shared["provenance"]["trials"] = {}
    validate_document(
        dict(schema_version=document["schema_version"], sessions={eid: shared})
    )
    identity = entry.get("recovery", {}).get("alf_table_fingerprint")
    if identity is None or identity != alf_table_fingerprint(table):
        raise ValueError(
            "Recovered parameters require a matching source trial-table fingerprint"
        )
    return (document["schema_version"], entry), hashlib.sha256(raw).hexdigest()


def _parameters(manifest, eid, trial_id):
    if manifest is None:
        return {}, {}
    version, entry = manifest
    key = str(trial_id)
    if key not in entry["trials"]:
        raise UnavailableInput("Parameter manifest has no original trial row")
    single = dict(entry, trials={key: entry["trials"][key]})
    if version == 2:
        single["provenance"] = dict(
            entry["provenance"], trials={key: entry["provenance"]["trials"][key]}
        )
    validate_document(dict(schema_version=version, sessions={eid: single}))
    values, evidence = resolved_evidence(single, trial_id)
    result = {}
    for field, item in evidence.items():
        trial_origin = single.get("provenance", {}).get("trials", {}).get(key, {}).get(field)
        is_trial = field in single["trials"][key] and (version == 1 or trial_origin == item)
        result[field] = dict(item, effective_value=values[field], scope="trial" if is_trial else "shared")
    return values, result


def _merge(values, evidence, field, value, origin, *, session_default=False):
    old = evidence.get(field)
    if old is not None:
        rank, old_rank = (
            SOURCE_ORDER[origin["source_kind"]],
            SOURCE_ORDER[old["source_kind"]],
        )
        if rank > old_rank:
            return
        if rank == old_rank and session_default and old.get("scope") == "trial":
            return
        if rank == old_rank and not math.isclose(
            value, values[field], rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError(f"Conflicting equally ranked evidence for {field}")
    values[field], evidence[field] = value, origin


def _event(row, field):
    value = row.get(field)
    if value is None or (isinstance(value, (float, np.floating)) and math.isnan(value)):
        raise UnavailableInput(f"Missing {field}")
    return _number(value, field)


def resolve_reconstruction_plan(
    access,
    eid,
    configuration,
    *,
    stimulus_parameters=None,
    trial_ids=None,
    revision=None,
    wheel_revision=None,
    settings_collection="raw_behavior_data",
    settings_revision=None,
):
    """Resolve one session. Trial IDs are original zero-based ALF row positions.

    Global configuration/identity/source conflicts raise; trial-local missing or
    contradictory inputs yield unavailable/invalid records with their identities.
    Prepared trials still require timeline/coverage evaluation before rendering.
    """
    eid = access.canonical_eid(eid)
    config = _configuration(configuration, eid)
    profile = behavior_profile(config["profile"])
    metadata = access.metadata(eid)
    table = access.load_dataset(
        eid, "_ibl_trials.table.pqt", collection="alf", revision=revision
    )
    frame = table.data
    ids = list(range(len(frame))) if trial_ids is None else list(trial_ids)
    if (
        not ids
        or any(type(i) is not int or not 0 <= i < len(frame) for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Request unique original trial IDs within the source table")
    manifest, manifest_hash = _manifest(stimulus_parameters, eid, frame)
    sources = [_source(table)]
    settings, settings_missing = _optional(
        access,
        eid,
        "_iblrig_taskSettings.raw.json",
        settings_collection,
        settings_revision,
    )
    task_settings = {} if settings is None else settings.data
    if not isinstance(task_settings, dict):
        raise ValueError("Task settings must be a decoded mapping")
    if settings is not None:
        sources.append(_source(settings))
    protocol = metadata.reported.get("task_protocol")
    if protocol and (
        "choiceworld" not in protocol.lower()
        or any(x in protocol.lower() for x in ("passive", "habituation"))
    ):
        raise ValueError(
            f"Unsupported task protocol for active choice-world behavior: {protocol}"
        )
    workflow = task_settings.get("VISUAL_STIMULUS")
    if workflow is not None and workflow != profile["display"]["workflow"]:
        raise ValueError(
            f"Task settings select a different display workflow: {workflow}"
        )
    geometry = dict(profile["nominal_display_geometry"])
    geometry["provenance"] = dict(config["applicability"], source=WORKFLOW)
    if "STIM_TRANSLATION_Z" in task_settings:
        geometry["distance"] = _number(task_settings["STIM_TRANSLATION_Z"], "STIM_TRANSLATION_Z", positive=True)
        geometry["distance_provenance"] = dict(kind="session", evidence=f"{settings.source.dataset_id}:STIM_TRANSLATION_Z")
    geometry["horizontal_fov_deg"] = math.degrees(2 * math.atan(geometry["width"] / (2 * geometry["distance"])))
    # Preserve distinct event columns. Missing auxiliary datasets stay missing;
    # no feedback/response/onset substitution is performed here.
    event_fields = {
        "stimOff_times",
        "stimFreeze_times",
        "closedLoop_times",
        config["events"]["closed_loop"]["field"],
        config["events"]["freeze"]["field"],
    }
    missing_events = {}
    frame = frame.copy()
    for field in sorted(event_fields):
        if field not in frame:
            loaded, missing = _optional(
                access, eid, f"_ibl_trials.{field}.npy", "alf", table.source.revision
            )
            if loaded is None:
                missing_events[field] = missing
            else:
                if np.asarray(loaded.data).shape != (len(frame),):
                    raise ValueError(f"{field} does not match original trial rows")
                frame[field] = loaded.data
                sources.append(_source(loaded))
    wheel_error = None
    times = positions = np.empty(0, dtype=np.float64)
    try:
        ts, pos = access.load_datasets(
            eid,
            ["_ibl_wheel.timestamps.npy", "_ibl_wheel.position.npy"],
            collection="alf",
            revision=wheel_revision,
        )
        sources.extend([_source(ts), _source(pos)])
        times, positions = (
            np.asarray(ts.data, dtype=np.float64),
            np.asarray(pos.data, dtype=np.float64),
        )
        if (
            times.ndim != 1
            or positions.shape != times.shape
            or len(times) < 2
            or not np.isfinite(times).all()
            or not np.isfinite(positions).all()
            or np.any(np.diff(times) < 0)
        ):
            wheel_error = dict(
                status="invalid",
                reason="Malformed, nonfinite or backward wheel samples",
            )
    except SessionAccessError as exc:
        if exc.reason not in {"source_unavailable", "local_source_unavailable"}:
            raise
        wheel_error = dict(status="unavailable", reason=str(exc))
    records = []
    for trial_id in ids:
        record = dict(trial_id=trial_id, status="prepared", reason=None)
        try:
            values, evidence = _parameters(manifest, eid, trial_id)
            row = frame.iloc[trial_id]
            contrasts = np.asarray(
                [row.get("contrastLeft", np.nan), row.get("contrastRight", np.nan)],
                dtype=float,
            )
            if np.isnan(contrasts).all():
                raise UnavailableInput("Missing stimulus contrast/side")
            if np.isfinite(contrasts).sum() != 1 or np.isinf(contrasts).any():
                raise ValueError("Exactly one finite contrast side required")
            side = "left" if np.isfinite(contrasts[0]) else "right"
            contrast = float(contrasts[0 if side == "left" else 1])
            if not 0 <= contrast <= 1:
                raise ValueError("Contrast must be in [0, 1]")
            _merge(
                values,
                evidence,
                "contrast",
                contrast,
                _evidence(
                    contrast,
                    "fraction",
                    f"{table.source.dataset_id} row {trial_id}",
                    "session",
                    "Observed ALF contrast",
                ),
            )
            defaults = dict(
                profile["nominal_parameters"],
                initial_azimuth_deg=(-1 if side == "left" else 1)
                * profile["nominal_initial_magnitude_deg"],
            )
            if "STIM_REVERSE" in task_settings:
                if type(task_settings["STIM_REVERSE"]) is not bool:
                    raise ValueError("STIM_REVERSE must be a boolean")
                defaults["gain_deg_per_mm"] *= -1 if task_settings["STIM_REVERSE"] else 1
            fallback_kind = "project_assumption" if config["applicability"]["kind"] == "project_assumption" else "task_configuration"
            for field, value in defaults.items():
                if field in evidence and SOURCE_ORDER[evidence[field]["source_kind"]] <= SOURCE_ORDER[fallback_kind]:
                    continue
                _merge(
                    values,
                    evidence,
                    field,
                    value,
                    _evidence(
                        value,
                        UNITS[field],
                        WORKFLOW if field == "wheel_radius_mm" else PARAMETERS,
                        fallback_kind,
                        "Nominal reference value; " + config["applicability"]["evidence"],
                    ),
                )
            if evidence["gain_deg_per_mm"]["source_kind"] == fallback_kind:
                evidence["gain_deg_per_mm"]["reversal"] = task_settings.get(
                    "STIM_REVERSE", "assumed false with nominal profile")
                if task_settings.get("STIM_REVERSE") is True and evidence["gain_deg_per_mm"]["source"] == PARAMETERS:
                    evidence["gain_deg_per_mm"].update(
                        original_value=profile["nominal_parameters"]["gain_deg_per_mm"],
                        conversion="negate nominal magnitude for session STIM_REVERSE")
            settings_fields = {
                "STIM_FREQ": "spatial_frequency_cpd",
                "STIM_SIGMA": "sigma_deg",
                "STIM_ANGLE": "orientation_deg",
                "STIM_GAIN": "gain_deg_per_mm",
            }
            for setting, field in settings_fields.items():
                if setting in task_settings:
                    value = _number(task_settings[setting], setting)
                    if setting == "STIM_GAIN":
                        reverse = task_settings.get("STIM_REVERSE")
                        if type(reverse) is not bool:
                            raise UnavailableInput(
                                "Session STIM_GAIN requires explicit STIM_REVERSE"
                            )
                        value *= -1 if reverse else 1
                    origin = _evidence(
                        value,
                        UNITS[field],
                        f"{settings.source.dataset_id}:{setting}",
                        "session",
                        "Session task configuration",
                    )
                    origin.update(
                        original_value=task_settings[setting],
                        conversion="apply STIM_REVERSE once"
                        if setting == "STIM_GAIN"
                        else "identity",
                    )
                    _merge(values, evidence, field, value, origin, session_default=True)
            if "STIM_POSITIONS" in task_settings:
                candidates = [
                    _number(x, "STIM_POSITIONS")
                    for x in task_settings["STIM_POSITIONS"]
                ]
                candidates = [
                    x for x in candidates if (x < 0 if side == "left" else x > 0)
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Session STIM_POSITIONS must identify exactly one position for this side"
                    )
                value = candidates[0]
                _merge(
                    values,
                    evidence,
                    "initial_azimuth_deg",
                    value,
                    _evidence(
                        value,
                        "deg",
                        f"{settings.source.dataset_id}:STIM_POSITIONS",
                        "session",
                        "Session positions with ALF side",
                    ),
                    session_default=True,
                )
            if "horizontal_fov_deg" in values and not math.isclose(values["horizontal_fov_deg"], geometry["horizontal_fov_deg"], abs_tol=1e-6):
                raise ValueError("Normalized FOV contradicts source ViewWindow geometry")
            values["horizontal_fov_deg"] = geometry["horizontal_fov_deg"]
            evidence["horizontal_fov_deg"] = dict(
                _evidence(values["horizontal_fov_deg"], "deg", WORKFLOW, fallback_kind,
                          "Derived from resolved display geometry"),
                conversion=profile["display"]["visual_span"], geometry=geometry)
            if (values["initial_azimuth_deg"] < 0) != (side == "left") or values[
                "initial_azimuth_deg"
            ] == 0:
                raise ValueError("Resolved stimulus side contradicts ALF")
            for field in ("wheel_radius_mm", "sigma_deg", "spatial_frequency_cpd"):
                _number(values[field], field, positive=True)
            if values["gain_deg_per_mm"] == 0:
                raise ValueError("Gain must be nonzero")
            if "phase_rad" not in values:
                seed_input = f"{config['phase_seed']}:{eid}:{trial_id}"
                digest = hashlib.sha256(seed_input.encode()).digest()
                # A fixed integer-to-uniform mapping avoids RNG/package dependence.
                phase = int.from_bytes(digest[:8], "big") / 2**64 * (2 * math.pi)
                values["phase_rad"] = phase
                evidence["phase_rad"] = dict(
                    _evidence(
                        phase,
                        "rad",
                        "SHA256 seed/EID/original-row",
                        "synthetic",
                        "Task phase unavailable",
                    ),
                    seed_input=seed_input,
                    algorithm="uint64 big-endian first 8 digest bytes / 2**64 * 2*pi",
                )
            choice, feedback = _event(row, "choice"), _event(row, "feedbackType")
            if choice not in {-1, 0, 1} or feedback not in {-1, 1} or (choice == 0 and feedback == 1):
                raise ValueError("Invalid choice/feedback outcome")
            closed_loop_field = config["events"]["closed_loop"]["field"]
            if "closedLoop_times" in row and np.isfinite(row["closedLoop_times"]):
                closed_loop_field = "closedLoop_times"
            events = {
                name: _event(row, field)
                for name, field in {
                    "onset": "stimOn_times",
                    "offset": "stimOff_times",
                    "closed_loop": closed_loop_field,
                }.items()
            }
            event_evidence = deepcopy(config["events"])
            if "closedLoop_times" in row and np.isfinite(row["closedLoop_times"]):
                events["closed_loop"] = float(row["closedLoop_times"])
                event_evidence["closed_loop"] = dict(
                    field="closedLoop_times",
                    provenance=dict(kind="session", evidence="Recorded ALF closed-loop event"))
            if choice != 0:
                freeze_field = config["events"]["freeze"]["field"]
                if "stimFreeze_times" in row and np.isfinite(row["stimFreeze_times"]):
                    freeze_field = "stimFreeze_times"
                    event_evidence["freeze"] = dict(field=freeze_field, provenance=dict(kind="session", evidence="Recorded ALF display freeze"))
                events["freeze"] = _event(row, freeze_field)
            movement_end = events.get("freeze", events["offset"])
            if (
                not events["onset"]
                <= events["closed_loop"]
                <= movement_end
                <= events["offset"]
                or events["onset"] == events["offset"]
            ):
                raise ValueError("Contradictory onset/closed-loop/freeze/offset order")
            # Preserve response and feedback separately even when they are not bindings.
            for field in (
                "response_times",
                "feedback_times",
                "stimFreeze_times",
                "goCue_times",
            ):
                value = row.get(field)
                if value is not None and np.isfinite(value):
                    events[field] = float(value)
            record.update(
                side=side,
                parameters=values,
                parameter_evidence=evidence,
                events=events,
                event_evidence=event_evidence,
                outcome="no_go"
                if choice == 0
                else "reward"
                if feedback == 1
                else "error",
                requested_domain=[events["onset"], events["offset"]],
            )
            # A source trial interval bounds any claim of pre/post-stimulus blank
            # content. Without it, visibility outside onset/offset stays unknown.
            if "intervals_0" in row and "intervals_1" in row:
                start, end = _event(row, "intervals_0"), _event(row, "intervals_1")
                if not start <= events["onset"] < events["offset"] <= end:
                    raise ValueError("Stimulus visibility lies outside the source trial interval")
                record["trial_interval"] = [start, end]
            if wheel_error is not None:
                record.update(wheel_error)
        except UnavailableInput as exc:
            record.update(status="unavailable", reason=str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            record.update(status="invalid", reason=str(exc))
        records.append(record)
    # Keep raw wheel evidence unchanged; interpolation and duplicate handling are VR02.
    times, positions = times.copy(), positions.copy()
    times.setflags(write=False)
    positions.setflags(write=False)
    definition = dict(
        schema_version=1,
        kind="resolved_reconstruction_inputs",
        eid=eid,
        profile=profile,
        configuration=config,
        display_geometry=geometry,
        session_evidence=dict(
            task_protocol=protocol,
            metadata_origin=metadata.origin,
            task_settings=task_settings,
            settings_unavailable=settings_missing,
        ),
        source_datasets=sources,
        trial_table_fingerprint=alf_table_fingerprint(table.data),
        original_trial_count=len(frame),
        requested_trial_ids=ids,
        parameter_manifest_sha256=manifest_hash,
        implementation_sha256={
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("preparation.py", "profiles.py", "timeline.py", "display.py", "scene.py", "stream.py")
        },
        missing_events=missing_events,
        wheel=dict(
            samples=len(times),
            timestamps_unit="session_seconds",
            position_unit="rad",
            timestamps_sha256=hashlib.sha256(times.astype("<f8").tobytes()).hexdigest(),
            positions_sha256=hashlib.sha256(
                positions.astype("<f8").tobytes()
            ).hexdigest(),
        ),
        trials=records,
    )
    return ReconstructionPlan(definition, times, positions)
