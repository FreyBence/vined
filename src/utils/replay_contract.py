"""Versioned replay records shared by renderers, publication and frame consumers.

This contract describes both future source renderers and the current explicitly
approximate adapter. It does not implement source timing or a mouse-view scene.
"""
import hashlib
from importlib.metadata import distribution
import platform
from pathlib import PurePosixPath

import numpy as np

from utils.paths import REPO_ROOT
from utils.provenance import file_hash, fingerprint

REPLAY_SCHEMA_VERSION = 1
LEGACY_BACKEND = "legacy_python_gabor"
STATES = {"hidden": 0, "stationary_visible": 1, "closed_loop": 2,
          "freeze_in_place": 3, "freeze_at_center": 4, "terminated": 5,
          "legacy_onset_coupled": 240, "legacy_response_hold": 241}
RENDER_FILES = ("src/visual_stim_gen.py", "src/utils/stimulus_parameters.py",
                "src/utils/replay_contract.py", "src/utils/provenance.py",
                "src/utils/paths.py", "src/utils/sessions.py")


def rendering_dependencies():
    """Local dependency closure plus runtime/package artifact-manifest identities."""
    import cv2
    packages = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless"):
        from importlib.metadata import PackageNotFoundError
        try:
            package = distribution(name)
        except PackageNotFoundError:
            continue
        record = package.read_text("RECORD")
        if record is None:
            raise ValueError(f"Rendering package {name} lacks its installed artifact manifest")
        packages[name] = {"version": package.version,
                          "record_sha256": hashlib.sha256(record.encode()).hexdigest()}
    if "numpy" not in packages or not any(name.startswith("opencv-") for name in packages):
        raise ValueError("Cannot fingerprint NumPy/OpenCV distributions")
    return dict(files={name: file_hash(REPO_ROOT / name) for name in RENDER_FILES},
                packages=packages, python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, opencv=cv2.__version__,
                opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest())


def legacy_contract(fps):
    contract = dict(schema_version=REPLAY_SCHEMA_VERSION,
                    backend=dict(id=LEGACY_BACKEND, version=1,
                                 capabilities=dict(stimulus_frames=True, scene_frames=False,
                                                   direct_frames=True, capture_storage=False)),
                    dependencies=rendering_dependencies(), scene_profile=None,
                    output_space="stimulus", capture_storage=None,
                    source_clock="session_seconds", source_time_kind="reconstructed",
                    timing_evidence=f"Legacy {fps:g} Hz onset-aligned reconstruction; experiment projection rate confirmed as 60 Hz; not recorded display refreshes",
                    video_mapping=dict(policy="identity", fps=float(fps)))
    contract["fingerprint"] = fingerprint(contract)
    validate_contract(contract)
    return contract


def validate_contract(contract, *, direct=False, check_dependencies=False):
    if not isinstance(contract, dict) or contract.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("Unsupported/unversioned replay contract; regenerate replays")
    required = {"schema_version", "backend", "dependencies", "scene_profile", "output_space",
                "capture_storage", "source_clock", "source_time_kind", "timing_evidence", "video_mapping", "fingerprint"}
    if set(contract) != required or type(contract["schema_version"]) is not int:
        raise ValueError("Malformed replay contract")
    if contract["fingerprint"] != fingerprint({key: value for key, value in contract.items() if key != "fingerprint"}):
        raise ValueError("Replay contract fingerprint mismatch")
    backend = contract["backend"]
    if (not isinstance(backend, dict) or set(backend) != {"id", "version", "capabilities"}
            or not isinstance(backend["id"], str) or not backend["id"]
            or type(backend["version"]) is not int or backend["version"] < 1):
        raise ValueError("Backend requires explicit identity and version")
    capabilities = backend["capabilities"]
    if (not isinstance(capabilities, dict) or set(capabilities) != {"stimulus_frames", "scene_frames", "direct_frames", "capture_storage"}
            or any(type(value) is not bool for value in capabilities.values())):
        raise ValueError("Backend capabilities must be explicit booleans")
    if contract["output_space"] not in ("stimulus", "scene") or not capabilities[contract["output_space"] + "_frames"]:
        raise ValueError("Backend cannot produce the requested frame space")
    if contract["source_clock"] != "session_seconds" or contract["source_time_kind"] not in ("recorded_display", "reconstructed"):
        raise ValueError("Unknown source clock or timing classification")
    if not isinstance(contract["timing_evidence"], str) or not contract["timing_evidence"].strip():
        raise ValueError("Source timing requires evidence or an explicit reconstruction declaration")
    scene = contract["scene_profile"]
    if scene is not None:
        if (not isinstance(scene, dict) or set(scene) != {"id", "schema_version", "geometry", "fingerprint"}
                or not isinstance(scene["id"], str) or not scene["id"]
                or type(scene["schema_version"]) is not int or scene["schema_version"] < 1
                or not isinstance(scene["geometry"], dict) or not scene["geometry"]
                or scene["fingerprint"] != fingerprint({key: value for key, value in scene.items() if key != "fingerprint"})):
            raise ValueError("Scene requires a versioned, fingerprinted concrete profile")
    if contract["output_space"] == "scene" and scene is None:
        raise ValueError("Scene output requires geometry from VR11")
    capture = contract["capture_storage"]
    if capture is not None:
        if (not capabilities["capture_storage"] or not isinstance(capture, dict)
                or set(capture) != {"format", "index_path", "index_sha256"}
                or capture["format"] not in ("png_rgb8", "npy_rgb8")
                or not isinstance(capture["index_path"], str) or not capture["index_path"]
                or PurePosixPath(capture["index_path"]).is_absolute()
                or ".." in PurePosixPath(capture["index_path"]).parts
                or "\\" in capture["index_path"] or ":" in capture["index_path"]
                or not _hash(capture["index_sha256"])):
            raise ValueError("Capture storage requires a lossless format and hashed frame index")
    mapping = contract["video_mapping"]
    if (not isinstance(mapping, dict) or set(mapping) != {"policy", "fps"}
            or mapping["policy"] not in ("identity", "hold_previous")
            or isinstance(mapping["fps"], bool) or not isinstance(mapping["fps"], (int, float))
            or not np.isfinite(mapping["fps"]) or mapping["fps"] <= 0):
        raise ValueError("Unsupported video/source mapping")
    dependencies = contract["dependencies"]
    if (not isinstance(dependencies, dict) or not isinstance(dependencies.get("files"), dict)
            or not dependencies["files"] or any(not _hash(value) for value in dependencies["files"].values())):
        raise ValueError("Rendering dependency hashes are required")
    if direct and not capabilities["direct_frames"]:
        raise ValueError("Capture-only backend cannot directly regenerate frames")
    if check_dependencies and dependencies != rendering_dependencies():
        raise ValueError("Rendering dependencies changed; regenerate replays before direct extraction")


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def record_fingerprint(record):
    return fingerprint({key: value for key, value in record.items() if key != "record_fingerprint"})


def validate_record(record, contract):
    if (not isinstance(record, dict) or type(record.get("trial_id")) is not int or record["trial_id"] < 0
            or type(record.get("valid")) is not bool
            or record.get("contract_fingerprint") != contract["fingerprint"]
            or record.get("record_fingerprint") != record_fingerprint(record)):
        raise ValueError("Invalid trial identity, validity or record fingerprint")
    if not isinstance(record.get("resolved_parameters"), dict):
        raise ValueError("Trial requires resolved parameter values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
           for value in record["resolved_parameters"].values()):
        raise ValueError("Resolved trial parameters must be finite numeric values")
    if not record["valid"] and (not isinstance(record.get("reason"), str) or not record["reason"]):
        raise ValueError("Invalid trial requires a reason")
    if record["valid"] and (type(record.get("frame_count")) is not int or record["frame_count"] <= 0):
        raise ValueError("Valid trial requires source frames")
    if record["valid"]:
        if type(record.get("video_written")) is not bool or not record["resolved_parameters"]:
            raise ValueError("Valid trial requires resolved parameters and explicit video availability")
        for field in ("stim_on", "stim_off"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Valid trial requires finite session-second interval boundaries")
        if record["stim_off"] <= record["stim_on"]:
            raise ValueError("Trial interval must have positive duration")


def validate_manifest(manifest, *, direct=False):
    contract = manifest.get("replay_contract")
    validate_contract(contract, direct=direct)
    if manifest.get("fps") != contract["video_mapping"]["fps"]:
        raise ValueError("Manifest FPS and contract disagree")
    ids = []
    for record in manifest["trials"]:
        validate_record(record, contract)
        ids.append(record["trial_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate original trial IDs, including invalid trials")
    if direct:
        validate_contract(contract, direct=True, check_dependencies=True)
    return contract


def validate_legacy_adapter(contract):
    validate_contract(contract)
    if (contract["backend"]["id"] != LEGACY_BACKEND or contract["backend"]["version"] != 1
            or contract["output_space"] != "stimulus" or contract["scene_profile"] is not None
            or contract["capture_storage"] is not None or contract["source_time_kind"] != "reconstructed"
            or contract["video_mapping"]["policy"] != "identity"
            or contract["backend"]["capabilities"] != dict(stimulus_frames=True, scene_frames=False,
                                                            direct_frames=True, capture_storage=False)):
        raise ValueError("No legacy adapter for this backend, timing or scene; no automatic fallback")


def legacy_sidecar(contract, record, frame_times, relative_times, wheel_delta, eid):
    validate_legacy_adapter(contract)
    validate_record(record, contract)
    times = np.asarray(frame_times, dtype=np.float64)
    n = len(times)
    config = record["effective_config"]
    arrays = dict(schema_version=np.int64(REPLAY_SCHEMA_VERSION), eid=np.asarray(eid),
                  trial_id=np.int64(record["trial_id"]), contract_fingerprint=np.asarray(contract["fingerprint"]),
                  record_fingerprint=np.asarray(record["record_fingerprint"]),
                  frame_times=times, relative_times=np.asarray(relative_times, dtype=np.float64),
                  source_frame_index=np.arange(n, dtype=np.int64),
                  valid=np.ones(n, dtype=bool), invalid_reason=np.full(n, "", dtype="U1"),
                  stimulus_state=np.where(times < record["freeze"], STATES["legacy_onset_coupled"],
                                          STATES["legacy_response_hold"]).astype(np.uint8),
                  stimulus_azimuth_deg=record["initial_azimuth_deg"] - wheel_delta * config["wheel_radius_mm"] * config["gain_deg_per_mm"],
                  stimulus_contrast=np.full(n, record["contrast"], dtype=np.float64),
                  stimulus_phase_rad=np.full(n, record["phase_rad"], dtype=np.float64),
                  wheel_delta=np.asarray(wheel_delta, dtype=np.float64),
                  video_pts=np.arange(n if record["video_written"] else 0, dtype=np.float64) / contract["video_mapping"]["fps"],
                  video_source_index=np.arange(n if record["video_written"] else 0, dtype=np.int64))
    validate_sidecar(arrays, contract, record, eid)
    return arrays


def validate_sidecar(data, contract, record, eid):
    """Validate numeric/Unicode arrays without pickle; times remain float64."""
    required = {"schema_version", "eid", "trial_id", "contract_fingerprint", "record_fingerprint",
                "frame_times", "relative_times", "source_frame_index", "valid", "invalid_reason",
                "stimulus_state", "stimulus_azimuth_deg", "stimulus_contrast", "stimulus_phase_rad",
                "video_pts", "video_source_index"}
    if not required.issubset(data.keys()):
        raise ValueError("Legacy/incomplete replay sidecar; regenerate it with its manifest")
    for name, expected in (("schema_version", REPLAY_SCHEMA_VERSION), ("eid", eid), ("trial_id", record["trial_id"]),
                           ("contract_fingerprint", contract["fingerprint"]), ("record_fingerprint", record["record_fingerprint"])):
        if data[name].shape != () or data[name].item() != expected:
            raise ValueError(f"Sidecar {name} mismatch")
    if (data["schema_version"].dtype.kind not in "iu" or data["trial_id"].dtype.kind not in "iu"
            or any(data[name].dtype.kind != "U" for name in ("eid", "contract_fingerprint", "record_fingerprint"))):
        raise ValueError("Sidecar identity fields require integer or Unicode scalar types")
    times, ids, valid = data["frame_times"], data["source_frame_index"], data["valid"]
    n = record["frame_count"]
    if (times.dtype != np.dtype("float64") or times.shape != (n,) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or ids.dtype != np.dtype("int64") or not np.array_equal(ids, np.arange(n))
            or valid.dtype != np.dtype("bool") or valid.shape != (n,)):
        raise ValueError("Invalid source frame indices, float64 session times or validity")
    reasons, states = data["invalid_reason"], data["stimulus_state"]
    if (reasons.dtype.kind != "U" or reasons.shape != (n,)
            or np.any((reasons != "") != ~valid) or states.dtype != np.dtype("uint8") or states.shape != (n,)
            or not np.isin(states, list(STATES.values())).all()):
        raise ValueError("Invalid stimulus state or missing invalid-frame reason")
    for name in ("relative_times", "stimulus_azimuth_deg", "stimulus_contrast", "stimulus_phase_rad"):
        values = data[name]
        if values.dtype != np.dtype("float64") or values.shape != (n,) or not np.isfinite(values[valid]).all():
            raise ValueError(f"Invalid frame field {name}")
    if np.any((data["stimulus_contrast"][valid] < 0) | (data["stimulus_contrast"][valid] > 1)):
        raise ValueError("Frame contrast must be fractional")
    if contract["backend"]["id"] == LEGACY_BACKEND:
        if "wheel_delta" not in data:
            raise ValueError("Legacy adapter requires its wheel trajectory")
        delta = data["wheel_delta"]
        if delta.dtype != np.dtype("float64") or delta.shape != (n,) or not np.isfinite(delta).all():
            raise ValueError("Invalid legacy wheel trajectory")
        config = record["effective_config"]
        expected_position = record["initial_azimuth_deg"] - delta * config["wheel_radius_mm"] * config["gain_deg_per_mm"]
        expected_state = np.where(times < record["freeze"], STATES["legacy_onset_coupled"], STATES["legacy_response_hold"])
        if (not np.allclose(data["stimulus_azimuth_deg"], expected_position, atol=1e-9, rtol=0)
                or not np.array_equal(states, expected_state)
                or not np.all(data["stimulus_contrast"] == record["contrast"])
                or not np.all(data["stimulus_phase_rad"] == record["phase_rad"])):
            raise ValueError("Legacy trajectory and declared frame stimulus state disagree")
    if (not np.allclose(data["relative_times"], times - record["stim_on"], atol=1e-9, rtol=0)
            or np.any(times < record["stim_on"]) or np.any(times >= record["stim_off"])):
        raise ValueError("Frame times must lie in the trial's half-open visible interval")
    pts, mapping = data["video_pts"], data["video_source_index"]
    if (pts.dtype != np.dtype("float64") or pts.ndim != 1 or mapping.dtype != np.dtype("int64")
            or mapping.shape != pts.shape or not np.isfinite(pts).all()
            or not np.allclose(pts, np.arange(len(pts)) / contract["video_mapping"]["fps"], rtol=0, atol=1e-9)
            or np.any(mapping < 0) or np.any(mapping >= n) or np.any(np.diff(mapping) < 0)
            or bool(len(pts)) != record["video_written"]):
        raise ValueError("Invalid video PTS/source mapping")
    if len(pts):
        policy = contract["video_mapping"]["policy"]
        query = record["stim_on"] + pts
        expected = np.searchsorted(times, query, side="right") - 1
        if policy == "identity":
            if not np.array_equal(mapping, ids) or not np.allclose(query, times, atol=1e-7, rtol=0):
                raise ValueError("Identity video mapping requires regular one-to-one source frames")
        elif np.any(query < times[0]) or np.any(query >= record["stim_off"]) or not np.array_equal(mapping, expected):
            raise ValueError("Hold-previous mapping must select the most recent source frame within the visible interval")
