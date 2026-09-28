"""Lossless replay publication and verified streaming readback."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .stream import ReplayObservation, ReplayStream, TrialOutcome, _bind, _digest, _json


def _publish_rename(source, destination):
    """Keep publication atomic while tolerating short Windows file locks."""
    if source.parent.resolve() != destination.parent.resolve():
        raise ValueError("Publication must rename within the same directory")
    deadline = time.monotonic() + 5.0
    delay = 0.1
    while True:
        if destination.exists():
            raise FileExistsError(f"Publication destination already exists: {destination}")
        try:
            source.rename(destination)
            return
        except OSError as exc:
            remaining = deadline - time.monotonic()
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            if remaining <= 0:
                raise PermissionError(
                    f"Publication remained locked for 5 seconds: {source} -> {destination}. "
                    "The rename was not completed and staged files were preserved. "
                    "Check for a process holding this path open."
                ) from exc
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 1.0)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with path.open("xb") as target:
        target.write(_json(value) + b"\n")
        target.flush()
        os.fsync(target.fileno())


def _read_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its generation directory")
    return path


def _video_entry(metadata, fps):
    index = metadata["schedule_index"]
    return dict(encoded_index=index, presentation_time=index / fps,
                observation_id=metadata["observation_id"], session_time=metadata["session_time"])


def _load_image(directory, index, image_format):
    """Decode one lossless frame in memory, including legacy raw arrays."""
    if image_format == "npy_rgb8":
        return np.load(_inside(directory, f"images/{index}.npy"), allow_pickle=False)
    if image_format == "npz_rgb8":
        with np.load(_inside(directory, f"images/{index}.npz"), allow_pickle=False) as archive:
            if archive.files != ["rgb"]:
                raise ValueError("Compressed image must contain exactly one RGB array")
            return archive["rgb"]
    raise ValueError("Unsupported replay image format")


def _encode_video(directory, outcome, image_format):
    """One encoded frame per observation, only for complete trials."""
    import cv2

    fps = outcome["timing"]["cadence_hz"]
    writer = None
    temporary = directory / ".video-staging.mp4"
    mapping_path = directory / ".video-mapping-staging.jsonl"
    try:
        with (directory / "observations.jsonl").open(encoding="utf-8") as records, mapping_path.open("xb") as mapping:
            for line in records:
                metadata = json.loads(line)
                rgb = _load_image(directory, metadata["schedule_index"], image_format)
                height, width = rgb.shape[:2]
                if writer is None:
                    if width % 2 or height % 2:
                        raise ValueError("MP4 requires even image dimensions; canonical images are unchanged")
                    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 encoder could not be opened")
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                mapping.write(_json(_video_entry(metadata, fps)) + b"\n")
        if writer is None:
            raise ValueError("No images to encode")
        writer.release()
        writer = None
        capture = cv2.VideoCapture(str(temporary))
        count = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if image.shape != (height, width, 3):
                    raise RuntimeError("Encoded video dimensions changed")
                count += 1
        finally:
            capture.release()
        if count != outcome["image_count"]:
            raise RuntimeError("Encoded video does not contain every source observation")
        _publish_rename(temporary, directory / "video.mp4")
        _publish_rename(mapping_path, directory / "video-mapping.jsonl")
        return dict(status="available", codec="mp4v", fps=fps, frame_count=count,
                    policy="one frame per observation; no repeats or drops",
                    video_sha256=_file_hash(directory / "video.mp4"),
                    mapping_sha256=_file_hash(directory / "video-mapping.jsonl"))
    except cv2.error as exc:
        raise RuntimeError(str(exc)) from exc
    finally:
        if writer is not None:
            writer.release()


def write_replay(replay, output, *, video=False, on_trial_published=None):
    """Consume a fresh ReplayStream into a new directory; publish manifest last.

    Existing output is never replaced. Storage errors or interruption propagate,
    leaving published trials inspectable but no completed generation manifest.
    Optional video failure is explicit and does not invalidate lossless images.
    """
    if not isinstance(replay, ReplayStream) or replay.state != "pending":
        raise ValueError("Publication requires a fresh ReplayStream")
    if type(video) is not bool:
        raise TypeError("video must be a boolean")
    if on_trial_published is not None and not callable(on_trial_published):
        raise TypeError("on_trial_published must be callable")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "trials").mkdir()
    definition = replay.definition
    _write_json(root / "definition.json", definition)
    published = []
    records = None
    staging = None
    try:
        with replay:
            for item in replay:
                metadata = item.metadata
                trial_id = metadata["trial_id"]
                if staging is None:
                    staging = root / "trials" / f".staging-{trial_id}"
                    staging.mkdir()
                    (staging / "images").mkdir()
                    records = (staging / "observations.jsonl").open("xb")
                if isinstance(item, ReplayObservation):
                    if item.rgb is not None:
                        with (staging / "images" / f"{metadata['schedule_index']}.npz").open("xb") as image:
                            np.savez_compressed(image, rgb=item.rgb)
                    records.write(_json(metadata) + b"\n")
                else:
                    records.flush()
                    os.fsync(records.fileno())
                    records.close()
                    records = None
                    _write_json(staging / "outcome.json", metadata)
                    video_result = dict(status="not_requested")
                    if video:
                        if metadata["status"] != "complete":
                            video_result = dict(status="skipped", reason="Trial reconstruction is incomplete; gaps are not encoded")
                        else:
                            try:
                                video_result = _encode_video(staging, metadata, "npz_rgb8")
                            except (ImportError, ValueError, OSError, RuntimeError) as exc:
                                video_result = dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
                    entry = dict(trial_id=trial_id,
                                 observations_file_sha256=_file_hash(staging / "observations.jsonl"),
                                 outcome_file_sha256=_file_hash(staging / "outcome.json"), video=video_result)
                    _publish_rename(staging, root / "trials" / str(trial_id))
                    staging = None
                    published.append(entry)
                    if on_trial_published is not None:
                        on_trial_published(deepcopy(metadata))
        completion = replay.completion
        if completion is None or [item["trial_id"] for item in published] != completion["requested_trial_ids"]:
            raise ValueError("Cannot publish an unfinished or unaccounted replay")
        manifest = dict(schema_version=1, kind="replay_artifacts", format="npz_rgb8",
                        writer_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        definition_id=replay.definition_id, completion=completion, trials=published)
        manifest["artifact_id"] = _digest(manifest)
        _write_json(root / ".manifest-staging.json", manifest)
        _publish_rename(root / ".manifest-staging.json", root / "manifest.json")
        return manifest
    finally:
        if records is not None:
            records.close()


class ReplayArtifactReader:
    """Single-use verified readback, yielding the same records as ReplayStream.

    A published manifest is required. Completion stays absent until every record
    and lossless image has been checked and the iterator reaches exhaustion.
    """

    def __init__(self, directory):
        self._root = Path(directory).resolve()
        manifest = _read_json(_inside(self._root, "manifest.json"))
        if (manifest.get("schema_version") != 1 or manifest.get("kind") != "replay_artifacts"
                or manifest.get("format") not in ("npy_rgb8", "npz_rgb8")
                or manifest.get("artifact_id") != _digest({k: v for k, v in manifest.items() if k != "artifact_id"})):
            raise ValueError("Invalid replay artifact manifest")
        definition = _read_json(_inside(self._root, "definition.json"))
        completion = manifest["completion"]
        ids = definition["inputs"]["requested_trial_ids"]
        if (definition.get("schema_version") != 1 or definition.get("kind") != "replay_stream_definition"
                or not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)
                or _digest(definition) != manifest["definition_id"]
                or completion["definition_id"] != manifest["definition_id"]
                or completion.get("schema_version") != 1 or completion.get("kind") != "replay_completion"
                or completion.get("accounting_complete") is not True
                or completion["eid"] != definition["inputs"]["eid"]
                or completion["trial_table_fingerprint"] != definition["inputs"]["trial_table_fingerprint"]
                or completion["image_space"] != definition["image_space"]
                or completion["requested_trial_ids"] != ids
                or [x["trial_id"] for x in completion["trials"]] != ids
                or [x["trial_id"] for x in manifest["trials"]] != ids
                or completion["generation_id"] != _digest({k: v for k, v in completion.items() if k != "generation_id"})):
            raise ValueError("Replay definition/completion identity mismatch")
        self._manifest, self._definition = manifest, definition
        self._completion = None
        self._state = "pending"
        self._iterator = self._run()

    @property
    def definition_id(self):
        return self._manifest["definition_id"]

    @property
    def definition(self):
        return deepcopy(self._definition)

    @property
    def artifact_manifest(self):
        return deepcopy(self._manifest)

    @property
    def completion(self):
        return deepcopy(self._completion)

    @property
    def state(self):
        return self._state

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self._iterator.close()
        if self._completion is None:
            self._state = "interrupted"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self):
        self._state = "running"
        digest = hashlib.sha256()
        declared = self._manifest["completion"]
        try:
            for entry, expected in zip(self._manifest["trials"], declared["trials"]):
                directory = _inside(self._root, f"trials/{entry['trial_id']}")
                records_path = _inside(directory, "observations.jsonl")
                outcome_path = _inside(directory, "outcome.json")
                if (_file_hash(records_path) != entry["observations_file_sha256"]
                        or _file_hash(outcome_path) != entry["outcome_file_sha256"]):
                    raise ValueError("Trial metadata file hash mismatch")
                outcome = _read_json(outcome_path)
                if outcome != expected:
                    raise ValueError("Trial outcome differs from completed generation")
                trial_digest = hashlib.sha256()
                counts = dict(valid=0, unavailable=0, invalid=0, failed=0)
                count = 0
                previous = -math.inf
                with ExitStack() as stack:
                    records = stack.enter_context(records_path.open(encoding="utf-8"))
                    mapping = None
                    video = entry["video"]
                    if video["status"] == "available":
                        video_path = _inside(directory, "video.mp4")
                        mapping_path = _inside(directory, "video-mapping.jsonl")
                        if (_file_hash(video_path) != video["video_sha256"]
                                or _file_hash(mapping_path) != video["mapping_sha256"]
                                or outcome["status"] != "complete"
                                or video["frame_count"] != outcome["image_count"]):
                            raise ValueError("Video artifact mismatch")
                        mapping = stack.enter_context(mapping_path.open(encoding="utf-8"))
                    for line in records:
                        metadata = json.loads(line)
                        if (metadata["kind"] != "observation" or metadata["definition_id"] != self.definition_id
                                or metadata["eid"] != declared["eid"]
                                or metadata["trial_id"] != entry["trial_id"]
                                or metadata["trial_table_fingerprint"] != declared["trial_table_fingerprint"]
                                or metadata["image_space"] != declared["image_space"]
                                or metadata["schedule_index"] != count
                                or metadata["observation_id"] != f"trial/{entry['trial_id']}/observation/{count}"
                                or not math.isfinite(metadata["session_time"])
                                or metadata["session_time"] <= previous
                                or metadata["timing"] != outcome["timing"]
                                or metadata["format"] != "RGB8" or metadata["row_origin"] != "top"
                                or metadata["value_range"] != [0, 255] or metadata["status"] not in counts):
                            raise ValueError("Invalid observation association or format")
                        rgb = None
                        if metadata["status"] == "valid":
                            pixels = _load_image(directory, count, self._manifest["format"])
                            if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
                                    or list(pixels.shape) != metadata["shape"]
                                    or hashlib.sha256(pixels.tobytes(order="C")).hexdigest() != metadata["rgb_sha256"]):
                                raise ValueError("Lossless image content differs from its observation")
                            rgb = np.frombuffer(pixels.tobytes(order="C"), dtype=np.uint8).reshape(pixels.shape)
                        elif metadata["rgb_sha256"] is not None or metadata["shape"] is not None or metadata["known_blank"]:
                            raise ValueError("Unavailable observation claims valid image content")
                        if mapping is not None and json.loads(mapping.readline()) != _video_entry(metadata, video["fps"]):
                            raise ValueError("Encoded-frame/source-observation mapping mismatch")
                        previous = metadata["session_time"]
                        counts[metadata["status"]] += 1
                        count += 1
                        _bind(trial_digest, metadata)
                        _bind(digest, metadata)
                        yield ReplayObservation(metadata, rgb)
                    if mapping is not None and mapping.readline():
                        raise ValueError("Video mapping has extra frames")
                if (trial_digest.hexdigest() != outcome["observations_sha256"]
                        or count != outcome["emitted_count"] or counts != outcome["observation_status_counts"]
                        or counts["valid"] != outcome["image_count"]):
                    raise ValueError("Trial content/accounting mismatch")
                _bind(digest, outcome)
                yield TrialOutcome(deepcopy(outcome))
            results = declared["trials"]
            status = ("success" if all(x["status"] == "complete" for x in results)
                      else "partial" if any(x["image_count"] for x in results) else "failed")
            if (digest.hexdigest() != declared["records_sha256"]
                    or sum(x["image_count"] for x in results) != declared["image_count"]
                    or sum(x["emitted_count"] for x in results) != declared["observation_count"]
                    or status != declared["reconstruction_status"]):
                raise ValueError("Generation content digest mismatch")
            self._completion = deepcopy(declared)
            self._state = "completed"
        finally:
            if self._completion is None:
                self._state = "interrupted"
