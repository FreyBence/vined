"""Compressed, content-bound feature publication and bounded verified readback."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np

from utils.provenance import file_hash, fingerprint
from utils.progress import Progress, iter_progress, logger
from visual_replay import TrialOutcome
from .encoder import EncodedObservation, iter_encoded_observations


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _bind(digest, value):
    encoded = _json(value)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _check_identity(value, field):
    if value.get(field) != fingerprint({key: item for key, item in value.items() if key != field}):
        raise ValueError(f"Invalid {field}")


def write_features(observations, encoder, output, *, batch_size=32, workers=1):
    """Publish a new archive only after extraction, source completion and readback.

    Existing destinations are refused. A failed run never returns an old result.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Feature output already exists: {output}; select a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    definition = dict(schema_version=1, kind="visual_feature_definition",
        replay_definition=observations.definition, replay_definition_id=observations.definition_id,
        selection=observations.selection, representation=encoder.provenance,
        batch_size=batch_size, writer_sha256=file_hash(__file__))
    definition_id = fingerprint(definition)
    count = 0
    summaries = []
    trial_count = 0
    selected_count = 0
    eid = observations.definition["inputs"]["eid"]
    progress = Progress(f"visual-features {eid}: extracting", unit="observations")
    trials_progress = Progress(f"visual-features {eid}: extracting trials",
        len(observations.definition["inputs"]["requested_trial_ids"]), "trials")
    observation_count = 0
    with tempfile.TemporaryDirectory(prefix=".features-", dir=output.parent) as temporary:
        staging = Path(temporary)
        with (staging / "records.jsonl").open("wb") as records, (staging / "pixels-free-features").open("wb") as values:
            with closing(iter_encoded_observations(observations, encoder, batch_size=batch_size,
                                                   workers=workers)) as stream:
                for item in stream:
                    if isinstance(item, EncodedObservation):
                        index = None
                        selected_count += int(item.selected)
                        if item.feature is not None:
                            index = count
                            values.write(item.feature.astype("<f4", copy=False).tobytes())
                            count += 1
                            trial_count += 1
                        record = dict(kind="observation", metadata=item.metadata, selected=item.selected,
                                      status=item.status, feature_index=index)
                    else:
                        summaries.append(dict(trial_id=item.metadata["trial_id"], feature_count=trial_count,
                                              selected_count=selected_count))
                        trial_count = selected_count = 0
                        record = dict(kind="trial_outcome", metadata=item.metadata)
                    records.write(_json(record) + b"\n")
                    if isinstance(item, EncodedObservation):
                        observation_count += 1
                        progress.update(observation_count,
                            f"{count} features, {len(summaries)} trials completed, trial_id={item.metadata['trial_id']}")
                    else:
                        trials_progress.update(len(summaries), f"{count} features")
        completion = observations.completion
        if completion is None:
            raise ValueError("Cannot publish features without verified replay completion")
        progress.finish()
        trials_progress.finish()
        logger.info("visual-features %s: extraction complete (%d/%d trials, %d features); hashing and compressing %s",
                    eid, len(summaries), len(completion["requested_trial_ids"]), count, output)
        manifest = dict(schema_version=1, kind="visual_feature_artifacts", definition_id=definition_id,
            replay_completion=completion, feature_count=count, trials=summaries,
            records_sha256=file_hash(staging / "records.jsonl"),
            features_sha256=file_hash(staging / "pixels-free-features"))
        manifest["generation_id"] = fingerprint(manifest)
        archive_path = staging / "features.npz"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            archive.writestr("definition.json", _json(definition))
            archive.write(staging / "records.jsonl", "records.jsonl")
            with archive.open("features.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array_header_2_0(member, dict(
                    descr="<f4", fortran_order=False, shape=(count, 768)))
                with (staging / "pixels-free-features").open("rb") as source:
                    shutil.copyfileobj(source, member, length=1024 * 1024)
            archive.writestr("manifest.json", _json(manifest))
        # Exercise the public reader before exposing the completed generation.
        logger.info("visual-features %s: verifying compressed archive", eid)
        with FeatureArtifactReader(archive_path) as reader:
            for _ in iter_progress(reader, f"visual-features {eid}: verifying archive",
                                   total=observation_count + len(summaries), unit="records"):
                pass
        # Same-filesystem hard-link publication is atomic and refuses replacement.
        os.link(archive_path, output)
    logger.info("visual-features %s: published %s", eid, output)
    return manifest


class FeatureArtifactReader:
    """Single-use readback; completion is withheld until every record is verified."""

    def __init__(self, path, *, eid=None, replay_generation_id=None,
                 representation=None, selection=None):
        self._archive = zipfile.ZipFile(path)
        try:
            if sorted(self._archive.namelist()) != sorted([
                    "definition.json", "manifest.json", "records.jsonl", "features.npy"]):
                raise ValueError("Unsupported or legacy feature archive; regenerate through the replay interface")
            self._definition = json.loads(self._archive.read("definition.json"))
            self._manifest = json.loads(self._archive.read("manifest.json"))
            definition, manifest = self._definition, self._manifest
            if (manifest.get("schema_version") != 1 or manifest.get("kind") != "visual_feature_artifacts"
                    or definition.get("schema_version") != 1 or definition.get("kind") != "visual_feature_definition"
                    or fingerprint(definition) != manifest["definition_id"]):
                raise ValueError("Invalid feature definition/manifest")
            _check_identity(manifest, "generation_id")
            source = manifest["replay_completion"]
            _check_identity(source, "generation_id")
            replay_definition = definition["replay_definition"]
            if (fingerprint(replay_definition) != definition["replay_definition_id"]
                    or source["definition_id"] != definition["replay_definition_id"]
                    or not source["accounting_complete"] or source["image_space"] != "mouse_view"
                    or replay_definition["image_space"] != "mouse_view"
                    or source["eid"] != replay_definition["inputs"]["eid"]
                    or source["trial_table_fingerprint"] != replay_definition["inputs"]["trial_table_fingerprint"]
                    or source["requested_trial_ids"] != replay_definition["inputs"]["requested_trial_ids"]
                    or [x["trial_id"] for x in source["trials"]] != source["requested_trial_ids"]
                    or [x["trial_id"] for x in manifest["trials"]] != source["requested_trial_ids"]):
                raise ValueError("Feature/replay identity or accounting mismatch")
            rep = definition["representation"]
            required = {"model", "revision", "artifact_hashes", "output", "feature_width", "dtype",
                        "normalization", "preparation", "processor", "packages", "sources", "device"}
            if (not required.issubset(rep) or not re.fullmatch(r"[0-9a-f]{40}", rep["revision"])
                    or not rep["artifact_hashes"] or rep["feature_width"] != 768
                    or rep["dtype"] != "float32" or rep["normalization"] != "L2 per observation"):
                raise ValueError("Missing or unsupported feature representation")
            for requested, actual, name in (
                (eid, source["eid"], "EID"),
                (replay_generation_id, source["generation_id"], "replay generation"),
                (representation, rep, "representation"), (selection, definition["selection"], "selection")):
                if requested is not None and requested != actual:
                    raise ValueError(f"Incompatible feature {name}")
        except BaseException:
            self._archive.close()
            raise
        self._completion = None
        self._state = "pending"
        self._iterator = self._run()

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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._iterator.close()
        self._archive.close()
        if self._completion is None:
            self._state = "interrupted"

    def _run(self):
        self._state = "running"
        manifest = self._manifest
        source = manifest["replay_completion"]
        records_hash, features_hash, source_hash = (hashlib.sha256() for _ in range(3))
        count = trial_index = trial_features = trial_selected = observation_index = 0
        previous = -math.inf
        try:
            with self._archive.open("records.jsonl") as records, self._archive.open("features.npy") as features:
                if np.lib.format.read_magic(features) != (2, 0):
                    raise ValueError("Unsupported feature array header")
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(features)
                if shape != (manifest["feature_count"], 768) or fortran or dtype != np.dtype("<f4"):
                    raise ValueError("Invalid feature array shape/type")
                for line in records:
                    records_hash.update(line)
                    record = json.loads(line)
                    metadata = record["metadata"]
                    if trial_index >= len(source["trials"]):
                        raise ValueError("Extra feature records")
                    outcome = source["trials"][trial_index]
                    if (metadata["trial_id"] != outcome["trial_id"] or metadata["eid"] != source["eid"]
                            or metadata["definition_id"] != source["definition_id"]
                            or metadata["trial_table_fingerprint"] != source["trial_table_fingerprint"]):
                        raise ValueError("Feature observation identity mismatch")
                    _bind(source_hash, metadata)
                    if record["kind"] == "trial_outcome":
                        if (metadata != outcome or observation_index != outcome["emitted_count"]
                                or manifest["trials"][trial_index] != dict(trial_id=outcome["trial_id"],
                                    feature_count=trial_features, selected_count=trial_selected)):
                            raise ValueError("Feature trial accounting mismatch")
                        trial_index += 1
                        trial_features = trial_selected = observation_index = 0
                        previous = -math.inf
                        yield TrialOutcome(deepcopy(metadata))
                        continue
                    if (record["kind"] != "observation" or metadata["kind"] != "observation"
                            or type(record["selected"]) is not bool
                            or metadata["schedule_index"] != observation_index
                            or metadata["observation_id"] != f"trial/{outcome['trial_id']}/observation/{observation_index}"
                            or not math.isfinite(metadata["session_time"]) or metadata["session_time"] <= previous
                            or metadata["image_space"] != "mouse_view"
                            or metadata["status"] not in ("valid", "unavailable", "invalid", "failed")):
                        raise ValueError("Invalid feature observation")
                    previous = metadata["session_time"]
                    observation_index += 1
                    selected = record["selected"]
                    trial_selected += int(selected)
                    expected_status = ("not_selected" if not selected else
                                       "encoded" if metadata["status"] == "valid" else "source_unavailable")
                    if record["status"] != expected_status:
                        raise ValueError("Feature/source validity mismatch")
                    vector = None
                    if expected_status == "encoded":
                        if record["feature_index"] != count:
                            raise ValueError("Feature association mismatch")
                        raw = features.read(768 * 4)
                        features_hash.update(raw)
                        vector = np.frombuffer(raw, dtype="<f4")
                        if (vector.shape != (768,) or not np.isfinite(vector).all()
                                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)):
                            raise ValueError("Invalid normalized feature vector")
                        count += 1
                        trial_features += 1
                    elif record["feature_index"] is not None:
                        raise ValueError("Unencoded observation claims a feature")
                    yield EncodedObservation(deepcopy(metadata), selected, vector, expected_status)
                if (features.read(1) or count != manifest["feature_count"]
                        or trial_index != len(source["trials"])
                        or records_hash.hexdigest() != manifest["records_sha256"]
                        or features_hash.hexdigest() != manifest["features_sha256"]
                        or source_hash.hexdigest() != source["records_sha256"]):
                    raise ValueError("Incomplete or corrupted feature generation")
            self._completion = deepcopy(manifest)
            self._state = "completed"
        finally:
            self._archive.close()
            if self._completion is None:
                self._state = "interrupted"
