"""Bounded replay observations and content-bound completion, without storage."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from .display import DisplayRenderer
from .preparation import ReconstructionPlan
from .scene import SceneProjector
from .timeline import prepare_trial_timelines, schedule_trial


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _bind(digest, metadata):
    encoded = _json(metadata)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


@dataclass(frozen=True)
class ReplayObservation:
    metadata: dict
    rgb: np.ndarray | None


@dataclass(frozen=True)
class TrialOutcome:
    metadata: dict


class ReplayStream:
    """Single-use iterator of observations and trial outcomes.

    Images are provisional until exhaustion publishes completion. Closing early,
    interruption, or an unexpected exception leaves completion absent. Expected
    trial-local input/rendering errors stop that trial and preserve later trials.
    """

    def __init__(self, plan, *, image_space="mouse_view"):
        if image_space not in {"mouse_view", "display"}:
            raise ValueError("image_space must be mouse_view or display")
        inputs = deepcopy(plan.definition)
        ids = inputs["requested_trial_ids"]
        if (not inputs.get("eid") or not inputs.get("trial_table_fingerprint")
                or not ids or len(set(ids)) != len(ids)
                or any(type(i) is not int or not 0 <= i < inputs["original_trial_count"] for i in ids)):
            raise ValueError("Replay requires source identity and unique original trial IDs")
        arrays = []
        for name, values in (("timestamps", plan.wheel_timestamps), ("positions", plan.wheel_positions)):
            array = np.array(values, dtype="<f8", copy=True)
            if (array.ndim != 1 or len(array) != inputs["wheel"]["samples"]
                    or hashlib.sha256(array.tobytes()).hexdigest() != inputs["wheel"][name + "_sha256"]):
                raise ValueError("Wheel evidence differs from the resolved definition")
            array.setflags(write=False)
            arrays.append(array)
        snapshot = ReconstructionPlan(inputs, *arrays)
        self._plan = snapshot
        self._timelines = prepare_trial_timelines(snapshot)
        self._renderer = DisplayRenderer(snapshot)
        self._projector = SceneProjector(snapshot) if image_space == "mouse_view" else None
        self._image_space = image_space
        self._inputs = inputs
        self._definition = dict(
            schema_version=1, kind="replay_stream_definition", inputs=inputs,
            image_space=image_space, display=self._renderer.provenance,
            scene=None if self._projector is None else self._projector.provenance,
            stream_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime=dict(python=platform.python_version(), numpy=np.__version__),
            digest_policy="sha256; canonical JSON; length-prefixed records; RGB8 C-order image hashes",
        )
        self._definition_id = _digest(self._definition)
        self._completion = None
        self._state = "pending"
        self._iterator = self._run()

    @property
    def definition_id(self):
        return self._definition_id

    @property
    def definition(self):
        return deepcopy(self._definition)

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

    def _observation(self, timeline, time, index, timing):
        metadata = dict(
            kind="observation", definition_id=self.definition_id,
            observation_id=f"trial/{timeline.trial_id}/observation/{index}",
            eid=timeline.eid, trial_id=timeline.trial_id,
            trial_table_fingerprint=timeline.trial_table_fingerprint,
            schedule_index=index, session_time=float(time), timing=deepcopy(timing),
            image_space=self._image_space, format="RGB8", row_origin="top", value_range=[0, 255],
            shape=None, rgb_sha256=None, status="failed", reason=None,
            known_blank=False, stimulus_state=None, render_parameters=None,
        )
        rgb = None
        try:
            state = timeline.evaluate(time)
            state_metadata = asdict(state)
            _json(state_metadata)  # Non-finite evaluated state is a trial failure.
            metadata.update(status=state.status, reason=state.reason,
                            known_blank=state.known_blank, stimulus_state=state_metadata)
            if state.status == "valid":
                display = self._renderer.render(state)
                frame = self._projector.project(display) if self._projector is not None else display
                # Bytes-backed pixels cannot be made writable by a consumer.
                pixels = frame.rgb.tobytes(order="C")
                rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(frame.rgb.shape)
                metadata.update(shape=list(rgb.shape), rgb_sha256=hashlib.sha256(pixels).hexdigest(),
                                render_parameters=deepcopy(display.render_parameters),
                                known_blank=state.known_blank or display.render_parameters.get("opacity") == 0)
        except (ValueError, TypeError, ArithmeticError, OSError) as exc:
            rgb = None
            metadata.update(status="failed", reason=f"{type(exc).__name__}: {exc}", known_blank=False,
                            shape=None, rgb_sha256=None, render_parameters=None)
        return ReplayObservation(metadata, rgb)

    def _run(self):
        self._state = "running"
        digest = hashlib.sha256()
        outcomes = []
        try:
            for timeline in self._timelines:
                trial_digest = hashlib.sha256()
                counts = dict(valid=0, unavailable=0, invalid=0, failed=0)
                result = dict(
                    kind="trial_outcome", definition_id=self.definition_id,
                    eid=timeline.eid, trial_id=timeline.trial_id,
                    trial_table_fingerprint=timeline.trial_table_fingerprint,
                    requested_domain=deepcopy(timeline.record.get("requested_domain")),
                    timing=None, input_coverage=[], input_coverage_status=None,
                    wheel_policy=deepcopy(timeline.wheel_policy),
                    scheduled_count=None, schedule_sha256=None, emitted_count=0,
                    unattempted_count=None, image_count=0, observation_status_counts=counts,
                    status="failed", reason=None,
                )
                try:
                    schedule = schedule_trial(timeline)
                except (ValueError, TypeError, ArithmeticError, OSError) as exc:
                    result["reason"] = f"Scheduling failed: {type(exc).__name__}: {exc}"
                else:
                    result.update(timing=deepcopy(schedule.timing),
                                  input_coverage=[asdict(part) for part in schedule.coverage],
                                  input_coverage_status=schedule.status,
                                  scheduled_count=len(schedule.times),
                                  schedule_sha256=hashlib.sha256(schedule.times.astype("<f8").tobytes()).hexdigest())
                    for index, time in enumerate(schedule.times):
                        observation = self._observation(timeline, time, index, schedule.timing)
                        status = observation.metadata["status"]
                        counts[status] += 1
                        result["emitted_count"] += 1
                        _bind(trial_digest, observation.metadata)
                        _bind(digest, observation.metadata)
                        # Bind content and accounting before exposing mutable metadata.
                        failed_reason = observation.metadata["reason"] if status == "failed" else None
                        yield observation
                        if status == "failed":
                            result.update(status="failed", reason=failed_reason)
                            break
                    else:
                        result["status"] = (
                            "complete" if schedule.status == "available" and counts["valid"] == len(schedule.times)
                            else "partial" if counts["valid"] else
                            "invalid" if schedule.status == "invalid" else "unavailable")
                        result["reason"] = schedule.reason
                    result["unattempted_count"] = result["scheduled_count"] - result["emitted_count"]
                result.update(image_count=counts["valid"], observations_sha256=trial_digest.hexdigest())
                _bind(digest, result)
                outcomes.append(deepcopy(result))
                yield TrialOutcome(result)
            # Reached only after every trial result has been consumed and the
            # iterator was resumed to exhaustion. No finally block may complete.
            self._finish(outcomes, digest)
        finally:
            if self._completion is None:
                self._state = "interrupted"

    def _finish(self, outcomes, digest):
        if [item["trial_id"] for item in outcomes] != self._inputs["requested_trial_ids"]:
            raise ValueError("Cannot complete a replay with missing or reordered trial outcomes")
        status = ("success" if all(x["status"] == "complete" for x in outcomes)
                  else "partial" if any(x["image_count"] for x in outcomes) else "failed")
        completion = dict(
            schema_version=1, kind="replay_completion", definition_id=self.definition_id,
            eid=self._inputs["eid"], trial_table_fingerprint=self._inputs["trial_table_fingerprint"],
            image_space=self._image_space, requested_trial_ids=self._inputs["requested_trial_ids"],
            accounting_complete=True, reconstruction_status=status,
            observation_count=sum(x["emitted_count"] for x in outcomes),
            image_count=sum(x["image_count"] for x in outcomes),
            records_sha256=digest.hexdigest(), trials=outcomes,
        )
        completion["generation_id"] = _digest(completion)
        self._completion = completion
        self._state = "completed"
