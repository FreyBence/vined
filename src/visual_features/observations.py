"""Bounded selection that preserves every source observation's metadata."""

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from visual_replay import ReplayArtifactReader, ReplayObservation, ReplayStream, TrialOutcome


@dataclass(frozen=True)
class FeatureObservation:
    """Source metadata and selection; pixels exist only for selected valid input.

    An unselected record is not invalid. Consult metadata['status'] for source
    validity and selected for the independent feature-selection decision.
    """

    metadata: dict
    selected: bool
    rgb: np.ndarray | None


class ObservationSelection:
    """Single-use mouse-view input, owning and exhausting a fresh replay reader.

    Every observation emits metadata, even when not selected, so sampling cannot
    erase invalid regions. TrialOutcome records pass through without filtering.
    Completion remains absent until the upstream iterator is fully exhausted.
    """

    def __init__(self, replay, *, sample_fps=None):
        if not isinstance(replay, (ReplayStream, ReplayArtifactReader)):
            raise TypeError("Expected ReplayStream or ReplayArtifactReader")
        if replay.state != "pending":
            raise ValueError("Selection requires a fresh replay")
        if sample_fps is not None:
            if isinstance(sample_fps, bool) or not isinstance(sample_fps, (int, float)):
                raise TypeError("sample_fps must be a positive number or None")
            if not math.isfinite(sample_fps) or sample_fps <= 0:
                raise ValueError("sample_fps must be finite and positive")
        definition = replay.definition
        if definition["image_space"] != "mouse_view":
            raise ValueError("Visual features require mouse_view observations")
        self._replay = replay
        self._definition = definition
        self._sample_fps = sample_fps
        self._completion = None
        self._state = "pending"
        self._iterator = self._run()

    @property
    def definition(self):
        return deepcopy(self._definition)

    @property
    def definition_id(self):
        return self._replay.definition_id

    @property
    def selection(self):
        return dict(sample_fps=self._sample_fps,
                    policy="all observations" if self._sample_fps is None else
                    "first observation at or after each target; coalesce targets passed by one observation",
                    anchor="first source observation in each trial",
                    time_reference="session_seconds")

    @property
    def completion(self):
        """Unchanged upstream completion, not a completed feature generation."""
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
        self._replay.close()
        if self._completion is None:
            self._state = "interrupted"

    def _run(self):
        self._state = "running"
        anchor = None
        target_index = 0
        try:
            with self._replay as replay:
                for item in replay:
                    if isinstance(item, TrialOutcome):
                        anchor = None
                        target_index = 0
                        yield TrialOutcome(deepcopy(item.metadata))
                        continue
                    if not isinstance(item, ReplayObservation):
                        raise TypeError("Unexpected replay record")
                    metadata = deepcopy(item.metadata)
                    time = metadata["session_time"]
                    if anchor is None:
                        anchor = time
                    selected = self._sample_fps is None
                    if self._sample_fps is not None:
                        elapsed = time - anchor
                        selected = elapsed >= target_index / self._sample_fps
                        if selected:
                            # One source observation may pass several targets,
                            # but it must never be emitted more than once.
                            target_index = max(target_index + 1,
                                               math.floor(elapsed * self._sample_fps) + 1)
                    yield FeatureObservation(metadata, selected, item.rgb if selected else None)
                completion = replay.completion
                if completion is None or not completion["accounting_complete"]:
                    raise ValueError("Replay did not complete trial accounting")
                self._completion = completion
                self._state = "completed"
        finally:
            if self._completion is None:
                self._state = "interrupted"
