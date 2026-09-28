"""Display-space functional reconstruction of the pinned IBL/BonVision path.

Derived from IBL Gabor2D.bonsai and BonVision DrawGratings, Gratings.frag,
CreateSphereGrid, SphereMapping and ViewWindow. Immutable references and the
explicit continuous-sampling approximation are recorded in profiles.py.
This is not a historical GPU rasterizer or a mouse-perspective projector.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from .profiles import behavior_profile
from .timeline import StimulusState


@dataclass(frozen=True)
class DisplayFrame:
    rgb: np.ndarray
    eid: str
    trial_id: int
    session_time: float
    image_space: str
    render_parameters: dict


def _finite(value, field, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid {field}")
    return value


class DisplayRenderer:
    """Render resolved states to RGB8 display frames without reading wheel data.

    Construct once per reconstruction plan; angular pixel coordinates are reused.
    Invalid/unavailable state raises ValueError, including zero-contrast state.
    """

    def __init__(self, plan):
        definition = plan.definition
        profile = behavior_profile(definition["profile"]["id"])
        if definition["profile"] != profile:
            raise ValueError("Resolved display profile differs from the supported profile")
        self.eid = definition["eid"]
        self.profile_id = profile["id"]
        self.trial_table_fingerprint = definition["trial_table_fingerprint"]
        self._trial_ids = frozenset(definition["requested_trial_ids"])
        self._profile = profile
        self._geometry = deepcopy(definition["display_geometry"])
        self._policy = deepcopy(profile["display"]["renderer"])
        size = definition["configuration"]["scene"]["display_size"]
        if len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
            raise ValueError("Display size requires [width, height] positive integers")
        self.width, self.height = size
        width = _finite(self._geometry["width"], "ViewWindow width", positive=True)
        height = _finite(self._geometry["height"], "ViewWindow height", positive=True)
        distance = _finite(self._geometry["distance"], "ViewWindow distance", positive=True)
        self._span = math.degrees(2 * math.atan(width / (2 * distance)))
        if not math.isclose(self._span, self._geometry["horizontal_fov_deg"], rel_tol=0, abs_tol=1e-9):
            raise ValueError("Resolved visual span disagrees with ViewWindow geometry")
        # Invert the front-facing source ViewWindow rays (x,y,-distance) into
        # the equirectangular stimulus texture used by CreateSphereGrid. This
        # source display mapping is distinct from the physical scene camera.
        x = ((np.arange(self.width, dtype=np.float64) + 0.5) / self.width - 0.5) * width
        y = (0.5 - (np.arange(self.height, dtype=np.float64) + 0.5) / self.height) * height
        self._azimuth = np.degrees(np.arctan2(x, distance))[None, :]
        self._elevation = np.degrees(np.arctan2(y[:, None], np.hypot(x, distance)[None, :]))
        self._azimuth.setflags(write=False)
        self._elevation.setflags(write=False)
        self.provenance = dict(
            profile_id=self.profile_id,
            references=deepcopy(profile["references"]),
            shader_reference=deepcopy(profile["display"]["shader_reference"]),
            display_geometry=deepcopy(self._geometry),
            policy=deepcopy(self._policy),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            output=dict(image_space="display", format="RGB8", shape=[self.height, self.width, 3],
                        row_origin="top", value_range=[0, 255]),
        )

    def render(self, state: StimulusState):
        if not isinstance(state, StimulusState):
            raise TypeError("Expected a resolved StimulusState")
        if (state.eid != self.eid or state.trial_id not in self._trial_ids
                or state.profile_id != self.profile_id
                or state.trial_table_fingerprint != self.trial_table_fingerprint):
            raise ValueError("State does not belong to this replay source/profile")
        if state.status != "valid" or type(state.visible) is not bool:
            raise ValueError(f"Cannot render {state.status} stimulus state: {state.reason}")
        _finite(state.session_time, "session time")
        background = np.asarray(self._policy["background_rgb"], dtype=np.uint8)
        rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        rgb[:] = background
        applied = dict(visible=state.visible, renderer_id=self._policy["id"],
                       background_rgb=background.tolist())
        if state.visible:
            parameters = state.parameters
            contrast = _finite(parameters["contrast"], "contrast")
            if not 0 <= contrast <= 1:
                raise ValueError("Contrast must be in [0, 1]")
            position = _finite(state.azimuth_deg, "resolved azimuth")
            frequency = _finite(parameters["spatial_frequency_cpd"], "frequency", positive=True)
            sigma = _finite(parameters["sigma_deg"], "task sigma", positive=True)
            phase = _finite(parameters["phase_rad"], "task phase")
            logged_angle = _finite(parameters["orientation_deg"], "logged orientation")
            if not math.isclose(parameters["horizontal_fov_deg"], self._span, rel_tol=0, abs_tol=1e-9):
                raise ValueError("State visual span differs from resolved display geometry")
            # IBL squares the transmitted task sigma *before* the linear
            # StimSize/VisualSpan rescale into both Radius and Aperture.
            radius = aperture = sigma ** self._policy["sigma_power"] / self._span
            if not math.isfinite(radius * aperture) or radius * aperture <= 0:
                raise ValueError("Task sigma is outside the supported numerical range")
            phase_cycles = self._policy["phase_sign"] * phase / (2 * math.pi)
            applied.update(azimuth_deg=position, contrast=contrast, spatial_frequency_cpd=frequency,
                           task_sigma_deg=sigma, radius=radius, aperture=aperture,
                           task_phase_rad=phase, shader_phase_cycles=phase_cycles,
                           logged_orientation_deg=logged_angle,
                           effective_orientation_deg=self._policy["effective_orientation_deg"],
                           angular_extent_deg=self._span,
                           opacity=0 if contrast <= self._policy["opacity_threshold"] else 1)
            if applied["opacity"]:
                dx = self._azimuth - position
                dy = self._elevation  # Source LocationY and Angle are fixed at zero.
                # Quad clipping precedes sphere mapping; do not wrap the quad
                # around the angular texture or clamp its center to the display.
                inside = (np.abs(dx) <= self._span / 2) & (np.abs(dy) <= self._span / 2)
                distance = 2 * np.hypot(dx, dy) / self._span / radius / aperture
                envelope = np.exp(-0.5 * distance * distance)
                envelope = np.where(inside, envelope, 0.0)
                carrier = np.sin(2 * math.pi * (dx * frequency + phase_cycles))
                shader_gray = 0.5 + 0.5 * contrast * carrier * envelope
                # Gratings -> RGBA texture with SrcAlpha/OneMinusSrcAlpha.
                # Then MeshMap blends that texture into an RGB cubemap. Its
                # sampled alpha is one during final ViewWindow presentation.
                gray = background[0] / 255.0
                texture_rgb = shader_gray * envelope + gray * (1 - envelope)
                texture_alpha = envelope * envelope + (1 - envelope)
                display_gray = texture_rgb * texture_alpha + gray * (1 - texture_alpha)
                channel = np.rint(255 * display_gray).clip(0, 255).astype(np.uint8)
                rgb[:] = channel[:, :, None]
        rgb.setflags(write=False)
        return DisplayFrame(rgb, state.eid, state.trial_id, state.session_time, "display", applied)
