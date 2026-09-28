"""Project completed display pixels onto the resolved frontal physical screen."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from .display import DisplayFrame, _finite


@dataclass(frozen=True)
class SceneFrame:
    rgb: np.ndarray
    eid: str
    trial_id: int
    session_time: float
    image_space: str
    display_render_parameters: dict


class SceneProjector:
    """Optional display-to-mouse-view stage, with no task or wheel evaluation.

    A fixed frontal pinhole camera sees an unlit rectangular screen. Projection
    samples the completed RGB display texture bilinearly at output pixel centers.
    Construct once per plan and retain provenance alongside returned frames.
    """

    def __init__(self, plan):
        definition = plan.definition
        scene = deepcopy(definition["configuration"]["scene"])
        self.eid = definition["eid"]
        self._trial_ids = frozenset(definition["requested_trial_ids"])
        for name in ("display_size", "image_size"):
            size = scene[name]
            if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
                raise ValueError(f"{name} requires positive integer width and height")
        self._display_width, self._display_height = scene["display_size"]
        self.width, self.height = scene["image_size"]
        screen_width = _finite(scene["screen_width_mm"], "screen width", positive=True)
        screen_height = _finite(scene["screen_height_mm"], "screen height", positive=True)
        distance = _finite(scene["distance_mm"], "screen distance", positive=True)
        fov = _finite(scene["horizontal_fov_deg"], "camera horizontal FOV", positive=True)
        if fov >= 180:
            raise ValueError("Perspective field of view must be below 180 degrees")
        pose = dict(camera_position_mm=[0.0, 0.0, 0.0],
                    camera_direction=[0.0, 0.0, 1.0], camera_up=[0.0, 1.0, 0.0],
                    screen_center_mm=[0.0, 0.0, distance], screen_normal=[0.0, 0.0, -1.0])
        if any(scene[key] != value for key, value in pose.items()):
            raise ValueError("Only the resolved fixed frontal screen/camera pose is supported")
        if scene["surround_rgb"] != [128, 128, 128]:
            raise ValueError("Unsupported neutral surround")
        if not math.isclose(screen_width / screen_height,
                            self._display_width / self._display_height, rel_tol=0.01):
            raise ValueError("Display raster must preserve physical screen aspect ratio")
        view_width = 2 * distance * math.tan(math.radians(fov / 2))
        view_height = view_width * self.height / self.width
        if not math.isfinite(view_width) or not math.isfinite(view_height):
            raise ValueError("Camera dimensions are outside the supported numerical range")
        if view_width < screen_width or view_height < screen_height:
            raise ValueError("Mouse camera must contain the complete screen")

        # Intersect pinhole rays with z=distance. Square output pixels give the
        # same millimeters-per-pixel on both axes, preserving screen aspect.
        x = ((np.arange(self.width) + 0.5) / self.width - 0.5) * view_width
        y = ((np.arange(self.height) + 0.5) / self.height - 0.5) * view_height
        self._cols = np.flatnonzero(np.abs(x) <= screen_width / 2)
        self._rows = np.flatnonzero(np.abs(y) <= screen_height / 2)
        if not self._cols.size or not self._rows.size:
            raise ValueError("Configured screen covers no output pixel centers")
        u = (x[self._cols] / screen_width + 0.5) * self._display_width - 0.5
        v = (y[self._rows] / screen_height + 0.5) * self._display_height - 0.5
        u = np.clip(u, 0, self._display_width - 1)
        v = np.clip(v, 0, self._display_height - 1)
        self._x0 = np.floor(u).astype(np.intp)
        self._y0 = np.floor(v).astype(np.intp)
        self._x1 = np.minimum(self._x0 + 1, self._display_width - 1)
        self._y1 = np.minimum(self._y0 + 1, self._display_height - 1)
        self._wx = (u - self._x0)[None, :, None]
        self._wy = (v - self._y0)[:, None, None]
        projected_width = self.width * screen_width / view_width
        projected_height = self.height * screen_height / view_height
        self.provenance = dict(
            scene=scene, projection="fixed frontal pinhole camera; unlit planar RGB texture",
            sampling="bilinear at pixel centers; clamp texture edges; round to RGB8",
            screen_bounds_pixels=[(self.width - projected_width) / 2,
                                  (self.height - projected_height) / 2,
                                  (self.width + projected_width) / 2,
                                  (self.height + projected_height) / 2],
            bounds_convention="left, top, right, bottom in pixel-edge coordinates",
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            output=dict(image_space="mouse_view", format="RGB8",
                        shape=[self.height, self.width, 3], row_origin="top", value_range=[0, 255]),
        )

    def project(self, frame: DisplayFrame):
        """Return a distinct scene frame or raise; never substitute display output."""
        if not isinstance(frame, DisplayFrame):
            raise TypeError("Expected a completed DisplayFrame")
        if frame.image_space != "display":
            raise ValueError("Scene projection requires display-space input")
        if frame.eid != self.eid or frame.trial_id not in self._trial_ids:
            raise ValueError("Display frame does not belong to this replay request")
        _finite(frame.session_time, "session time")
        texture = frame.rgb
        if (not isinstance(texture, np.ndarray) or texture.dtype != np.uint8
                or texture.shape != (self._display_height, self._display_width, 3)):
            raise ValueError("Display frame must have the configured RGB8 shape")
        top = (texture[self._y0[:, None], self._x0] * (1 - self._wx)
               + texture[self._y0[:, None], self._x1] * self._wx)
        bottom = (texture[self._y1[:, None], self._x0] * (1 - self._wx)
                  + texture[self._y1[:, None], self._x1] * self._wx)
        sampled = np.rint(top * (1 - self._wy) + bottom * self._wy).clip(0, 255).astype(np.uint8)
        rgb = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        rgb[self._rows[:, None], self._cols] = sampled
        rgb.setflags(write=False)
        return SceneFrame(rgb, frame.eid, frame.trial_id, frame.session_time,
                          "mouse_view", deepcopy(frame.render_parameters))
