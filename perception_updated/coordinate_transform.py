import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """
    Converts pixel (u, v) + depth → 3D point (x, y, z) in camera frame.

    Formula:
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

    Fixes vs original:
    - from_camera() classmethod — reads real intrinsics from RealSenseCamera
      instead of using hardcoded values everywhere
    - update_intrinsics() — allows runtime update if camera profile changes
    - pixel_to_3d() validates depth > 0 before computing
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        logger.debug(
            "CoordinateTransformer: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
            fx, fy, cx, cy
        )

    # ------------------------------------------------------------------ #
    #  Constructor helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_camera(cls, camera) -> "CoordinateTransformer":
        """
        Create transformer using real intrinsics from a RealSenseCamera instance.
        Use this instead of hardcoding fx/fy/cx/cy.

        Example:
            camera = RealSenseCamera()
            coord  = CoordinateTransformer.from_camera(camera)
        """
        intr = camera.get_intrinsics()
        return cls(
            fx=intr["fx"],
            fy=intr["fy"],
            cx=intr["cx"],
            cy=intr["cy"],
        )

    @classmethod
    def from_dict(cls, intr: dict) -> "CoordinateTransformer":
        """Create from intrinsics dict {fx, fy, cx, cy}."""
        return cls(fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"])

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Update intrinsics at runtime."""
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    # ------------------------------------------------------------------ #
    #  Core transform
    # ------------------------------------------------------------------ #

    def pixel_to_3d(self, u: Union[int, float], v: Union[int, float],
                    depth: float) -> np.ndarray:
        """
        Convert pixel + depth to 3D point in camera coordinate frame.

        Args:
            u:     pixel column (x in image)
            v:     pixel row    (y in image)
            depth: distance in METRES (already converted from raw uint16)

        Returns:
            np.array([x, y, z]) in metres
            Returns [0, 0, 0] if depth is zero or negative.
        """
        if depth <= 0:
            logger.debug("CoordinateTransformer.pixel_to_3d: depth=%.4f invalid", depth)
            return np.zeros(3, dtype=np.float32)

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return np.array([x, y, z], dtype=np.float32)

    def project_3d_to_pixel(self, xyz: np.ndarray):
        """
        Inverse: project a 3D point back to pixel coordinates.
        Returns (u, v) pixel tuple.
        """
        if xyz[2] <= 0:
            return None
        u = int(xyz[0] * self.fx / xyz[2] + self.cx)
        v = int(xyz[1] * self.fy / xyz[2] + self.cy)
        return u, v
