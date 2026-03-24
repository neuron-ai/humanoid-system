import logging
import os

import cv2
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """
    Intel RealSense D435/D455 camera wrapper.

    Fixes vs original:
    - get_intrinsics() exposes real camera calibration for pointcloud_fusion
    - get_frames() has try/except — timeout doesn't crash the camera loop
    - show_stream() guarded by SHOW_DISPLAY env var — safe in headless Docker
    - stop() is idempotent — safe to call multiple times
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps
        self._running = False

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8,  fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16,   fps)

        self.profile = self.pipeline.start(config)
        self._running = True

        # Real depth scale from sensor (typically ~0.001 m/unit)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Align depth frame to colour frame
        self.align = rs.align(rs.stream.color)

        # Cache real intrinsics — used by PointCloudFusion and CoordinateTransformer
        self._intrinsics = self._read_intrinsics()

        logger.info(
            "RealSenseCamera: started %dx%d@%dfps depth_scale=%.6f",
            width, height, fps, self.depth_scale
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_frames(self):
        """
        Returns (color_bgr, depth_uint16) numpy arrays, or (None, None) on failure.
        depth values are raw uint16 — multiply by depth_scale to get metres.
        """
        try:
            frames   = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned  = self.align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            depth = np.asanyarray(depth_frame.get_data())   # uint16, raw units
            color = np.asanyarray(color_frame.get_data())   # uint8 BGR

            return color, depth

        except Exception as e:
            logger.error("RealSenseCamera.get_frames: %s", e)
            return None, None

    def get_intrinsics(self) -> dict:
        """
        Return real camera intrinsics dict:
            {fx, fy, cx, cy, width, height}
        Use these in CoordinateTransformer and PointCloudFusion instead of
        hardcoded values — hardcoded values cause wrong 3D positions.
        """
        return dict(self._intrinsics)

    def show_stream(self):
        """
        Display live RGB + depth — only works when SHOW_DISPLAY=1 env var is set.
        Safe to call in headless Docker: just logs and returns.
        """
        if os.environ.get("SHOW_DISPLAY", "0") != "1":
            logger.info("RealSenseCamera.show_stream: SHOW_DISPLAY not set — skipping")
            return

        while True:
            color, depth = self.get_frames()
            if color is None:
                continue

            depth_display = cv2.convertScaleAbs(depth, alpha=0.03)
            cv2.imshow("RealSense RGB",   color)
            cv2.imshow("RealSense Depth", depth_display)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.stop()
        cv2.destroyAllWindows()

    def stop(self):
        """Stop pipeline — safe to call multiple times."""
        if self._running:
            try:
                self.pipeline.stop()
                self._running = False
                logger.info("RealSenseCamera: stopped")
            except Exception as e:
                logger.error("RealSenseCamera.stop: %s", e)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _read_intrinsics(self) -> dict:
        """Read real intrinsics from the RealSense profile."""
        try:
            stream  = self.profile.get_stream(rs.stream.color)
            intr    = stream.as_video_stream_profile().get_intrinsics()
            return {
                "fx": intr.fx, "fy": intr.fy,
                "cx": intr.ppx, "cy": intr.ppy,
                "width": intr.width, "height": intr.height,
            }
        except Exception as e:
            logger.warning("RealSenseCamera: could not read intrinsics (%s) — using defaults", e)
            return {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0,
                    "width": self.width, "height": self.height}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    camera = RealSenseCamera()
    camera.show_stream()
