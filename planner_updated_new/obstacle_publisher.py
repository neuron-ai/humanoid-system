"""
obstacle_publisher.py
=====================
Publishes REAL 3D point cloud from RealSense depth frame to Nav2 costmap.

Previous approach (WRONG for 3D camera):
    - Took YOLO bounding box centres
    - Generated fake clusters of points at fixed heights (0.0, 0.3, 0.6m)
    - Threw away all real depth data

This approach (CORRECT):
    - Takes the actual raw depth frame from RealSense
    - Converts every depth pixel to a real 3D point using camera intrinsics
    - Downsamples to keep message size manageable
    - Publishes the real point cloud to /obstacle_pointcloud
    - Nav2 gets accurate 3D obstacle data at all heights

This is what a 3D depth camera is for. Use it properly.

ROS2 topic published:
    /obstacle_pointcloud  (sensor_msgs/PointCloud2, frame: camera_link)

Nav2 setup required (nav2_params.yaml) — Sanjay's side:
    local_costmap:
      local_costmap:
        plugins: ["voxel_layer", "inflation_layer"]
        voxel_layer:
          topic: /obstacle_pointcloud
          data_type: PointCloud2
          marking: true
          clearing: true
"""

from __future__ import annotations

import logging
import os
import struct
import time
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ObstaclePublisher: rclpy not available — mock mode")


class ObstaclePublisher:
    """
    Converts real RealSense depth frames → ROS2 PointCloud2 → Nav2 3D costmap.

    Parameters
    ----------
    mock : bool | None
        True = log only, no ROS2. Auto-detects if None.
    voxel_size : float
        Downsampling voxel size in metres. 0.05 = 5cm grid — good balance
        of detail vs message size for a warehouse robot.
    max_range_m : float
        Discard points beyond this distance. Default 4m.
    """

    TOPIC        = "/obstacle_pointcloud"
    FRAME_ID     = "camera_link"   # depth camera frame — Nav2 uses TF to map frame
    VOXEL_SIZE   = float(os.environ.get("OBSTACLE_VOXEL_SIZE", "0.05"))   # 5cm
    MAX_RANGE_M  = float(os.environ.get("OBSTACLE_MAX_RANGE",  "4.0"))
    MIN_RANGE_M  = float(os.environ.get("OBSTACLE_MIN_RANGE",  "0.15"))   # ignore < 15cm

    def __init__(self, mock: Optional[bool] = None):
        if mock is None:
            mock = (
                not ROS2_AVAILABLE
                or os.environ.get("EXECUTOR_MOCK", "0") == "1"
            )
        self.mock = mock

        self._node: Optional[Any]      = None
        self._publisher: Optional[Any] = None

        if not self.mock and ROS2_AVAILABLE:
            self._init_ros2()

        logger.info(
            "ObstaclePublisher: initialised (mock=%s, topic=%s, "
            "voxel=%.2fm, range=%.1f-%.1fm)",
            self.mock, self.TOPIC, self.VOXEL_SIZE,
            self.MIN_RANGE_M, self.MAX_RANGE_M,
        )

    # ------------------------------------------------------------------ #
    #  Public API — called from main.py perception_loop
    # ------------------------------------------------------------------ #

    def publish_from_depth(
        self,
        depth_frame: np.ndarray,
        intrinsics: dict,
        depth_scale: float = 0.001,
    ) -> int:
        """
        Convert real depth frame to point cloud and publish to Nav2.

        Parameters
        ----------
        depth_frame : np.ndarray
            Raw uint16 depth image from RealSense (H x W).
        intrinsics : dict
            Camera intrinsics: {fx, fy, cx, cy, width, height}
            from camera.get_intrinsics()
        depth_scale : float
            Metres per depth unit. RealSense default is 0.001 (1mm per unit).

        Returns number of points published.
        """
        if depth_frame is None:
            return 0

        try:
            points = self._depth_to_pointcloud(depth_frame, intrinsics, depth_scale)
        except Exception as e:
            logger.error("ObstaclePublisher.publish_from_depth: %s", e)
            return 0

        if len(points) == 0:
            return 0

        if self.mock:
            logger.debug(
                "[MOCK] ObstaclePublisher: would publish %d points", len(points)
            )
            return len(points)

        self._publish_pointcloud(points)
        return len(points)

    # ------------------------------------------------------------------ #
    #  Depth frame → real 3D points
    # ------------------------------------------------------------------ #

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,
        intrinsics: dict,
        depth_scale: float,
    ) -> List[Tuple[float, float, float]]:
        """
        Vectorised conversion: depth image → list of (x, y, z) in metres.

        Uses real camera intrinsics so every point is geometrically accurate.
        Applies voxel downsampling to keep the point count manageable.
        """
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

        h, w = depth.shape

        # Build pixel coordinate grids
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        # Convert raw depth to metres
        z = depth.astype(np.float32) * depth_scale

        # Mask: keep only points within valid range
        valid = (z > self.MIN_RANGE_M) & (z < self.MAX_RANGE_M)

        z  = z[valid]
        uu = uu[valid]
        vv = vv[valid]

        if len(z) == 0:
            return []

        # Back-project to 3D using pinhole camera model
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        # z stays as-is (forward = depth direction)

        # Stack into Nx3 array
        pts = np.stack([x, y, z], axis=1)

        # Voxel downsample — snap to grid and deduplicate
        if self.VOXEL_SIZE > 0:
            voxel_idx = np.floor(pts / self.VOXEL_SIZE).astype(np.int32)
            _, unique = np.unique(voxel_idx, axis=0, return_index=True)
            pts = pts[unique]

        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]

    # ------------------------------------------------------------------ #
    #  ROS2 PointCloud2 publishing
    # ------------------------------------------------------------------ #

    def _init_ros2(self) -> None:
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("obstacle_publisher")
            self._publisher = self._node.create_publisher(
                PointCloud2, self.TOPIC, 10
            )
            logger.info(
                "ObstaclePublisher: ROS2 publisher ready → %s", self.TOPIC
            )
        except Exception as e:
            logger.error(
                "ObstaclePublisher: ROS2 init failed (%s) — mock mode", e
            )
            self.mock = True

    def _publish_pointcloud(self, points: List[Tuple[float, float, float]]) -> None:
        if self._publisher is None:
            return
        try:
            fields = [
                PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
            ]

            data = bytearray()
            for x, y, z in points:
                data += struct.pack("fff", x, y, z)

            msg                 = PointCloud2()
            msg.header          = Header()
            msg.header.frame_id = self.FRAME_ID
            msg.header.stamp    = self._node.get_clock().now().to_msg()
            msg.height          = 1
            msg.width           = len(points)
            msg.fields          = fields
            msg.is_bigendian    = False
            msg.point_step      = 12
            msg.row_step        = 12 * len(points)
            msg.data            = bytes(data)
            msg.is_dense        = True

            self._publisher.publish(msg)
            logger.debug(
                "ObstaclePublisher: published %d real depth points → %s",
                len(points), self.TOPIC,
            )

        except Exception as e:
            logger.error("ObstaclePublisher._publish_pointcloud: %s", e)

    def shutdown(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
