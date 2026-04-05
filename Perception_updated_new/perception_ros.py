"""
perception_ros.py
=================
Proper ROS2 perception node — the single source of truth for all
camera data in the system.

Architecture:
    RealSense ROS2 driver
          ↓  (ROS2 topics)
    PerceptionNode  ← THIS FILE
          ↓  publishes to:
    /perception/detections        (std_msgs/String JSON) → World Model
    /perception/annotated_image   (CompressedImage)      → RViz
    /perception/camera_info       (CameraInfo)           → RViz
    /obstacle_pointcloud          (PointCloud2)          → Nav2 costmap

Why everything goes through ROS2:
    - One camera connection, no conflicts
    - All modules on same data
    - Proper timestamping and synchronisation
    - Works with Sanjay's sensor fusion
"""

import json
import logging
import os
import struct
import time

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import (CameraInfo, CompressedImage, Image,
                              PointCloud2, PointField)
from std_msgs.msg import Header, String

import math
try:
    from tf2_ros import Buffer, TransformListener
    from tf2_geometry_msgs import do_transform_point
    from geometry_msgs.msg import PointStamped
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False

from Perception.object_detection import ObjectDetector
from Perception.object_tracker import ObjectTracker
from Perception.depth_processing import DepthProcessor

logger = logging.getLogger(__name__)


class PerceptionNode(Node):
    """
    Unified ROS2 perception node.
    Subscribes to RealSense topics, runs YOLO+DeepSORT+depth,
    publishes detections for world model AND point cloud for Nav2.
    """

    # Depth range for obstacle pointcloud
    MIN_DEPTH_M  = float(os.environ.get("OBSTACLE_MIN_RANGE", "0.15"))
    MAX_DEPTH_M  = float(os.environ.get("OBSTACLE_MAX_RANGE", "4.0"))
    VOXEL_SIZE   = float(os.environ.get("OBSTACLE_VOXEL_SIZE", "0.05"))

    def __init__(self):
        super().__init__("perception_pipeline_node")

        # ── Core perception ───────────────────────────────────────────
        self.detector        = ObjectDetector(conf_threshold=0.1)
        self.tracker         = ObjectTracker(max_age=10)
        self.depth_processor = DepthProcessor()
        self.bridge          = CvBridge()

        self.latest_info      = None
        # TF2 — for converting camera frame to map frame
        self._tf_buffer   = None
        self._tf_listener = None
        if TF2_AVAILABLE:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self.get_logger().info('TF2 listener ready')
        else:
            self.get_logger().warning('TF2 not available — storing camera frame positions')
        self.latest_depth_raw = None   # kept for pointcloud generation
        self.latest_intrinsics = None

        # ── QoS ───────────────────────────────────────────────────────
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        # ── Publishers ────────────────────────────────────────────────
        # 1. Detections JSON → World Model (your main.py subscribes here)
        self.det_pub = self.create_publisher(
            String, "/perception/detections", 10
        )
        # 2. Annotated image → RViz
        self.image_pub = self.create_publisher(
            CompressedImage, "/perception/annotated_image/compressed", qos
        )
        # 3. Camera info → RViz
        self.info_pub = self.create_publisher(
            CameraInfo, "/perception/camera_info", qos
        )
        # 4. Real 3D point cloud → Nav2 costmap
        self.pc_pub = self.create_publisher(
            PointCloud2, "/obstacle_pointcloud", 10
        )

        # ── Subscribers ───────────────────────────────────────────────
        # Topic names — configurable via env vars
        # Check actual topics with: ros2 topic list | grep camera
        rgb_topic   = os.environ.get("RGB_TOPIC",   "/camera/camera/color/image_raw")
        depth_topic = os.environ.get("DEPTH_TOPIC", "/camera/camera/depth/image_rect_raw")
        info_topic  = os.environ.get("INFO_TOPIC",  "/camera/camera/color/camera_info")

        self.get_logger().info(f"Subscribing to RGB: {rgb_topic}")
        self.get_logger().info(f"Subscribing to depth: {depth_topic}")

        self.rgb_sub = message_filters.Subscriber(
            self, Image, rgb_topic
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, depth_topic
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

        self.info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self._save_info,
            10,
        )

        self.get_logger().info(
            "PerceptionNode ready — publishing detections + pointcloud"
        )

    # ------------------------------------------------------------------ #
    #  Camera info
    # ------------------------------------------------------------------ #

    def _save_info(self, msg: CameraInfo) -> None:
        self.latest_info = msg
        # Extract intrinsics for pointcloud generation
        if self.latest_intrinsics is None:
            k = msg.k  # 3x3 row-major
            self.latest_intrinsics = {
                "fx": k[0], "fy": k[4],
                "cx": k[2], "cy": k[5],
                "width":  msg.width,
                "height": msg.height,
            }
            self.get_logger().info(
                f"Intrinsics received: fx={k[0]:.1f} fy={k[4]:.1f} cx={k[2]:.1f} cy={k[5]:.1f}"
            )

    # ------------------------------------------------------------------ #
    #  Main callback — runs every synced RGB+depth frame
    # ------------------------------------------------------------------ #

    def _to_map_frame(self, x_c, y_c, z_c, stamp):
        """
        Convert camera-frame point to map-frame using TF2.
        Falls back to simple formula if TF2 unavailable.
        """
        if TF2_AVAILABLE and self._tf_buffer is not None:
            try:
                pt = PointStamped()
                pt.header.frame_id = os.environ.get('CAMERA_FRAME', 'camera_link')
                pt.header.stamp    = stamp
                pt.point.x = float(x_c)
                pt.point.y = float(y_c)
                pt.point.z = float(z_c)
                tf = self._tf_buffer.lookup_transform(
                    'map', 'camera_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                mp = do_transform_point(pt, tf)
                return [round(mp.point.x, 3), round(mp.point.y, 3), round(mp.point.z, 3)]
            except Exception as e:
                self.get_logger().debug(f'TF2 transform failed: {e} — using camera frame')
        # Fallback: simple formula (only correct at map origin)
        return [round(z_c, 3), round(-x_c, 3), 0.0]

    def callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            rgb_frame   = self.bridge.imgmsg_to_cv2(rgb_msg,   "bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            self.latest_depth_raw = depth_frame   # keep for pointcloud

            # ── Detect + Track ────────────────────────────────────────
            detections = self.detector.detect(rgb_frame)
            tracks     = self.tracker.update(detections, rgb_frame)

            results = []
            for obj in tracks[:5]:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                label    = obj["label"]
                track_id = obj.get("track_id", -1)
                conf     = obj.get("confidence", 0.0)

                raw_depth = self.depth_processor.get_bbox_depth(
                    depth_frame, (x1, y1, x2, y2)
                )
                raw_depth = float(raw_depth) if raw_depth else 0.0

                # Guard: if > 100 it's raw mm, convert to metres
                depth_m = raw_depth / 1000.0 if raw_depth > 100 else raw_depth

                # Compute x offset in metres using intrinsics
                cx = self.latest_intrinsics["cx"] if self.latest_intrinsics else 320.0
                fx = self.latest_intrinsics["fx"] if self.latest_intrinsics else 615.0
                x_off = ((x1 + x2) / 2 - cx) * depth_m / fx

                # Convert to map frame using TF2
                # World model stores map-frame positions so they stay
                # correct even after robot moves
                map_pos = self._to_map_frame(
                    x_off, 0.0, depth_m, rgb_msg.header.stamp
                )

                results.append({
                    "id":         track_id,
                    "label":      label,
                    "position":   map_pos,   # map frame — always correct
                    "depth_m":    depth_m,
                    "confidence": conf,
                    "bbox":       [x1, y1, x2, y2],
                    "timestamp":  time.time(),
                })

                # Annotate frame
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    rgb_frame,
                    f"{label} {depth_m:.2f}m",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

            # ── Publish detections JSON → World Model ─────────────────
            det_msg      = String()
            det_msg.data = json.dumps(results)
            self.det_pub.publish(det_msg)

            # ── Publish annotated image → RViz ────────────────────────
            ok, buf = cv2.imencode(
                ".jpg", rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if ok:
                img_msg        = CompressedImage()
                img_msg.header = rgb_msg.header
                img_msg.format = "jpeg"
                img_msg.data   = np.array(buf).tobytes()
                self.image_pub.publish(img_msg)

                if self.latest_info:
                    self.latest_info.header = rgb_msg.header
                    self.info_pub.publish(self.latest_info)

            # ── Publish real 3D point cloud → Nav2 ────────────────────
            if self.latest_intrinsics is not None:
                self._publish_pointcloud(depth_frame, rgb_msg.header)

        except Exception as e:
            self.get_logger().error(f"PerceptionNode callback error: {e}")

    # ------------------------------------------------------------------ #
    #  Real depth → PointCloud2 → Nav2
    # ------------------------------------------------------------------ #

    def _publish_pointcloud(self, depth: np.ndarray, header) -> None:
        try:
            intr = self.latest_intrinsics
            fx, fy = intr["fx"], intr["fy"]
            cx, cy = intr["cx"], intr["cy"]
            h, w   = depth.shape

            # Vectorised back-projection
            u  = np.arange(w, dtype=np.float32)
            v  = np.arange(h, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)

            z = depth.astype(np.float32) * 0.001  # mm → metres
            valid = (z > self.MIN_DEPTH_M) & (z < self.MAX_DEPTH_M)

            z  = z[valid];  uu = uu[valid];  vv = vv[valid]
            if len(z) == 0:
                return

            x = (uu - cx) * z / fx
            y = (vv - cy) * z / fy

            pts = np.stack([x, y, z], axis=1)

            # Voxel downsample
            if self.VOXEL_SIZE > 0:
                idx = np.floor(pts / self.VOXEL_SIZE).astype(np.int32)
                _, uniq = np.unique(idx, axis=0, return_index=True)
                pts = pts[uniq]

            # Pack as PointCloud2
            fields = [
                PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
            ]
            data = bytearray()
            for p in pts:
                data += struct.pack("fff", float(p[0]), float(p[1]), float(p[2]))

            pc_msg               = PointCloud2()
            pc_msg.header        = header
            pc_msg.header.frame_id = "camera_link"
            pc_msg.height        = 1
            pc_msg.width         = len(pts)
            pc_msg.fields        = fields
            pc_msg.is_bigendian  = False
            pc_msg.point_step    = 12
            pc_msg.row_step      = 12 * len(pts)
            pc_msg.data          = bytes(data)
            pc_msg.is_dense      = True
            self.pc_pub.publish(pc_msg)

        except Exception as e:
            self.get_logger().error(f"PerceptionNode pointcloud error: {e}")


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
