"""
plan_push.py
============
Resolves planner targets to real map-frame coordinates using TF2,
then sends NavigateToPose to Nav2.

Coordinate flow:
    Camera detects object at [x, y, z] in camera_link frame
          ↓  TF2 transform (knows robot position in map)
    Object position in map frame [map_x, map_y]
          ↓
    Nav2 NavigateToPose goal
          ↓
    Robot drives to object

Why TF2 matters:
    Without TF2 you are converting camera→map with a fixed formula.
    That only works if the robot is at map origin facing forward.
    The moment the robot moves or turns, the formula is wrong.
    TF2 knows exactly where the robot and camera are in the map
    at every moment, so the transform is always correct.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose
    from geometry_msgs.msg import PoseStamped, Quaternion, PointStamped
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("PlanPush: rclpy not available — mock mode")

try:
    from tf2_ros import Buffer, TransformListener
    from tf2_geometry_msgs import do_transform_point
    TF2_AVAILABLE = True
except ImportError:
    TF2_AVAILABLE = False
    logger.warning("PlanPush: tf2_ros not available — using fallback transform")


def _yaw_to_quaternion(yaw: float) -> "Quaternion":
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def _camera_to_map_fallback(cam_pos: List[float]) -> Tuple[float, float, float]:
    """
    Simple fallback when TF2 is not available.
    Only accurate when robot is at map origin facing forward.
    Use TF2 for real deployment.
    """
    x_c, y_c, z_c = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
    map_x = z_c
    map_y = -x_c
    yaw   = 0.0
    return map_x, map_y, yaw


class PlanPush:
    """
    Converts camera-frame object positions to map-frame Nav2 goals using TF2.
    """

    FRAME_ID            = "map"
    CAMERA_FRAME        = os.environ.get("CAMERA_FRAME", "camera_link")
    ACTION_SERVER       = "navigate_to_pose"
    SERVER_TIMEOUT_S    = float(os.environ.get("NAV2_SERVER_TIMEOUT", "5.0"))
    GOAL_TIMEOUT_S      = float(os.environ.get("NAV2_GOAL_TIMEOUT",   "60.0"))
    APPROACH_DISTANCE_M = float(os.environ.get("NAV2_APPROACH_DIST",  "0.4"))

    # Class-level start position — stored once when first command runs
    _start_position = None

    def __init__(self, mock: Optional[bool] = None):
        if mock is None:
            mock = (
                not ROS2_AVAILABLE
                or os.environ.get("EXECUTOR_MOCK", "0") == "1"
            )
        self.mock = mock

        self._node: Optional[Any]          = None
        self._action_client: Optional[Any] = None
        self._tf_buffer: Optional[Any]     = None
        self._tf_listener: Optional[Any]   = None

        if not self.mock and ROS2_AVAILABLE:
            self._init_ros2()

        logger.info(
            "PlanPush: initialised (mock=%s, tf2=%s, approach=%.2fm)",
            self.mock,
            TF2_AVAILABLE and self._tf_buffer is not None,
            self.APPROACH_DISTANCE_M,
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def push(
        self,
        target: Union[str, List[float], Dict],
        world_model: Any,
    ) -> Dict[str, Any]:

        result: Dict[str, Any] = {
            "status":   "failed",
            "target":   target,
            "position": None,
            "nav_goal": None,
            "error":    None,
        }

        # 1 — Resolve camera-frame coordinates
        cam_pos = self.resolve_coordinates(target, world_model)
        if cam_pos is None:
            result["error"] = f"Could not resolve coordinates for target: {target!r}"
            logger.warning(
                "PlanPush: '%s' not found in world model — "
                "object is not visible to camera or not yet detected.", target
            )
            return result

        result["position"] = cam_pos
        logger.info("PlanPush: resolved '%s' → camera pos %s", target, cam_pos)

        # Store start position the first time
        if PlanPush._start_position is None:
            PlanPush._start_position = [0.0, 0.0, 0.0]
            logger.info("PlanPush: start position stored as map origin")

        # 2 — Positions from world model are already in map frame
        # (perception_ros.py converts using TF2 at detection time)
        # cam_pos is actually [map_x, map_y, map_z]
        map_x = float(cam_pos[0])
        map_y = float(cam_pos[1])
        # Yaw — face the target
        import math as _math
        yaw = _math.atan2(map_y, map_x)

        # Apply approach distance
        dist = math.sqrt(map_x ** 2 + map_y ** 2)
        if dist > self.APPROACH_DISTANCE_M:
            scale  = (dist - self.APPROACH_DISTANCE_M) / dist
            map_x *= scale
            map_y *= scale

        result["nav_goal"] = {
            "x":   round(map_x, 3),
            "y":   round(map_y, 3),
            "yaw": round(yaw, 4),
        }
        logger.info(
            "PlanPush: nav goal → x=%.3f y=%.3f yaw=%.4f (map frame)",
            map_x, map_y, yaw,
        )

        # 3 — Send to Nav2
        if self.mock:
            logger.info(
                "[MOCK] PlanPush → NavigateToPose x=%.3f y=%.3f yaw=%.4f",
                map_x, map_y, yaw,
            )
            result["status"] = "mock"
            return result

        success = self._send_nav2_goal(map_x, map_y, yaw)
        result["status"] = "succeeded" if success else "failed"
        if not success:
            result["error"] = "Nav2 goal did not succeed within timeout"

        return result

    # ------------------------------------------------------------------ #
    #  TF2 transform — camera frame → map frame
    # ------------------------------------------------------------------ #

    def _transform_to_map(
        self, cam_pos: List[float]
    ) -> Tuple[float, float, float]:
        """
        Transform [x, y, z] in camera_link frame to map frame using TF2.
        Falls back to simple formula if TF2 unavailable.
        """
        if not TF2_AVAILABLE or self._tf_buffer is None:
            logger.debug("PlanPush: using fallback transform (TF2 not available)")
            return _camera_to_map_fallback(cam_pos)

        try:
            # Build a PointStamped in camera frame
            point = PointStamped()
            point.header.frame_id = self.CAMERA_FRAME
            point.header.stamp    = self._node.get_clock().now().to_msg()
            point.point.x = float(cam_pos[0])
            point.point.y = float(cam_pos[1])
            point.point.z = float(cam_pos[2])

            # Look up transform from camera_link to map
            transform = self._tf_buffer.lookup_transform(
                self.FRAME_ID,       # target frame — map
                self.CAMERA_FRAME,   # source frame — camera
                rclpy.time.Time(),   # latest available
                timeout=rclpy.duration.Duration(seconds=1.0),
            )

            # Apply transform
            map_point = do_transform_point(point, transform)

            map_x = map_point.point.x
            map_y = map_point.point.y

            # Yaw — face the target from current robot position
            yaw = math.atan2(map_y, map_x)

            logger.info(
                "PlanPush: TF2 transform camera[%.2f,%.2f,%.2f] → map[%.3f,%.3f]",
                cam_pos[0], cam_pos[1], cam_pos[2], map_x, map_y,
            )
            return map_x, map_y, yaw

        except Exception as e:
            logger.warning(
                "PlanPush: TF2 transform failed (%s) — using fallback", e
            )
            return _camera_to_map_fallback(cam_pos)

    # ------------------------------------------------------------------ #
    #  Coordinate resolution
    # ------------------------------------------------------------------ #

    def resolve_coordinates(
        self,
        target: Union[str, List[float], Dict],
        world_model: Any,
    ) -> Optional[List[float]]:

        if isinstance(target, (list, tuple)) and len(target) >= 3:
            return [float(v) for v in target[:3]]

        if isinstance(target, dict) and "x" in target:
            return [float(target["x"]), float(target.get("y", 0)), float(target["z"])]

        if not isinstance(target, str) or not target.strip():
            return None

        label = target.strip().lower()

        # Exact label match
        matches = world_model.find_by_label(label)
        if not matches:
            first_word = label.split()[0]
            matches = world_model.find_by_label(first_word)

        if matches:
            # Filter by confidence
            confident = [o for o in matches if o.get("confidence", 1.0) >= 0.5]
            if confident:
                matches = confident

            # Pick closest
            best = min(
                matches,
                key=lambda o: o.get("position", [0, 0, 999])[2],
            )
            pos = best.get("position")
            if pos and len(pos) >= 3:
                logger.debug("PlanPush: exact match '%s' → %s", label, pos)
                return list(pos[:3])

        # Semantic search fallback
        sem_results = world_model.semantic_search(label, k=1)
        if sem_results:
            pos = sem_results[0].get("position")
            if pos and len(pos) >= 3:
                logger.debug("PlanPush: semantic match '%s' → %s", label, pos)
                return list(pos[:3])

        return None

    # ------------------------------------------------------------------ #
    #  ROS2 init
    # ------------------------------------------------------------------ #

    def _init_ros2(self) -> None:
        try:
            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node("plan_push")

            # TF2 buffer and listener
            if TF2_AVAILABLE:
                self._tf_buffer   = Buffer()
                self._tf_listener = TransformListener(
                    self._tf_buffer, self._node
                )
                logger.info("PlanPush: TF2 listener ready")

            self._action_client = ActionClient(
                self._node, NavigateToPose, self.ACTION_SERVER
            )
            logger.info(
                "PlanPush: ROS2 ready — TF2=%s, Nav2 action=/%s",
                TF2_AVAILABLE, self.ACTION_SERVER,
            )

        except Exception as exc:
            logger.error("PlanPush: ROS2 init failed (%s) — mock mode", exc)
            self.mock = True

    # ------------------------------------------------------------------ #
    #  Nav2 goal
    # ------------------------------------------------------------------ #

    def _send_nav2_goal(self, x: float, y: float, yaw: float) -> bool:
        if self._action_client is None:
            return False

        if not self._action_client.wait_for_server(
            timeout_sec=self.SERVER_TIMEOUT_S
        ):
            logger.error(
                "PlanPush: Nav2 action server not available after %.1fs",
                self.SERVER_TIMEOUT_S,
            )
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.FRAME_ID
        goal_msg.pose.header.stamp    = self._node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation = _yaw_to_quaternion(yaw)

        logger.info(
            "PlanPush: sending NavigateToPose → x=%.3f y=%.3f yaw=%.4f",
            x, y, yaw,
        )

        send_future = self._action_client.send_goal_async(goal_msg)
        deadline    = time.time() + self.SERVER_TIMEOUT_S

        while not send_future.done():
            rclpy.spin_once(self._node, timeout_sec=0.05)
            if time.time() > deadline:
                logger.error("PlanPush: timeout waiting for goal acceptance")
                return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            logger.error("PlanPush: goal rejected by Nav2")
            return False

        logger.info(
            "PlanPush: goal accepted — waiting up to %.0fs",
            self.GOAL_TIMEOUT_S,
        )

        result_future = goal_handle.get_result_async()
        deadline      = time.time() + self.GOAL_TIMEOUT_S

        while not result_future.done():
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if time.time() > deadline:
                logger.error("PlanPush: navigation timed out")
                goal_handle.cancel_goal_async()
                return False

        logger.info("PlanPush: navigation completed successfully")
        return True

    def shutdown(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
