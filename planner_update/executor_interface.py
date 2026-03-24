import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try importing ROS2 rclpy — only available on Jetson / ROS2 environment
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ExecutorInterface: rclpy not available — using mock mode")


class ExecutorInterface:
    """
    Translates skill actions into ROS2 messages sent to the Raspberry Pi.

    In mock mode (no ROS2 installed) it simulates execution and logs actions.
    In real mode it publishes to the appropriate ROS2 topics.

    Action → ROS2 topic mapping:
        move_base           → /cmd_vel          (geometry_msgs/Twist)
        arm_extend          → /arm/cmd          (std_msgs/String)
        arm_retract         → /arm/cmd
        align_gripper       → /gripper/align    (std_msgs/String)
        close_gripper       → /gripper/cmd
        open_gripper        → /gripper/cmd
        arm_home            → /arm/cmd
        move_arm_to_target  → /arm/cmd
        head_rotate_*       → /head/cmd         (std_msgs/String)
        head_tilt_*         → /head/cmd
        head_center         → /head/cmd
        speak_question      → /audio/speak      (std_msgs/String)
        wait_for_response   → internal wait
        idle                → no-op
    """

    ACTION_TIMEOUT: Dict[str, float] = {
        "move_base":           5.0,
        "arm_extend":          2.0,
        "arm_retract":         2.0,
        "align_gripper":       1.5,
        "close_gripper":       1.0,
        "open_gripper":        1.0,
        "arm_home":            2.0,
        "move_arm_to_target":  3.0,
        "head_rotate_left":    1.0,
        "head_rotate_center":  0.5,
        "head_rotate_right":   1.0,
        "head_tilt_down":      0.8,
        "head_tilt_center":    0.5,
        "head_center":         0.5,
        "speak_question":      3.0,
        "wait_for_response":  10.0,
        "idle":                0.5,
    }

    def __init__(self, mock: Optional[bool] = None):
        if mock is None:
            mock = not ROS2_AVAILABLE or os.environ.get("EXECUTOR_MOCK", "0") == "1"
        self.mock = mock

        self._node = None
        self._publishers: Dict[str, Any] = {}

        if not self.mock and ROS2_AVAILABLE:
            self._init_ros2()
        else:
            logger.info("ExecutorInterface: running in MOCK mode")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def execute(self, actions: List[str], target: Optional[str] = None,
                confidence_required: float = 0.0) -> List[Dict]:
        """
        Execute a list of low-level actions in order.
        Stops and returns on the first failure.

        Returns:
            List of {action, status, duration_ms, error} dicts.
        """
        results = []
        for action in actions:
            result = self._execute_one(action, target)
            results.append(result)
            if result["status"] != "success":
                logger.error("ExecutorInterface: action '%s' failed — stopping sequence", action)
                break
        return results

    def execute_emergency_stop(self) -> None:
        """Immediately stop all movement and open gripper."""
        logger.warning("ExecutorInterface: EMERGENCY STOP")
        self._execute_one("open_gripper", None)
        self._execute_one("arm_home", None)
        self._publish_twist(0.0, 0.0)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _execute_one(self, action: str, target: Optional[str]) -> Dict:
        start = time.time()
        try:
            if self.mock:
                success = self._mock_execute(action, target)
            else:
                success = self._ros2_execute(action, target)

            duration_ms = int((time.time() - start) * 1000)
            status = "success" if success else "failed"
            logger.debug("ExecutorInterface: %s → %s (%dms)", action, status, duration_ms)
            return {"action": action, "status": status, "duration_ms": duration_ms, "error": None}

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            logger.error("ExecutorInterface: %s raised exception: %s", action, e)
            return {"action": action, "status": "error", "duration_ms": duration_ms, "error": str(e)}

    def _mock_execute(self, action: str, target: Optional[str]) -> bool:
        timeout = self.ACTION_TIMEOUT.get(action, 1.0)
        logger.info("[MOCK] Executing: %-25s target=%-20s (simulating %.1fs)", action, target or "-", timeout)
        time.sleep(min(timeout * 0.1, 0.3))   # simulate briefly in mock
        return True

    def _ros2_execute(self, action: str, target: Optional[str]) -> bool:
        if action == "move_base":
            # Publish a short forward movement — real nav2 integration would go here
            self._publish_twist(linear=0.2, angular=0.0)
            time.sleep(self.ACTION_TIMEOUT.get(action, 3.0))
            self._publish_twist(0.0, 0.0)
            return True

        if action in ("arm_extend", "arm_retract", "arm_home", "move_arm_to_target"):
            self._publish_string("/arm/cmd", action if not target else f"{action}:{target}")
            time.sleep(self.ACTION_TIMEOUT.get(action, 2.0))
            return True

        if action in ("align_gripper", "close_gripper", "open_gripper"):
            self._publish_string("/gripper/cmd", action)
            time.sleep(self.ACTION_TIMEOUT.get(action, 1.0))
            return True

        if action.startswith("head_"):
            self._publish_string("/head/cmd", action)
            time.sleep(self.ACTION_TIMEOUT.get(action, 1.0))
            return True

        if action == "speak_question":
            msg = target if target else "I need your help."
            self._publish_string("/audio/speak", msg)
            time.sleep(self.ACTION_TIMEOUT.get(action, 3.0))
            return True

        if action == "wait_for_response":
            time.sleep(self.ACTION_TIMEOUT.get(action, 10.0))
            return True

        if action == "idle":
            time.sleep(0.5)
            return True

        logger.warning("ExecutorInterface: no ROS2 handler for action '%s'", action)
        return False

    # ------------------------------------------------------------------ #
    #  ROS2 helpers
    # ------------------------------------------------------------------ #

    def _init_ros2(self):
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = rclpy.create_node("executor_interface")
            self._publishers["/cmd_vel"] = self._node.create_publisher(Twist, "/cmd_vel", 10)
            self._publishers["/arm/cmd"] = self._node.create_publisher(String, "/arm/cmd", 10)
            self._publishers["/gripper/cmd"] = self._node.create_publisher(String, "/gripper/cmd", 10)
            self._publishers["/head/cmd"] = self._node.create_publisher(String, "/head/cmd", 10)
            self._publishers["/audio/speak"] = self._node.create_publisher(String, "/audio/speak", 10)
            logger.info("ExecutorInterface: ROS2 node initialised")
        except Exception as e:
            logger.error("ExecutorInterface: ROS2 init failed (%s) — switching to mock", e)
            self.mock = True

    def _publish_string(self, topic: str, data: str):
        if topic in self._publishers and not self.mock:
            msg = String()
            msg.data = data
            self._publishers[topic].publish(msg)
        else:
            logger.debug("ExecutorInterface: publish %s → '%s'", topic, data)

    def _publish_twist(self, linear: float, angular: float):
        if "/cmd_vel" in self._publishers and not self.mock:
            msg = Twist()
            msg.linear.x = linear
            msg.angular.z = angular
            self._publishers["/cmd_vel"].publish(msg)
        else:
            logger.debug("ExecutorInterface: cmd_vel linear=%.2f angular=%.2f", linear, angular)
