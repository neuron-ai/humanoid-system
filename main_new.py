import json
import logging
import os
import signal
import sys
import time
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def main():
    logger.info("=" * 60)
    logger.info("AMR Robot Brain starting")
    logger.info("LLM_MODE      = %s", os.environ.get("LLM_MODE", "auto"))
    logger.info("EXECUTOR_MOCK = %s", os.environ.get("EXECUTOR_MOCK", "1"))
    logger.info("=" * 60)

    from world_model.world_model import WorldModel
    from planner.llm_client import LLMClient
    from planner.planner import Planner

    #  World model 
    logger.info("Initialising world model...")
    world_model = WorldModel()

    # LLM 
    logger.info("Initialising LLM client...")
    llm_client = LLMClient()
    logger.info("LLM mode: %s", llm_client.mode)

    # Planner 
    logger.info("Initialising planner...")
    planner = Planner(
        llm_client=llm_client,
        world_model=world_model,
        perception=None,
    )
    logger.info("Planner ready")

    # ROS2 detection subscriber 
    # Receives detections from perception_ros.py via /perception/detections
    # and feeds them into the world model
    stop_event = threading.Event()

    try:
        import rclpy
        from std_msgs.msg import String as RosString

        def ros_spin():
            try:
                if not rclpy.ok():
                    rclpy.init()

                node = rclpy.create_node("amr_brain_node")

                def detection_callback(msg):
                    try:
                        detections = json.loads(msg.data)
                        if not detections:
                            return

                        objects_for_wm = []
                        for obj in detections:
                            depth_m  = obj.get("depth_m", 0.0)
                            position = obj.get("position", [0.0, 0.0, depth_m])

                            objects_for_wm.append({
                                "id":         obj.get("id", -1),
                                "label":      obj.get("label", "unknown"),
                                "position":   position,
                                "depth_m":    depth_m,
                                "confidence": obj.get("confidence", 0.0),
                                "timestamp":  time.time(),
                            })

                        world_model.update_batch(objects_for_wm)
                        logger.debug(
                            "WorldModel: %d objects updated from camera",
                            len(objects_for_wm)
                        )

                    except Exception as e:
                        logger.error("detection_callback error: %s", e)

                node.create_subscription(
                    RosString,
                    "/perception/detections",
                    detection_callback,
                    10,
                )
                logger.info("ROS2: subscribed to /perception/detections")

                while not stop_event.is_set() and rclpy.ok():
                    rclpy.spin_once(node, timeout_sec=0.05)

                node.destroy_node()

            except Exception as e:
                logger.error("ROS2 spin error: %s", e)

        threading.Thread(
            target=ros_spin,
            daemon=True,
            name="ros2_subscriber"
        ).start()

    except ImportError:
        logger.warning(
            "ROS2 not available — run perception_ros.py separately "
            "and make sure it publishes to /perception/detections"
        )

    # Shutdown 
    def shutdown(sig=None, frame=None):
        logger.info("Shutting down...")
        stop_event.set()
        try:
            planner.plan_push.shutdown()
        except Exception:
            pass
        logger.info("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Command loop 
    logger.info("Robot ready. Type commands.")
    logger.info("")
    logger.info("Flow: camera sees object → world model stores position")
    logger.info("      → LLM plans → navigate to object → pick it up")
    logger.info("")

    while True:
        try:
            goal = input("\nCommand: ").strip()

            if not goal:
                continue

            if goal.lower() in ("exit", "quit", "q"):
                shutdown()

            # Show what's currently visible before planning
            objects = world_model.get_current_detections()
            if objects:
                logger.info(
                    "Camera sees %d objects: %s",
                    len(objects),
                    [o["label"] for o in objects]
                )
            else:
                logger.warning(
                    "World model is empty — camera sees nothing yet. "
                    "Make sure perception_ros.py is running."
                )

            logger.info("Running planner: %s", goal)
            result = planner.run(goal)

            if result.get("error"):
                logger.warning("Plan error: %s", result["error"])
            elif result.get("failure_reason"):
                reason = result["failure_reason"]
                if "Could not resolve coordinates" in str(reason):
                    # Extract object name from error
                    obj_name = goal.split()[-1] if goal else "object"
                    logger.warning("")
                    logger.warning("  !! OBJECT NOT FOUND !!")
                    logger.warning("  The camera cannot see '%s'.", obj_name)
                    logger.warning("  Please place it in front of the camera and try again.")
                    logger.warning("")
                else:
                    logger.warning("Plan failed: %s", reason)
            else:
                logger.info("Plan completed successfully")

        except EOFError:
            time.sleep(1)
        except Exception as e:
            logger.error("Main loop error: %s", e)
            time.sleep(1)


if __name__ == "__main__":
    main()
