"""
main.py — AMR Robot Brain startup
Jetson Orin Nano 8GB / L4T 36.4.7 / JetPack 6.2.1

Wires all 6 handshakes explicitly:
  1. Camera thread       → perception pipeline (threading lock)
  2. Perception          → world model (update_batch call loop)
  3. World model         → planner (passed at init)
  4. Planner             → ROS2  → Raspberry Pi (executor_interface)
  5. Docker containers   → Redis / Neo4j (env vars + healthcheck in compose)
  6. Local LLM container → llm_client (auto-detected via _try_local)

Usage:
  python3 -m planner   (inside Docker container)
  OR
  python3 main.py
"""

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
    logger.info("LLM_MODE     = %s", os.environ.get("LLM_MODE",    "auto"))
    logger.info("REDIS_URL    = %s", os.environ.get("REDIS_URL",   "redis://localhost:6379"))
    logger.info("NEO4J_URI    = %s", os.environ.get("NEO4J_URI",   "bolt://localhost:7687"))
    logger.info("SHOW_DISPLAY = %s", os.environ.get("SHOW_DISPLAY","0"))
    logger.info("=" * 60)

    # ── Import modules ───────────────────────────────────────────
    from Perception.perception_pipeline import PerceptionPipeline
    from world_model.world_model import WorldModel
    from planner.llm_client import LLMClient
    from planner.planner import Planner

    # ── HANDSHAKE 1: Init world model ────────────────────────────
    logger.info("Initialising world model...")
    world_model = WorldModel()

    # ── HANDSHAKE 2: Init perception pipeline ────────────────────
    logger.info("Initialising perception pipeline (starts camera thread)...")
    perception = PerceptionPipeline()
    # Camera thread starts automatically inside PerceptionPipeline.__init__
    # Warm up — wait for first frame
    time.sleep(2.0)
    logger.info("Perception pipeline ready")

    # ── HANDSHAKE 3: Init LLM client ─────────────────────────────
    logger.info("Initialising LLM client (tries local first, then cloud)...")
    llm_client = LLMClient()
    logger.info("LLM mode: %s", llm_client.mode)

    # ── HANDSHAKE 4: Init planner (wires world_model in) ─────────
    logger.info("Initialising planner (LangGraph)...")
    planner = Planner(llm_client=llm_client, world_model=world_model, perception=perception)
    logger.info("Planner ready")

    # ── HANDSHAKE 5: Perception → World Model update loop ────────
    # Runs in background thread — feeds world model with every perception frame
    stop_event = threading.Event()

    def perception_loop():
        logger.info("Perception → World Model loop started (30fps)")
        while not stop_event.is_set():
            try:
                objects = perception.step()
                if objects:
                    world_model.update_batch(objects)
            except Exception as e:
                logger.error("perception_loop error: %s", e)
            time.sleep(0.033)   # ~30fps

    perc_thread = threading.Thread(target=perception_loop, daemon=True, name="perception_loop")
    perc_thread.start()

    # ── Graceful shutdown ─────────────────────────────────────────
    def shutdown(sig, frame):
        logger.info("Shutdown signal received")
        stop_event.set()
        perception.shutdown()
        logger.info("Robot brain stopped cleanly")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── HANDSHAKE 6: Main command loop ───────────────────────────
    logger.info("Robot brain ready. Waiting for commands...")
    logger.info("Type a command and press Enter (or send via API)")

    while True:
        try:
            # In production: replace input() with ROS2 subscriber,
            # REST API endpoint, or voice recognition module
            goal = input("\nCommand: ").strip()
            if not goal:
                continue
            if goal.lower() in ("exit", "quit", "q"):
                shutdown(None, None)

            logger.info("Running planner for goal: '%s'", goal)
            result = planner.run(goal)

            if result.get("error"):
                logger.warning("Plan completed with error: %s", result["error"])
            else:
                logger.info("Plan completed successfully")

        except EOFError:
            # Running non-interactively (Docker, systemd) — sleep and wait
            time.sleep(1.0)
        except Exception as e:
            logger.error("Main loop error: %s", e)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
