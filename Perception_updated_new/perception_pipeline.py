import os
import logging
import threading
import time
from typing import List

import cv2
cv2.setNumThreads(1)

from Perception.camera_realsense import RealSenseCamera
from Perception.object_detection import ObjectDetector
from Perception.object_tracker import ObjectTracker
from Perception.depth_processing import DepthProcessor

logger = logging.getLogger(__name__)


class PerceptionPipeline:

    def __init__(self):
        self.camera         = RealSenseCamera()
        self.detector       = ObjectDetector(conf_threshold=0.25)
        self.tracker        = ObjectTracker(max_age=10)
        self.depth_processor= DepthProcessor()

        self.rgb_frame   = None
        self.depth_frame = None
        self.lock        = threading.Lock()

        self.max_objects  = 5
        self.frame_count  = 0

        # FIXED: headless by default — set SHOW_DISPLAY=1 to enable window
        self.show_display = os.environ.get("SHOW_DISPLAY", "0") == "1"

        # Cache last detections so context_builder can call get_current_detections()
        self._last_detections: List[dict] = []
        self._detections_lock = threading.Lock()

        self.running = True
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True,
        )
        self.camera_thread.start()

        logger.info("PerceptionPipeline: started (display=%s)", self.show_display)

    def _camera_loop(self):
        """Dedicated thread — keeps camera buffer fresh at full frame rate."""
        while self.running:
            try:
                rgb, depth = self.camera.get_frames()
                if rgb is None or depth is None:
                    time.sleep(0.01)
                    continue
                with self.lock:
                    self.rgb_frame   = rgb
                    self.depth_frame = depth
            except Exception as e:
                logger.error("PerceptionPipeline._camera_loop: %s", e)
                time.sleep(0.1)

    def step(self) -> List[dict]:
        """
        Run one detection + tracking cycle.
        Returns list of {id, label, bbox, depth_m, confidence}.
        Called by main.py perception_loop at ~30 Hz.
        """
        self.frame_count += 1

        with self.lock:
            rgb   = None if self.rgb_frame   is None else self.rgb_frame.copy()
            depth = None if self.depth_frame is None else self.depth_frame.copy()

        if rgb is None or depth is None:
            return []

        # ── Detect ────────────────────────────────────────────────────
        detections = self.detector.detect(rgb)

        # ── Track ─────────────────────────────────────────────────────
        tracks = self.tracker.update(detections, rgb)
        tracks = tracks[:self.max_objects]

        # ── Depth per track ───────────────────────────────────────────
        results = []
        for obj in tracks:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            label    = obj["label"]
            track_id = obj.get("track_id", -1)
            conf     = obj.get("confidence", 0.0)

            raw_depth = self.depth_processor.get_bbox_depth(depth, (x1, y1, x2, y2))
            raw_depth = float(raw_depth) if raw_depth else 0.0

            # FIXED: depth from DepthProcessor is already in metres via depth_scale.
            # Guard: if value > 100 it accidentally arrived as raw mm — convert.
            depth_m = raw_depth / 1000.0 if raw_depth > 100 else raw_depth

            results.append({
                "id":         track_id,
                "label":      label,
                "bbox":       [x1, y1, x2, y2],
                "depth_m":    depth_m,          # always metres
                "confidence": conf,
                # x offset from centre (metres) — used for better 3-D position
                "x_offset_m": ((x1 + x2) / 2 - 320) * depth_m / 615.0,
            })

            if self.show_display:
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb, f"{label} {depth_m:.2f}m",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.show_display:
            cv2.putText(rgb, "PIPELINE RUNNING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Perception Output", rgb)
            cv2.waitKey(1)

        # Cache for context_builder
        with self._detections_lock:
            self._last_detections = results

        return results

    def get_current_detections(self) -> List[dict]:
        """
        Return most recent detections without blocking step().
        Called by ContextBuilder so the planner sees live objects.
        """
        with self._detections_lock:
            return list(self._last_detections)

    def shutdown(self):
        self.running = False
        try:
            self.camera.stop()
        except Exception:
            pass
        if self.show_display:
            cv2.destroyAllWindows()
        logger.info("PerceptionPipeline: shutdown complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = PerceptionPipeline()
    time.sleep(2)
    try:
        while True:
            results = pipeline.step()
            if results:
                for obj in results:
                    print(f"  {obj['label']} | depth={obj['depth_m']:.2f}m | id={obj['id']}")
            elif pipeline.frame_count % 30 == 0:
                print("No objects detected")
            time.sleep(0.033)
    except KeyboardInterrupt:
        pipeline.shutdown()
