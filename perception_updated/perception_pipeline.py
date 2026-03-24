import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cv2

cv2.setNumThreads(1)   # prevent OpenCV from spawning threads that fight with ThreadPoolExecutor

from .camera_realsense import RealSenseCamera
from .object_detection import ObjectDetector
from .object_tracker import ObjectTracker
from .depth_processing import DepthProcessor
from .coordinate_transform import CoordinateTransformer
from .clip_embeddings import CLIPEncoder
from .texture_analysis import TextureAnalyzer

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """
    Full perception pipeline: camera → detect → track → depth → 3D → embed → texture

    Output per object (per frame):
        {id, label, bbox, position:[x,y,z], embedding:[512 floats], texture:{...}, timestamp}

    Fixes vs original:
    1. Depth double-scaling removed — get_bbox_depth() returns raw units,
       pipeline multiplies by depth_scale ONCE → correct metres
    2. CoordinateTransformer uses real camera intrinsics (not hardcoded 615/320/240)
    3. cv2.imshow gated by SHOW_DISPLAY env var — safe in headless Docker
    4. bare except → specific exception logging
    5. CLIP and texture timeouts increased to 1.0s (was 0.5s — too tight on Jetson)
    6. Camera loop sleep added to prevent CPU spin when no new frames
    7. get_current_detections() exposed for world_model integration
    """

    def __init__(self):
        # ── Camera ──────────────────────────────────────────────────────
        self.camera = RealSenseCamera()

        # ── Vision ──────────────────────────────────────────────────────
        self.detector = ObjectDetector(conf_threshold=0.25)

        # ── Tracking ────────────────────────────────────────────────────
        self.tracker = ObjectTracker(max_age=10)

        # ── Depth ───────────────────────────────────────────────────────
        self.depth_processor = DepthProcessor()

        # FIX: Use real intrinsics from camera instead of hardcoded values
        self.coord = CoordinateTransformer.from_camera(self.camera)

        # ── Feature models ───────────────────────────────────────────────
        self.clip    = CLIPEncoder()
        self.texture = TextureAnalyzer()

        # ── Parallel workers ─────────────────────────────────────────────
        self.executor = ThreadPoolExecutor(max_workers=2)

        # ── Frame buffers ────────────────────────────────────────────────
        self.rgb_frame   = None
        self.depth_frame = None
        self.lock        = threading.Lock()

        # ── Latest detections (for world_model.get_current_detections) ──
        self._latest_detections: List[dict] = []
        self._detections_lock = threading.Lock()

        # ── Controls ─────────────────────────────────────────────────────
        self.max_objects = 5
        self.frame_count = 0
        self.show_display = os.environ.get("SHOW_DISPLAY", "0") == "1"

        # ── Camera capture thread ─────────────────────────────────────────
        self.running = True
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True,
            name="camera_loop",
        )
        self.camera_thread.start()
        logger.info("PerceptionPipeline: started (display=%s)", self.show_display)

    # ------------------------------------------------------------------ #
    #  Camera loop — runs in background thread
    # ------------------------------------------------------------------ #

    def _camera_loop(self):
        while self.running:
            try:
                rgb, depth = self.camera.get_frames()

                if rgb is None or depth is None:
                    time.sleep(0.005)   # FIX: don't spin CPU when no frame
                    continue

                with self.lock:
                    self.rgb_frame   = rgb
                    self.depth_frame = depth

            except Exception as e:
                logger.error("PerceptionPipeline._camera_loop: %s", e)
                time.sleep(0.1)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def step(self) -> List[dict]:
        """
        Run one perception cycle.
        Returns list of detected+tracked objects with 3D position and embeddings.
        Call this in a loop at your desired rate (e.g. 10–30 Hz).
        """
        self.frame_count += 1

        with self.lock:
            rgb_frame   = None if self.rgb_frame   is None else self.rgb_frame.copy()
            depth_frame = None if self.depth_frame is None else self.depth_frame.copy()

        if rgb_frame is None or depth_frame is None:
            return []

        # ── Detection ───────────────────────────────────────────────────
        detections = self.detector.detect(rgb_frame)

        # ── Tracking ────────────────────────────────────────────────────
        tracks = self.tracker.update(detections, rgb_frame)
        tracks = tracks[:self.max_objects]

        perception_output = []
        h, w = rgb_frame.shape[:2]

        for obj in tracks:
            x1, y1, x2, y2 = map(int, obj["bbox"])

            # Clamp to frame bounds
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(w, x2);  y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            label    = obj["label"]
            track_id = obj["track_id"]

            # ── Depth → metres ────────────────────────────────────────────
            # get_bbox_depth returns raw uint16 sensor units
            # multiply by depth_scale (≈0.001) to get metres
            # FIX: original also divided by 1000 inside get_bbox_depth → double scaling
            depth_raw = self.depth_processor.get_bbox_depth(
                depth_frame, (x1, y1, x2, y2)
            )
            if depth_raw is None:
                continue

            depth_metres = depth_raw * self.camera.depth_scale   # now correct metres

            if depth_metres <= 0 or depth_metres > 5.0:   # sanity: 0–5m range
                continue

            # ── 3D position ───────────────────────────────────────────────
            cx  = int((x1 + x2) / 2)
            cy  = int((y1 + y2) / 2)
            xyz = self.coord.pixel_to_3d(cx, cy, depth_metres)

            # ── Crop ──────────────────────────────────────────────────────
            crop = rgb_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ── CLIP + texture in parallel ────────────────────────────────
            clip_future    = self.executor.submit(self.clip.encode_image, crop)
            texture_future = self.executor.submit(self.texture.analyze,   crop)

            embedding        = None
            texture_features = None
            try:
                embedding        = clip_future.result(timeout=1.0)    # FIX: was 0.5 — too tight
                texture_features = texture_future.result(timeout=1.0)
            except Exception as e:
                logger.debug("PerceptionPipeline.step: feature timeout (%s)", e)

            obj_state = {
                "id":        track_id,
                "label":     label,
                "bbox":      [x1, y1, x2, y2],
                "position":  xyz.tolist(),
                "embedding": embedding.tolist() if embedding is not None else None,
                "texture":   texture_features,
                "timestamp": time.time(),
            }
            perception_output.append(obj_state)

            # ── Visualisation ─────────────────────────────────────────────
            if self.show_display:
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    rgb_frame,
                    f"{label} {xyz[2]:.2f}m",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

        # ── Display (headless-safe) ───────────────────────────────────────
        if self.show_display:
            cv2.imshow("Perception", rgb_frame)
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03),
                cv2.COLORMAP_JET,
            )
            cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)

        # ── Update shared detections buffer ──────────────────────────────
        with self._detections_lock:
            self._latest_detections = perception_output

        return perception_output

    def get_current_detections(self) -> List[dict]:
        """
        Return the most recent detection results without running a new step.
        Used by world_model.get_current_detections() / context_builder.
        Thread-safe.
        """
        with self._detections_lock:
            return list(self._latest_detections)

    def shutdown(self):
        """Cleanly shut down camera, executor, and display windows."""
        logger.info("PerceptionPipeline: shutting down")
        self.running = False
        self.executor.shutdown(wait=False)
        self.camera.stop()
        if self.show_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
