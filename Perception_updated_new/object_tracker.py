import logging
from typing import List

from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Multi-object tracker using DeepSORT.

    Fixes vs original:
    - Negative/zero width-height bbox guard — bad YOLO boxes crash DeepSORT
    - try/except around update_tracks — one bad frame doesn't kill the pipeline
    - max_age configurable at init
    - Returns empty list (not crash) on any failure
    """

    def __init__(self, max_age: int = 10):
        self.tracker = DeepSort(max_age=max_age)
        logger.info("ObjectTracker: initialised max_age=%d", max_age)

    def update(self, detections: List[dict], frame) -> List[dict]:
        """
        Update tracker with new detections and return confirmed tracks.

        Args:
            detections: list of {label, confidence, bbox:[x1,y1,x2,y2]}
            frame:      BGR numpy frame (used by DeepSORT for Re-ID features)

        Returns:
            list of {track_id, bbox:[x1,y1,x2,y2], label}
        """
        if frame is None:
            return []

        boxes = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1

            # Guard: DeepSORT crashes on zero/negative width or height
            if w <= 0 or h <= 0:
                logger.debug("ObjectTracker: skipping invalid bbox %s", d["bbox"])
                continue

            boxes.append(([x1, y1, w, h], d["confidence"], d["label"]))

        if not boxes:
            return []

        try:
            tracks = self.tracker.update_tracks(boxes, frame=frame)
        except Exception as e:
            logger.error("ObjectTracker.update_tracks: %s", e)
            return []

        tracked = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            try:
                x1, y1, x2, y2 = map(int, t.to_ltrb())
                tracked.append({
                    "track_id": t.track_id,
                    "bbox":     [x1, y1, x2, y2],
                    "label":    t.det_class,
                })
            except Exception as e:
                logger.debug("ObjectTracker: skipping track (%s)", e)
                continue

        return tracked
