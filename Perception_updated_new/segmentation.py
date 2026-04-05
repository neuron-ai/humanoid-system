import logging
from typing import List

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Segmenter:
    """
    YOLOv8n-seg instance segmentation.

    Fixes vs original:
    - conf_threshold now passed directly to model() call — no redundant post-filter loop
    - FP16 half precision used on GPU (was missing — detection had it, segmentation didn't)
    - try/except — bad frames don't crash the pipeline
    - class_id stored as int not numpy int — JSON serialisable
    - mask stored as list of [x,y] pairs (polygon points)
    """

    def __init__(self, model_path: str = "yolov8n-seg.pt", conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device.startswith("cuda")

        logger.info("Segmenter: loading %s on %s", model_path, self.device)
        self.model = YOLO(model_path)
        self.names = self.model.names
        logger.info("Segmenter: ready — %d classes, conf=%.2f, half=%s",
                    len(self.names), conf_threshold, self.use_half)

    def segment(self, frame) -> List[dict]:
        """
        Run segmentation on a BGR frame.

        Returns list of:
            {label, class_id, confidence, bbox:[x1,y1,x2,y2], mask:[[x,y],...]}
        """
        if frame is None:
            return []

        segments = []

        try:
            with torch.no_grad():
                results = self.model(
                    frame,
                    device=self.device,
                    conf=self.conf_threshold,   # FIX: was not passed in original
                    half=self.use_half,         # FIX: was missing in original
                    verbose=False,
                )

            for r in results:
                if r.boxes is None or r.masks is None:
                    continue

                boxes   = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                scores  = r.boxes.conf.cpu().numpy()
                masks   = r.masks.xy   # list of polygon point arrays

                for box, cls, score, mask in zip(boxes, classes, scores, masks):
                    x1, y1, x2, y2 = map(int, box)
                    segments.append({
                        "label":      self.names[int(cls)],
                        "class_id":   int(cls),          # FIX: int not numpy.int64
                        "confidence": float(score),
                        "bbox":       [x1, y1, x2, y2],
                        "mask":       mask.tolist(),      # [[x,y], ...] polygon
                    })

        except Exception as e:
            logger.error("Segmenter.segment: %s", e)

        return segments
