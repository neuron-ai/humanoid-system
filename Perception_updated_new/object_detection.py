import logging
from typing import List

import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    YOLOv8n object detector with TensorRT-ready FP16 inference on Jetson.

    Fixes vs original:
    - Uses logging not print()
    - model.to("cuda") call removed — YOLO handles device internally via device= param
      Calling .to("cuda") separately can cause tensor device mismatches
    - conf_threshold passed to model() call — no redundant post-filter needed
    - try/except around each inference — bad frames don't crash the pipeline
    - export_tensorrt() helper — converts model to TRT engine for faster Jetson inference
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info("ObjectDetector: loading %s on %s", model_path, self.device)

        self.model = YOLO(model_path)
        self.names = self.model.names

        # FP16 only on GPU — CPU doesn't support half precision
        self.use_half = self.device.startswith("cuda")

        logger.info(
            "ObjectDetector: ready — %d classes, conf=%.2f, half=%s",
            len(self.names), conf_threshold, self.use_half
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def detect(self, frame) -> List[dict]:
        """
        Run inference on a BGR frame.

        Returns list of:
            {label, class_id, confidence, bbox: [x1,y1,x2,y2]}
        """
        if frame is None:
            return []

        detections = []

        try:
            with torch.no_grad():
                results = self.model(
                    frame,
                    device=self.device,
                    conf=self.conf_threshold,   # filter at model level — no post-filter
                    half=self.use_half,
                    verbose=False,
                )

            for r in results:
                if r.boxes is None:
                    continue

                boxes   = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                scores  = r.boxes.conf.cpu().numpy()

                for box, cls, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        "label":      self.names[int(cls)],
                        "class_id":   int(cls),
                        "confidence": float(score),
                        "bbox":       [x1, y1, x2, y2],
                    })

        except Exception as e:
            logger.error("ObjectDetector.detect: %s", e)

        return detections

    def export_tensorrt(self, output_path: str = "yolov8n.engine") -> str:
        """
        Export model to TensorRT engine for faster inference on Jetson.
        Run this ONCE after deployment — then load the .engine file instead of .pt

        Usage:
            detector = ObjectDetector("yolov8n.pt")
            detector.export_tensorrt("yolov8n.engine")
            # Then restart with:
            detector = ObjectDetector("yolov8n.engine")
        """
        logger.info("ObjectDetector: exporting to TensorRT — this takes ~5 min on Jetson")
        self.model.export(format="engine", half=True, device=self.device)
        logger.info("ObjectDetector: TRT engine saved to %s", output_path)
        return output_path
