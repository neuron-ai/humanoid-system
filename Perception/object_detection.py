from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="yolov8n.pt", conf=0.5):

        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):

        results = self.model(frame)

        detections = []

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):

                if score < self.conf:
                    continue

                x1, y1, x2, y2 = map(int, box)

                detections.append({
                    "label": self.model.names[int(cls)],
                    "confidence": float(score),
                    "bbox": [x1, y1, x2, y2]
                })

        return detections