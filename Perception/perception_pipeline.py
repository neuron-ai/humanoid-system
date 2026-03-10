from Perception.object_detection import ObjectDetector
from Perception.object_tracker import ObjectTracker
from Perception.depth_processing import DepthProcessor
from Perception.coordinate_transform import CoordinateTransformer
from Perception.segmentation import Segmenter
from Perception.clip_embeddings import CLIPEncoder


class PerceptionPipeline:

    def __init__(self, camera):

        self.detector = ObjectDetector()

        self.tracker = ObjectTracker()

        self.depth = DepthProcessor()

        self.transform = CoordinateTransformer(
            camera.fx, camera.fy, camera.cx, camera.cy
        )

        self.segmenter = Segmenter()

        self.clip = CLIPEncoder()

    def run(self, rgb, depth):

        detections = self.detector.detect(rgb)

        tracked = self.tracker.update(detections)

        objects = []

        for obj in tracked:

            x1, y1, x2, y2 = obj["bbox"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            d = self.depth.get_depth(depth, cx, cy)

            world = self.transform.pixel_to_world(cx, cy, d)

            embedding = self.clip.encode_image(rgb)

            objects.append({
                "id": obj["id"],
                "bbox": obj["bbox"],
                "depth": d,
                "world_coord": world,
                "embedding": embedding
            })

        return objects