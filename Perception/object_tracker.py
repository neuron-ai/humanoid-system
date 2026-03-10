from ultralytics.trackers.byte_tracker import BYTETracker


class ObjectTracker:

    def __init__(self):

        self.tracker = BYTETracker()

    def update(self, detections):

        tracks = self.tracker.update(detections)

        objects = []

        for track in tracks:

            obj = {
                "id": int(track.track_id),
                "bbox": track.tlbr
            }

            objects.append(obj)

        return objects