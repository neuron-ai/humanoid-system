from fastsam import FastSAM


class Segmenter:

    def __init__(self):

        self.model = FastSAM("FastSAM-x.pt")

    def segment(self, frame, bbox):

        results = self.model(frame)

        return results