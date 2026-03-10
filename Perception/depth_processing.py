class DepthProcessor:

    def __init__(self, depth_scale=0.001):

        self.depth_scale = depth_scale

    def get_depth(self, depth_frame, x, y):

        depth_value = depth_frame[y, x]

        if depth_value == 0:
            return None

        return float(depth_value * self.depth_scale)

    def get_bbox_depth(self, depth_frame, bbox):

        x1, y1, x2, y2 = bbox

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        return self.get_depth(depth_frame, cx, cy)