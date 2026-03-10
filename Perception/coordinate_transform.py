class CoordinateTransformer:

    def __init__(self, fx, fy, cx, cy):

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def pixel_to_world(self, u, v, depth):

        if depth is None:
            return None

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return [float(x), float(y), float(z)]