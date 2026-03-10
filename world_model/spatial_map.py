import math


class SpatialMap:

    def __init__(self):
        self.positions = {}

    def update(self, obj_id, coord):

        if coord is None:
            return

        self.positions[obj_id] = coord

    def get(self, obj_id):
        return self.positions.get(obj_id)

    def nearby(self, coord, radius=1.0):

        result = []

        for obj_id, pos in self.positions.items():

            dx = pos[0] - coord[0]
            dy = pos[1] - coord[1]
            dz = pos[2] - coord[2]

            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist < radius:
                result.append(obj_id)

        return result