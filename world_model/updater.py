class WorldUpdater:

    def __init__(self, memory, spatial, vectors):

        self.memory = memory
        self.spatial = spatial
        self.vectors = vectors

    def update(self, objects):

        for obj in objects:

            self.memory.update(obj)

            if "world_coord" in obj:
                self.spatial.update(obj["id"], obj["world_coord"])

            if "embedding" in obj:
                self.vectors.add(obj["id"], obj["embedding"])

        self.memory.remove_stale()