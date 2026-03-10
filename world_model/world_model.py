from world_model.object_memory import ObjectMemory
from world_model.spatial_map import SpatialMap
from world_model.vector_memory import VectorMemory
from world_model.product_memory import ProductMemory
from world_model.knowledge_base import KnowledgeBase
from world_model.scene_graph import SceneGraph
from world_model.updater import WorldUpdater


class WorldModel:

    def __init__(self):

        self.memory = ObjectMemory()
        self.spatial = SpatialMap()
        self.vectors = VectorMemory()
        self.products = ProductMemory()
        self.kb = KnowledgeBase()
        self.scene = SceneGraph()

        self.updater = WorldUpdater(
            self.memory,
            self.spatial,
            self.vectors
        )

    def update(self, objects):

        self.updater.update(objects)

    def get_objects(self):

        return self.memory.all()

    def find_object_by_text(self, embedding):

        obj_id, score = self.vectors.search(embedding)

        return self.memory.get(obj_id)

    def get_position(self, obj_id):

        return self.spatial.get(obj_id)