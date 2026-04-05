import logging
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional

from world_model.object_memory   import ObjectMemory
from world_model.product_memory  import ProductMemory
from world_model.spatial_map     import SpatialMap
from world_model.temporal_memory import TemporalMemory
from world_model.scene_graph     import SceneGraph

logger = logging.getLogger(__name__)

try:
    from world_model.vector_memory import VectorMemory
    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    VECTOR_MEMORY_AVAILABLE = False
    logger.warning("WorldModel: VectorMemory not available (faiss missing)")

try:
    from Perception.clip_embeddings import CLIPEncoder
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("WorldModel: CLIPEncoder not available (clip missing)")


class WorldModel:
    """
    Central world state  single source of truth for the robot.

    Changes vs original:
    1. Absolute imports  works when running files directly on Jetson
       without a package structure (was relative from .object_memory etc.)
    2. VectorMemory and CLIPEncoder are optional  world model starts
       cleanly even when faiss or clip are not installed.
    3. semantic_search() falls back to label-word matching when CLIP
       unavailable  planner still works, just less semantically rich.
    4. update_batch() calls object_memory.clear_stale() every cycle 
       removes ghost objects not seen for STALE_OBJECT_AGE_S seconds.
    5. person_in_scene property  fast check for planner halt logic.
    6. Stale age configurable via STALE_OBJECT_AGE env var (default 5s).
    """

    STALE_OBJECT_AGE_S: float = float(os.environ.get("STALE_OBJECT_AGE", "5.0"))

    def __init__(self):
        self.object_memory   = ObjectMemory()
        self.product_memory  = ProductMemory()
        self.spatial_map     = SpatialMap()
        self.temporal_memory = TemporalMemory()
        self.scene_graph     = SceneGraph()

        self.vector_memory: Optional[Any] = None
        if VECTOR_MEMORY_AVAILABLE:
            try:
                self.vector_memory = VectorMemory(dim=512)
                logger.info("WorldModel: VectorMemory ready (FAISS)")
            except Exception as e:
                logger.warning("WorldModel: VectorMemory init failed (%s)", e)

        self._clip_encoder: Optional[Any] = None
        self.lock = Lock()

        logger.info(
            "WorldModel: initialised (vector=%s clip=%s)",
            self.vector_memory is not None,
            CLIP_AVAILABLE,
        )

    # ---------------------------------------------------------------- #
    #  Write — called by perception pipeline at 20 Hz
    # ---------------------------------------------------------------- #

    def update_batch(self, objects: List[dict]) -> None:
        """
        Ingest fresh detections from perception_pipeline.step().
        Each object must have at minimum: {id, label, position, timestamp}
        """
        if not objects:
            return

        with self.lock:
            for obj in objects:
                if not obj.get("position"):
                    continue
                self.object_memory.update(obj)
                self.spatial_map.update(obj)
                self.temporal_memory.update(obj)
                if self.vector_memory and obj.get("embedding") is not None:
                    self.vector_memory.update(obj)

            # Remove objects not seen recently
            self.object_memory.clear_stale(self.STALE_OBJECT_AGE_S)

            # Rebuild scene graph
            all_objs = {
                f"{o['label']}_{o['id']}": o
                for o in self.object_memory.get_all()
            }
            self.scene_graph.update(all_objs)

        logger.debug("WorldModel.update_batch: %d objects", len(objects))

    # ---------------------------------------------------------------- #
    #  Read — called by planner context_builder
    # ---------------------------------------------------------------- #

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "objects":   self.object_memory.get_all(),
                "positions": self.spatial_map.get_all(),
                "relations": self.scene_graph.get_all(),
            }

    def get_current_detections(self) -> List[dict]:
        with self.lock:
            return self.object_memory.get_all()

    @property
    def person_in_scene(self) -> bool:
        """True if any person is currently tracked. Used by planner halt."""
        with self.lock:
            return len(self.object_memory.get_by_label("person")) > 0

    def find_by_label(self, label: str) -> List[dict]:
        with self.lock:
            return self.object_memory.get_by_label(label)

    def find_by_id(self, obj_id: str) -> Optional[dict]:
        with self.lock:
            return self.object_memory.get(obj_id)

    def get_object_history(self, obj_id: str) -> List[dict]:
        with self.lock:
            return self.temporal_memory.get_history(obj_id)

    def get_object_velocity(self, obj_id: str) -> Optional[List[float]]:
        with self.lock:
            return self.temporal_memory.get_velocity(obj_id)

    def get_neighbors(self, obj_id: str) -> List[dict]:
        with self.lock:
            return self.scene_graph.get_neighbors(obj_id)

    def add_product(self, product: dict) -> None:
        with self.lock:
            self.product_memory.update(product)

    def find_product(self, label: str) -> List[dict]:
        with self.lock:
            return self.product_memory.get_by_label(label)

    # ---------------------------------------------------------------- #
    #  Semantic search
    # ---------------------------------------------------------------- #

    def semantic_search(self, text: str, k: int = 3) -> List[dict]:
        """
        CLIP path when available, label-word fallback otherwise.
        Fallback saves ~450MB (300MB CLIP + 150MB FAISS).
        """
        if self.vector_memory and self.vector_memory.is_available:
            encoder = self._get_clip_encoder()
            if encoder is not None:
                try:
                    emb = encoder.encode_text(text)
                    if emb is not None:
                        with self.lock:
                            ids, distances = self.vector_memory.search(emb.tolist(), k=k)
                        results = []
                        for obj_id, dist in zip(ids, distances):
                            obj = self.object_memory.get(obj_id)
                            if obj:
                                results.append({
                                    "id":       obj_id,
                                    "label":    obj.get("label", "unknown"),
                                    "distance": dist,
                                    "position": obj.get("position"),
                                })
                        return results
                except Exception as e:
                    logger.error("WorldModel.semantic_search: CLIP failed (%s)", e)

        # Label-word fallback
        words = text.lower().split()
        with self.lock:
            all_objects = self.object_memory.get_all()
        results = []
        for obj in all_objects:
            if any(w in obj.get("label", "").lower() for w in words):
                results.append({
                    "id":       f"{obj['label']}_{obj['id']}",
                    "label":    obj["label"],
                    "distance": 0.0,
                    "position": obj.get("position"),
                })
        return results[:k]

    def _get_clip_encoder(self) -> Optional[Any]:
        if self._clip_encoder is not None:
            return self._clip_encoder
        if not CLIP_AVAILABLE:
            return None
        try:
            self._clip_encoder = CLIPEncoder()
            logger.info("WorldModel: CLIP encoder loaded")
        except Exception as e:
            logger.warning("WorldModel: CLIPEncoder load failed (%s)", e)
            self._clip_encoder = None
        return self._clip_encoder


# ── Standalone test ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wm = WorldModel()

    fake = [
        {"id": 1, "label": "bottle", "position": [0.1, 0.0, 1.2], "timestamp": time.time()},
        {"id": 2, "label": "cup",    "position": [0.3, 0.0, 1.5], "timestamp": time.time()},
        {"id": 3, "label": "person", "position": [0.0, 0.0, 2.0], "timestamp": time.time()},
    ]
    wm.update_batch(fake)

    state = wm.get_state()
    print(f"Objects: {len(state['objects'])}")
    for o in state["objects"]:
        print(f"  {o['label']} @ {o['position']}")

    print(f"\nperson_in_scene: {wm.person_in_scene}")

    print("\nsemantic_search('bottle'):")
    for r in wm.semantic_search("bottle"):
        print(f"  {r['label']} dist={r['distance']} pos={r['position']}")
