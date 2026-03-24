import logging
import os
from threading import Lock
from typing import Any, Dict, List, Optional

from .object_memory import ObjectMemory
from .product_memory import ProductMemory
from .vector_memory import VectorMemory
from .spatial_map import SpatialMap
from .temporal_memory import TemporalMemory
from .scene_graph import SceneGraph

logger = logging.getLogger(__name__)


class WorldModel:
    """
    Central world model — single source of truth for the robot's understanding
    of the environment.

    Aggregates:
        ObjectMemory    — live detected objects (Redis + RAM, TTL 5min)
        ProductMemory   — known product catalogue (Redis, permanent)
        VectorMemory    — CLIP embeddings for semantic search (FAISS + disk)
        SpatialMap      — 3D positions (Redis + RAM, TTL 5min)
        TemporalMemory  — position history + velocity (Redis Streams + RAM)
        SceneGraph      — spatial relations between objects (Neo4j or RAM)

    Thread-safe: all public methods use self.lock.

    get_state() returns:
        {
            "objects":   List[dict]       ← list, not dict (planner expects list)
            "positions": Dict[str, list]  ← {obj_id: [x,y,z]}
            "relations": List[dict]       ← [{from_id, to_id, relation_type, distance}]
        }
    """

    def __init__(self):
        self.object_memory   = ObjectMemory()
        self.product_memory  = ProductMemory()
        self.vector_memory   = VectorMemory(dim=512)   # CLIP ViT-B/32 = 512
        self.spatial_map     = SpatialMap()
        self.temporal_memory = TemporalMemory()
        self.scene_graph     = SceneGraph()

        self.lock = Lock()

        # Lazy-load CLIP encoder for semantic_search()
        self._clip_encoder = None

        logger.info("WorldModel: initialised")

    # ------------------------------------------------------------------ #
    #  Write path — called by perception pipeline
    # ------------------------------------------------------------------ #

    def update_batch(self, objects: List[dict]) -> None:
        """
        Ingest a fresh batch of detected objects from the perception pipeline.
        Each object must have: {id, label, bbox, position, embedding, texture, timestamp}
        """
        if not objects:
            return

        with self.lock:
            for obj in objects:
                # Skip objects with no position (depth unavailable)
                if not obj.get("position"):
                    continue

                self.object_memory.update(obj)
                self.spatial_map.update(obj)
                self.temporal_memory.update(obj)

                # Only index objects that have CLIP embeddings
                if obj.get("embedding") is not None:
                    self.vector_memory.update(obj)

            # Rebuild scene graph relations from current object set
            all_objects_dict = {
                f"{o['label']}_{o['id']}": o
                for o in self.object_memory.get_all()
            }
            self.scene_graph.update(all_objects_dict)

        logger.debug("WorldModel.update_batch: processed %d objects", len(objects))

    # ------------------------------------------------------------------ #
    #  Read path — called by planner context_builder
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict[str, Any]:
        """
        Return full world state snapshot.

        Returns dict with:
            objects   → List[dict]        (was dict in original — broke planner)
            positions → Dict[str, list]   (was numpy arrays — broke JSON serialisation)
            relations → List[dict]        (was missing get_all() in scene_graph)
        """
        with self.lock:
            return {
                "objects":   self.object_memory.get_all(),    # List[dict]
                "positions": self.spatial_map.get_all(),      # Dict[str, list]
                "relations": self.scene_graph.get_all(),      # List[dict]
            }

    def get_current_detections(self) -> List[dict]:
        """
        Alias used by context_builder's perception integration.
        Returns same as get_state()['objects'].
        """
        with self.lock:
            return self.object_memory.get_all()

    def find_by_label(self, label: str) -> List[dict]:
        """
        Find all current objects matching a label.
        Fixed: original iterated dict.values() which is fine,
        but now correctly uses get_all() which returns a list.
        """
        with self.lock:
            return self.object_memory.get_by_label(label)

    def find_by_id(self, obj_id: str) -> Optional[dict]:
        with self.lock:
            return self.object_memory.get(obj_id)

    def semantic_search(self, text: str, k: int = 3) -> List[dict]:
        """
        Find objects semantically similar to a text description.

        FIXED: original passed raw text string directly to vector_memory.search()
        which expects a float32 numpy array — would crash or return garbage.
        Now properly encodes text to CLIP embedding first.

        Returns list of {id, label, distance, position} dicts.
        """
        encoder = self._get_clip_encoder()
        if encoder is None:
            logger.warning("WorldModel.semantic_search: CLIP encoder unavailable")
            return []

        try:
            import torch
            with torch.no_grad():
                import clip
                tokens = clip.tokenize([text]).to(encoder.device)
                text_emb = encoder.model.encode_text(tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                query = text_emb.cpu().numpy()[0]
        except Exception as e:
            logger.error("WorldModel.semantic_search: encoding failed (%s)", e)
            return []

        with self.lock:
            ids, distances = self.vector_memory.search(query, k=k)

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

    def get_object_history(self, obj_id: str) -> List[dict]:
        """Return temporal history for an object."""
        with self.lock:
            return self.temporal_memory.get_history(obj_id)

    def get_object_velocity(self, obj_id: str):
        """Return latest velocity estimate for an object."""
        with self.lock:
            return self.temporal_memory.get_velocity(obj_id)

    def get_neighbors(self, obj_id: str) -> List[dict]:
        """Return spatially related objects from scene graph."""
        with self.lock:
            return self.scene_graph.get_neighbors(obj_id)

    def add_product(self, product: dict) -> None:
        """Add a known product to the product catalogue."""
        with self.lock:
            self.product_memory.update(product)

    def find_product(self, label: str) -> List[dict]:
        """Search product catalogue by label."""
        with self.lock:
            return self.product_memory.get_by_label(label)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _get_clip_encoder(self):
        """Lazy-load CLIP encoder — only instantiated if semantic_search is called."""
        if self._clip_encoder is not None:
            return self._clip_encoder
        try:
            from .clip_embeddings import CLIPEncoder  # relative import from Perception
            self._clip_encoder = CLIPEncoder()
            logger.info("WorldModel: CLIP encoder loaded for semantic search")
        except Exception as e:
            logger.warning("WorldModel: cannot load CLIPEncoder (%s)", e)
            self._clip_encoder = None
        return self._clip_encoder
