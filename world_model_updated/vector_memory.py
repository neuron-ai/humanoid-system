import json
import logging
import os
from typing import List, Optional, Tuple

import faiss
import numpy as np
import redis

logger = logging.getLogger(__name__)


class VectorMemory:
    """
    CLIP embedding index for semantic object search.

    Dimension must match your encoder:
        CLIP ViT-B/32         → dim = 512
        sentence-transformers → dim = 384
        CLIP ViT-L/14         → dim = 768

    Persistence:
        FAISS index saved to disk (faiss_index.bin) on every N updates.
        ID map saved to Redis so index survives restarts.

    search() returns (ids, distances) so callers can apply confidence thresholds.
    """

    SAVE_EVERY = 50    # save index to disk every N additions
    INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/data/object_index.faiss")
    IDS_KEY    = "vector_memory:ids"

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._count = 0

        # id_map: FAISS row index → object id string
        self.ids: List[str] = []

        # Try loading existing index from disk
        self.index = self._load_index()

        # Redis for id map persistence
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
            self._load_ids_from_redis()
            logger.info("VectorMemory: Redis connected, %d IDs loaded", len(self.ids))
        except Exception as e:
            self._redis_ok = False
            logger.warning("VectorMemory: Redis unavailable (%s)", e)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def update(self, obj: dict) -> None:
        """
        Add or update a CLIP embedding for an object.
        obj must have: {id, embedding}  where embedding is a list of floats.
        """
        emb = obj.get("embedding")
        if emb is None:
            logger.debug("VectorMemory.update: object %s has no embedding — skipping", obj.get("id"))
            return

        emb_array = np.array(emb, dtype="float32")

        # Guard: check embedding dimension matches index
        if emb_array.shape[0] != self.dim:
            logger.error(
                "VectorMemory.update: embedding dim %d != index dim %d — "
                "check your CLIP model. ViT-B/32=512, sentence-transformers=384.",
                emb_array.shape[0], self.dim
            )
            return

        # Normalise to unit vector (cosine similarity via L2 on normalised vectors)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm

        self.index.add(emb_array.reshape(1, -1))
        obj_id = str(obj["id"])
        self.ids.append(obj_id)
        self._count += 1

        # Persist id map to Redis
        if self._redis_ok:
            try:
                self._redis.rpush(self.IDS_KEY, obj_id)
            except Exception as e:
                logger.error("VectorMemory.update: Redis rpush failed (%s)", e)

        # Periodically save index to disk
        if self._count % self.SAVE_EVERY == 0:
            self._save_index()

    def search(self, query_embedding: list, k: int = 3,
               distance_threshold: float = 0.5) -> Tuple[List[str], List[float]]:
        """
        Search for k nearest objects to a query embedding.

        Returns:
            (ids, distances) — both lists, filtered by distance_threshold.
            Distances are L2 on normalised vectors (0=identical, 2=opposite).
            Lower distance = better match.

        Original code only returned ids — distances dropped.
        Callers need distances to apply confidence thresholds.
        """
        if self.index.ntotal == 0:
            logger.debug("VectorMemory.search: index is empty")
            return [], []

        query = np.array(query_embedding, dtype="float32")

        # Normalise query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Guard dim
        if query.shape[0] != self.dim:
            logger.error("VectorMemory.search: query dim %d != index dim %d",
                         query.shape[0], self.dim)
            return [], []

        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query.reshape(1, -1), actual_k)

        result_ids = []
        result_dists = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            if dist > distance_threshold:
                continue
            result_ids.append(self.ids[idx])
            result_dists.append(round(float(dist), 4))

        return result_ids, result_dists

    def save(self) -> None:
        """Manually trigger index save to disk."""
        self._save_index()

    # ------------------------------------------------------------------ #
    #  Persistence helpers
    # ------------------------------------------------------------------ #

    def _save_index(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.INDEX_PATH), exist_ok=True)
            faiss.write_index(self.index, self.INDEX_PATH)
            logger.debug("VectorMemory: saved FAISS index to %s (%d vectors)",
                         self.INDEX_PATH, self.index.ntotal)
        except Exception as e:
            logger.error("VectorMemory._save_index: failed (%s)", e)

    def _load_index(self) -> faiss.IndexFlatL2:
        if os.path.exists(self.INDEX_PATH):
            try:
                index = faiss.read_index(self.INDEX_PATH)
                logger.info("VectorMemory: loaded FAISS index from %s (%d vectors)",
                            self.INDEX_PATH, index.ntotal)
                return index
            except Exception as e:
                logger.warning("VectorMemory: could not load index (%s) — creating new", e)
        return faiss.IndexFlatL2(self.dim)

    def _load_ids_from_redis(self) -> None:
        try:
            stored = self._redis.lrange(self.IDS_KEY, 0, -1)
            if stored:
                self.ids = stored
                logger.debug("VectorMemory: loaded %d IDs from Redis", len(self.ids))
        except Exception as e:
            logger.error("VectorMemory._load_ids_from_redis: failed (%s)", e)
