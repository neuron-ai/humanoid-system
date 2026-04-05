import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "VectorMemory: faiss not installed — vector search disabled.\n"
        "Install: pip install faiss-cpu --break-system-packages"
    )

try:
    import redis as _redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("VectorMemory: redis not installed — id map RAM only")


class VectorMemory:
    """
    CLIP embedding index for semantic object search.

    Dimension must match your encoder:
        CLIP ViT-B/32         → dim = 512  (default)
        sentence-transformers → dim = 384
        CLIP ViT-L/14         → dim = 768

    Changes vs original:
    - faiss import guarded — won't crash if faiss not installed
    - redis import guarded — won't crash if redis not installed
    - INDEX_PATH default changed from /app/data/ (Docker path, doesn't
      exist on Jetson) to /tmp/faiss_index.faiss — always writable.
      Override with FAISS_INDEX_PATH env var for persistent storage.
    - update() silently skips if FAISS unavailable instead of crashing
    - search() returns ([], []) if FAISS unavailable instead of crashing
    """

    SAVE_EVERY = 50
    # Default to /tmp so it always works on Jetson without setup.
    # For persistence across reboots set: FAISS_INDEX_PATH=/ssd/ai/faiss_index.faiss
    INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/tmp/faiss_index.faiss")
    IDS_KEY    = "vector_memory:ids"

    def __init__(self, dim: int = 512):
        self.dim    = dim
        self._count = 0
        self.ids: List[str] = []

        # FAISS index
        self.index = None
        if FAISS_AVAILABLE:
            self.index = self._load_index()
        else:
            logger.warning("VectorMemory: FAISS unavailable — semantic search disabled")

        # Redis for id map persistence
        self._redis_ok = False
        self._redis    = None

        if REDIS_AVAILABLE:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            try:
                self._redis = _redis_lib.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                self._redis_ok = True
                self._load_ids_from_redis()
                logger.info("VectorMemory: Redis connected, %d IDs loaded", len(self.ids))
            except Exception as e:
                self._redis_ok = False
                logger.warning("VectorMemory: Redis unavailable (%s)", e)

    @property
    def is_available(self) -> bool:
        return FAISS_AVAILABLE and self.index is not None

    def update(self, obj: dict) -> None:
        """Add a CLIP embedding. Silently skips if FAISS not installed."""
        if not self.is_available:
            return

        emb = obj.get("embedding")
        if emb is None:
            return

        emb_array = np.array(emb, dtype="float32")

        if emb_array.shape[0] != self.dim:
            logger.error(
                "VectorMemory.update: embedding dim %d != index dim %d",
                emb_array.shape[0], self.dim,
            )
            return

        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm

        self.index.add(emb_array.reshape(1, -1))
        obj_id = str(obj["id"])
        self.ids.append(obj_id)
        self._count += 1

        if self._redis_ok:
            try:
                self._redis.rpush(self.IDS_KEY, obj_id)
            except Exception as e:
                logger.error("VectorMemory.update: Redis rpush failed (%s)", e)

        if self._count % self.SAVE_EVERY == 0:
            self._save_index()

    def search(
        self,
        query_embedding: list,
        k: int = 3,
        distance_threshold: float = 0.5,
    ) -> Tuple[List[str], List[float]]:
        """
        Search k nearest embeddings. Returns ([], []) if FAISS unavailable.
        Lower distance = better match (0 = identical, 2 = opposite).
        """
        if not self.is_available or self.index.ntotal == 0:
            return [], []

        query = np.array(query_embedding, dtype="float32")
        norm  = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        if query.shape[0] != self.dim:
            logger.error(
                "VectorMemory.search: query dim %d != index dim %d",
                query.shape[0], self.dim,
            )
            return [], []

        actual_k          = min(k, self.index.ntotal)
        distances, indices = self.index.search(query.reshape(1, -1), actual_k)

        result_ids   = []
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
        self._save_index()

    def _save_index(self) -> None:
        if not self.is_available:
            return
        try:
            parent = os.path.dirname(self.INDEX_PATH)
            if parent:
                os.makedirs(parent, exist_ok=True)
            faiss.write_index(self.index, self.INDEX_PATH)
            logger.debug(
                "VectorMemory: saved FAISS index to %s (%d vectors)",
                self.INDEX_PATH, self.index.ntotal,
            )
        except Exception as e:
            logger.error("VectorMemory._save_index: failed (%s)", e)

    def _load_index(self):
        if os.path.exists(self.INDEX_PATH):
            try:
                index = faiss.read_index(self.INDEX_PATH)
                logger.info(
                    "VectorMemory: loaded FAISS index from %s (%d vectors)",
                    self.INDEX_PATH, index.ntotal,
                )
                return index
            except Exception as e:
                logger.warning(
                    "VectorMemory: could not load index (%s) — creating new", e
                )
        return faiss.IndexFlatL2(self.dim)

    def _load_ids_from_redis(self) -> None:
        try:
            stored = self._redis.lrange(self.IDS_KEY, 0, -1)
            if stored:
                self.ids = stored
        except Exception as e:
            logger.error("VectorMemory._load_ids_from_redis: failed (%s)", e)
