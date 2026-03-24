import json
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not installed — using in-memory fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not installed — using exact-match fallback")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence_transformers not installed — using exact-match fallback")


class ExperienceMemory:

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

    def __init__(self):
        # Redis connection
        self._redis = None
        if REDIS_AVAILABLE:
            try:
                redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("ExperienceMemory: Redis connected at %s", redis_url)
            except Exception as e:
                logger.warning("ExperienceMemory: Redis unavailable (%s) — using RAM fallback", e)
                self._redis = None

        # Sentence encoder
        self._encoder = None
        if ST_AVAILABLE:
            try:
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("ExperienceMemory: SentenceTransformer loaded")
            except Exception as e:
                logger.warning("ExperienceMemory: SentenceTransformer load failed (%s)", e)

        # FAISS vector index
        self._index = None
        self._id_map: list = []   # maps faiss row index → redis key / memory list index
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatL2(self.EMBEDDING_DIM)

        # RAM-only fallback list (used when Redis is down)
        self._memory: list = []

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def store(self, goal: str, plan: list, outcome: str = "success",
              object_id: Optional[str] = None) -> None:
        """Persist a goal→plan pair with its outcome."""
        record = {
            "goal": goal,
            "plan": json.dumps(plan),
            "outcome": outcome,
            "object_id": object_id or "",
        }

        if self._redis:
            try:
                key = f"exp:{self._redis.incr('exp:counter')}"
                self._redis.hset(key, mapping=record)
                self._add_to_index(goal, key)
                logger.debug("ExperienceMemory: stored to Redis key=%s", key)
                return
            except Exception as e:
                logger.error("ExperienceMemory: Redis store failed (%s) — falling back to RAM", e)

        # RAM fallback
        idx = len(self._memory)
        self._memory.append(record)
        self._add_to_index(goal, str(idx))

    def retrieve(self, goal: str, similarity_threshold: float = 0.7) -> Optional[list]:
        """
        Return the plan for the most similar past goal.
        Uses FAISS vector similarity when available, otherwise exact string match.
        Returns None if nothing found above threshold.
        """
        if self._index is not None and self._encoder is not None and len(self._id_map) > 0:
            return self._retrieve_by_vector(goal, similarity_threshold)

        return self._retrieve_exact(goal)

    def retrieve_by_object(self, object_id: str) -> Optional[dict]:
        """Find past experiences involving a specific object_id."""
        if self._redis:
            try:
                keys = self._redis.keys("exp:*")
                for key in keys:
                    record = self._redis.hgetall(key)
                    if record.get("object_id") == object_id:
                        return {
                            "goal": record["goal"],
                            "plan": json.loads(record["plan"]),
                            "outcome": record["outcome"],
                        }
            except Exception as e:
                logger.error("ExperienceMemory: Redis retrieve_by_object failed (%s)", e)

        for record in self._memory:
            if record.get("object_id") == object_id:
                return {
                    "goal": record["goal"],
                    "plan": json.loads(record["plan"]),
                    "outcome": record["outcome"],
                }
        return None

    def all_goals(self) -> list:
        """Return list of all stored goal strings (for debugging)."""
        if self._redis:
            try:
                keys = self._redis.keys("exp:*")
                return [self._redis.hget(k, "goal") for k in keys]
            except Exception:
                pass
        return [r["goal"] for r in self._memory]

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _encode(self, text: str) -> np.ndarray:
        return self._encoder.encode([text])[0].astype("float32")

    def _add_to_index(self, goal: str, storage_key: str) -> None:
        if self._index is None or self._encoder is None:
            return
        try:
            emb = self._encode(goal)
            self._index.add(emb.reshape(1, -1))
            self._id_map.append(storage_key)
        except Exception as e:
            logger.error("ExperienceMemory: FAISS add failed (%s)", e)

    def _retrieve_by_vector(self, goal: str, threshold: float) -> Optional[list]:
        try:
            emb = self._encode(goal)
            distances, indices = self._index.search(emb.reshape(1, -1), 1)
            dist = float(distances[0][0])
            idx = int(indices[0][0])

            # L2 distance: lower = more similar. Convert threshold: typical good match < 0.5
            l2_threshold = 1.0 - threshold  # rough mapping
            if dist > l2_threshold or idx < 0:
                logger.debug("ExperienceMemory: no vector match (dist=%.3f threshold=%.3f)", dist, l2_threshold)
                return None

            storage_key = self._id_map[idx]
            logger.debug("ExperienceMemory: vector match key=%s dist=%.3f", storage_key, dist)

            if self._redis:
                record = self._redis.hgetall(storage_key)
                if record:
                    return json.loads(record["plan"])

            # RAM fallback
            int_idx = int(storage_key)
            if 0 <= int_idx < len(self._memory):
                return json.loads(self._memory[int_idx]["plan"])

        except Exception as e:
            logger.error("ExperienceMemory: vector retrieve failed (%s)", e)

        return None

    def _retrieve_exact(self, goal: str) -> Optional[list]:
        """Exact string match fallback."""
        if self._redis:
            try:
                keys = self._redis.keys("exp:*")
                for key in keys:
                    stored_goal = self._redis.hget(key, "goal")
                    if stored_goal == goal:
                        plan_str = self._redis.hget(key, "plan")
                        return json.loads(plan_str)
            except Exception as e:
                logger.error("ExperienceMemory: Redis exact retrieve failed (%s)", e)

        for record in self._memory:
            if record["goal"] == goal:
                return json.loads(record["plan"])

        return None
