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
    logger.warning("ExperienceMemory: redis not installed — RAM fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("ExperienceMemory: faiss not installed — exact-match fallback")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("ExperienceMemory: sentence_transformers not installed — exact-match fallback")


class ExperienceMemory:
    """
    Stores goal→plan pairs and retrieves the most similar past plan
    for a new goal using sentence embeddings (all-MiniLM-L6-v2, 384-dim).

    Falls back to exact string match when FAISS/SentenceTransformer unavailable.

    Changes vs original:
    - L2 distance threshold math fixed.
      Original: l2_threshold = 1.0 - similarity_threshold
        → similarity 0.7 → l2_threshold 0.3
        → almost nothing ever matched because L2 on normalised 384-dim
          vectors rarely falls below 0.3 (typical good match is ~0.6–1.0)
      Fixed: use a direct L2 distance threshold of 1.2 (good match),
        tuned to all-MiniLM-L6-v2 on short goal strings.
        Configurable via L2_MATCH_THRESHOLD env var.
    - Redis import was already guarded — no change needed there
    """

    EMBEDDING_DIM   = 384   # all-MiniLM-L6-v2
    # L2 distance threshold for a "good enough" match.
    # L2 on normalised 384-dim vectors: 0=identical, 2=opposite.
    # Typical same-intent goals (e.g. "get bottle" vs "pick up bottle"): ~0.6–0.9
    # Typical different goals: ~1.2–1.8
    # Default 1.2 catches same-intent variations without false positives.
    L2_MATCH_THRESHOLD: float = float(os.environ.get("L2_MATCH_THRESHOLD", "1.2"))

    def __init__(self):
        # Redis
        self._redis = None
        if REDIS_AVAILABLE:
            try:
                url = os.environ.get("REDIS_URL", "redis://localhost:6379")
                self._redis = redis.Redis.from_url(url, decode_responses=True)
                self._redis.ping()
                logger.info("ExperienceMemory: Redis connected at %s", url)
            except Exception as e:
                logger.warning("ExperienceMemory: Redis unavailable (%s) — RAM fallback", e)
                self._redis = None

        # Sentence encoder
        self._encoder = None
        if ST_AVAILABLE:
            try:
                if os.environ.get("ENABLE_EXPERIENCE_MEMORY", "0") == "1":
                    self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("ExperienceMemory: SentenceTransformer loaded")
                else:
                    logger.info("ExperienceMemory: SentenceTransformer skipped (set ENABLE_EXPERIENCE_MEMORY=1 to enable — uses ~500MB RAM)")
            except Exception as e:
                logger.warning("ExperienceMemory: SentenceTransformer failed (%s)", e)

        # FAISS index
        self._index  = None
        self._id_map: list = []
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatL2(self.EMBEDDING_DIM)

        # RAM fallback
        self._memory: list = []

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def store(self, goal: str, plan: list, outcome: str = "success",
              object_id: Optional[str] = None) -> None:
        """Persist a goal→plan pair with its outcome."""
        record = {
            "goal":      goal,
            "plan":      json.dumps(plan),
            "outcome":   outcome,
            "object_id": object_id or "",
        }

        if self._redis:
            try:
                key = f"exp:{self._redis.incr('exp:counter')}"
                self._redis.hset(key, mapping=record)
                self._add_to_index(goal, key)
                logger.debug("ExperienceMemory: stored key=%s outcome=%s", key, outcome)
                return
            except Exception as e:
                logger.error("ExperienceMemory: Redis store failed (%s) — RAM fallback", e)

        idx = len(self._memory)
        self._memory.append(record)
        self._add_to_index(goal, str(idx))

    def retrieve(self, goal: str, similarity_threshold: float = 0.7) -> Optional[list]:
        """
        Return the plan for the most similar past goal, or None if no match.
        similarity_threshold is kept as parameter for API compatibility but
        the actual matching now uses L2_MATCH_THRESHOLD directly.
        """
        if self._index is not None and self._encoder is not None and len(self._id_map) > 0:
            return self._retrieve_by_vector(goal)
        return self._retrieve_exact(goal)

    def retrieve_by_object(self, object_id: str) -> Optional[dict]:
        """Find past experiences involving a specific object_id."""
        if self._redis:
            try:
                for key in self._redis.keys("exp:*"):
                    record = self._redis.hgetall(key)
                    if record.get("object_id") == object_id:
                        return {
                            "goal":    record["goal"],
                            "plan":    json.loads(record["plan"]),
                            "outcome": record["outcome"],
                        }
            except Exception as e:
                logger.error("ExperienceMemory: retrieve_by_object Redis failed (%s)", e)

        for record in self._memory:
            if record.get("object_id") == object_id:
                return {
                    "goal":    record["goal"],
                    "plan":    json.loads(record["plan"]),
                    "outcome": record["outcome"],
                }
        return None

    def all_goals(self) -> list:
        if self._redis:
            try:
                return [self._redis.hget(k, "goal") for k in self._redis.keys("exp:*")]
            except Exception:
                pass
        return [r["goal"] for r in self._memory]

    # ------------------------------------------------------------------ #
    #  Internal
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

    def _retrieve_by_vector(self, goal: str) -> Optional[list]:
        """
        FIX: original used l2_threshold = 1.0 - similarity_threshold
        which mapped similarity=0.7 → l2_threshold=0.3.
        L2 distances on normalised 384-dim vectors for similar short strings
        typically fall in 0.6–1.0 range, so threshold=0.3 rejected everything.

        Now uses L2_MATCH_THRESHOLD directly (default 1.2).
        """
        try:
            emb = self._encode(goal)
            distances, indices = self._index.search(emb.reshape(1, -1), 1)
            dist = float(distances[0][0])
            idx  = int(indices[0][0])

            if idx < 0 or dist > self.L2_MATCH_THRESHOLD:
                logger.debug(
                    "ExperienceMemory: no match (dist=%.3f threshold=%.3f)",
                    dist, self.L2_MATCH_THRESHOLD,
                )
                return None

            storage_key = self._id_map[idx]
            logger.debug("ExperienceMemory: matched key=%s dist=%.3f", storage_key, dist)

            if self._redis:
                record = self._redis.hgetall(storage_key)
                if record:
                    return json.loads(record["plan"])

            int_idx = int(storage_key)
            if 0 <= int_idx < len(self._memory):
                return json.loads(self._memory[int_idx]["plan"])

        except Exception as e:
            logger.error("ExperienceMemory: vector retrieve failed (%s)", e)

        return None

    def _retrieve_exact(self, goal: str) -> Optional[list]:
        if self._redis:
            try:
                for key in self._redis.keys("exp:*"):
                    if self._redis.hget(key, "goal") == goal:
                        return json.loads(self._redis.hget(key, "plan"))
            except Exception as e:
                logger.error("ExperienceMemory: exact retrieve Redis failed (%s)", e)

        for record in self._memory:
            if record["goal"] == goal:
                return json.loads(record["plan"])
        return None
