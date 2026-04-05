import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("ObjectMemory: redis not installed — RAM only")


class ObjectMemory:
    """
    Stores detected objects in both RAM cache and Redis.
    Redis key format:  obj:{label}_{id}
    Redis set key:     obj:all_ids

    Changes vs original:
    - redis import guarded — won't crash if redis not installed
    - clear_stale() added — removes objects not seen for N seconds
      Called by WorldModel.update_batch() to stop stale detections
      from accumulating and confusing the planner
    """

    KEY_PREFIX  = "obj:"
    SET_KEY     = "obj:all_ids"
    DEFAULT_TTL = 300

    def __init__(self):
        self.cache: Dict[str, dict] = {}
        self._redis_ok = False
        self._redis    = None

        if not REDIS_AVAILABLE:
            return

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = _redis_lib.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
            logger.info("ObjectMemory: Redis connected at %s", redis_url)
        except Exception as e:
            self._redis_ok = False
            logger.warning("ObjectMemory: Redis unavailable (%s) — RAM only", e)

    def update(self, obj: dict) -> None:
        """Store or update a detected object. Resets TTL on each update."""
        obj_id = f"{obj['label']}_{obj['id']}"
        self.cache[obj_id] = obj

        if self._redis_ok:
            try:
                key = f"{self.KEY_PREFIX}{obj_id}"
                self._redis.setex(key, self.DEFAULT_TTL, json.dumps(obj))
                self._redis.sadd(self.SET_KEY, obj_id)
            except Exception as e:
                logger.error("ObjectMemory.update: Redis write failed (%s)", e)

    def get(self, obj_id: str) -> Optional[dict]:
        if obj_id in self.cache:
            return self.cache[obj_id]

        if self._redis_ok:
            try:
                data = self._redis.get(f"{self.KEY_PREFIX}{obj_id}")
                if data:
                    obj = json.loads(data)
                    self.cache[obj_id] = obj
                    return obj
            except Exception as e:
                logger.error("ObjectMemory.get: Redis read failed (%s)", e)

        return None

    def get_all(self) -> List[dict]:
        return list(self.cache.values())

    def get_by_label(self, label: str) -> List[dict]:
        return [o for o in self.cache.values() if o.get("label") == label]

    def clear_stale(self, max_age_s: float = 5.0) -> int:
        """
        Remove objects not updated for more than max_age_s seconds.
        Returns count of removed objects.
        Called every update_batch() cycle to prevent ghost objects
        from accumulating in the planner context.
        """
        now   = time.time()
        stale = [
            k for k, v in self.cache.items()
            if now - v.get("timestamp", now) > max_age_s
        ]
        for k in stale:
            self.cache.pop(k, None)
            if self._redis_ok:
                try:
                    self._redis.delete(f"{self.KEY_PREFIX}{k}")
                    self._redis.srem(self.SET_KEY, k)
                except Exception:
                    pass
        if stale:
            logger.debug("ObjectMemory.clear_stale: removed %d stale objects", len(stale))
        return len(stale)

    def remove(self, obj_id: str) -> None:
        self.cache.pop(obj_id, None)
        if self._redis_ok:
            try:
                self._redis.delete(f"{self.KEY_PREFIX}{obj_id}")
                self._redis.srem(self.SET_KEY, obj_id)
            except Exception as e:
                logger.error("ObjectMemory.remove: Redis delete failed (%s)", e)

    def clear(self) -> None:
        self.cache.clear()
        if self._redis_ok:
            try:
                keys = self._redis.keys(f"{self.KEY_PREFIX}*")
                if keys:
                    self._redis.delete(*keys)
                self._redis.delete(self.SET_KEY)
            except Exception as e:
                logger.error("ObjectMemory.clear: Redis clear failed (%s)", e)
