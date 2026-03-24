import json
import logging
import os
from typing import Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class ObjectMemory:
    """
    Stores detected objects in both RAM cache and Redis.
    Redis key format:  obj:{label}_{id}
    Redis set key:     obj:all_ids   (tracks all known object IDs)

    get_all() returns a LIST of objects — compatible with planner context_builder.
    """

    KEY_PREFIX = "obj:"
    SET_KEY = "obj:all_ids"
    DEFAULT_TTL = 300   # seconds — objects expire after 5 min if not refreshed

    def __init__(self):
        self.cache: Dict[str, dict] = {}

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
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
        """Get a single object by id. Checks RAM first, then Redis."""
        if obj_id in self.cache:
            return self.cache[obj_id]

        if self._redis_ok:
            try:
                data = self._redis.get(f"{self.KEY_PREFIX}{obj_id}")
                if data:
                    obj = json.loads(data)
                    self.cache[obj_id] = obj   # warm RAM cache
                    return obj
            except Exception as e:
                logger.error("ObjectMemory.get: Redis read failed (%s)", e)

        return None

    def get_all(self) -> List[dict]:
        """
        Return all known objects as a LIST.
        Planner context_builder expects List[dict], not Dict.
        """
        return list(self.cache.values())

    def get_by_label(self, label: str) -> List[dict]:
        """Return all objects matching a label string."""
        return [o for o in self.cache.values() if o.get("label") == label]

    def remove(self, obj_id: str) -> None:
        """Remove an object from cache and Redis."""
        self.cache.pop(obj_id, None)
        if self._redis_ok:
            try:
                self._redis.delete(f"{self.KEY_PREFIX}{obj_id}")
                self._redis.srem(self.SET_KEY, obj_id)
            except Exception as e:
                logger.error("ObjectMemory.remove: Redis delete failed (%s)", e)

    def clear(self) -> None:
        """Clear all objects."""
        self.cache.clear()
        if self._redis_ok:
            try:
                keys = self._redis.keys(f"{self.KEY_PREFIX}*")
                if keys:
                    self._redis.delete(*keys)
                self._redis.delete(self.SET_KEY)
            except Exception as e:
                logger.error("ObjectMemory.clear: Redis clear failed (%s)", e)
