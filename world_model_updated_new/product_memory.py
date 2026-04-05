import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("ProductMemory: redis not installed — RAM only")


class ProductMemory:
    """
    Stores product catalogue — known item types the robot handles.
    Persistent: survives restarts via Redis (no TTL).

    Changes vs original:
    - redis import guarded — won't crash if redis not installed
    """

    KEY_PREFIX = "product:"
    SET_KEY    = "product:all_ids"

    def __init__(self):
        self.products: Dict[str, dict] = {}
        self._redis_ok = False
        self._redis    = None

        if not REDIS_AVAILABLE:
            return

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = _redis_lib.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
            logger.info("ProductMemory: Redis connected at %s", redis_url)
            self._load_from_redis()
        except Exception as e:
            self._redis_ok = False
            logger.warning("ProductMemory: Redis unavailable (%s) — RAM only", e)

    def update(self, obj: dict) -> None:
        product_id = str(obj["id"])
        record = {
            "id":            product_id,
            "label":         obj.get("label",        "unknown"),
            "description":   obj.get("description",  ""),
            "category":      obj.get("category",     ""),
            "color":         obj.get("color",         ""),
            "size":          obj.get("size",          ""),
            "location_hint": obj.get("location_hint", ""),
        }
        self.products[product_id] = record

        if self._redis_ok:
            try:
                self._redis.set(f"{self.KEY_PREFIX}{product_id}", json.dumps(record))
                self._redis.sadd(self.SET_KEY, product_id)
            except Exception as e:
                logger.error("ProductMemory.update: Redis write failed (%s)", e)

    def get(self, product_id: str) -> Optional[dict]:
        pid = str(product_id)
        if pid in self.products:
            return self.products[pid]

        if self._redis_ok:
            try:
                data = self._redis.get(f"{self.KEY_PREFIX}{pid}")
                if data:
                    record = json.loads(data)
                    self.products[pid] = record
                    return record
            except Exception as e:
                logger.error("ProductMemory.get: Redis read failed (%s)", e)

        return None

    def get_by_label(self, label: str) -> List[dict]:
        label_lower = label.lower()
        return [p for p in self.products.values()
                if label_lower in p.get("label", "").lower()]

    def get_all(self) -> List[dict]:
        return list(self.products.values())

    def remove(self, product_id: str) -> None:
        pid = str(product_id)
        self.products.pop(pid, None)
        if self._redis_ok:
            try:
                self._redis.delete(f"{self.KEY_PREFIX}{pid}")
                self._redis.srem(self.SET_KEY, pid)
            except Exception as e:
                logger.error("ProductMemory.remove: Redis delete failed (%s)", e)

    def _load_from_redis(self) -> None:
        try:
            ids = self._redis.smembers(self.SET_KEY)
            for pid in ids:
                data = self._redis.get(f"{self.KEY_PREFIX}{pid}")
                if data:
                    self.products[pid] = json.loads(data)
            logger.info("ProductMemory: loaded %d products from Redis", len(self.products))
        except Exception as e:
            logger.error("ProductMemory._load_from_redis: failed (%s)", e)
