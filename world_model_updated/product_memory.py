import json
import logging
import os
from typing import Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class ProductMemory:
    """
    Stores product catalogue — known item types the robot is expected to handle.
    Unlike ObjectMemory (which stores live detections), ProductMemory stores
    persistent product definitions that survive restarts.

    Redis key format:  product:{id}
    Redis set key:     product:all_ids
    """

    KEY_PREFIX = "product:"
    SET_KEY = "product:all_ids"

    def __init__(self):
        self.products: Dict[str, dict] = {}

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
            logger.info("ProductMemory: Redis connected at %s", redis_url)
            self._load_from_redis()   # warm RAM from Redis on startup
        except Exception as e:
            self._redis_ok = False
            logger.warning("ProductMemory: Redis unavailable (%s) — RAM only", e)

    def update(self, obj: dict) -> None:
        """
        Store or update a product entry.
        obj must have at minimum: {id, label}
        Optional fields: {description, category, color, size, location_hint}
        """
        product_id = str(obj["id"])
        record = {
            "id":          product_id,
            "label":       obj.get("label", "unknown"),
            "description": obj.get("description", ""),
            "category":    obj.get("category", ""),
            "color":       obj.get("color", ""),
            "size":        obj.get("size", ""),
            "location_hint": obj.get("location_hint", ""),
        }
        self.products[product_id] = record

        if self._redis_ok:
            try:
                key = f"{self.KEY_PREFIX}{product_id}"
                self._redis.set(key, json.dumps(record))   # no TTL — products are permanent
                self._redis.sadd(self.SET_KEY, product_id)
            except Exception as e:
                logger.error("ProductMemory.update: Redis write failed (%s)", e)

    def get(self, product_id: str) -> Optional[dict]:
        """Get product by id."""
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
        """Find all products matching a label (case-insensitive)."""
        label_lower = label.lower()
        return [p for p in self.products.values()
                if label_lower in p.get("label", "").lower()]

    def get_all(self) -> List[dict]:
        """Return all known products as a list."""
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
        """On startup, load all products from Redis into RAM."""
        try:
            ids = self._redis.smembers(self.SET_KEY)
            for pid in ids:
                data = self._redis.get(f"{self.KEY_PREFIX}{pid}")
                if data:
                    self.products[pid] = json.loads(data)
            logger.info("ProductMemory: loaded %d products from Redis", len(self.products))
        except Exception as e:
            logger.error("ProductMemory._load_from_redis: failed (%s)", e)
