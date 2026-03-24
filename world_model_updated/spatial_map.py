import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import redis

logger = logging.getLogger(__name__)


class SpatialMap:
    """
    Stores 3D positions of all tracked objects.

    Positions stored as plain Python lists (JSON serialisable),
    not numpy arrays — so they can be sent over Redis and REST APIs.

    Redis key format:  spatial:{obj_id}
    TTL: 300s (positions expire if object not re-detected)
    """

    KEY_PREFIX = "spatial:"
    DEFAULT_TTL = 300

    def __init__(self):
        self.positions: Dict[str, list] = {}   # obj_id → [x, y, z]

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
            logger.info("SpatialMap: Redis connected at %s", redis_url)
        except Exception as e:
            self._redis_ok = False
            logger.warning("SpatialMap: Redis unavailable (%s) — RAM only", e)

    def update(self, obj: dict) -> None:
        """
        Update position for an object.
        Stores as list, not numpy array — JSON serialisable.
        """
        obj_id = f"{obj['label']}_{obj['id']}"
        position = obj.get("position")

        if position is None:
            return

        # Convert to plain list regardless of input type
        if isinstance(position, np.ndarray):
            pos_list = position.tolist()
        else:
            pos_list = list(position)

        self.positions[obj_id] = pos_list

        if self._redis_ok:
            try:
                self._redis.setex(
                    f"{self.KEY_PREFIX}{obj_id}",
                    self.DEFAULT_TTL,
                    json.dumps(pos_list)
                )
            except Exception as e:
                logger.error("SpatialMap.update: Redis write failed (%s)", e)

    def get_position(self, obj_id: str) -> Optional[list]:
        """Get position as [x, y, z] list. Returns None if unknown."""
        if obj_id in self.positions:
            return self.positions[obj_id]

        if self._redis_ok:
            try:
                data = self._redis.get(f"{self.KEY_PREFIX}{obj_id}")
                if data:
                    pos = json.loads(data)
                    self.positions[obj_id] = pos
                    return pos
            except Exception as e:
                logger.error("SpatialMap.get_position: Redis read failed (%s)", e)

        return None

    def get_position_np(self, obj_id: str) -> Optional[np.ndarray]:
        """Get position as numpy array. Useful for internal distance calculations."""
        pos = self.get_position(obj_id)
        return np.array(pos) if pos is not None else None

    def get_all(self) -> Dict[str, list]:
        """
        Return all positions as {obj_id: [x,y,z]} dict.
        Values are plain lists — JSON serialisable.
        """
        return dict(self.positions)

    def get_nearest(self, position: list, n: int = 3) -> List[dict]:
        """
        Return n nearest objects to a given [x,y,z] position.
        Returns list of {obj_id, distance, position} sorted by distance.
        """
        query = np.array(position[:3])
        results = []
        for obj_id, pos in self.positions.items():
            dist = float(np.linalg.norm(np.array(pos[:3]) - query))
            results.append({"obj_id": obj_id, "distance": round(dist, 3), "position": pos})
        results.sort(key=lambda x: x["distance"])
        return results[:n]

    def remove(self, obj_id: str) -> None:
        self.positions.pop(obj_id, None)
        if self._redis_ok:
            try:
                self._redis.delete(f"{self.KEY_PREFIX}{obj_id}")
            except Exception as e:
                logger.error("SpatialMap.remove: Redis delete failed (%s)", e)
