import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class TemporalMemory:
    """
    Tracks position history of every object over time.

    RAM: uses collections.deque(maxlen=100) — O(1) append and trim.
         Original used list.pop(0) which is O(n) — slow for large histories.

    Redis: uses Redis Streams (XADD/XRANGE) for persistent time-series.
           Each object has its own stream: temporal:{obj_id}
           Stream is capped at MAX_STREAM_LEN entries automatically.

    On restart: recent history is reloaded from Redis Streams into RAM.
    """

    MAX_HISTORY = 100          # RAM entries per object
    MAX_STREAM_LEN = 500       # Redis stream entries per object
    RELOAD_ON_START = 20       # how many entries to reload from Redis on startup

    def __init__(self):
        # RAM: obj_id → deque of {position, timestamp, velocity}
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
            logger.info("TemporalMemory: Redis connected at %s", redis_url)
        except Exception as e:
            self._redis_ok = False
            logger.warning("TemporalMemory: Redis unavailable (%s) — RAM only", e)

    def update(self, obj: dict) -> None:
        """
        Record a new observation for an object.
        Computes velocity if previous position is available.
        """
        obj_id = f"{obj['label']}_{obj['id']}"
        position = obj.get("position", [0, 0, 0])
        timestamp = obj.get("timestamp", time.time())

        # Compute velocity from previous entry
        velocity = [0.0, 0.0, 0.0]
        if self.history[obj_id]:
            prev = self.history[obj_id][-1]
            dt = timestamp - prev["timestamp"]
            if dt > 0:
                prev_pos = prev["position"]
                velocity = [
                    round((position[i] - prev_pos[i]) / dt, 4)
                    for i in range(min(3, len(position), len(prev_pos)))
                ]

        entry = {
            "position":  list(position),
            "velocity":  velocity,
            "timestamp": timestamp,
        }

        # RAM — O(1) thanks to deque(maxlen=...)
        self.history[obj_id].append(entry)

        # Redis Stream — persistent, capped
        if self._redis_ok:
            try:
                stream_key = f"temporal:{obj_id}"
                self._redis.xadd(
                    stream_key,
                    {
                        "position":  json.dumps(entry["position"]),
                        "velocity":  json.dumps(entry["velocity"]),
                        "timestamp": str(entry["timestamp"]),
                    },
                    maxlen=self.MAX_STREAM_LEN,
                    approximate=True,
                )
            except Exception as e:
                logger.error("TemporalMemory.update: Redis XADD failed (%s)", e)

    def get_history(self, obj_id: str, n: Optional[int] = None) -> List[dict]:
        """
        Return position history for an object.
        If n is given, return last n entries only.
        Falls back to Redis Stream if RAM cache is empty.
        """
        ram = list(self.history.get(obj_id, []))
        if ram:
            return ram[-n:] if n else ram

        # Try Redis Stream
        if self._redis_ok:
            return self._load_from_stream(obj_id, count=n or self.RELOAD_ON_START)

        return []

    def get_velocity(self, obj_id: str) -> Optional[List[float]]:
        """Return the most recent velocity estimate for an object."""
        hist = self.history.get(obj_id)
        if hist:
            return hist[-1].get("velocity", [0.0, 0.0, 0.0])
        return None

    def get_all_ids(self) -> List[str]:
        """Return list of all tracked object IDs."""
        return list(self.history.keys())

    def _load_from_stream(self, obj_id: str, count: int = 20) -> List[dict]:
        """Load recent entries from Redis Stream into RAM."""
        try:
            stream_key = f"temporal:{obj_id}"
            entries = self._redis.xrevrange(stream_key, count=count)
            result = []
            for _, fields in reversed(entries):
                try:
                    entry = {
                        "position":  json.loads(fields["position"]),
                        "velocity":  json.loads(fields["velocity"]),
                        "timestamp": float(fields["timestamp"]),
                    }
                    result.append(entry)
                    self.history[obj_id].append(entry)
                except Exception:
                    continue
            return result
        except Exception as e:
            logger.error("TemporalMemory._load_from_stream: failed (%s)", e)
            return []
