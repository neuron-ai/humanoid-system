import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("TemporalMemory: redis not installed — RAM only")


class TemporalMemory:
    """
    Tracks position history and velocity of every object over time.

    RAM: deque(maxlen=100) per object — O(1) append/trim.

    Redis: Redis Streams (XADD/XRANGE), capped at MAX_STREAM_LEN.

    Changes vs original:
    - redis import guarded — won't crash if redis not installed
    - REDIS_WRITE_INTERVAL added — perception runs at 20Hz but writing
      every frame to Redis Streams wastes CPU and memory bandwidth.
      Default: write to Redis at most once per second per object.
      RAM deque still updates every frame — velocity stays accurate.
    """

    MAX_HISTORY          = 100
    MAX_STREAM_LEN       = 500
    RELOAD_ON_START      = 20
    REDIS_WRITE_INTERVAL = 1.0   # seconds between Redis writes per object

    def __init__(self):
        self.history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.MAX_HISTORY)
        )
        # Track last Redis write time per object to rate-limit stream writes
        self._last_redis_write: Dict[str, float] = {}

        self._redis_ok = False
        self._redis    = None

        if not REDIS_AVAILABLE:
            return

        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        try:
            self._redis = _redis_lib.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            self._redis_ok = True
            logger.info("TemporalMemory: Redis connected at %s", redis_url)
        except Exception as e:
            self._redis_ok = False
            logger.warning("TemporalMemory: Redis unavailable (%s) — RAM only", e)

    def update(self, obj: dict) -> None:
        """
        Record a new observation. Computes velocity from previous position.
        RAM updated every call. Redis written at most once per second per object.
        """
        obj_id    = f"{obj['label']}_{obj['id']}"
        position  = obj.get("position", [0, 0, 0])
        timestamp = obj.get("timestamp", time.time())

        # Velocity from previous entry
        velocity = [0.0, 0.0, 0.0]
        if self.history[obj_id]:
            prev = self.history[obj_id][-1]
            dt   = timestamp - prev["timestamp"]
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

        # RAM — always updated (needed for accurate velocity)
        self.history[obj_id].append(entry)

        # Redis — rate-limited to avoid 20Hz stream writes
        if self._redis_ok:
            now       = time.time()
            last_write = self._last_redis_write.get(obj_id, 0.0)
            if now - last_write >= self.REDIS_WRITE_INTERVAL:
                try:
                    self._redis.xadd(
                        f"temporal:{obj_id}",
                        {
                            "position":  json.dumps(entry["position"]),
                            "velocity":  json.dumps(entry["velocity"]),
                            "timestamp": str(entry["timestamp"]),
                        },
                        maxlen=self.MAX_STREAM_LEN,
                        approximate=True,
                    )
                    self._last_redis_write[obj_id] = now
                except Exception as e:
                    logger.error("TemporalMemory.update: Redis XADD failed (%s)", e)

    def get_history(self, obj_id: str, n: Optional[int] = None) -> List[dict]:
        ram = list(self.history.get(obj_id, []))
        if ram:
            return ram[-n:] if n else ram
        if self._redis_ok:
            return self._load_from_stream(obj_id, count=n or self.RELOAD_ON_START)
        return []

    def get_velocity(self, obj_id: str) -> Optional[List[float]]:
        hist = self.history.get(obj_id)
        if hist:
            return hist[-1].get("velocity", [0.0, 0.0, 0.0])
        return None

    def get_all_ids(self) -> List[str]:
        return list(self.history.keys())

    def _load_from_stream(self, obj_id: str, count: int = 20) -> List[dict]:
        try:
            entries = self._redis.xrevrange(f"temporal:{obj_id}", count=count)
            result  = []
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
