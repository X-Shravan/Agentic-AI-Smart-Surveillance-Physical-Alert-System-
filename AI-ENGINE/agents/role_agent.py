from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Tuple
import math
import time


class RoleAgent:
    """Classifies track role by movement speed (prototype heuristic)."""

    def __init__(self, speed_threshold_px_per_sec: float = 120.0, window_size: int = 15):
        self.speed_threshold = speed_threshold_px_per_sec
        self._history: Dict[str, Deque[Tuple[float, Tuple[float, float]]]] = defaultdict(lambda: deque(maxlen=window_size))

    def classify(self, track_id: str, centroid: Tuple[float, float], timestamp: float | None = None) -> str:
        ts = timestamp or time.time()
        history = self._history[track_id]
        history.append((ts, centroid))

        speed = self._estimate_speed(history)
        return "invigilator" if speed > self.speed_threshold else "student"

    @staticmethod
    def _estimate_speed(history: Deque[Tuple[float, Tuple[float, float]]]) -> float:
        if len(history) < 2:
            return 0.0

        (t0, c0), (t1, c1) = history[0], history[-1]
        dt = max(t1 - t0, 1e-6)
        distance = math.dist(c0, c1)
        return distance / dt
