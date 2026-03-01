from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List
import time
import numpy as np


@dataclass
class BufferedFrame:
    timestamp: float
    frame: np.ndarray


class VideoBuffer:
    """Keeps a rolling time window of frames for evidence export."""

    def __init__(self, seconds: float = 10.0):
        self.seconds = seconds
        self._frames: Deque[BufferedFrame] = deque()

    def append(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        self._frames.append(BufferedFrame(timestamp=ts, frame=frame.copy()))
        self._trim(ts)

    def _trim(self, current_ts: float) -> None:
        while self._frames and current_ts - self._frames[0].timestamp > self.seconds:
            self._frames.popleft()

    def snapshot(self) -> List[BufferedFrame]:
        return list(self._frames)
