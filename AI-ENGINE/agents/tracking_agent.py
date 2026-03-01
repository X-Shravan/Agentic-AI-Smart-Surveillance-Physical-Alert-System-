from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import time


@dataclass
class Track:
    track_id: str
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    last_seen: float = field(default_factory=time.time)


class TrackingAgent:
    """Simple nearest-centroid tracking for person detections."""

    def __init__(self, max_distance: float = 90.0, timeout_seconds: float = 1.5):
        self.max_distance = max_distance
        self.timeout_seconds = timeout_seconds
        self._tracks: Dict[str, Track] = {}
        self._next_id = 1

    @staticmethod
    def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, person_detections: List[Dict], timestamp: float | None = None) -> List[Dict]:
        ts = timestamp or time.time()
        active_tracks = []

        for detection in person_detections:
            bbox = detection["bbox"]
            centroid = self._centroid(bbox)
            matched_track = self._match_track(centroid)

            if matched_track is None:
                track_id = f"ID_{self._next_id}"
                self._next_id += 1
                matched_track = Track(track_id=track_id, centroid=centroid, bbox=bbox, last_seen=ts)
                self._tracks[track_id] = matched_track
            else:
                matched_track.centroid = centroid
                matched_track.bbox = bbox
                matched_track.last_seen = ts

            active_tracks.append(
                {
                    "track_id": matched_track.track_id,
                    "bbox": bbox,
                    "centroid": centroid,
                    "confidence": detection["confidence"],
                }
            )

        self._purge_stale_tracks(ts)
        return active_tracks

    def _match_track(self, centroid: Tuple[float, float]) -> Track | None:
        best_track = None
        best_dist = self.max_distance

        for track in self._tracks.values():
            dist = math.dist(track.centroid, centroid)
            if dist <= best_dist:
                best_track = track
                best_dist = dist

        return best_track

    def _purge_stale_tracks(self, now: float) -> None:
        stale_ids = [tid for tid, t in self._tracks.items() if now - t.last_seen > self.timeout_seconds]
        for track_id in stale_ids:
            del self._tracks[track_id]
