from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
import time


EVENT_SCORES = {
    "mobile_detected": 60,
    "downward_head": 20,
    "talking": 15,
}


@dataclass
class EventState:
    start_time: float
    last_seen: float
    confidence: float


class DecisionAgent:
    """Risk-based reasoning with persistence and confidence gating."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        persistence_seconds: float = 2.0,
        escalation_threshold: int = 75,
    ):
        self.confidence_threshold = confidence_threshold
        self.persistence_seconds = persistence_seconds
        self.escalation_threshold = escalation_threshold
        self._events: Dict[str, Dict[str, EventState]] = defaultdict(dict)

    def update_event(self, track_id: str, event_name: str, confidence: float, timestamp: float | None = None) -> None:
        ts = timestamp or time.time()
        track_events = self._events[track_id]

        if event_name not in track_events:
            track_events[event_name] = EventState(start_time=ts, last_seen=ts, confidence=confidence)
        else:
            state = track_events[event_name]
            state.last_seen = ts
            state.confidence = max(state.confidence, confidence)

    def decay_events(self, now: float | None = None, stale_after: float = 1.0) -> None:
        ts = now or time.time()
        for track_id in list(self._events.keys()):
            for event_name in list(self._events[track_id].keys()):
                if ts - self._events[track_id][event_name].last_seen > stale_after:
                    del self._events[track_id][event_name]
            if not self._events[track_id]:
                del self._events[track_id]

    def evaluate(self, track_id: str, timestamp: float | None = None) -> Dict:
        ts = timestamp or time.time()
        risk_score = 0
        top_event = None

        for event_name, state in self._events.get(track_id, {}).items():
            persisted_for = ts - state.start_time
            if persisted_for < self.persistence_seconds:
                continue
            if state.confidence < self.confidence_threshold:
                continue

            risk_score += EVENT_SCORES.get(event_name, 0)
            if top_event is None or EVENT_SCORES[event_name] > EVENT_SCORES.get(top_event, 0):
                top_event = event_name

        return {
            "track_id": track_id,
            "risk_score": risk_score,
            "event": top_event,
            "escalate": risk_score >= self.escalation_threshold,
        }
