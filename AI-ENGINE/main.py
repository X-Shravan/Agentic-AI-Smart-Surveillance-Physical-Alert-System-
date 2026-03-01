from __future__ import annotations

import time
from collections import defaultdict

import cv2

from agents.alert_agent import AlertAgent
from agents.decision_agent import DecisionAgent
from agents.evidence_agent import EvidenceAgent
from agents.role_agent import RoleAgent
from agents.surveillance_agent import SurveillanceAgent
from agents.tracking_agent import TrackingAgent
from utils.config import (
    ALERT_ENDPOINT,
    BUFFER_SECONDS,
    CAMERA_INDEX,
    CLIP_DIR,
    MIN_CONFIDENCE,
    MODEL_PATH,
    RISK_CONFIDENCE_THRESHOLD,
    RISK_ESCALATION_THRESHOLD,
    RISK_EVENT_PERSISTENCE_SECONDS,
    SCREENSHOT_DIR,
)
from utils.video_buffer import VideoBuffer


def bbox_overlap_ratio(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = max((ax2 - ax1) * (ay2 - ay1), 1)
    return inter_area / a_area


def run() -> None:
    surveillance = SurveillanceAgent(str(MODEL_PATH), confidence_threshold=MIN_CONFIDENCE)
    tracker = TrackingAgent()
    role_agent = RoleAgent()
    decision_agent = DecisionAgent(
        confidence_threshold=RISK_CONFIDENCE_THRESHOLD,
        persistence_seconds=RISK_EVENT_PERSISTENCE_SECONDS,
        escalation_threshold=RISK_ESCALATION_THRESHOLD,
    )
    evidence_agent = EvidenceAgent(SCREENSHOT_DIR, CLIP_DIR)
    alert_agent = AlertAgent(ALERT_ENDPOINT)
    video_buffer = VideoBuffer(seconds=BUFFER_SECONDS)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam. Check camera permissions and CAMERA_INDEX.")

    last_alert_time = defaultdict(float)
    print("[INFO] Surveillance loop started. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        video_buffer.append(frame, now)

        detections = surveillance.detect(frame)
        people = [d for d in detections if d["class"] == "person"]
        phones = [d for d in detections if d["class"] == "cell phone"]

        tracked_people = tracker.update(people, now)
        decision_agent.decay_events(now)

        for person in tracked_people:
            track_id = person["track_id"]
            role = role_agent.classify(track_id, person["centroid"], now)

            if role != "student":
                continue

            # Event 1: mobile detected near student body.
            mobile_conf = 0.0
            for phone in phones:
                overlap = bbox_overlap_ratio(person["bbox"], phone["bbox"])
                if overlap > 0.03:
                    mobile_conf = max(mobile_conf, phone["confidence"])
            if mobile_conf > 0:
                decision_agent.update_event(track_id, "mobile_detected", mobile_conf, now)

            # Prototype placeholders for behavior events. Replace with dedicated CV estimators.
            # They are left at low confidence by default to avoid accidental escalation.
            downward_head_conf = 0.0
            talking_conf = 0.0
            if downward_head_conf > 0:
                decision_agent.update_event(track_id, "downward_head", downward_head_conf, now)
            if talking_conf > 0:
                decision_agent.update_event(track_id, "talking", talking_conf, now)

            decision = decision_agent.evaluate(track_id, now)
            if decision["escalate"] and now - last_alert_time[track_id] > 8:
                evidence = evidence_agent.save(frame, video_buffer.snapshot(), track_id, decision["risk_score"])
                payload = {
                    "track_id": track_id,
                    "risk_score": decision["risk_score"],
                    "event": decision["event"] or "unknown",
                }
                response = alert_agent.send(payload)
                print(f"[ALERT] {payload} | evidence={evidence['screenshot']} | backend={response}")
                last_alert_time[track_id] = now

            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 120), 2)
            cv2.putText(
                frame,
                f"{track_id} {role}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Agentic AI Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
