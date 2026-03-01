from __future__ import annotations

from typing import Dict, List

from ultralytics import YOLO


class SurveillanceAgent:
    """Detects people and mobile phones from webcam frames."""

    TARGET_CLASSES = {"person", "cell phone"}

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame) -> List[Dict]:
        results = self.model.predict(source=frame, verbose=False, conf=self.confidence_threshold)
        detections: List[Dict] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                class_name = names[int(box.cls[0])]
                confidence = float(box.conf[0])
                if class_name not in self.TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        return detections
