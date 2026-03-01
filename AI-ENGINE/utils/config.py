from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
CLIP_DIR = OUTPUT_DIR / "clips"
MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"

CAMERA_INDEX = 0
MIN_CONFIDENCE = 0.5
RISK_CONFIDENCE_THRESHOLD = 0.7
RISK_EVENT_PERSISTENCE_SECONDS = 2.0
RISK_ESCALATION_THRESHOLD = 75
BUFFER_SECONDS = 10
ALERT_ENDPOINT = "http://localhost:8080/api/alerts"

for path in (OUTPUT_DIR, SCREENSHOT_DIR, CLIP_DIR):
    path.mkdir(parents=True, exist_ok=True)
