from __future__ import annotations

from typing import Dict
import requests


class AlertAgent:
    """Dispatches escalation payloads to Spring Boot backend."""

    def __init__(self, endpoint: str, timeout_seconds: int = 3):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def send(self, payload: Dict) -> Dict:
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            return {"success": True, "status_code": response.status_code, "body": response.text}
        except requests.RequestException as exc:
            return {"success": False, "error": str(exc)}
