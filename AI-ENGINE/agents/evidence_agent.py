from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict

import cv2

from utils.video_buffer import BufferedFrame


class EvidenceAgent:
    """Stores screenshot and buffered clip on escalation."""

    def __init__(self, screenshot_dir: Path, clip_dir: Path, fps: int = 20):
        self.screenshot_dir = screenshot_dir
        self.clip_dir = clip_dir
        self.fps = fps

    def save(self, frame, buffer_frames: List[BufferedFrame], track_id: str, risk_score: int) -> Dict[str, str]:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        prefix = f"{track_id}_risk{risk_score}_{timestamp}"

        screenshot_path = self.screenshot_dir / f"{prefix}.jpg"
        cv2.imwrite(str(screenshot_path), frame)

        clip_path = self.clip_dir / f"{prefix}.mp4"
        if buffer_frames:
            height, width = buffer_frames[0].frame.shape[:2]
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            for buffered in buffer_frames:
                writer.write(buffered.frame)
            writer.release()

        return {
            "track_id": track_id,
            "risk_score": str(risk_score),
            "timestamp": timestamp,
            "screenshot": str(screenshot_path),
            "clip": str(clip_path),
        }
