"""
Clip Buffer
===========
Ring buffer of recent frames for extracting 5-second clips on interaction detection.
"""

import os
import time
from collections import deque

import cv2

from config.settings import CLIP_DURATION_SECONDS, CLIP_SAVE_DIR


class ClipBuffer:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        max_frames = int(fps * CLIP_DURATION_SECONDS * 2)  # keep 2x clip length
        self.buffer: deque = deque(maxlen=max_frames)
        self.timestamps: deque = deque(maxlen=max_frames)

    def add_frame(self, frame, timestamp: float = None):
        self.buffer.append(frame.copy())
        self.timestamps.append(timestamp or time.time())

    def extract_clip_frames(self, event_time: float, n_frames: int = 4) -> list:
        """Extract n evenly-spaced frames from a window centered on event_time.

        Returns list of numpy frames for VLM verification.
        """
        half_window = CLIP_DURATION_SECONDS / 2
        start = event_time - half_window
        end = event_time + half_window

        window_frames = [
            (ts, frame)
            for ts, frame in zip(self.timestamps, self.buffer)
            if start <= ts <= end
        ]

        if not window_frames:
            return []

        step = max(1, len(window_frames) // n_frames)
        return [window_frames[i][1] for i in range(0, len(window_frames), step)][
            :n_frames
        ]

    def save_clip(self, event_time: float, event_id: str) -> str | None:
        """Save full clip to disk for training data. Returns file path or None."""
        half_window = CLIP_DURATION_SECONDS / 2
        start = event_time - half_window
        end = event_time + half_window

        window_frames = [
            frame
            for ts, frame in zip(self.timestamps, self.buffer)
            if start <= ts <= end
        ]

        if not window_frames:
            return None

        os.makedirs(CLIP_SAVE_DIR, exist_ok=True)
        path = os.path.join(CLIP_SAVE_DIR, f"{event_id}.mp4")

        h, w = window_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        for frame in window_frames:
            writer.write(frame)
        writer.release()
        return path
