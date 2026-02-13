"""
Event Deduplicator
==================
Sliding-window deduplication for verified events.
"""

import time
from collections import defaultdict

from config.settings import DEDUP_WINDOW_SECONDS


class EventDeduplicator:
    def __init__(self, window: float = DEDUP_WINDOW_SECONDS):
        self.window = window
        self.recent: dict[str, list[dict]] = defaultdict(list)

    def should_emit(self, zone_id: str, action: str, confidence: float) -> bool:
        """Returns True if this event should be emitted (not a duplicate).

        Within the sliding window, keeps the highest-confidence version.
        """
        now = time.time()
        key = f"{zone_id}:{action}"

        # Purge expired
        self.recent[key] = [
            e for e in self.recent[key] if now - e["ts"] < self.window
        ]

        if self.recent[key]:
            # Duplicate within window — update confidence if higher
            if confidence > self.recent[key][-1]["conf"]:
                self.recent[key][-1] = {"ts": now, "conf": confidence}
            return False

        self.recent[key].append({"ts": now, "conf": confidence})
        return True
