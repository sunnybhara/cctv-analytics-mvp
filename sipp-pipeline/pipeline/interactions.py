"""
Interaction Detection + Trigger Debouncing
==========================================
Detects person-object interactions in bar zone with state-based debouncing.
"""

import time
from dataclasses import dataclass
from config.settings import (
    IOU_THRESHOLD, TRIGGER_COOLDOWN_SECONDS,
    OBJECT_VELOCITY_THRESHOLD, MAX_VLM_CALLS_PER_HOUR,
)


@dataclass
class PendingEvent:
    person_track_id: int
    object_track_id: int
    object_class: str
    zone: str
    timestamp: float
    person_bbox: tuple
    object_bbox: tuple


class InteractionDetector:
    def __init__(self):
        self.active_interactions: dict[str, float] = {}  # pair_id -> last_trigger_time
        self.calls_this_hour: int = 0
        self.hour_start: float = time.time()
        self.prev_positions: dict[int, tuple] = {}  # track_id -> (cx, cy)

    def detect(self, persons: list, objects: list) -> list[PendingEvent]:
        """Find person-object interactions in bar zone. Apply debouncing.

        Returns list of PendingEvent for VLM verification.
        """
        now = time.time()
        self._reset_hourly_counter(now)
        events = []

        bar_persons = [p for p in persons if p.zone and "bar" in p.zone]
        bar_objects = [o for o in objects if o.zone and "bar" in o.zone]

        for person in bar_persons:
            for obj in bar_objects:
                if self._bbox_iou(person.bbox, obj.bbox) < IOU_THRESHOLD:
                    continue
                if not self._should_trigger(person.track_id, obj.track_id, obj, now):
                    continue
                events.append(
                    PendingEvent(
                        person_track_id=person.track_id,
                        object_track_id=obj.track_id,
                        object_class=obj.class_name,
                        zone=person.zone,
                        timestamp=now,
                        person_bbox=person.bbox,
                        object_bbox=obj.bbox,
                    )
                )

        # Update position history for velocity calculation
        for obj in objects:
            cx = (obj.bbox[0] + obj.bbox[2]) / 2
            cy = (obj.bbox[1] + obj.bbox[3]) / 2
            self.prev_positions[obj.track_id] = (cx, cy)

        return events

    def _should_trigger(self, person_id: int, obj_id: int, obj, now: float) -> bool:
        """State-based debouncing. Only trigger on rising edge or state change."""
        if self.calls_this_hour >= MAX_VLM_CALLS_PER_HOUR:
            return False

        pair_id = f"{person_id}_{obj_id}"
        last_time = self.active_interactions.get(pair_id, 0)
        if now - last_time < TRIGGER_COOLDOWN_SECONDS:
            return False

        # Motion gate: only trigger if object is moving
        prev = self.prev_positions.get(obj.track_id)
        if prev:
            cx = (obj.bbox[0] + obj.bbox[2]) / 2
            cy = (obj.bbox[1] + obj.bbox[3]) / 2
            velocity = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
            if velocity < OBJECT_VELOCITY_THRESHOLD:
                return False

        self.active_interactions[pair_id] = now
        self.calls_this_hour += 1
        return True

    def _bbox_iou(self, a: tuple, b: tuple) -> float:
        """Calculate IoU between two bounding boxes (x1, y1, x2, y2)."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _reset_hourly_counter(self, now: float):
        if now - self.hour_start >= 3600:
            self.calls_this_hour = 0
            self.hour_start = now
