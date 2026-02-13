"""
Dual-Tracker Detector
=====================
YOLO11s with two tracker instances: BoT-SORT for persons, ByteTrack for objects.
"""

from dataclasses import dataclass
from ultralytics import YOLO
from config.settings import (
    YOLO_MODEL, YOLO_CONF_THRESHOLD, PERSON_CLASS_ID,
    DRINK_CLASS_IDS, DRINK_CLASS_NAMES, PERSON_TRACKER, OBJECT_TRACKER,
)
from pipeline.zones import get_zone_for_bbox


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    zone: str | None = None


class DualTracker:
    """Two YOLO instances for independent tracker state.

    Trade-off: 2x inference cost, but BoT-SORT appearance features for persons
    are critical for handling bar counter occlusion. Net throughput ~30 FPS on A100.
    """

    def __init__(self, model_path: str = YOLO_MODEL):
        self.person_model = YOLO(model_path)
        self.object_model = YOLO(model_path)

    def process_frame(
        self, frame, zones: dict
    ) -> tuple[list[TrackedObject], list[TrackedObject]]:
        """Run detection + tracking on a single frame.

        Returns: (persons, objects) with persistent track_ids and zone assignments.
        """
        person_results = self.person_model.track(
            source=frame,
            tracker=PERSON_TRACKER,
            classes=[PERSON_CLASS_ID],
            conf=YOLO_CONF_THRESHOLD,
            persist=True,
            verbose=False,
        )

        object_results = self.object_model.track(
            source=frame,
            tracker=OBJECT_TRACKER,
            classes=DRINK_CLASS_IDS,
            conf=YOLO_CONF_THRESHOLD,
            persist=True,
            verbose=False,
        )

        persons = self._extract_tracks(person_results, zones)
        objects = self._extract_tracks(object_results, zones)
        return persons, objects

    def _extract_tracks(self, results, zones: dict) -> list[TrackedObject]:
        """Convert YOLO results to TrackedObject list with zone assignment."""
        tracks = []
        if not results or not results[0].boxes:
            return tracks
        for box in results[0].boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            tracks.append(
                TrackedObject(
                    track_id=int(box.id[0]),
                    class_id=class_id,
                    class_name=DRINK_CLASS_NAMES.get(
                        class_id, "person" if class_id == 0 else "unknown"
                    ),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    zone=get_zone_for_bbox(x1, y1, x2, y2, zones),
                )
            )
        return tracks
