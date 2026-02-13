"""Tests for pipeline/interactions.py"""

import time

import pytest

from pipeline.interactions import InteractionDetector, PendingEvent


class FakeTrackedObject:
    """Lightweight stand-in for TrackedObject (avoids YOLO import)."""

    def __init__(self, track_id, class_name, bbox, zone, confidence=0.8):
        self.track_id = track_id
        self.class_id = 0
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.zone = zone


def _make_person(track_id=1, bbox=(100, 100, 300, 400), zone="bar_zone"):
    return FakeTrackedObject(track_id, "person", bbox, zone)


def _make_object(track_id=100, bbox=(150, 150, 250, 350), zone="bar_zone", class_name="bottle"):
    return FakeTrackedObject(track_id, class_name, bbox, zone)


class TestInteractionDetection:
    def test_overlapping_bboxes_in_bar_zone_triggers(self):
        """Two overlapping bboxes in bar zone should trigger an event."""
        det = InteractionDetector()
        person = _make_person()
        obj = _make_object()
        # Seed previous position with motion so velocity gate passes
        det.prev_positions[obj.track_id] = (0, 0)
        events = det.detect([person], [obj])
        assert len(events) == 1
        assert events[0].person_track_id == 1
        assert events[0].object_track_id == 100

    def test_same_pair_within_cooldown_no_retrigger(self):
        """Same pair within cooldown should NOT re-trigger."""
        det = InteractionDetector()
        person = _make_person()
        obj = _make_object()
        det.prev_positions[obj.track_id] = (0, 0)

        events1 = det.detect([person], [obj])
        assert len(events1) == 1

        # Immediately try again — cooldown blocks
        det.prev_positions[obj.track_id] = (0, 0)
        events2 = det.detect([person], [obj])
        assert len(events2) == 0

    def test_static_object_no_trigger(self):
        """Object with no velocity should NOT trigger."""
        det = InteractionDetector()
        person = _make_person()
        obj = _make_object()
        # Seed position at the same location — no movement
        cx = (obj.bbox[0] + obj.bbox[2]) / 2
        cy = (obj.bbox[1] + obj.bbox[3]) / 2
        det.prev_positions[obj.track_id] = (cx, cy)

        events = det.detect([person], [obj])
        assert len(events) == 0

    def test_budget_cap_prevents_triggers(self):
        """After MAX_VLM_CALLS_PER_HOUR, no more triggers."""
        det = InteractionDetector()
        det.calls_this_hour = 100  # at budget

        person = _make_person()
        obj = _make_object()
        det.prev_positions[obj.track_id] = (0, 0)

        events = det.detect([person], [obj])
        assert len(events) == 0

    def test_non_bar_zone_not_detected(self):
        """Interactions outside bar zone are ignored."""
        det = InteractionDetector()
        person = _make_person(zone="queue_zone")
        obj = _make_object(zone="queue_zone")
        det.prev_positions[obj.track_id] = (0, 0)

        events = det.detect([person], [obj])
        assert len(events) == 0


class TestIoUCalculation:
    def test_identical_boxes_iou_is_one(self):
        det = InteractionDetector()
        box = (100, 100, 200, 200)
        assert det._bbox_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap_iou_is_zero(self):
        det = InteractionDetector()
        a = (0, 0, 100, 100)
        b = (200, 200, 300, 300)
        assert det._bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        det = InteractionDetector()
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        # intersection = 50*50 = 2500
        # union = 10000 + 10000 - 2500 = 17500
        assert det._bbox_iou(a, b) == pytest.approx(2500 / 17500)

    def test_contained_box(self):
        det = InteractionDetector()
        outer = (0, 0, 200, 200)
        inner = (50, 50, 100, 100)
        # intersection = 50*50 = 2500, union = 40000 + 2500 - 2500 = 40000
        assert det._bbox_iou(outer, inner) == pytest.approx(2500 / 40000)

    def test_zero_area_box(self):
        det = InteractionDetector()
        a = (100, 100, 100, 100)  # zero area
        b = (50, 50, 150, 150)
        assert det._bbox_iou(a, b) == pytest.approx(0.0)


class TestHourlyReset:
    def test_counter_resets_after_hour(self):
        det = InteractionDetector()
        det.calls_this_hour = 50
        det.hour_start = time.time() - 3601  # over an hour ago
        det._reset_hourly_counter(time.time())
        assert det.calls_this_hour == 0
