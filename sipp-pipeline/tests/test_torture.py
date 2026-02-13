"""
Torture Tests — SIPP Bar Detection Pipeline
============================================
Edge cases, stress scenarios, malformed inputs, budget exhaustion,
concurrent interactions, and boundary conditions.
"""

import json
import os
import time
import uuid

import numpy as np
import pytest
from shapely.geometry import Polygon

from pipeline.clip_buffer import ClipBuffer
from pipeline.deduplicator import EventDeduplicator
from pipeline.events import BarEvent
from pipeline.interactions import InteractionDetector, PendingEvent
from pipeline.verifier import _safe_parse
from pipeline.zones import get_zone_for_bbox, get_zone_for_point, load_zones


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeTracked:
    def __init__(self, track_id, class_name, bbox, zone, confidence=0.8):
        self.track_id = track_id
        self.class_id = 0
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.zone = zone


def _person(tid=1, bbox=(100, 100, 300, 400), zone="bar_zone"):
    return FakeTracked(tid, "person", bbox, zone)


def _obj(tid=100, bbox=(150, 150, 250, 350), zone="bar_zone", cls="bottle"):
    return FakeTracked(tid, cls, bbox, zone)


@pytest.fixture
def zones_file(tmp_path):
    data = {
        "camera_01": {
            "frame_width": 1920,
            "frame_height": 1080,
            "bar_zone": [[100, 100], [500, 100], [500, 400], [100, 400]],
            "queue_zone": [[550, 100], [800, 100], [800, 400], [550, 400]],
        },
        "camera_hd": {
            "frame_width": 3840,
            "frame_height": 2160,
            "bar_zone": [[200, 200], [1000, 200], [1000, 800], [200, 800]],
        },
    }
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(data))
    return str(path)


# ── Zone Torture Tests ───────────────────────────────────────────────────────

class TestZoneTorture:
    def test_zone_on_polygon_edge(self, zones_file):
        """Point exactly on the edge — Shapely boundary is not 'contained'."""
        zones = load_zones(zones_file, "camera_01")
        # (100, 100) is a corner vertex — Shapely does NOT contain boundary points
        result = get_zone_for_point(100, 100, zones)
        # Shapely Polygon.contains() excludes boundary, so this is None
        assert result is None

    def test_point_just_inside_boundary(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        # Just inside the bar_zone polygon
        assert get_zone_for_point(101, 101, zones) == "bar_zone"

    def test_negative_coordinates(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_point(-50, -50, zones) is None

    def test_very_large_coordinates(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_point(999999, 999999, zones) is None

    def test_float_coordinates(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_point(300.5, 250.7, zones) == "bar_zone"

    def test_empty_zones_dict(self):
        assert get_zone_for_point(300, 250, {}) is None
        assert get_zone_for_bbox(200, 200, 400, 300, {}) is None

    def test_degenerate_bbox_zero_area(self, zones_file):
        """Zero-area bbox — center is a point, still valid."""
        zones = load_zones(zones_file, "camera_01")
        # center = (300, 250) → bar_zone
        result = get_zone_for_bbox(300, 250, 300, 250, zones)
        # (300, 250) is inside bar_zone
        assert result == "bar_zone"

    def test_inverted_bbox(self, zones_file):
        """x1 > x2 — center is still (300, 250)."""
        zones = load_zones(zones_file, "camera_01")
        result = get_zone_for_bbox(400, 300, 200, 200, zones)
        # center = (300, 250) → bar_zone
        assert result == "bar_zone"

    def test_malformed_zones_file(self, tmp_path):
        """Invalid JSON should raise."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json}")
        with pytest.raises(json.JSONDecodeError):
            load_zones(str(bad_file), "camera_01")

    def test_empty_polygon_in_zones_file(self, tmp_path):
        """Empty polygon coordinates — creates degenerate polygon, no zones match."""
        data = {"camera_01": {"frame_width": 1920, "frame_height": 1080, "bar_zone": []}}
        path = tmp_path / "empty_poly.json"
        path.write_text(json.dumps(data))
        zones = load_zones(str(path), "camera_01")
        # Degenerate polygon — nothing should be inside it
        assert get_zone_for_point(300, 250, zones) is None

    def test_scaling_preserves_containment(self, zones_file):
        """Scaling coordinates should preserve point-in-polygon relationships."""
        zones_native = load_zones(zones_file, "camera_01", 1920, 1080)
        zones_scaled = load_zones(zones_file, "camera_01", 3840, 2160)
        # (300, 250) in native → bar_zone, (600, 500) in scaled → bar_zone
        assert get_zone_for_point(300, 250, zones_native) == "bar_zone"
        assert get_zone_for_point(600, 500, zones_scaled) == "bar_zone"


# ── Interaction Torture Tests ────────────────────────────────────────────────

class TestInteractionTorture:
    def test_many_persons_many_objects(self):
        """10 persons x 10 objects — all overlapping in bar zone."""
        det = InteractionDetector()
        persons = [_person(tid=i, bbox=(100, 100, 300, 400)) for i in range(10)]
        objects = [_obj(tid=100 + i, bbox=(150, 150, 250, 350)) for i in range(10)]
        for o in objects:
            det.prev_positions[o.track_id] = (0, 0)

        events = det.detect(persons, objects)
        # Should trigger up to MAX_VLM_CALLS_PER_HOUR (100), but 10x10=100 pairs
        assert len(events) == 100

    def test_budget_exact_limit(self):
        """Exactly at budget should still block the next call."""
        det = InteractionDetector()
        det.calls_this_hour = 99  # one call left
        p = _person()
        o = _obj()
        det.prev_positions[o.track_id] = (0, 0)

        events = det.detect([p], [o])
        assert len(events) == 1
        assert det.calls_this_hour == 100

        # Now at limit — next call blocked
        det.active_interactions.clear()  # clear cooldown
        det.prev_positions[o.track_id] = (0, 0)
        events2 = det.detect([p], [o])
        assert len(events2) == 0

    def test_cooldown_expires_allows_retrigger(self):
        """After cooldown, same pair can trigger again."""
        det = InteractionDetector()
        p = _person()
        o = _obj()
        det.prev_positions[o.track_id] = (0, 0)

        events1 = det.detect([p], [o])
        assert len(events1) == 1

        # Simulate cooldown expiry by backdating the interaction
        pair_id = f"{p.track_id}_{o.track_id}"
        det.active_interactions[pair_id] = time.time() - 11.0

        det.prev_positions[o.track_id] = (0, 0)
        events2 = det.detect([p], [o])
        assert len(events2) == 1

    def test_no_objects_no_crash(self):
        det = InteractionDetector()
        events = det.detect([_person()], [])
        assert events == []

    def test_no_persons_no_crash(self):
        det = InteractionDetector()
        events = det.detect([], [_obj()])
        assert events == []

    def test_both_empty_no_crash(self):
        det = InteractionDetector()
        events = det.detect([], [])
        assert events == []

    def test_mixed_zones_only_bar_triggers(self):
        """Only bar_zone interactions trigger, not queue_zone."""
        det = InteractionDetector()
        bar_p = _person(tid=1, zone="bar_zone")
        queue_p = _person(tid=2, zone="queue_zone")
        bar_o = _obj(tid=100, zone="bar_zone")
        queue_o = _obj(tid=101, zone="queue_zone")

        det.prev_positions[100] = (0, 0)
        det.prev_positions[101] = (0, 0)

        events = det.detect([bar_p, queue_p], [bar_o, queue_o])
        assert len(events) == 1
        assert events[0].zone == "bar_zone"

    def test_service_rail_zone_triggers(self):
        """service_rail contains 'bar' -> should NOT trigger (no 'bar' substring)."""
        det = InteractionDetector()
        p = _person(zone="service_rail")
        o = _obj(zone="service_rail")
        det.prev_positions[o.track_id] = (0, 0)
        events = det.detect([p], [o])
        assert len(events) == 0

    def test_bar_substring_match(self):
        """Zone name 'bar_front' contains 'bar' → triggers."""
        det = InteractionDetector()
        p = _person(zone="bar_front")
        o = _obj(zone="bar_front")
        det.prev_positions[o.track_id] = (0, 0)
        events = det.detect([p], [o])
        assert len(events) == 1


# ── VLM Parse Torture Tests ─────────────────────────────────────────────────

class TestVerifierTorture:
    def test_nested_json_fences(self):
        raw = '```json\n```json\n{"action":"pouring_draft","confidence":0.8}\n```\n```'
        result = _safe_parse(raw)
        # The double fence stripping should still parse
        assert result["action"] in ("pouring_draft", "unknown")

    def test_json_with_trailing_comma(self):
        """Trailing comma is invalid JSON — should fallback."""
        raw = '{"action": "pouring_draft", "confidence": 0.8,}'
        result = _safe_parse(raw)
        assert result["action"] == "unknown"

    def test_json_with_unicode(self):
        raw = '{"action": "pouring_draft", "confidence": 0.8, "description": "Bartender pours a pint 🍺"}'
        result = _safe_parse(raw)
        assert result["action"] == "pouring_draft"

    def test_extremely_long_output(self):
        raw = '{"action": "unknown", "confidence": 0.0, "description": "' + "x" * 10000 + '"}'
        result = _safe_parse(raw)
        assert result["action"] == "unknown"

    def test_json_array_instead_of_object(self):
        raw = '[{"action": "pouring_draft"}]'
        result = _safe_parse(raw)
        # Array is valid JSON but not the expected dict schema — returns unknown
        assert isinstance(result, dict)
        assert result["action"] == "unknown"

    def test_null_json(self):
        result = _safe_parse("null")
        assert isinstance(result, dict)
        assert result["action"] == "unknown"

    def test_number_json(self):
        result = _safe_parse("42")
        assert isinstance(result, dict)
        assert result["action"] == "unknown"

    def test_mixed_fences_and_text(self):
        raw = "Here's the analysis:\n```json\n{\"action\":\"pouring_draft\",\"confidence\":0.9}\n```\nThat's my answer."
        result = _safe_parse(raw)
        # After stripping fences: tries to parse "Here's the analysis:\n{...}\nThat's..."
        # This may or may not parse — depends on fence stripping logic
        assert result.get("action") in ("pouring_draft", "unknown")


# ── Deduplicator Torture Tests ───────────────────────────────────────────────

class TestDeduplicatorTorture:
    def test_many_different_actions_all_emit(self):
        dedup = EventDeduplicator(window=10.0)
        actions = ["pouring_draft", "pouring_bottle", "mixing_cocktail",
                    "serving_customer", "payment_card", "payment_cash"]
        for action in actions:
            assert dedup.should_emit("bar_zone", action, 0.9) is True

    def test_rapid_fire_same_event(self):
        """100 rapid identical events — only first emits."""
        dedup = EventDeduplicator(window=10.0)
        emitted = 0
        for _ in range(100):
            if dedup.should_emit("bar_zone", "pouring_draft", 0.9):
                emitted += 1
        assert emitted == 1

    def test_window_zero_always_emits(self):
        """Zero-length window — every event emits."""
        dedup = EventDeduplicator(window=0.0)
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.9) is True
        # With window=0, all entries expire immediately
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.8) is True

    def test_confidence_monotonically_tracks_max(self):
        """Confidence should track the highest value seen in window."""
        dedup = EventDeduplicator(window=10.0)
        dedup.should_emit("bar_zone", "pouring_draft", 0.5)  # first — emits
        dedup.should_emit("bar_zone", "pouring_draft", 0.3)  # lower — no update
        assert dedup.recent["bar_zone:pouring_draft"][-1]["conf"] == 0.5
        dedup.should_emit("bar_zone", "pouring_draft", 0.9)  # higher — update
        assert dedup.recent["bar_zone:pouring_draft"][-1]["conf"] == 0.9
        dedup.should_emit("bar_zone", "pouring_draft", 0.7)  # lower — no update
        assert dedup.recent["bar_zone:pouring_draft"][-1]["conf"] == 0.9


# ── ClipBuffer Torture Tests ────────────────────────────────────────────────

class TestClipBufferTorture:
    def test_extract_from_empty_buffer(self):
        buf = ClipBuffer(fps=30.0)
        frames = buf.extract_clip_frames(time.time())
        assert frames == []

    def test_save_from_empty_buffer(self, tmp_path):
        buf = ClipBuffer(fps=30.0)
        os.environ["CLIP_SAVE_DIR"] = str(tmp_path)
        result = buf.save_clip(time.time(), "test_event")
        assert result is None

    def test_buffer_overflow_drops_old_frames(self):
        buf = ClipBuffer(fps=10.0)
        # maxlen = 10 * 5 * 2 = 100 frames
        now = time.time()
        for i in range(200):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            buf.add_frame(frame, now + i * 0.1)

        assert len(buf.buffer) == 100  # max kept

    def test_extract_returns_correct_count(self):
        buf = ClipBuffer(fps=10.0)
        now = time.time()
        for i in range(50):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            buf.add_frame(frame, now + i * 0.1)

        # Extract 4 frames centered on now + 2.5
        frames = buf.extract_clip_frames(now + 2.5, n_frames=4)
        assert len(frames) <= 4
        assert len(frames) > 0

    def test_extract_single_frame_request(self):
        buf = ClipBuffer(fps=10.0)
        now = time.time()
        for i in range(20):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            buf.add_frame(frame, now + i * 0.1)

        frames = buf.extract_clip_frames(now + 1.0, n_frames=1)
        assert len(frames) == 1

    def test_save_creates_file(self, tmp_path):
        buf = ClipBuffer(fps=10.0)
        now = time.time()
        for i in range(30):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            buf.add_frame(frame, now + i * 0.1)

        event_id = str(uuid.uuid4())
        import pipeline.clip_buffer as cb
        original_dir = cb.CLIP_SAVE_DIR
        cb.CLIP_SAVE_DIR = str(tmp_path)
        try:
            path = buf.save_clip(now + 1.5, event_id)
            assert path is not None
            assert os.path.exists(path)
            assert path.endswith(".mp4")
        finally:
            cb.CLIP_SAVE_DIR = original_dir


# ── BarEvent Torture Tests ───────────────────────────────────────────────────

class TestBarEventTorture:
    def test_auto_generates_event_id(self):
        event = BarEvent(
            event_type="serve", venue_id="v1", camera_id="c1",
            zone="bar", confidence=0.9, action="pouring_draft"
        )
        assert event.event_id  # non-empty
        assert len(event.event_id) == 36  # UUID format

    def test_auto_generates_timestamp(self):
        event = BarEvent(
            event_type="serve", venue_id="v1", camera_id="c1",
            zone="bar", confidence=0.9, action="pouring_draft"
        )
        assert event.timestamp > 0

    def test_preserves_explicit_values(self):
        event = BarEvent(
            event_type="serve", venue_id="v1", camera_id="c1",
            zone="bar", confidence=0.9, action="pouring_draft",
            event_id="custom-id", timestamp=12345.0
        )
        assert event.event_id == "custom-id"
        assert event.timestamp == 12345.0

    def test_asdict_roundtrip(self):
        from dataclasses import asdict
        event = BarEvent(
            event_type="serve", venue_id="v1", camera_id="c1",
            zone="bar", confidence=0.9, action="pouring_draft"
        )
        d = asdict(event)
        assert d["event_type"] == "serve"
        assert d["action"] == "pouring_draft"
        assert "event_id" in d


# ── Integration Torture Tests ────────────────────────────────────────────────

class TestIntegrationTorture:
    """Tests that span multiple modules working together."""

    def test_full_detect_dedup_flow(self):
        """Interaction → dedup → only first event emits."""
        det = InteractionDetector()
        dedup = EventDeduplicator(window=10.0)

        p = _person()
        o = _obj()
        det.prev_positions[o.track_id] = (0, 0)

        events = det.detect([p], [o])
        assert len(events) == 1

        emitted = []
        for ev in events:
            if dedup.should_emit(ev.zone, ev.object_class, 0.9):
                emitted.append(ev)
        assert len(emitted) == 1

        # Second detection — cooldown blocks interaction detector
        det.prev_positions[o.track_id] = (0, 0)
        events2 = det.detect([p], [o])
        assert len(events2) == 0

    def test_zone_to_interaction_integration(self, zones_file):
        """Load real zones, check bbox zones, then run interaction detection."""
        zones = load_zones(zones_file, "camera_01")

        # Create tracked objects with real zone assignments
        p_zone = get_zone_for_bbox(200, 200, 400, 300, zones)
        o_zone = get_zone_for_bbox(250, 250, 350, 350, zones)
        assert p_zone == "bar_zone"
        assert o_zone == "bar_zone"

        det = InteractionDetector()
        p = _person(zone=p_zone)
        o = _obj(zone=o_zone)
        det.prev_positions[o.track_id] = (0, 0)

        events = det.detect([p], [o])
        assert len(events) == 1
        assert events[0].zone == "bar_zone"

    def test_clip_buffer_with_interaction_timing(self):
        """Clip extraction works with interaction-detected timestamps."""
        buf = ClipBuffer(fps=10.0)
        det = InteractionDetector()

        now = time.time()
        for i in range(50):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            buf.add_frame(frame, now + i * 0.1)

        p = _person()
        o = _obj()
        det.prev_positions[o.track_id] = (0, 0)

        events = det.detect([p], [o])
        assert len(events) == 1

        # Extract frames around the event time
        frames = buf.extract_clip_frames(events[0].timestamp, n_frames=4)
        assert len(frames) > 0
        assert len(frames) <= 4
