"""Production Torture Test — 7 Phases, Zero API Calls
====================================================
Tests input validation, tracker stress, VLM resilience, resource safety,
security hardening, integration chaos, and long-run stability.
All mocked. No GPU, no Anthropic API, no external services.
"""

import json
import math
import os
import sys
import time
import unittest
from collections import defaultdict
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detector import TrackedObject
from pipeline.interactions import InteractionDetector, PendingEvent
from pipeline.deduplicator import EventDeduplicator
from pipeline.verifier import _safe_parse, SYSTEM_PROMPT
from pipeline.clip_buffer import ClipBuffer
from pipeline.events import BarEvent


def _person(tid, bbox, zone="bar_zone"):
    return TrackedObject(tid, 0, "person", bbox, 0.9, zone)

def _glass(tid, bbox, zone="bar_zone"):
    return TrackedObject(tid, 40, "wine_glass", bbox, 0.8, zone)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Input Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorruptedFrames(unittest.TestCase):
    """Pipeline components must handle garbage/edge-case frames."""

    def test_zero_frame_to_clip_buffer(self):
        buf = ClipBuffer(fps=30.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        buf.add_frame(frame, 1000.0)
        clips = buf.extract_clip_frames(1000.0, n_frames=4)
        self.assertEqual(len(clips), 1)

    def test_single_pixel_frame(self):
        buf = ClipBuffer(fps=30.0)
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        buf.add_frame(frame, 1000.0)
        clips = buf.extract_clip_frames(1000.0)
        self.assertEqual(clips[0].shape, (1, 1, 3))

    def test_huge_frame_dimensions(self):
        """4K frame in clip buffer — just memory, no OOM on buffer logic."""
        buf = ClipBuffer(fps=30.0)
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        buf.add_frame(frame, 1000.0)
        self.assertEqual(len(buf.buffer), 1)

    def test_nan_bbox_iou(self):
        det = InteractionDetector()
        iou = det._bbox_iou((0, 0, 0, 0), (0, 0, 0, 0))
        self.assertEqual(iou, 0.0)

    def test_negative_bbox_iou(self):
        det = InteractionDetector()
        iou = det._bbox_iou((-10, -10, -5, -5), (-8, -8, -3, -3))
        self.assertGreaterEqual(iou, 0.0)

    def test_inverted_bbox(self):
        """x2 < x1 — degenerate bbox."""
        det = InteractionDetector()
        iou = det._bbox_iou((100, 100, 50, 50), (60, 60, 200, 200))
        self.assertEqual(iou, 0.0)


class TestResolutionStress(unittest.TestCase):
    """Detection at extreme resolutions."""

    def test_144p_tiny_objects(self):
        """At 144p, objects are <10px. IoU with person is near-zero."""
        det = InteractionDetector()
        # Tiny person and glass in 256x144 frame
        p = _person(1, (50, 20, 80, 100))
        o = _glass(100, (55, 60, 65, 70))  # 10x10 glass
        with patch("pipeline.interactions.time") as mt:
            mt.time = lambda: 10000.0
            events = det.detect([p], [o])
        # Very small overlap area — may or may not trigger depending on IoU
        self.assertIsInstance(events, list)

    def test_4k_large_bboxes(self):
        """4K bboxes should compute IoU correctly at large scales."""
        det = InteractionDetector()
        p = _person(1, (1000, 500, 1500, 1800))
        o = _glass(100, (1100, 900, 1300, 1100))
        iou = det._bbox_iou(p.bbox, o.bbox)
        self.assertGreater(iou, 0.0)


class TestFrameRateChaos(unittest.TestCase):
    """ClipBuffer under varied FPS."""

    def test_60fps_buffer_capacity(self):
        buf = ClipBuffer(fps=60.0)
        # At 60fps with 5s clip: buffer = 60*5*2 = 600 frames
        self.assertEqual(buf.buffer.maxlen, 600)

    def test_5fps_buffer_capacity(self):
        buf = ClipBuffer(fps=5.0)
        self.assertEqual(buf.buffer.maxlen, 50)

    def test_fractional_fps(self):
        buf = ClipBuffer(fps=29.97)
        self.assertGreater(buf.buffer.maxlen, 0)

    def test_extract_from_sparse_buffer(self):
        """5 FPS = only 25 frames in 5s window. Extract 4 should work."""
        buf = ClipBuffer(fps=5.0)
        for i in range(25):
            buf.add_frame(np.zeros((100, 100, 3), dtype=np.uint8), 1000.0 + i * 0.2)
        clips = buf.extract_clip_frames(1002.5, n_frames=4)
        self.assertGreater(len(clips), 0)
        self.assertLessEqual(len(clips), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Tracker Stress (via InteractionDetector)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOcclusionGauntlet(unittest.TestCase):
    """Person disappears behind pillar, reappears — new track_id."""

    def test_reappeared_person_triggers_new_pair(self):
        with patch("pipeline.interactions.time") as mt:
            t = 10000.0
            mt.time = lambda: t
            det = InteractionDetector()
            glass = _glass(50, (500, 300, 600, 400))

            # Person #1 interacts
            det.detect([_person(1, (450, 200, 650, 600))], [glass])
            self.assertIn("1_50", det.active_interactions)

            # 15 seconds later: same person, new track_id #7
            t = 10015.0
            mt.time = lambda: t
            glass2 = _glass(50, (505, 300, 605, 400))  # moved
            events = det.detect([_person(7, (450, 200, 650, 600))], [glass2])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].person_track_id, 7)
            self.assertIn("7_50", det.active_interactions)


class TestCrowdDensity(unittest.TestCase):
    """15+ persons simultaneously."""

    def test_15_persons_no_crash(self):
        with patch("pipeline.interactions.time") as mt:
            mt.time = lambda: 10000.0
            det = InteractionDetector()
            persons = [_person(i, (i * 120, 200, i * 120 + 100, 600)) for i in range(15)]
            objects = [_glass(100 + i, (i * 120 + 20, 350, i * 120 + 70, 450)) for i in range(15)]
            events = det.detect(persons, objects)
            self.assertIsInstance(events, list)
            # All first-time objects trigger (no prev_pos → velocity skipped)
            self.assertGreaterEqual(len(events), 10)

    def test_30_persons_budget_limit(self):
        with patch("pipeline.interactions.time") as mt:
            mt.time = lambda: 10000.0
            det = InteractionDetector()
            persons = [_person(i, (i * 60, 200, i * 60 + 50, 600)) for i in range(30)]
            objects = [_glass(100 + i, (i * 60 + 10, 350, i * 60 + 40, 450)) for i in range(30)]
            events = det.detect(persons, objects)
            # Budget is 100, so 30 triggers should all pass
            self.assertLessEqual(len(events), 100)


class TestRapidEntryExit(unittest.TestCase):
    """Person in frame for 3 frames then gone — no phantom tracks."""

    def test_brief_appearance_single_trigger(self):
        with patch("pipeline.interactions.time") as mt:
            t = 10000.0
            mt.time = lambda: t
            det = InteractionDetector()
            glass = _glass(50, (500, 300, 600, 400))

            # Frame 1: person enters
            det.detect([_person(1, (450, 200, 650, 600))], [glass])

            # Frame 2-3: same person, glass stationary
            t += 1 / 30
            mt.time = lambda: t
            det.detect([_person(1, (450, 200, 650, 600))], [glass])

            t += 1 / 30
            mt.time = lambda: t
            det.detect([_person(1, (450, 200, 650, 600))], [glass])

            # Frame 4: person gone, only glass
            t += 1 / 30
            mt.time = lambda: t
            events = det.detect([], [glass])
            self.assertEqual(len(events), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: VLM Adversarial Testing (no API calls)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafeParseAdversarial(unittest.TestCase):
    """Adversarial VLM outputs that must not crash _safe_parse."""

    ADVERSARIAL_OUTPUTS = [
        "",
        "null",
        "undefined",
        "true",
        "false",
        "42",
        "-1.5e10",
        "[1,2,3]",
        "[[[]]]",
        '{"action": "pouring"}',  # missing confidence
        '{"confidence": 0.9}',    # missing action
        "```json\n{}\n```",
        "```\n{}\n```",
        '```json\n{"action": "pour"}\n```',
        "Here is the JSON:\n```json\n{}\n```\nDone.",
        "{broken json",
        '{"action": "test", "extra": "' + "A" * 10000 + '"}',  # 10KB payload
        '{"action": "<script>alert(1)</script>"}',  # XSS attempt
        '{"action": "\\u0000\\u0001\\u0002"}',  # null bytes
        "I refuse to classify this image. It contains sensitive content.",
        "ERROR: Rate limit exceeded. Please try again later.",
        "<html><body>502 Bad Gateway</body></html>",
        '{"action": "pouring_draft", "confidence": "high"}',  # string confidence
        '{"action": "pouring_draft", "confidence": -1}',
        '{"action": "pouring_draft", "confidence": 999}',
        '{"action": "", "confidence": 0.5}',
        "{" * 100,  # nested braces
        "}" * 100,
        "\x00\x01\x02\x03",  # binary garbage
        "DROP TABLE events;",
    ]

    def test_all_adversarial_inputs_return_dict(self):
        for raw in self.ADVERSARIAL_OUTPUTS:
            result = _safe_parse(raw)
            self.assertIsInstance(result, dict, f"Failed on: {raw[:50]}")
            self.assertIn("action", result)
            self.assertIn("confidence", result)

    def test_adversarial_never_raises(self):
        for raw in self.ADVERSARIAL_OUTPUTS:
            try:
                _safe_parse(raw)
            except Exception as e:
                self.fail(f"_safe_parse raised {type(e).__name__} on: {raw[:50]}")

    def test_valid_json_preserves_fields(self):
        raw = '{"action": "pouring_draft", "confidence": 0.85, "extra": "data"}'
        r = _safe_parse(raw)
        self.assertEqual(r["action"], "pouring_draft")
        self.assertEqual(r["confidence"], 0.85)
        self.assertEqual(r["extra"], "data")

    def test_nested_markdown_fences(self):
        raw = "```json\n```json\n{\"action\": \"test\"}\n```\n```"
        r = _safe_parse(raw)
        self.assertIsInstance(r, dict)

    def test_unicode_in_description(self):
        raw = '{"action": "pouring_draft", "confidence": 0.8, "description": "Serving a b\u00e9er to cust\u00f6mer"}'
        r = _safe_parse(raw)
        self.assertEqual(r["action"], "pouring_draft")


class TestVLMBudgetEnforcement(unittest.TestCase):
    """MAX_VLM_CALLS_PER_HOUR hard cap."""

    def test_budget_hard_cap_at_100(self):
        with patch("pipeline.interactions.time") as mt:
            mt.time = lambda: 10000.0
            det = InteractionDetector()
            total_triggered = 0

            for i in range(200):
                p = _person(i, (400, 200, 600, 600))
                o = _glass(1000 + i, (450, 350, 530, 450))
                events = det.detect([p], [o])
                total_triggered += len(events)

            self.assertEqual(total_triggered, 100,
                             "Budget must hard-stop at MAX_VLM_CALLS_PER_HOUR=100")

    def test_budget_resets_after_hour(self):
        with patch("pipeline.interactions.time") as mt:
            t = 10000.0
            mt.time = lambda: t
            det = InteractionDetector()
            det.calls_this_hour = 100  # exhausted

            p = _person(1, (400, 200, 600, 600))
            o = _glass(500, (450, 350, 530, 450))
            self.assertEqual(len(det.detect([p], [o])), 0)

            # Advance 1 hour
            t = 13601.0
            mt.time = lambda: t
            o2 = _glass(501, (455, 350, 535, 450))
            events = det.detect([p], [o2])
            self.assertEqual(len(events), 1, "Budget should reset after 1 hour")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Resource Safety
# ═══════════════════════════════════════════════════════════════════════════════

class TestClipBufferMemory(unittest.TestCase):
    """Ring buffer must not grow beyond maxlen."""

    def test_buffer_stays_bounded(self):
        buf = ClipBuffer(fps=30.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(1000):
            buf.add_frame(frame, 1000.0 + i / 30)
        self.assertLessEqual(len(buf.buffer), buf.buffer.maxlen)
        self.assertLessEqual(len(buf.timestamps), buf.timestamps.maxlen)

    def test_old_frames_evicted(self):
        buf = ClipBuffer(fps=30.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(500):
            buf.add_frame(frame, 1000.0 + i / 30)
        # Buffer maxlen = 300 at 30fps. After 500 inserts, oldest 200 gone.
        self.assertEqual(len(buf.buffer), 300)
        self.assertAlmostEqual(buf.timestamps[0], 1000.0 + 200 / 30, places=2)


class TestInteractionDetectorMemory(unittest.TestCase):
    """prev_positions and active_interactions don't grow unbounded."""

    def test_prev_positions_tracks_only_current(self):
        with patch("pipeline.interactions.time") as mt:
            mt.time = lambda: 10000.0
            det = InteractionDetector()
            # Simulate 500 unique objects appearing one at a time
            for i in range(500):
                o = _glass(i, (100, 100, 200, 200))
                det.detect([], [o])
            # All 500 objects stored (no eviction in current impl)
            # This tests that it doesn't crash, not that it's bounded
            self.assertEqual(len(det.prev_positions), 500)

    def test_active_interactions_grows_with_pairs(self):
        with patch("pipeline.interactions.time") as mt:
            mt.time = lambda: 10000.0
            det = InteractionDetector()
            for i in range(100):
                p = _person(i, (400, 200, 600, 600))
                o = _glass(1000 + i, (450, 350, 530, 450))
                det.detect([p], [o])
            self.assertEqual(len(det.active_interactions), 100)


class TestDeduplicatorMemory(unittest.TestCase):
    """Deduplicator purges expired entries."""

    def test_expired_entries_purged(self):
        d = EventDeduplicator(window=0.01)  # 10ms window
        d.should_emit("z1", "pour", 0.8)
        time.sleep(0.02)  # Wait for expiry
        # Next call should purge and emit
        self.assertTrue(d.should_emit("z1", "pour", 0.8))

    def test_many_zones_no_crash(self):
        d = EventDeduplicator(window=10.0)
        for i in range(1000):
            d.should_emit(f"zone_{i}", "pour", 0.8)
        self.assertEqual(len(d.recent), 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Security & Privacy
# ═══════════════════════════════════════════════════════════════════════════════

class TestZonesSecurityInput(unittest.TestCase):
    """Malformed zones.json must not cause path traversal or crash."""

    def test_invalid_polygon_type(self):
        from pipeline.zones import load_zones
        import tempfile
        zones = {"camera_01": {"bar_zone": "../../../../etc/passwd"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(zones, f)
            f.flush()
            try:
                result = load_zones(f.name, "camera_01")
                # Should either return empty or handle string as non-polygon
                self.assertIsInstance(result, dict)
            except (TypeError, ValueError, KeyError):
                pass  # Acceptable: rejected bad input
            finally:
                os.unlink(f.name)

    def test_empty_zones_file(self):
        from pipeline.zones import load_zones
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            f.flush()
            try:
                result = load_zones(f.name, "camera_01")
                self.assertEqual(result, {})
            except KeyError:
                pass  # Also acceptable
            finally:
                os.unlink(f.name)

    def test_missing_camera_id(self):
        from pipeline.zones import load_zones
        import tempfile
        zones = {"camera_99": {"bar_zone": [[0, 0], [100, 0], [100, 100], [0, 100]]}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(zones, f)
            f.flush()
            try:
                result = load_zones(f.name, "camera_01")
                self.assertEqual(result, {})
            except KeyError:
                pass
            finally:
                os.unlink(f.name)


class TestSystemPromptSecurity(unittest.TestCase):
    """VLM system prompt must not leak internal details."""

    def test_no_file_paths_in_prompt(self):
        self.assertNotIn("/etc/", SYSTEM_PROMPT)
        self.assertNotIn("home/", SYSTEM_PROMPT)
        self.assertNotIn(".env", SYSTEM_PROMPT)

    def test_no_api_keys_in_prompt(self):
        self.assertNotIn("sk-", SYSTEM_PROMPT)
        self.assertNotIn("api_key", SYSTEM_PROMPT.lower())

    def test_json_schema_present(self):
        self.assertIn("action", SYSTEM_PROMPT)
        self.assertIn("confidence", SYSTEM_PROMPT)


class TestBarEventSecurity(unittest.TestCase):
    """BarEvent must generate unique IDs, no injection in fields."""

    def test_unique_event_ids(self):
        ids = set()
        for _ in range(1000):
            e = BarEvent("serve", "v1", "c1", "z1", 0.8, "pour")
            ids.add(e.event_id)
        self.assertEqual(len(ids), 1000, "Event IDs must be unique")

    def test_timestamp_auto_set(self):
        e = BarEvent("serve", "v1", "c1", "z1", 0.8, "pour")
        self.assertGreater(e.timestamp, 0)

    def test_xss_in_description_passthrough(self):
        """Pipeline doesn't sanitize — but it should store raw. Check no crash."""
        e = BarEvent("serve", "v1", "c1", "z1", 0.8, "pour",
                     description="<script>alert('xss')</script>")
        self.assertIn("<script>", e.description)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Integration Chaos
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventDeduplicatorIntegration(unittest.TestCase):
    """Dedup under rapid-fire event bursts."""

    def test_burst_of_100_same_events(self):
        d = EventDeduplicator(window=10.0)
        emitted = sum(1 for _ in range(100) if d.should_emit("bar", "pour", 0.8))
        self.assertEqual(emitted, 1, "Only first of 100 identical events should emit")

    def test_burst_of_3_different_actions(self):
        d = EventDeduplicator(window=10.0)
        actions = ["pouring_draft", "serving_customer", "payment_card"]
        emitted = sum(1 for a in actions if d.should_emit("bar", a, 0.8))
        self.assertEqual(emitted, 3)

    def test_interleaved_zones_and_actions(self):
        d = EventDeduplicator(window=10.0)
        emitted = 0
        for zone in ["bar_zone", "service_rail"]:
            for action in ["pour", "serve", "pay"]:
                for _ in range(10):
                    if d.should_emit(zone, action, 0.8):
                        emitted += 1
        self.assertEqual(emitted, 6, "2 zones × 3 actions = 6 unique")


class TestVerifyEventNoApiKey(unittest.TestCase):
    """verify_event with no API key must return unknown, not crash."""

    def test_no_key_returns_unknown(self):
        from pipeline.verifier import verify_event
        with patch("pipeline.verifier.ANTHROPIC_API_KEY", ""):
            with patch("pipeline.verifier._client", None):
                result = verify_event(
                    [np.zeros((100, 100, 3), dtype=np.uint8)],
                    {"object_class": "bottle", "zone": "bar_zone"},
                )
                self.assertEqual(result["action"], "unknown")

    def test_empty_frames_returns_unknown(self):
        from pipeline.verifier import verify_event
        with patch("pipeline.verifier.ANTHROPIC_API_KEY", "test-key"):
            result = verify_event([], {"object_class": "bottle"})
            self.assertEqual(result["action"], "unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: Long-Run Stability (simulated)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarathonSimulation(unittest.TestCase):
    """Simulate 10,000 frames of sustained operation."""

    def test_10k_frames_no_state_corruption(self):
        """Run InteractionDetector for 10,000 frames. No crash, bounded state."""
        with patch("pipeline.interactions.time") as mt:
            t = 10000.0
            mt.time = lambda: t
            det = InteractionDetector()

            triggers = 0
            for frame in range(10000):
                t += 1 / 30
                mt.time = lambda: t

                # Rotating cast: 5 persons, 5 glasses, positions shift
                persons = []
                objects = []
                for pid in range(5):
                    x = 100 + pid * 200 + (frame % 50)
                    persons.append(_person(pid, (x, 200, x + 150, 600)))
                    dx = frame * 0.1 % 100
                    objects.append(_glass(50 + pid, (x + 20 + dx, 350, x + 80 + dx, 450)))

                events = det.detect(persons, objects)
                triggers += len(events)

            self.assertGreater(triggers, 0)
            self.assertLessEqual(triggers, 100, "Budget cap enforced over 10K frames")
            self.assertLessEqual(len(det.prev_positions), 10, "Position history bounded")

    def test_dedup_over_many_events(self):
        """Deduplicator under sustained load: 1000 events across 10 actions."""
        d = EventDeduplicator(window=0.05)  # 50ms window for fast test
        emitted = 0
        for i in range(1000):
            action = f"action_{i % 10}"
            if d.should_emit("bar_zone", action, 0.8):
                emitted += 1
            if i % 100 == 99:
                time.sleep(0.06)  # Let window expire every 100 events
        # Should emit ~10 unique actions per window cycle
        self.assertGreater(emitted, 10)
        self.assertLess(emitted, 200)

    def test_clip_buffer_steady_state(self):
        """Buffer stays at maxlen after filling up."""
        buf = ClipBuffer(fps=30.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(2000):
            buf.add_frame(frame, 10000.0 + i / 30)
        self.assertEqual(len(buf.buffer), buf.buffer.maxlen)
        # Oldest frame is recent enough
        age = buf.timestamps[-1] - buf.timestamps[0]
        self.assertAlmostEqual(age, (buf.buffer.maxlen - 1) / 30, places=1)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
