"""Friday Night Rush — Chaos Engineering Simulation
=================================================
Monte Carlo simulation of 5,000 frames testing three dangerous failure modes:

1. Tracker Drift  — Bartender leaves frame, returns with new track ID (Scenario B)
2. Async Lag      — VLM 17x slower than frame loop, queue backpressure (all scenarios)
3. API Flakiness  — 10% of VLM calls return HTTP 500 or invalid JSON (Chaos Monkey)

Scenarios:
  A: "The Perfect Pour"       — Person + Glass overlap for 150 frames → expect triggers
  B: "The Occluded Bartender"  — Person #5 → gap → Person #6, same glass → ID switch
  C: "The Phantom Trigger"     — Person walks by stationary tap, no dwell → zero events
  D: "Rush Hour Stress"        — 10 persons × 10 glasses burst at frame 2500

No GPU required. Tests real InteractionDetector, EventDeduplicator, _safe_parse.
"""

import logging
import os
import queue
import random
import sys
import threading
import time
import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.detector import TrackedObject
from pipeline.interactions import InteractionDetector, PendingEvent
from pipeline.deduplicator import EventDeduplicator
from pipeline.verifier import _safe_parse
from pipeline.events import BarEvent

logger = logging.getLogger("torture_sim")

# ───────────────────────────────────────────────────────────────────────────────
# Simulation Clock
# ───────────────────────────────────────────────────────────────────────────────

class SimClock:
    """Deterministic clock advancing 1/FPS per tick."""

    def __init__(self, start: float = 10000.0, fps: float = 30.0):
        self.now = start
        self.dt = 1.0 / fps

    def tick(self):
        self.now += self.dt
        return self.now


# ───────────────────────────────────────────────────────────────────────────────
# Mock YOLO — Deterministic Scenario Generator
# ───────────────────────────────────────────────────────────────────────────────

def _person(track_id, bbox):
    return TrackedObject(
        track_id=track_id, class_id=0, class_name="person",
        bbox=bbox, confidence=0.92, zone="bar_zone",
    )

def _glass(track_id, bbox):
    return TrackedObject(
        track_id=track_id, class_id=40, class_name="wine_glass",
        bbox=bbox, confidence=0.78, zone="bar_zone",
    )


class MockYOLO:
    """Yields deterministic TrackedObject lists for each frame index."""

    @staticmethod
    def detect(frame_idx: int):
        persons, objects = [], []

        # ── Tap #300: always visible, STATIONARY (velocity = 0 always) ──
        objects.append(_glass(300, (1600, 350, 1700, 450)))

        # ── Scenario A: "The Perfect Pour" ──
        # Person #1 stands still, glass drifts through their bbox at 3 px/frame.
        # Glass exits person after ~50 frames. Trigger fires on first appearance.
        if 100 <= frame_idx <= 249:
            persons.append(_person(1, (400, 200, 600, 600)))
            dx = (frame_idx - 100) * 3.0
            objects.append(_glass(100, (450 + dx, 350, 530 + dx, 450)))

        # Second pour, different glass (#101), same bartender
        if 800 <= frame_idx <= 949:
            persons.append(_person(1, (400, 200, 600, 600)))
            dx = (frame_idx - 800) * 3.0
            objects.append(_glass(101, (450 + dx, 350, 530 + dx, 450)))

        # ── Scenario B: "The Occluded Bartender" ──
        # Person AND glass drift together at 3 px/frame so IoU stays constant.
        # Phase 1: Person #5 + Glass #200
        if 1500 <= frame_idx <= 1599:
            dx = (frame_idx - 1500) * 3.0
            persons.append(_person(5, (700 + dx, 200, 900 + dx, 600)))
            objects.append(_glass(200, (750 + dx, 350, 830 + dx, 450)))

        # Phase 2: Person #5 gone (occlusion), glass still visible
        if 1600 <= frame_idx <= 1629:
            dx = (frame_idx - 1500) * 3.0
            objects.append(_glass(200, (750 + dx, 350, 830 + dx, 450)))

        # Phase 3: Person #6 (new track ID!) + same Glass #200
        if 1630 <= frame_idx <= 1779:
            dx = (frame_idx - 1500) * 3.0
            persons.append(_person(6, (700 + dx, 200, 900 + dx, 600)))
            objects.append(_glass(200, (750 + dx, 350, 830 + dx, 450)))

        # ── Scenario C: "The Phantom Trigger" ──
        # Person #10 walks by Tap #300 for 3 frames. Tap is stationary → velocity=0.
        # Tap has prev_position from prior frames → velocity gate blocks.
        for start in [500, 1000, 2000, 3000, 4000, 4500]:
            if start <= frame_idx <= start + 2:
                persons.append(_person(10, (1550, 200, 1750, 600)))

        # ── Scenario D: "Rush Hour Stress" ──
        # 10 persons × 10 glasses, all overlapping. Glasses drift out after ~38 frames.
        # All 10 pairs fire on frame 2500 (first appearance, no prev_pos).
        if 2500 <= frame_idx <= 3500:
            for pid in range(20, 30):
                x = 100 + (pid - 20) * 140
                persons.append(_person(pid, (x, 200, x + 130, 600)))
                dx = (frame_idx - 2500) * 3.0
                objects.append(_glass(400 + (pid - 20), (x + 15 + dx, 350, x + 75 + dx, 450)))

        return persons, objects


# ───────────────────────────────────────────────────────────────────────────────
# Mock VLM — Latency + 10% Chaos Monkey
# ───────────────────────────────────────────────────────────────────────────────

VLM_GOOD = [
    '{"action":"pouring_draft","confidence":0.85,"drink_category":"beer","vessel_type":"pint_glass","description":"Pouring draft"}',
    '{"action":"serving_customer","confidence":0.82,"drink_category":"wine","vessel_type":"wine_glass","description":"Serving wine"}',
    '{"action":"mixing_cocktail","confidence":0.79,"drink_category":"cocktail","vessel_type":"shaker","description":"Mixing cocktail"}',
]

VLM_BAD = [
    "500 Internal Server Error",
    '{"error": "rate_limit_exceeded"}',
    "null",
    "[1, 2, 3]",
    '```json\n{"broken": true\n```',
    "",
    "I cannot process this image.",
    '{"confidence": 0.5}',
]


class MockVLM:
    """Simulates VLM with configurable latency and error injection."""

    def __init__(self, latency: float = 0.005, error_rate: float = 0.10, seed: int = 43):
        self.latency = latency
        self.error_rate = error_rate
        self.rng = random.Random(seed)
        self.call_count = 0
        self.error_count = 0
        self.lock = threading.Lock()

    def call(self, pending: PendingEvent) -> dict:
        time.sleep(self.latency)
        with self.lock:
            self.call_count += 1

        if self.rng.random() < self.error_rate:
            with self.lock:
                self.error_count += 1
            return _safe_parse(self.rng.choice(VLM_BAD))

        return _safe_parse(self.rng.choice(VLM_GOOD))


# ───────────────────────────────────────────────────────────────────────────────
# Mock Redis — Python Queue with size tracking
# ───────────────────────────────────────────────────────────────────────────────

class MockRedis:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.max_size = 0
        self.total_pushed = 0
        self._lock = threading.Lock()

    def push(self, item):
        self.q.put(item)
        with self._lock:
            self.total_pushed += 1
            size = self.q.qsize()
            if size > self.max_size:
                self.max_size = size

    def pop(self, timeout=0.1):
        return self.q.get(timeout=timeout)

    def empty(self):
        return self.q.empty()


# ───────────────────────────────────────────────────────────────────────────────
# Simulation Metrics
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class SimMetrics:
    frames_processed: int = 0
    frames_dropped: int = 0
    pending_total: int = 0
    pending_by_scenario: dict = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})
    vlm_calls: int = 0
    vlm_errors: int = 0
    events_emitted: int = 0
    events_filtered: int = 0
    events_deduped: int = 0
    emitted_by_scenario: dict = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})
    queue_max_size: int = 0
    worker_crashes: int = 0


def _classify_scenario(obj_track_id: int) -> str:
    if obj_track_id in (100, 101):
        return "A"
    if obj_track_id == 200:
        return "B"
    if obj_track_id == 300:
        return "C"
    return "D"


# ───────────────────────────────────────────────────────────────────────────────
# run_simulation — the Friday Night Rush
# ───────────────────────────────────────────────────────────────────────────────

def run_simulation(
    n_frames: int = 5000,
    vlm_latency: float = 0.005,
    vlm_error_rate: float = 0.10,
    frame_drop_count: int = 50,
    seed: int = 42,
) -> SimMetrics:
    """Run the Virtual Friday Night.  Returns collected metrics."""

    rng = random.Random(seed)
    clock = SimClock(start=10000.0, fps=30.0)
    yolo = MockYOLO()
    vlm = MockVLM(latency=vlm_latency, error_rate=vlm_error_rate, seed=seed + 1)
    redis = MockRedis()
    metrics = SimMetrics()

    dropped = set(rng.sample(range(n_frames), min(frame_drop_count, n_frames)))

    done = threading.Event()
    emit_lock = threading.Lock()

    # ── Shared deduplicator (used by worker, real wall-clock time) ──
    dedup = EventDeduplicator()

    # ── Worker thread (Slow Loop) ──
    def vlm_worker():
        while not done.is_set() or not redis.empty():
            try:
                pending = redis.pop(timeout=0.05)
            except queue.Empty:
                continue

            try:
                result = vlm.call(pending)
                action = result.get("action", "unknown")
                confidence = result.get("confidence", 0.0)

                if action in ("unknown", "idle", "cleaning") or confidence < 0.5:
                    with emit_lock:
                        metrics.events_filtered += 1
                    continue

                if not dedup.should_emit(pending.zone, action, confidence):
                    with emit_lock:
                        metrics.events_deduped += 1
                    continue

                scenario = _classify_scenario(pending.object_track_id)
                with emit_lock:
                    metrics.events_emitted += 1
                    metrics.emitted_by_scenario[scenario] += 1

            except Exception:
                with emit_lock:
                    metrics.worker_crashes += 1

    worker = threading.Thread(target=vlm_worker, daemon=True)
    worker.start()

    # ── Main Loop (Fast Loop) — patched time for InteractionDetector ──
    with patch("pipeline.interactions.time") as mock_time:
        mock_time.time = lambda: clock.now

        detector = InteractionDetector()

        print(f"\n{'='*60}")
        print(f"  FRIDAY NIGHT RUSH — {n_frames} frames, "
              f"{vlm_error_rate*100:.0f}% VLM chaos, {frame_drop_count} drops")
        print(f"{'='*60}")

        for fi in range(n_frames):
            clock.tick()

            if fi in dropped:
                metrics.frames_dropped += 1
                continue

            metrics.frames_processed += 1
            persons, objects = yolo.detect(fi)
            pending_events = detector.detect(persons, objects)

            for pe in pending_events:
                scenario = _classify_scenario(pe.object_track_id)
                metrics.pending_by_scenario[scenario] += 1
                metrics.pending_total += 1
                redis.push(pe)

            cur_q = redis.q.qsize()
            if cur_q > metrics.queue_max_size:
                metrics.queue_max_size = cur_q

            if fi > 0 and fi % 1000 == 0:
                print(f"  [{fi:>5}/{n_frames}]  pending={metrics.pending_total:>3}  "
                      f"queue={cur_q:>3}  emitted={metrics.events_emitted:>3}  "
                      f"filtered={metrics.events_filtered}  deduped={metrics.events_deduped}")

    # Signal worker to drain remaining items and stop
    done.set()
    worker.join(timeout=30.0)

    metrics.vlm_calls = vlm.call_count
    metrics.vlm_errors = vlm.error_count
    if redis.max_size > metrics.queue_max_size:
        metrics.queue_max_size = redis.max_size

    print(f"\n  {'─'*56}")
    print(f"  SHIFT REPORT")
    print(f"  {'─'*56}")
    print(f"  Frames processed : {metrics.frames_processed} "
          f"(dropped {metrics.frames_dropped})")
    print(f"  Pending events   : {metrics.pending_total}  "
          f"(A={metrics.pending_by_scenario['A']}  B={metrics.pending_by_scenario['B']}  "
          f"C={metrics.pending_by_scenario['C']}  D={metrics.pending_by_scenario['D']})")
    print(f"  VLM calls        : {metrics.vlm_calls}  "
          f"(errors={metrics.vlm_errors})")
    print(f"  Events emitted   : {metrics.events_emitted}  "
          f"(filtered={metrics.events_filtered}  deduped={metrics.events_deduped})")
    print(f"  Emitted by scene : A={metrics.emitted_by_scenario['A']}  "
          f"B={metrics.emitted_by_scenario['B']}  "
          f"C={metrics.emitted_by_scenario['C']}  "
          f"D={metrics.emitted_by_scenario['D']}")
    print(f"  Queue max size   : {metrics.queue_max_size}")
    print(f"  Worker crashes   : {metrics.worker_crashes}")
    print(f"  {'─'*56}\n")

    return metrics


# ───────────────────────────────────────────────────────────────────────────────
# Cached result — simulation runs once per test session
# ───────────────────────────────────────────────────────────────────────────────

_cached: SimMetrics | None = None

def _get_result() -> SimMetrics:
    global _cached
    if _cached is None:
        _cached = run_simulation()
    return _cached


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerfectPour(unittest.TestCase):
    """Scenario A: sustained person-glass overlap must trigger events."""

    def test_triggers_generated(self):
        m = _get_result()
        self.assertGreaterEqual(m.pending_by_scenario["A"], 1,
                                "Perfect Pour must generate at least 1 pending event")

    def test_triggers_bounded(self):
        m = _get_result()
        self.assertLessEqual(m.pending_by_scenario["A"], 4,
                             "Perfect Pour should not exceed 4 triggers (2 glasses × cooldown)")


class TestOccludedBartender(unittest.TestCase):
    """Scenario B: person ID switch during occlusion."""

    def test_both_phases_trigger(self):
        """Person #5 and Person #6 are different pairs → both should trigger."""
        m = _get_result()
        self.assertGreaterEqual(m.pending_by_scenario["B"], 2,
                                "Occluded bartender must trigger for both person IDs")

    def test_triggers_bounded(self):
        m = _get_result()
        self.assertLessEqual(m.pending_by_scenario["B"], 4,
                             "Scenario B triggers bounded by cooldown")


class TestPhantomTrigger(unittest.TestCase):
    """Scenario C: walk-by with stationary tap must produce ZERO events."""

    def test_zero_pending_events(self):
        m = _get_result()
        self.assertEqual(m.pending_by_scenario["C"], 0,
                         "Phantom trigger must be fully debounced (stationary tap, velocity=0)")

    def test_zero_emitted_events(self):
        m = _get_result()
        self.assertEqual(m.emitted_by_scenario["C"], 0,
                         "No Scenario C events should ever be emitted")


class TestRushHourStress(unittest.TestCase):
    """Scenario D: 10 persons × 10 glasses burst."""

    def test_burst_triggers(self):
        m = _get_result()
        self.assertGreaterEqual(m.pending_by_scenario["D"], 8,
                                "Rush hour burst should trigger most of 10 pairs")

    def test_budget_not_exceeded(self):
        """Total pending events across all scenarios must stay under hourly budget."""
        m = _get_result()
        self.assertLessEqual(m.pending_total, 100,
                             "Total triggers must respect MAX_VLM_CALLS_PER_HOUR")


class TestBackpressure(unittest.TestCase):
    """Queue must never explode under load."""

    def test_queue_bounded(self):
        m = _get_result()
        self.assertLess(m.queue_max_size, 500,
                        f"Queue hit {m.queue_max_size} — backpressure failed")

    def test_all_events_processed(self):
        """Everything pushed to queue must be consumed (emitted + filtered + deduped)."""
        m = _get_result()
        accounted = m.events_emitted + m.events_filtered + m.events_deduped
        self.assertEqual(accounted, m.pending_total,
                         f"Leak: {m.pending_total} pushed but only {accounted} accounted for")


class TestVLMResilience(unittest.TestCase):
    """10% error rate must not crash the worker."""

    def test_no_worker_crashes(self):
        m = _get_result()
        self.assertEqual(m.worker_crashes, 0, "Worker crashed on bad VLM response")

    def test_errors_injected(self):
        m = _get_result()
        self.assertGreater(m.vlm_errors, 0, "Chaos monkey should have injected errors")

    def test_errors_handled_gracefully(self):
        """Errored VLM calls produce 'unknown' action → filtered, not emitted."""
        m = _get_result()
        self.assertGreaterEqual(m.events_filtered, m.vlm_errors,
                                "VLM errors should be caught by the unknown/low-conf filter")

    def test_safe_parse_http_500(self):
        r = _safe_parse("500 Internal Server Error")
        self.assertEqual(r["action"], "unknown")

    def test_safe_parse_null(self):
        r = _safe_parse("null")
        self.assertEqual(r["action"], "unknown")

    def test_safe_parse_array(self):
        r = _safe_parse("[1, 2, 3]")
        self.assertEqual(r["action"], "unknown")

    def test_safe_parse_empty(self):
        r = _safe_parse("")
        self.assertEqual(r["action"], "unknown")

    def test_safe_parse_broken_fence(self):
        r = _safe_parse('```json\n{"broken": true\n```')
        self.assertEqual(r["action"], "unknown")

    def test_safe_parse_missing_action(self):
        r = _safe_parse('{"confidence": 0.9}')
        self.assertEqual(r["action"], "unknown")
        self.assertEqual(r["confidence"], 0.9)


class TestFrameDropResilience(unittest.TestCase):
    """Random frame drops must not corrupt pipeline state."""

    def test_drops_applied(self):
        m = _get_result()
        self.assertEqual(m.frames_dropped, 50, "Expected 50 frame drops")

    def test_total_frames_correct(self):
        m = _get_result()
        self.assertEqual(m.frames_processed + m.frames_dropped, 5000)

    def test_no_crashes_after_drops(self):
        m = _get_result()
        self.assertEqual(m.worker_crashes, 0)


class TestInteractionDetectorDirect(unittest.TestCase):
    """Direct unit tests for InteractionDetector edge cases."""

    def test_velocity_gate_blocks_stationary(self):
        """Stationary object with known prev_position → no trigger."""
        with patch("pipeline.interactions.time") as mt:
            t = 20000.0
            mt.time = lambda: t
            det = InteractionDetector()
            p = _person(1, (400, 200, 600, 600))
            o = _glass(100, (450, 350, 530, 450))

            # First call: establishes prev_pos, triggers (no prev)
            events = det.detect([p], [o])
            self.assertEqual(len(events), 1)

            # Advance past cooldown
            t += 15.0
            mt.time = lambda: t

            # Same positions → velocity = 0 → blocked
            events = det.detect([p], [o])
            self.assertEqual(len(events), 0, "Stationary object should be blocked")

    def test_cooldown_prevents_rapid_fire(self):
        with patch("pipeline.interactions.time") as mt:
            t = 20000.0
            mt.time = lambda: t
            det = InteractionDetector()

            # Frame 1: trigger
            p = _person(1, (400, 200, 600, 600))
            o = _glass(100, (450, 350, 530, 450))
            self.assertEqual(len(det.detect([p], [o])), 1)

            # Frame 2: 0.5s later, object moved but cooldown blocks
            t += 0.5
            mt.time = lambda: t
            o2 = _glass(100, (453, 350, 533, 450))
            self.assertEqual(len(det.detect([p], [o2])), 0)

    def test_different_pair_not_affected_by_cooldown(self):
        """Person #5 → Person #6 creates a new pair_id → independent cooldown."""
        with patch("pipeline.interactions.time") as mt:
            t = 20000.0
            mt.time = lambda: t
            det = InteractionDetector()

            p5 = _person(5, (400, 200, 600, 600))
            glass = _glass(200, (450, 350, 530, 450))
            self.assertEqual(len(det.detect([p5], [glass])), 1)

            # 2 seconds later: same glass, different person
            t += 2.0
            mt.time = lambda: t
            p6 = _person(6, (400, 200, 600, 600))
            glass2 = _glass(200, (456, 350, 536, 450))  # moved
            events = det.detect([p6], [glass2])
            self.assertEqual(len(events), 1, "New person_id = new pair = no cooldown")
            self.assertEqual(events[0].person_track_id, 6)

    def test_hourly_budget_cap(self):
        with patch("pipeline.interactions.time") as mt:
            t = 20000.0
            mt.time = lambda: t
            det = InteractionDetector()
            det.calls_this_hour = 99  # 1 below limit

            p = _person(1, (400, 200, 600, 600))
            o = _glass(100, (450, 350, 530, 450))
            self.assertEqual(len(det.detect([p], [o])), 1)  # uses last slot

            # Budget exhausted — new pair, but budget blocks
            t += 15.0
            mt.time = lambda: t
            p2 = _person(2, (400, 200, 600, 600))
            o2 = _glass(101, (453, 350, 533, 450))
            self.assertEqual(len(det.detect([p2], [o2])), 0, "Budget exhausted")

    def test_no_bar_zone_no_trigger(self):
        with patch("pipeline.interactions.time") as mt:
            mt.time = lambda: 20000.0
            det = InteractionDetector()
            p = TrackedObject(1, 0, "person", (400, 200, 600, 600), 0.9, zone="queue_zone")
            o = TrackedObject(100, 40, "wine_glass", (450, 350, 530, 450), 0.8, zone="queue_zone")
            self.assertEqual(len(det.detect([p], [o])), 0)


class TestDeduplicatorDirect(unittest.TestCase):
    """Direct tests for EventDeduplicator edge cases."""

    def test_first_event_emitted(self):
        d = EventDeduplicator(window=10.0)
        self.assertTrue(d.should_emit("bar_zone", "pouring_draft", 0.85))

    def test_duplicate_blocked(self):
        d = EventDeduplicator(window=10.0)
        d.should_emit("bar_zone", "pouring_draft", 0.85)
        self.assertFalse(d.should_emit("bar_zone", "pouring_draft", 0.80))

    def test_different_action_emitted(self):
        d = EventDeduplicator(window=10.0)
        d.should_emit("bar_zone", "pouring_draft", 0.85)
        self.assertTrue(d.should_emit("bar_zone", "serving_customer", 0.82))

    def test_different_zone_emitted(self):
        d = EventDeduplicator(window=10.0)
        d.should_emit("bar_zone", "pouring_draft", 0.85)
        self.assertTrue(d.should_emit("service_rail", "pouring_draft", 0.85))

    def test_higher_confidence_updates(self):
        d = EventDeduplicator(window=10.0)
        d.should_emit("bar_zone", "pouring_draft", 0.70)
        d.should_emit("bar_zone", "pouring_draft", 0.95)  # higher → updates
        self.assertEqual(d.recent["bar_zone:pouring_draft"][-1]["conf"], 0.95)


class TestFullSimulation(unittest.TestCase):
    """End-to-end assertions on the complete 5,000-frame simulation."""

    def test_events_produced(self):
        m = _get_result()
        self.assertGreater(m.events_emitted, 0, "Pipeline must produce at least 1 event")

    def test_no_phantom_events(self):
        m = _get_result()
        self.assertEqual(m.emitted_by_scenario["C"], 0)

    def test_accounting_balanced(self):
        """emitted + filtered + deduped == pending_total (no leaks)."""
        m = _get_result()
        total = m.events_emitted + m.events_filtered + m.events_deduped
        self.assertEqual(total, m.pending_total)

    def test_queue_never_exploded(self):
        m = _get_result()
        self.assertLess(m.queue_max_size, 500)

    def test_simulation_completes(self):
        """Implicit: if we get here, no deadlock occurred."""
        m = _get_result()
        self.assertEqual(m.frames_processed + m.frames_dropped, 5000)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
