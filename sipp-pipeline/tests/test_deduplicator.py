"""Tests for pipeline/deduplicator.py"""

import time

import pytest

from pipeline.deduplicator import EventDeduplicator


class TestEventDeduplicator:
    def test_first_event_always_emits(self):
        dedup = EventDeduplicator(window=10.0)
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.9) is True

    def test_same_zone_action_within_window_blocked(self):
        dedup = EventDeduplicator(window=10.0)
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.9) is True
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.8) is False

    def test_same_zone_action_after_window_emits(self):
        dedup = EventDeduplicator(window=0.01)  # very short window
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.9) is True
        time.sleep(0.02)  # wait for window to expire
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.8) is True

    def test_different_action_same_zone_emits(self):
        dedup = EventDeduplicator(window=10.0)
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.9) is True
        assert dedup.should_emit("bar_zone", "payment_card", 0.8) is True

    def test_same_action_different_zone_emits(self):
        dedup = EventDeduplicator(window=10.0)
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.9) is True
        assert dedup.should_emit("queue_zone", "pouring_draft", 0.8) is True

    def test_higher_confidence_updates_stored(self):
        dedup = EventDeduplicator(window=10.0)
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.7) is True
        # Lower confidence — blocked, no update
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.5) is False
        stored = dedup.recent["bar_zone:pouring_draft"][-1]["conf"]
        assert stored == 0.7

        # Higher confidence — blocked but updates stored value
        assert dedup.should_emit("bar_zone", "pouring_draft", 0.95) is False
        stored = dedup.recent["bar_zone:pouring_draft"][-1]["conf"]
        assert stored == 0.95
