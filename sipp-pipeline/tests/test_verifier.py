"""Tests for pipeline/verifier.py"""

import pytest

from pipeline.verifier import _safe_parse, verify_event


class TestSafeParse:
    def test_clean_json(self):
        result = _safe_parse('{"action": "pouring_draft", "confidence": 0.95}')
        assert result["action"] == "pouring_draft"
        assert result["confidence"] == 0.95

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"action": "serving_customer", "confidence": 0.8}\n```'
        result = _safe_parse(raw)
        assert result["action"] == "serving_customer"
        assert result["confidence"] == 0.8

    def test_json_with_plain_fences(self):
        raw = '```\n{"action": "payment_card", "confidence": 0.7}\n```'
        result = _safe_parse(raw)
        assert result["action"] == "payment_card"

    def test_garbage_input_returns_unknown(self):
        result = _safe_parse("this is not json at all")
        assert result["action"] == "unknown"
        assert result["confidence"] == 0.0
        assert "raw" in result

    def test_missing_action_field_gets_default(self):
        result = _safe_parse('{"confidence": 0.9, "description": "something"}')
        assert result["action"] == "unknown"
        assert result["confidence"] == 0.9  # confidence IS present in input

    def test_missing_confidence_field_gets_default(self):
        result = _safe_parse('{"action": "pouring_draft"}')
        assert result["action"] == "pouring_draft"
        assert result["confidence"] == 0.0

    def test_extra_fields_preserved(self):
        result = _safe_parse(
            '{"action": "pouring_draft", "confidence": 0.9, '
            '"drink_category": "beer", "vessel_type": "pint_glass"}'
        )
        assert result["drink_category"] == "beer"
        assert result["vessel_type"] == "pint_glass"

    def test_empty_string_returns_unknown(self):
        result = _safe_parse("")
        assert result["action"] == "unknown"

    def test_whitespace_only_returns_unknown(self):
        result = _safe_parse("   \n\n   ")
        assert result["action"] == "unknown"


class TestVerifyEvent:
    def test_no_api_key_returns_unknown(self):
        """When ANTHROPIC_API_KEY is empty, verify_event should skip gracefully."""
        result = verify_event([], {"object_class": "bottle", "zone": "bar_zone"})
        assert result["action"] == "unknown"
        assert result["confidence"] == 0.0

    def test_empty_frames_returns_unknown(self):
        result = verify_event([], {"object_class": "bottle"})
        assert result["action"] == "unknown"
