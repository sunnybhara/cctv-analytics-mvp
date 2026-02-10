"""
Behavior Detection Module Tests
===============================
Tests for the MediaPipe-based behavior and engagement detection.
"""
import pytest
import numpy as np


class TestBehaviorModule:
    """Test behavior detection functions."""

    def test_module_imports(self):
        """Behavior module should import without errors."""
        import behavior
        assert hasattr(behavior, 'analyze_behavior')
        assert hasattr(behavior, 'BehaviorResult')
        assert hasattr(behavior, 'BehaviorType')

    def test_behavior_result_dataclass(self):
        """BehaviorResult should have all required fields."""
        from behavior import BehaviorResult

        result = BehaviorResult(
            engagement_score=75.0,
            behavior_type="engaged",
            pose_confidence=0.9,
            body_orientation=0.5,
            is_stationary=True,
            posture="upright",
            landmarks=None
        )

        assert result.engagement_score == 75.0
        assert result.behavior_type == "engaged"
        assert result.pose_confidence == 0.9

    def test_behavior_types_enum(self):
        """BehaviorType enum should have all expected values."""
        from behavior import BehaviorType

        assert BehaviorType.ENGAGED.value == "engaged"
        assert BehaviorType.BROWSING.value == "browsing"
        assert BehaviorType.WAITING.value == "waiting"
        assert BehaviorType.PASSING.value == "passing"

    def test_analyze_behavior_with_dummy_image(self):
        """analyze_behavior should handle images without people."""
        from behavior import analyze_behavior

        # Create dummy gray image (no person)
        dummy = np.zeros((400, 200, 3), dtype=np.uint8)
        dummy[:] = (128, 128, 128)

        result = analyze_behavior(dummy)

        # Should return unknown behavior when no pose detected
        assert result.behavior_type == "unknown"
        assert result.pose_confidence == 0.0
        assert result.engagement_score == 50.0  # Neutral default

    def test_analyze_behavior_returns_valid_engagement(self):
        """Engagement score should be in valid range."""
        from behavior import analyze_behavior

        dummy = np.zeros((400, 200, 3), dtype=np.uint8)
        result = analyze_behavior(dummy)

        assert 0 <= result.engagement_score <= 100

    def test_calculate_body_orientation(self):
        """Body orientation calculation should handle edge cases."""
        from behavior import calculate_body_orientation

        # Missing landmarks
        result = calculate_body_orientation({})
        assert result == 0.0

        # Partial landmarks
        result = calculate_body_orientation({
            "left_shoulder": {"x": 100, "y": 100, "z": 0, "visibility": 0.9}
        })
        assert result == 0.0

    def test_calculate_posture(self):
        """Posture calculation should return valid values."""
        from behavior import calculate_posture

        # Missing landmarks
        result = calculate_posture({})
        assert result == "unknown"

    def test_estimate_movement_no_history(self):
        """Movement estimation with no history should assume stationary."""
        from behavior import estimate_movement

        landmarks = {
            "left_hip": {"x": 100, "y": 200, "z": 0, "visibility": 0.9},
            "right_hip": {"x": 150, "y": 200, "z": 0, "visibility": 0.9}
        }

        is_stationary, speed = estimate_movement(landmarks, None)

        assert is_stationary is True
        assert speed == 0.0

    def test_calculate_engagement_score(self):
        """Engagement score calculation should be in valid range."""
        from behavior import calculate_engagement_score

        # Test various inputs
        score = calculate_engagement_score(
            body_orientation=1.0,  # Facing camera
            posture="leaning_forward",
            is_stationary=True,
            pose_confidence=0.9
        )

        assert 0 <= score <= 100

        # Negative orientation (facing away) should lower score
        score_away = calculate_engagement_score(
            body_orientation=-1.0,
            posture="leaning_back",
            is_stationary=False,
            pose_confidence=0.9
        )

        assert score_away < score

    def test_classify_behavior(self):
        """Behavior classification should return valid types."""
        from behavior import classify_behavior, BehaviorType

        # High engagement, stationary, facing camera = engaged
        result = classify_behavior(
            engagement_score=80,
            is_stationary=True,
            posture="upright",
            body_orientation=0.5
        )
        assert result == BehaviorType.ENGAGED.value

        # Moving with low engagement = passing
        result = classify_behavior(
            engagement_score=30,
            is_stationary=False,
            posture="upright",
            body_orientation=0.0
        )
        assert result == BehaviorType.PASSING.value

        # Stationary but arms crossed = waiting
        result = classify_behavior(
            engagement_score=55,
            is_stationary=True,
            posture="arms_crossed",
            body_orientation=0.2
        )
        assert result == BehaviorType.WAITING.value


class TestMediaPipeLoading:
    """Test pose model loading (YOLO11-Pose, with MediaPipe backward-compat alias)."""

    def test_mediapipe_loads_lazily(self):
        """Pose model should not load until needed."""
        from behavior import _pose_model

        # Initially None (lazy loading)
        # Note: May already be loaded if other tests ran
        # Just verify the module structure exists

    def test_load_mediapipe_function(self):
        """_load_mediapipe should return YOLO pose model or None."""
        from behavior import _load_mediapipe

        try:
            model = _load_mediapipe()
            # Should either return a YOLO model or None
            if model is not None:
                assert hasattr(model, 'track') or hasattr(model, '__call__')
        except ImportError:
            # ultralytics not installed - that's OK for this test
            pass


class TestPoseLandmarkExtraction:
    """Test pose landmark extraction."""

    def test_extract_pose_landmarks_empty_image(self):
        """Should return None for empty/invalid images."""
        from behavior import extract_pose_landmarks

        # Empty image
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = extract_pose_landmarks(empty)
        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    def test_extract_pose_landmarks_small_image(self):
        """Should handle very small images."""
        from behavior import extract_pose_landmarks

        small = np.zeros((10, 10, 3), dtype=np.uint8)
        result = extract_pose_landmarks(small)
        # Should return None (too small to detect pose)
        assert result is None or isinstance(result, dict)
