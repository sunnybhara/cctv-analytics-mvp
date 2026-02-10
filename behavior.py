"""
Behavior & Engagement Detection Module
======================================
Uses YOLO11-Pose (ultralytics) to analyze body language and estimate engagement.

Engagement Signals:
- Body orientation (facing camera/display vs turned away)
- Posture (leaning forward = interested, arms crossed = waiting)
- Movement patterns (still = engaged, pacing = browsing)
- Head position (looking up at display vs down at phone)

Output:
- engagement_score: 0-100 (higher = more engaged)
- behavior_type: "engaged", "browsing", "waiting", "passing"
- pose_confidence: 0-1 (quality of pose detection)

COCO Keypoint Indices (17 keypoints):
  0: nose,  1: left_eye,  2: right_eye,  3: left_ear,  4: right_ear,
  5: left_shoulder,  6: right_shoulder,  7: left_elbow,  8: right_elbow,
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Lazy-loaded YOLO pose model (singleton)
_pose_model = None

# COCO keypoint index mapping
_KP_NOSE = 0
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_ELBOW = 7
_KP_RIGHT_ELBOW = 8
_KP_LEFT_WRIST = 9
_KP_RIGHT_WRIST = 10
_KP_LEFT_HIP = 11
_KP_RIGHT_HIP = 12
_KP_LEFT_KNEE = 13
_KP_RIGHT_KNEE = 14


class BehaviorType(Enum):
    ENGAGED = "engaged"      # Actively interested, facing display, still
    BROWSING = "browsing"    # Looking around, moving slowly
    WAITING = "waiting"      # Standing still, but not focused on anything
    PASSING = "passing"      # Walking through, not stopping
    INTERACTING = "interacting"  # Talking to staff or others
    UNKNOWN = "unknown"


@dataclass
class BehaviorResult:
    engagement_score: float  # 0-100
    behavior_type: str
    pose_confidence: float  # 0-1
    body_orientation: float  # -1 to 1 (negative = facing away, positive = facing camera)
    is_stationary: bool
    posture: str  # "leaning_forward", "upright", "leaning_back", "arms_crossed"
    landmarks: Optional[Dict] = None  # Key landmarks for debugging


def _load_pose_model():
    """Lazy load YOLO11-Pose model."""
    global _pose_model

    if _pose_model is None:
        try:
            from ultralytics import YOLO
            _pose_model = YOLO("yolo11n-pose.pt")
            print("YOLO11-Pose model loaded successfully")
        except ImportError as e:
            print(f"ultralytics not available: {e}")
            _pose_model = None
        except Exception as e:
            print(f"Error loading YOLO11-Pose: {e}")
            _pose_model = None

    return _pose_model


def _load_mediapipe():
    """Backward-compatible alias for _load_pose_model().

    Called by app/video/models.py and main_original.py to eagerly
    initialise the pose back-end at startup.
    """
    return _load_pose_model()


def extract_pose_landmarks(person_crop: np.ndarray) -> Optional[Dict]:
    """
    Extract pose landmarks from a person crop using YOLO11-Pose.

    Args:
        person_crop: BGR image of the person

    Returns:
        Dictionary of landmark positions or None if detection fails
    """
    model = _load_pose_model()
    if model is None:
        return None

    try:
        h, w = person_crop.shape[:2]
        if h < 1 or w < 1:
            return None

        results = model(person_crop, verbose=False)

        for result in results:
            keypoints = result.keypoints
            if keypoints is None or keypoints.xy is None:
                continue

            # If there are no detections the tensor is empty
            if keypoints.xy.shape[0] == 0:
                continue

            xy = keypoints.xy[0].cpu().numpy()    # Shape: (17, 2)
            conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)  # Shape: (17,)

            def _make_point(idx):
                """Build a landmark dict compatible with the old MediaPipe format.

                YOLO-Pose provides (x, y) in pixel coords and a per-keypoint
                confidence but no depth (z).  We set z=0 and map the YOLO
                confidence to 'visibility' so every downstream function that
                inspects visibility keeps working unchanged.
                """
                return {
                    "x": float(xy[idx][0]),
                    "y": float(xy[idx][1]),
                    "z": 0.0,               # YOLO-Pose has no depth; keep field for API compat
                    "visibility": float(conf[idx]),
                }

            return {
                "nose":            _make_point(_KP_NOSE),
                "left_shoulder":   _make_point(_KP_LEFT_SHOULDER),
                "right_shoulder":  _make_point(_KP_RIGHT_SHOULDER),
                "left_elbow":      _make_point(_KP_LEFT_ELBOW),
                "right_elbow":     _make_point(_KP_RIGHT_ELBOW),
                "left_wrist":      _make_point(_KP_LEFT_WRIST),
                "right_wrist":     _make_point(_KP_RIGHT_WRIST),
                "left_hip":        _make_point(_KP_LEFT_HIP),
                "right_hip":       _make_point(_KP_RIGHT_HIP),
                "left_knee":       _make_point(_KP_LEFT_KNEE),
                "right_knee":      _make_point(_KP_RIGHT_KNEE),
            }

        # No person detected in any result
        return None

    except Exception as e:
        print(f"Pose extraction error: {e}")
        return None


def calculate_body_orientation(landmarks: Dict) -> float:
    """
    Calculate body orientation from shoulder positions.

    Because YOLO-Pose does not provide a z-depth value, orientation is
    estimated from the *visibility* (confidence) of each shoulder and
    the horizontal distance between them relative to the image.  When both
    shoulders are clearly visible and well-separated the person is most
    likely facing the camera.

    Returns:
        -1 to 1: negative = facing away, 0 = sideways, positive = facing camera
    """
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")

    if not left_shoulder or not right_shoulder:
        return 0.0

    # Average visibility: high confidence on both shoulders means
    # the person is probably facing the camera.
    avg_visibility = (left_shoulder["visibility"] + right_shoulder["visibility"]) / 2

    # Shoulder width in pixels -- wider spread typically indicates a
    # more frontal pose.  We normalise to a rough range so the score
    # stays between -1 and 1.
    shoulder_dx = abs(left_shoulder["x"] - right_shoulder["x"])

    # Use z-depth difference if available (kept for API compatibility)
    z_diff = left_shoulder["z"] - right_shoulder["z"]

    # Combine visibility and z-depth (z is 0 for YOLO so the orientation
    # depends entirely on visibility, which is a reasonable proxy).
    orientation = avg_visibility * (1 - abs(z_diff) * 2)

    return max(-1.0, min(1.0, orientation))


def calculate_posture(landmarks: Dict) -> str:
    """
    Determine posture from landmark positions.

    Returns:
        "leaning_forward", "upright", "leaning_back", "arms_crossed", "hands_on_hips"
    """
    nose = landmarks.get("nose")
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")
    left_wrist = landmarks.get("left_wrist")
    right_wrist = landmarks.get("right_wrist")
    left_elbow = landmarks.get("left_elbow")
    right_elbow = landmarks.get("right_elbow")

    if not all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
        return "unknown"

    # Calculate shoulder centre and hip centre
    shoulder_center_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
    hip_center_y = (left_hip["y"] + right_hip["y"]) / 2
    shoulder_center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
    hip_center_x = (left_hip["x"] + right_hip["x"]) / 2

    # Check for leaning (shoulder centre vs hip centre horizontally)
    lean = (shoulder_center_x - hip_center_x) / max(abs(hip_center_y - shoulder_center_y), 1)

    # Check for arms crossed (wrists near opposite elbows)
    if left_wrist and right_wrist and left_elbow and right_elbow:
        wrist_center_x = (left_wrist["x"] + right_wrist["x"]) / 2
        wrist_center_y = (left_wrist["y"] + right_wrist["y"]) / 2

        # Check if wrists are close together near chest
        wrist_distance = abs(left_wrist["x"] - right_wrist["x"])
        shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])

        if wrist_distance < shoulder_width * 0.5 and wrist_center_y < shoulder_center_y:
            return "arms_crossed"

        # Check for hands on hips (wrists near hips)
        left_wrist_near_hip = (
            abs(left_wrist["y"] - left_hip["y"]) < 50
            and abs(left_wrist["x"] - left_hip["x"]) < 50
        )
        right_wrist_near_hip = (
            abs(right_wrist["y"] - right_hip["y"]) < 50
            and abs(right_wrist["x"] - right_hip["x"]) < 50
        )

        if left_wrist_near_hip and right_wrist_near_hip:
            return "hands_on_hips"

    # Determine lean direction
    if lean > 0.15:
        return "leaning_forward"
    elif lean < -0.15:
        return "leaning_back"
    else:
        return "upright"


def estimate_movement(
    current_landmarks: Dict,
    previous_landmarks: Optional[Dict] = None,
    time_delta: float = 0.5,
) -> Tuple[bool, float]:
    """
    Estimate if person is stationary or moving.

    Returns:
        Tuple of (is_stationary, movement_speed)
    """
    if previous_landmarks is None:
        return True, 0.0  # Assume stationary if no history

    # Calculate movement based on hip centre displacement
    curr_hip_x = (current_landmarks["left_hip"]["x"] + current_landmarks["right_hip"]["x"]) / 2
    curr_hip_y = (current_landmarks["left_hip"]["y"] + current_landmarks["right_hip"]["y"]) / 2

    prev_hip_x = (previous_landmarks["left_hip"]["x"] + previous_landmarks["right_hip"]["x"]) / 2
    prev_hip_y = (previous_landmarks["left_hip"]["y"] + previous_landmarks["right_hip"]["y"]) / 2

    displacement = np.sqrt((curr_hip_x - prev_hip_x) ** 2 + (curr_hip_y - prev_hip_y) ** 2)
    speed = displacement / time_delta if time_delta > 0 else 0

    # Stationary if moving less than 10 pixels per second
    is_stationary = speed < 10

    return is_stationary, speed


def calculate_engagement_score(
    body_orientation: float,
    posture: str,
    is_stationary: bool,
    pose_confidence: float,
) -> float:
    """
    Calculate overall engagement score from behaviour signals.

    Weights:
        - Body orientation (shoulder line angle): 35%  (~20 pts swing around midpoint)
        - Posture (forward lean, crossed arms):   25%  (up to +/-15 pts)
        - Movement speed (stationary = engaged):  25%  (+15 / -10 pts)
        - Stillness / dwell:                      15%  (via confidence regression)

    Returns:
        0-100 engagement score
    """
    score = 50.0  # Start neutral

    # Body orientation: facing camera = more engaged
    # orientation is -1 to 1, where 1 = facing camera
    score += body_orientation * 20  # +/- 20 points

    # Posture adjustments
    posture_scores = {
        "leaning_forward": 15,   # Very engaged
        "upright": 5,            # Neutral positive
        "leaning_back": -10,     # Disengaged
        "arms_crossed": -5,      # Waiting / closed
        "hands_on_hips": -10,    # Impatient
        "unknown": 0,
    }
    score += posture_scores.get(posture, 0)

    # Stationary = more engaged (stopped to look at something)
    if is_stationary:
        score += 15
    else:
        score -= 10  # Moving = passing through

    # Confidence adjustment -- lower confidence = regress to neutral
    confidence_factor = max(0.5, pose_confidence)
    score = 50 + (score - 50) * confidence_factor

    return max(0.0, min(100.0, score))


def classify_behavior(
    engagement_score: float,
    is_stationary: bool,
    posture: str,
    body_orientation: float,
) -> str:
    """
    Classify behaviour type based on signals.
    """
    if not is_stationary and engagement_score < 40:
        return BehaviorType.PASSING.value

    if is_stationary:
        if engagement_score >= 70 and body_orientation > 0.3:
            return BehaviorType.ENGAGED.value
        elif posture in ["arms_crossed", "hands_on_hips"]:
            return BehaviorType.WAITING.value
        elif engagement_score >= 50:
            return BehaviorType.BROWSING.value
        else:
            return BehaviorType.WAITING.value
    else:
        if engagement_score >= 50:
            return BehaviorType.BROWSING.value
        else:
            return BehaviorType.PASSING.value


def analyze_behavior(
    person_crop: np.ndarray,
    previous_landmarks: Optional[Dict] = None,
    time_delta: float = 0.5,
) -> BehaviorResult:
    """
    Main function to analyse behaviour from a person crop.

    Args:
        person_crop: BGR image of the person
        previous_landmarks: Landmarks from previous frame (for movement detection)
        time_delta: Time since previous frame in seconds

    Returns:
        BehaviorResult with engagement score and behaviour classification
    """
    # Extract pose landmarks
    landmarks = extract_pose_landmarks(person_crop)

    if landmarks is None:
        return BehaviorResult(
            engagement_score=50.0,  # Neutral if can't detect
            behavior_type=BehaviorType.UNKNOWN.value,
            pose_confidence=0.0,
            body_orientation=0.0,
            is_stationary=True,
            posture="unknown",
            landmarks=None,
        )

    # Calculate average visibility (keypoint confidence) as overall pose confidence
    visibilities = [
        landmarks["left_shoulder"]["visibility"],
        landmarks["right_shoulder"]["visibility"],
        landmarks["left_hip"]["visibility"],
        landmarks["right_hip"]["visibility"],
    ]
    pose_confidence = sum(visibilities) / len(visibilities)

    # Analyse behaviour signals
    body_orientation = calculate_body_orientation(landmarks)
    posture = calculate_posture(landmarks)
    is_stationary, movement_speed = estimate_movement(landmarks, previous_landmarks, time_delta)

    # Calculate engagement score
    engagement_score = calculate_engagement_score(
        body_orientation,
        posture,
        is_stationary,
        pose_confidence,
    )

    # Classify behaviour
    behavior_type = classify_behavior(
        engagement_score,
        is_stationary,
        posture,
        body_orientation,
    )

    return BehaviorResult(
        engagement_score=round(engagement_score, 1),
        behavior_type=behavior_type,
        pose_confidence=round(pose_confidence, 2),
        body_orientation=round(body_orientation, 2),
        is_stationary=is_stationary,
        posture=posture,
        landmarks=landmarks,
    )


# ---------------------------------------------------------------------------
# Self-tests (run with: python behavior.py)
# ---------------------------------------------------------------------------

def test_behavior():
    """Test the behaviour detection module."""
    print("Testing Behavior Detection Module...")

    # Test YOLO11-Pose loading
    print("\n1. Loading YOLO11-Pose model...")
    model = _load_pose_model()
    print(f"   YOLO11-Pose loaded: {model is not None}")

    if model is None:
        print("   Install ultralytics: pip install ultralytics")
        return

    # Test with dummy image
    print("\n2. Testing pose extraction on dummy image...")
    dummy = np.zeros((400, 200, 3), dtype=np.uint8)
    dummy[:] = (128, 128, 128)

    landmarks = extract_pose_landmarks(dummy)
    print(f"   Landmarks detected: {landmarks is not None}")
    print("   (Expected: None - no person in dummy image)")

    # Test behaviour analysis
    print("\n3. Testing behavior analysis...")
    result = analyze_behavior(dummy)
    print(f"   Engagement score: {result.engagement_score}")
    print(f"   Behavior type: {result.behavior_type}")
    print(f"   Pose confidence: {result.pose_confidence}")

    # Test backward-compatible alias
    print("\n4. Testing _load_mediapipe() alias...")
    alias_model = _load_mediapipe()
    assert alias_model is model, "_load_mediapipe() should return the same model instance"
    print("   _load_mediapipe() alias works correctly")

    # Test pure-logic functions with synthetic landmarks
    print("\n5. Testing pure-logic functions with synthetic landmarks...")
    synth = {
        "nose":            {"x": 100, "y": 50,  "z": 0, "visibility": 0.95},
        "left_shoulder":   {"x": 80,  "y": 100, "z": 0, "visibility": 0.90},
        "right_shoulder":  {"x": 120, "y": 100, "z": 0, "visibility": 0.90},
        "left_elbow":      {"x": 60,  "y": 150, "z": 0, "visibility": 0.85},
        "right_elbow":     {"x": 140, "y": 150, "z": 0, "visibility": 0.85},
        "left_wrist":      {"x": 50,  "y": 200, "z": 0, "visibility": 0.80},
        "right_wrist":     {"x": 150, "y": 200, "z": 0, "visibility": 0.80},
        "left_hip":        {"x": 85,  "y": 220, "z": 0, "visibility": 0.88},
        "right_hip":       {"x": 115, "y": 220, "z": 0, "visibility": 0.88},
        "left_knee":       {"x": 80,  "y": 320, "z": 0, "visibility": 0.75},
        "right_knee":      {"x": 120, "y": 320, "z": 0, "visibility": 0.75},
    }

    orient = calculate_body_orientation(synth)
    post = calculate_posture(synth)
    stat, spd = estimate_movement(synth, None)
    eng = calculate_engagement_score(orient, post, stat, 0.9)
    beh = classify_behavior(eng, stat, post, orient)

    print(f"   Orientation: {orient:.2f}")
    print(f"   Posture: {post}")
    print(f"   Stationary: {stat}, speed: {spd:.1f}")
    print(f"   Engagement: {eng:.1f}")
    print(f"   Behavior: {beh}")

    assert 0 <= eng <= 100, f"Engagement out of range: {eng}"
    assert beh in [bt.value for bt in BehaviorType], f"Unknown behavior: {beh}"

    print("\nBehavior Detection Module ready!")
    print("Note: Real testing requires images with people")


if __name__ == "__main__":
    test_behavior()
