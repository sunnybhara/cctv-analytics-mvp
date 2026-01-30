"""
Behavior & Engagement Detection Module
======================================
Uses MediaPipe Pose to analyze body language and estimate engagement.

Engagement Signals:
- Body orientation (facing camera/display vs turned away)
- Posture (leaning forward = interested, arms crossed = waiting)
- Movement patterns (still = engaged, pacing = browsing)
- Head position (looking up at display vs down at phone)

Output:
- engagement_score: 0-100 (higher = more engaged)
- behavior_type: "engaged", "browsing", "waiting", "passing"
- pose_confidence: 0-1 (quality of pose detection)
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Lazy-loaded MediaPipe
_mp_pose = None
_pose_detector = None


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


def _load_mediapipe():
    """Lazy load MediaPipe Pose."""
    global _mp_pose, _pose_detector

    if _mp_pose is None:
        try:
            import mediapipe as mp
            _mp_pose = mp.solutions.pose
            _pose_detector = _mp_pose.Pose(
                static_image_mode=True,  # For processing individual frames
                model_complexity=1,       # 0=lite, 1=full, 2=heavy
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Pose loaded successfully")
        except ImportError as e:
            print(f"MediaPipe not available: {e}")
            _mp_pose = None
            _pose_detector = None
        except Exception as e:
            print(f"Error loading MediaPipe: {e}")
            _mp_pose = None
            _pose_detector = None

    return _pose_detector


def extract_pose_landmarks(person_crop: np.ndarray) -> Optional[Dict]:
    """
    Extract pose landmarks from a person crop.

    Args:
        person_crop: BGR image of the person

    Returns:
        Dictionary of landmark positions or None if detection fails
    """
    import cv2

    pose = _load_mediapipe()
    if pose is None:
        return None

    try:
        # Convert BGR to RGB
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        # Process image
        results = pose.process(rgb)

        if not results.pose_landmarks:
            return None

        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        h, w = person_crop.shape[:2]

        # Key landmark indices (MediaPipe Pose)
        # 0: nose, 11: left_shoulder, 12: right_shoulder
        # 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
        # 23: left_hip, 24: right_hip

        def get_point(idx):
            lm = landmarks[idx]
            return {
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z,  # Depth (relative)
                "visibility": lm.visibility
            }

        return {
            "nose": get_point(0),
            "left_shoulder": get_point(11),
            "right_shoulder": get_point(12),
            "left_elbow": get_point(13),
            "right_elbow": get_point(14),
            "left_wrist": get_point(15),
            "right_wrist": get_point(16),
            "left_hip": get_point(23),
            "right_hip": get_point(24),
            "left_knee": get_point(25),
            "right_knee": get_point(26),
        }

    except Exception as e:
        print(f"Pose extraction error: {e}")
        return None


def calculate_body_orientation(landmarks: Dict) -> float:
    """
    Calculate body orientation from shoulder positions.

    Returns:
        -1 to 1: negative = facing away, 0 = sideways, positive = facing camera
    """
    left_shoulder = landmarks.get("left_shoulder")
    right_shoulder = landmarks.get("right_shoulder")

    if not left_shoulder or not right_shoulder:
        return 0.0

    # Use Z-depth difference between shoulders
    # If left shoulder is closer (more negative Z), person is facing right
    # If right shoulder is closer, person is facing left
    # If both similar, person is facing camera

    z_diff = left_shoulder["z"] - right_shoulder["z"]

    # Also check visibility - higher visibility = more facing camera
    avg_visibility = (left_shoulder["visibility"] + right_shoulder["visibility"]) / 2

    # Combine Z-depth and visibility for orientation score
    # Z-depth close to 0 = facing camera
    orientation = avg_visibility * (1 - abs(z_diff) * 2)

    return max(-1, min(1, orientation))


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

    # Calculate shoulder center and hip center
    shoulder_center_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
    hip_center_y = (left_hip["y"] + right_hip["y"]) / 2
    shoulder_center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
    hip_center_x = (left_hip["x"] + right_hip["x"]) / 2

    # Check for leaning (shoulder center vs hip center horizontally)
    lean = (shoulder_center_x - hip_center_x) / max(abs(hip_center_y - shoulder_center_y), 1)

    # Check for arms crossed (wrists near opposite elbows)
    if left_wrist and right_wrist and left_elbow and right_elbow:
        # Arms crossed if wrists are close to center body and near chest height
        wrist_center_x = (left_wrist["x"] + right_wrist["x"]) / 2
        wrist_center_y = (left_wrist["y"] + right_wrist["y"]) / 2

        # Check if wrists are close together near chest
        wrist_distance = abs(left_wrist["x"] - right_wrist["x"])
        shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])

        if wrist_distance < shoulder_width * 0.5 and wrist_center_y < shoulder_center_y:
            return "arms_crossed"

        # Check for hands on hips (wrists near hips)
        left_wrist_near_hip = abs(left_wrist["y"] - left_hip["y"]) < 50 and abs(left_wrist["x"] - left_hip["x"]) < 50
        right_wrist_near_hip = abs(right_wrist["y"] - right_hip["y"]) < 50 and abs(right_wrist["x"] - right_hip["x"]) < 50

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
    time_delta: float = 0.5
) -> Tuple[bool, float]:
    """
    Estimate if person is stationary or moving.

    Returns:
        Tuple of (is_stationary, movement_speed)
    """
    if previous_landmarks is None:
        return True, 0.0  # Assume stationary if no history

    # Calculate movement based on hip center displacement
    curr_hip_x = (current_landmarks["left_hip"]["x"] + current_landmarks["right_hip"]["x"]) / 2
    curr_hip_y = (current_landmarks["left_hip"]["y"] + current_landmarks["right_hip"]["y"]) / 2

    prev_hip_x = (previous_landmarks["left_hip"]["x"] + previous_landmarks["right_hip"]["x"]) / 2
    prev_hip_y = (previous_landmarks["left_hip"]["y"] + previous_landmarks["right_hip"]["y"]) / 2

    displacement = np.sqrt((curr_hip_x - prev_hip_x)**2 + (curr_hip_y - prev_hip_y)**2)
    speed = displacement / time_delta if time_delta > 0 else 0

    # Stationary if moving less than 10 pixels per second
    is_stationary = speed < 10

    return is_stationary, speed


def calculate_engagement_score(
    body_orientation: float,
    posture: str,
    is_stationary: bool,
    pose_confidence: float
) -> float:
    """
    Calculate overall engagement score from behavior signals.

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
        "arms_crossed": -5,      # Waiting/closed
        "hands_on_hips": -10,    # Impatient
        "unknown": 0
    }
    score += posture_scores.get(posture, 0)

    # Stationary = more engaged (stopped to look at something)
    if is_stationary:
        score += 15
    else:
        score -= 10  # Moving = passing through

    # Confidence adjustment - lower confidence = regress to neutral
    confidence_factor = max(0.5, pose_confidence)
    score = 50 + (score - 50) * confidence_factor

    return max(0, min(100, score))


def classify_behavior(
    engagement_score: float,
    is_stationary: bool,
    posture: str,
    body_orientation: float
) -> str:
    """
    Classify behavior type based on signals.
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
    time_delta: float = 0.5
) -> BehaviorResult:
    """
    Main function to analyze behavior from a person crop.

    Args:
        person_crop: BGR image of the person
        previous_landmarks: Landmarks from previous frame (for movement detection)
        time_delta: Time since previous frame in seconds

    Returns:
        BehaviorResult with engagement score and behavior classification
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
            landmarks=None
        )

    # Calculate average visibility as confidence
    visibilities = [
        landmarks["left_shoulder"]["visibility"],
        landmarks["right_shoulder"]["visibility"],
        landmarks["left_hip"]["visibility"],
        landmarks["right_hip"]["visibility"],
    ]
    pose_confidence = sum(visibilities) / len(visibilities)

    # Analyze behavior signals
    body_orientation = calculate_body_orientation(landmarks)
    posture = calculate_posture(landmarks)
    is_stationary, movement_speed = estimate_movement(landmarks, previous_landmarks, time_delta)

    # Calculate engagement score
    engagement_score = calculate_engagement_score(
        body_orientation,
        posture,
        is_stationary,
        pose_confidence
    )

    # Classify behavior
    behavior_type = classify_behavior(
        engagement_score,
        is_stationary,
        posture,
        body_orientation
    )

    return BehaviorResult(
        engagement_score=round(engagement_score, 1),
        behavior_type=behavior_type,
        pose_confidence=round(pose_confidence, 2),
        body_orientation=round(body_orientation, 2),
        is_stationary=is_stationary,
        posture=posture,
        landmarks=landmarks
    )


def test_behavior():
    """Test the behavior detection module."""
    print("Testing Behavior Detection Module...")

    # Test MediaPipe loading
    print("\n1. Loading MediaPipe Pose...")
    pose = _load_mediapipe()
    print(f"   MediaPipe loaded: {pose is not None}")

    if pose is None:
        print("   Install mediapipe: pip install mediapipe")
        return

    # Test with dummy image
    print("\n2. Testing pose extraction on dummy image...")
    dummy = np.zeros((400, 200, 3), dtype=np.uint8)
    dummy[:] = (128, 128, 128)

    landmarks = extract_pose_landmarks(dummy)
    print(f"   Landmarks detected: {landmarks is not None}")
    print("   (Expected: None - no person in dummy image)")

    # Test behavior analysis
    print("\n3. Testing behavior analysis...")
    result = analyze_behavior(dummy)
    print(f"   Engagement score: {result.engagement_score}")
    print(f"   Behavior type: {result.behavior_type}")
    print(f"   Pose confidence: {result.pose_confidence}")

    print("\nBehavior Detection Module ready!")
    print("Note: Real testing requires images with people")


if __name__ == "__main__":
    test_behavior()
