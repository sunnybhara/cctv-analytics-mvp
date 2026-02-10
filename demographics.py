"""
Demographics Module - InsightFace Edition
==========================================
Age and gender detection using InsightFace buffalo_l model which provides
face detection, age estimation, and gender classification in a single pass.

Replaces the previous MTCNN + HuggingFace pipeline approach with a single
unified model for faster and more consistent results.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from PIL import Image

# Shared InsightFace singleton
_face_analyzer = None

# Legacy globals kept for API compatibility (all point to _face_analyzer)
_face_detector = None
_age_classifier = None
_gender_classifier = None


def _get_face_analyzer():
    """Get or create the shared InsightFace FaceAnalysis singleton."""
    global _face_analyzer

    if _face_analyzer is not None:
        return _face_analyzer

    try:
        from insightface.app import FaceAnalysis

        _face_analyzer = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        _face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        print("InsightFace buffalo_l model loaded (face detection + age + gender)")
        return _face_analyzer

    except Exception as e:
        print(f"Failed to load InsightFace model: {e}")
        _face_analyzer = None
        return None


def _load_face_detector():
    """Load the face detector (InsightFace FaceAnalysis).

    Called by app/video/models.py preload_models.
    """
    global _face_detector
    analyzer = _get_face_analyzer()
    _face_detector = analyzer
    return _face_detector


def _load_age_classifier():
    """No-op: InsightFace provides age estimation as part of face analysis.

    Called by app/video/models.py preload_models.
    """
    global _age_classifier
    analyzer = _get_face_analyzer()
    _age_classifier = analyzer
    return _age_classifier


def _load_gender_classifier():
    """No-op: InsightFace provides gender classification as part of face analysis.

    Called by app/video/models.py preload_models.
    """
    global _gender_classifier
    analyzer = _get_face_analyzer()
    _gender_classifier = analyzer
    return _gender_classifier


def _age_to_bracket(age: float) -> str:
    """Map a continuous age value to a bracket string.

    Brackets:
        age < 18    -> '0-17'
        18 <= age < 20 -> '18-24'
        20 <= age < 30 -> '20-29'
        30 <= age < 40 -> '30-39'
        40 <= age < 50 -> '40-49'
        50 <= age < 60 -> '50-59'
        age >= 60      -> '60+'
    """
    if age < 18:
        return '0-17'
    elif age < 20:
        return '18-24'
    elif age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    else:
        return '60+'


def _gender_to_label(gender: int) -> str:
    """Map InsightFace gender int to 'M' or 'F'.

    InsightFace convention: 0 = female, 1 = male.
    """
    return 'M' if gender == 1 else 'F'


def detect_faces(person_crop: np.ndarray) -> List[Dict]:
    """
    Detect faces in a person crop using InsightFace.

    Args:
        person_crop: BGR image array of the person

    Returns:
        List of dicts with 'box' (x1,y1,x2,y2), 'prob' (confidence),
        'pil_image' (PIL face crop), 'age' (float), 'gender' (int 0/1)
    """
    analyzer = _get_face_analyzer()

    if analyzer is None:
        return []

    try:
        # InsightFace expects BGR (OpenCV default), which person_crop already is
        raw_faces = analyzer.get(person_crop)

        if not raw_faces:
            return []

        h, w = person_crop.shape[:2]
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        faces = []
        for face in raw_faces:
            det_score = float(face.det_score)

            if det_score < 0.5:
                continue

            x1, y1, x2, y2 = [int(b) for b in face.bbox]

            # Bounds check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                face_rgb = rgb[y1:y2, x1:x2]
                face_pil = Image.fromarray(face_rgb)
                faces.append({
                    'box': (x1, y1, x2, y2),
                    'prob': det_score,
                    'pil_image': face_pil,
                    'age': float(face.age),
                    'gender': int(face.gender),
                })

        return faces

    except Exception as e:
        print(f"InsightFace detection error: {e}")
        return []


def classify_age(face_image: Image.Image) -> Tuple[Optional[str], float]:
    """
    Classify age from a face image.

    When called after detect_faces(), the age is already known from InsightFace.
    This function runs a fresh InsightFace pass on the provided PIL image to
    extract the age, then maps it to a bracket.

    Args:
        face_image: PIL Image of a face crop

    Returns:
        Tuple of (age_bracket, confidence)
    """
    analyzer = _get_face_analyzer()

    if analyzer is None:
        return None, 0.0

    try:
        # Convert PIL -> BGR numpy for InsightFace
        rgb = np.array(face_image)
        if rgb.ndim == 2:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raw_faces = analyzer.get(bgr)

        if not raw_faces:
            return None, 0.0

        # Use highest-confidence face
        best = max(raw_faces, key=lambda f: f.det_score)
        confidence = float(best.det_score)
        age_bracket = _age_to_bracket(float(best.age))

        return age_bracket, confidence

    except Exception as e:
        print(f"Age classification error: {e}")
        return None, 0.0


def classify_gender(face_image: Image.Image) -> Tuple[Optional[str], float]:
    """
    Classify gender from a face image.

    When called after detect_faces(), the gender is already known from InsightFace.
    This function runs a fresh InsightFace pass on the provided PIL image to
    extract the gender.

    Args:
        face_image: PIL Image of a face crop

    Returns:
        Tuple of (gender, confidence) - gender is 'M' or 'F'
    """
    analyzer = _get_face_analyzer()

    if analyzer is None:
        return None, 0.0

    try:
        # Convert PIL -> BGR numpy for InsightFace
        rgb = np.array(face_image)
        if rgb.ndim == 2:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raw_faces = analyzer.get(bgr)

        if not raw_faces:
            return None, 0.0

        # Use highest-confidence face
        best = max(raw_faces, key=lambda f: f.det_score)
        confidence = float(best.det_score)
        gender = _gender_to_label(int(best.gender))

        return gender, confidence

    except Exception as e:
        print(f"Gender classification error: {e}")
        return None, 0.0


def estimate_demographics(
    person_crop: np.ndarray,
    box_height: float = 0,
    frame_height: float = 0
) -> Tuple[Optional[str], Optional[str]]:
    """
    Estimate demographics from a person crop.

    Uses InsightFace to detect faces and extract age + gender in a single pass.

    Args:
        person_crop: BGR image array of the person
        box_height: Height of bounding box (unused, for API compatibility)
        frame_height: Height of frame (unused, for API compatibility)

    Returns:
        Tuple of (age_bracket, gender) - either can be None if not detected
    """
    if person_crop is None or person_crop.size == 0:
        return None, None

    # Minimum size check
    if person_crop.shape[0] < 50 or person_crop.shape[1] < 30:
        return None, None

    # Detect faces (InsightFace gives age + gender in same call)
    faces = detect_faces(person_crop)

    if not faces:
        return None, None

    # Use highest confidence face
    best_face = max(faces, key=lambda f: f['prob'])

    age_bracket = _age_to_bracket(best_face['age'])
    gender = _gender_to_label(best_face['gender'])

    return age_bracket, gender


def estimate_demographics_detailed(person_crop: np.ndarray) -> Dict:
    """
    Get detailed demographics with confidence scores.

    Returns:
        Dict with age, gender, confidence scores, and face detection info
    """
    empty_result = {
        'age': None, 'gender': None,
        'age_confidence': 0.0, 'gender_confidence': 0.0,
        'face_detected': False, 'face_confidence': 0.0
    }

    if person_crop is None or person_crop.size == 0:
        return empty_result

    if person_crop.shape[0] < 50 or person_crop.shape[1] < 30:
        return empty_result

    faces = detect_faces(person_crop)

    if not faces:
        return empty_result

    best_face = max(faces, key=lambda f: f['prob'])

    age_bracket = _age_to_bracket(best_face['age'])
    gender = _gender_to_label(best_face['gender'])
    det_conf = best_face['prob']

    return {
        'age': age_bracket,
        'gender': gender,
        'age_confidence': det_conf,
        'gender_confidence': det_conf,
        'face_detected': True,
        'face_confidence': det_conf
    }


def test_demographics():
    """Test the demographics module."""
    print("=" * 50)
    print("Demographics Module Test (InsightFace)")
    print("=" * 50)

    # Test face analyzer loading
    print("\n1. Loading InsightFace buffalo_l model...")
    analyzer = _get_face_analyzer()
    print(f"   FaceAnalysis: {'OK' if analyzer else 'FAILED'}")

    # Test loader functions (called by preload_models)
    print("\n2. Testing _load_face_detector()...")
    det = _load_face_detector()
    print(f"   Face detector: {'OK' if det else 'FAILED'}")

    print("\n3. Testing _load_age_classifier() (no-op)...")
    age_clf = _load_age_classifier()
    print(f"   Age classifier: {'OK' if age_clf else 'FAILED'}")

    print("\n4. Testing _load_gender_classifier() (no-op)...")
    gender_clf = _load_gender_classifier()
    print(f"   Gender classifier: {'OK' if gender_clf else 'FAILED'}")

    # Test with dummy image (no face expected)
    print("\n5. Testing with dummy image (no face expected)...")
    dummy = np.zeros((200, 100, 3), dtype=np.uint8)
    dummy[:] = (128, 128, 128)

    age, gender = estimate_demographics(dummy)
    print(f"   Result: age={age}, gender={gender}")
    print("   Expected: None, None (no face)")

    # Test age bracket mapping
    print("\n6. Testing age bracket mapping...")
    test_ages = [5, 15, 19, 25, 35, 45, 55, 65, 80]
    for a in test_ages:
        bracket = _age_to_bracket(a)
        print(f"   age={a} -> bracket='{bracket}'")

    # Summary
    print("\n" + "=" * 50)
    all_ok = analyzer is not None
    if all_ok:
        print("Demographics module ready!")
        print("- Backend: InsightFace buffalo_l")
        print("- Face detection + age + gender in single pass")
    else:
        print("WARNING: InsightFace failed to load")
    print("=" * 50)


if __name__ == "__main__":
    test_demographics()
