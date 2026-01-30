"""
Demographics Module - Production Edition
========================================
Real age and gender detection using:
- MTCNN for accurate face detection (better than Haar cascade)
- ViT-Age-Classifier for age estimation (9 age brackets)
- Gender-Classification model for gender

Both models are well-tested and available on HuggingFace.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from PIL import Image

# Lazy-loaded models
_face_detector = None
_age_classifier = None
_gender_classifier = None


def _load_face_detector():
    """Load MTCNN face detector for better accuracy."""
    global _face_detector

    if _face_detector is not None:
        return _face_detector

    try:
        from facenet_pytorch import MTCNN
        import torch

        # Use CPU for compatibility (MPS has issues with some operations)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _face_detector = MTCNN(
            image_size=224,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=device,
            keep_all=True  # Return all faces
        )
        print("MTCNN face detector loaded")
        return _face_detector

    except ImportError:
        print("MTCNN not available, using Haar cascade fallback")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_detector = cv2.CascadeClassifier(cascade_path)
        return _face_detector


def _load_age_classifier():
    """Load ViT age classification model."""
    global _age_classifier

    if _age_classifier is not None:
        return _age_classifier

    try:
        from transformers import pipeline

        _age_classifier = pipeline(
            "image-classification",
            model="nateraw/vit-age-classifier",
            device=-1  # CPU for compatibility
        )
        print("Age classifier loaded (nateraw/vit-age-classifier)")

    except Exception as e:
        print(f"Failed to load age classifier: {e}")
        _age_classifier = None

    return _age_classifier


def _load_gender_classifier():
    """Load gender classification model."""
    global _gender_classifier

    if _gender_classifier is not None:
        return _gender_classifier

    try:
        from transformers import pipeline

        _gender_classifier = pipeline(
            "image-classification",
            model="rizvandwiki/gender-classification",
            device=-1  # CPU for compatibility
        )
        print("Gender classifier loaded (rizvandwiki/gender-classification)")

    except Exception as e:
        print(f"Failed to load gender classifier: {e}")
        _gender_classifier = None

    return _gender_classifier


def detect_faces(person_crop: np.ndarray) -> List[Dict]:
    """
    Detect faces in a person crop.

    Returns:
        List of dicts with 'box' (x1,y1,x2,y2), 'prob' (confidence), 'pil_image'
    """
    detector = _load_face_detector()

    if detector is None:
        return []

    # Check if MTCNN or Haar cascade
    if hasattr(detector, 'detect'):
        # MTCNN
        try:
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            boxes, probs = detector.detect(pil_image)

            if boxes is None:
                return []

            faces = []
            for box, prob in zip(boxes, probs):
                if prob > 0.9:  # High confidence only
                    x1, y1, x2, y2 = [int(b) for b in box]

                    # Bounds check
                    h, w = person_crop.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        face_rgb = rgb[y1:y2, x1:x2]
                        face_pil = Image.fromarray(face_rgb)
                        faces.append({
                            'box': (x1, y1, x2, y2),
                            'prob': float(prob),
                            'pil_image': face_pil
                        })

            return faces

        except Exception as e:
            print(f"MTCNN detection error: {e}")
            return []

    else:
        # Haar cascade fallback
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        faces = []
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in detections:
            face_rgb = rgb[y:y+h, x:x+w]
            face_pil = Image.fromarray(face_rgb)
            faces.append({
                'box': (x, y, x + w, y + h),
                'prob': 0.9,
                'pil_image': face_pil
            })

        return faces


def classify_age(face_image: Image.Image) -> Tuple[Optional[str], float]:
    """
    Classify age from face image.

    Returns:
        Tuple of (age_bracket, confidence)
    """
    classifier = _load_age_classifier()

    if classifier is None:
        return None, 0.0

    try:
        results = classifier(face_image)

        if not results:
            return None, 0.0

        top = results[0]
        label = top['label']
        confidence = top['score']

        if confidence < 0.3:
            return None, confidence

        # Map to our standard brackets
        age_mapping = {
            '0-2': '0-17',
            '3-9': '0-17',
            '10-19': '10-19',
            '20-29': '20-29',
            '30-39': '30-39',
            '40-49': '40-49',
            '50-59': '50-59',
            '60-69': '60+',
            'more than 70': '60+'
        }

        return age_mapping.get(label, '30-39'), confidence

    except Exception as e:
        print(f"Age classification error: {e}")
        return None, 0.0


def classify_gender(face_image: Image.Image) -> Tuple[Optional[str], float]:
    """
    Classify gender from face image.

    Returns:
        Tuple of (gender, confidence) - gender is 'M' or 'F'
    """
    classifier = _load_gender_classifier()

    if classifier is None:
        return None, 0.0

    try:
        results = classifier(face_image)

        if not results:
            return None, 0.0

        top = results[0]
        label = top['label'].lower()
        confidence = top['score']

        if confidence < 0.6:
            return None, confidence

        if 'male' in label and 'female' not in label:
            return 'M', confidence
        elif 'female' in label:
            return 'F', confidence
        else:
            return None, confidence

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

    # Detect faces
    faces = detect_faces(person_crop)

    if not faces:
        return None, None

    # Use highest confidence face
    best_face = max(faces, key=lambda f: f['prob'])
    face_pil = best_face['pil_image']

    # Classify
    age, age_conf = classify_age(face_pil)
    gender, gender_conf = classify_gender(face_pil)

    return age, gender


def estimate_demographics_detailed(person_crop: np.ndarray) -> Dict:
    """
    Get detailed demographics with confidence scores.

    Returns:
        Dict with age, gender, confidence scores, and face detection info
    """
    if person_crop is None or person_crop.size == 0:
        return {
            'age': None, 'gender': None,
            'age_confidence': 0.0, 'gender_confidence': 0.0,
            'face_detected': False, 'face_confidence': 0.0
        }

    if person_crop.shape[0] < 50 or person_crop.shape[1] < 30:
        return {
            'age': None, 'gender': None,
            'age_confidence': 0.0, 'gender_confidence': 0.0,
            'face_detected': False, 'face_confidence': 0.0
        }

    faces = detect_faces(person_crop)

    if not faces:
        return {
            'age': None, 'gender': None,
            'age_confidence': 0.0, 'gender_confidence': 0.0,
            'face_detected': False, 'face_confidence': 0.0
        }

    best_face = max(faces, key=lambda f: f['prob'])
    face_pil = best_face['pil_image']

    age, age_conf = classify_age(face_pil)
    gender, gender_conf = classify_gender(face_pil)

    return {
        'age': age,
        'gender': gender,
        'age_confidence': age_conf,
        'gender_confidence': gender_conf,
        'face_detected': True,
        'face_confidence': best_face['prob']
    }


def test_demographics():
    """Test the demographics module."""
    print("=" * 50)
    print("Demographics Module Test")
    print("=" * 50)

    # Test face detector
    print("\n1. Loading face detector...")
    detector = _load_face_detector()
    detector_type = 'MTCNN' if hasattr(detector, 'detect') else 'Haar cascade'
    print(f"   Detector: {detector_type}")

    # Test age classifier
    print("\n2. Loading age classifier...")
    age_clf = _load_age_classifier()
    print(f"   Age classifier: {'OK' if age_clf else 'FAILED'}")

    # Test gender classifier
    print("\n3. Loading gender classifier...")
    gender_clf = _load_gender_classifier()
    print(f"   Gender classifier: {'OK' if gender_clf else 'FAILED'}")

    # Test with dummy image
    print("\n4. Testing with dummy image (no face expected)...")
    dummy = np.zeros((200, 100, 3), dtype=np.uint8)
    dummy[:] = (128, 128, 128)

    age, gender = estimate_demographics(dummy)
    print(f"   Result: age={age}, gender={gender}")
    print("   Expected: None, None (no face)")

    # Summary
    print("\n" + "=" * 50)
    all_ok = detector and age_clf and gender_clf
    if all_ok:
        print("Demographics module ready!")
        print(f"- Face detection: {detector_type}")
        print("- Age: nateraw/vit-age-classifier (9 brackets)")
        print("- Gender: rizvandwiki/gender-classification")
    else:
        print("WARNING: Some components failed to load")
    print("=" * 50)


if __name__ == "__main__":
    test_demographics()
