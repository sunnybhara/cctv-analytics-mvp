"""
Demographics Module - Phase 2
=============================
Real age and gender detection using:
- OpenCV DNN for face detection (no extra dependencies)
- HuggingFace ViT models for age/gender classification
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image
import os

# Lazy-loaded models
_face_detector = None
_age_classifier = None
_gender_classifier = None
_models_loaded = False


def _load_face_detector():
    """
    Load OpenCV's DNN face detector.
    Uses the built-in Haar cascade as fallback.
    """
    global _face_detector

    if _face_detector is not None:
        return _face_detector

    # Try to use Haar cascade (built into OpenCV, no download needed)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    _face_detector = cv2.CascadeClassifier(cascade_path)

    return _face_detector


def _load_age_classifier():
    """Lazy load age classification model."""
    global _age_classifier
    if _age_classifier is None:
        try:
            from transformers import pipeline
            import torch

            # Determine device
            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1

            _age_classifier = pipeline(
                "image-classification",
                model="nateraw/vit-age-classifier",
                device=device
            )
        except Exception as e:
            print(f"Failed to load age classifier: {e}")
            _age_classifier = None
    return _age_classifier


def _load_gender_classifier():
    """Lazy load gender classification model."""
    global _gender_classifier
    if _gender_classifier is None:
        try:
            from transformers import pipeline
            import torch

            # Determine device
            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1

            _gender_classifier = pipeline(
                "image-classification",
                model="rizvandwiki/gender-classification",
                device=device
            )
        except Exception as e:
            print(f"Failed to load gender classifier: {e}")
            _gender_classifier = None
    return _gender_classifier


def detect_faces_in_crop(person_crop: np.ndarray) -> List[Image.Image]:
    """
    Detect faces within a person crop using OpenCV.

    Args:
        person_crop: BGR image array of the person

    Returns:
        List of face crops (RGB PIL Images)
    """
    detector = _load_face_detector()

    if detector is None:
        return []

    # Convert to grayscale for detection
    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return []

    face_crops = []
    for (x, y, w, h) in faces:
        # Add some padding around the face
        pad = int(w * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(person_crop.shape[1], x + w + pad)
        y2 = min(person_crop.shape[0], y + h + pad)

        # Extract face crop
        face_bgr = person_crop[y1:y2, x1:x2]

        # Convert BGR to RGB and to PIL
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        face_crops.append(face_pil)

    return face_crops


def classify_age(face_image: Image.Image) -> Optional[str]:
    """
    Classify age from face image.

    Returns:
        Age bracket: "20s", "30s", "40s", "50+"
    """
    classifier = _load_age_classifier()
    if classifier is None:
        return None

    try:
        results = classifier(face_image)

        if not results:
            return None

        # Get top prediction
        top = results[0]
        label = top['label']
        confidence = top['score']

        if confidence < 0.3:
            return None

        # Map model labels to our age brackets
        # nateraw/vit-age-classifier labels
        age_mapping = {
            '0-2': '20s',
            '3-9': '20s',
            '10-19': '20s',
            '20-29': '20s',
            '30-39': '30s',
            '40-49': '40s',
            '50-59': '50+',
            '60-69': '50+',
            '70+': '50+',
            'more than 70': '50+'
        }

        return age_mapping.get(label, '30s')

    except Exception as e:
        print(f"Age classification error: {e}")
        return None


def classify_gender(face_image: Image.Image) -> Optional[str]:
    """
    Classify gender from face image.

    Returns:
        "M" or "F", or None if uncertain
    """
    classifier = _load_gender_classifier()
    if classifier is None:
        return None

    try:
        results = classifier(face_image)

        if not results:
            return None

        # Get top prediction
        top = results[0]
        label = top['label'].lower()
        confidence = top['score']

        if confidence < 0.6:
            return None

        if 'male' in label and 'female' not in label:
            return 'M'
        elif 'female' in label:
            return 'F'
        elif 'man' in label and 'woman' not in label:
            return 'M'
        elif 'woman' in label:
            return 'F'
        else:
            return None

    except Exception as e:
        print(f"Gender classification error: {e}")
        return None


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

    # Detect faces
    face_crops = detect_faces_in_crop(person_crop)

    if not face_crops:
        return None, None

    # Use the first/largest face
    face = face_crops[0]

    # Classify
    age = classify_age(face)
    gender = classify_gender(face)

    return age, gender


def test_demographics():
    """Test the demographics module."""
    print("Testing demographics module...")

    # Test face detector loading
    detector = _load_face_detector()
    print(f"Face detector loaded: {detector is not None}")

    # Create test image (won't have a face, but tests the pipeline)
    dummy = np.zeros((200, 100, 3), dtype=np.uint8)
    dummy[:] = (128, 128, 128)

    age, gender = estimate_demographics(dummy)
    print(f"Dummy image result: age={age}, gender={gender}")
    print("(Expected: None, None - no face in dummy image)")

    # Test classifier loading (this will download models)
    print("\nLoading age classifier (may download model)...")
    age_clf = _load_age_classifier()
    print(f"Age classifier loaded: {age_clf is not None}")

    print("\nLoading gender classifier (may download model)...")
    gender_clf = _load_gender_classifier()
    print(f"Gender classifier loaded: {gender_clf is not None}")

    print("\nDemographics module ready!")


if __name__ == "__main__":
    test_demographics()
