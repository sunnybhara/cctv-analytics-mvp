"""
ML Model Management
===================
YOLO model cache, shared InsightFace singleton, preloading, and warmup.
"""

import numpy as np

from app.video.deps import load_video_deps, YOLO, reid_module, behavior_module

# Pre-loaded model cache
_yolo_model = None
_shared_insightface = None


def get_shared_insightface():
    """Get or create the shared InsightFace FaceAnalysis singleton.

    Used by both demographics (age/gender) and reid (embedding) to avoid
    loading the ~250MB buffalo_l model twice.
    """
    global _shared_insightface
    if _shared_insightface is None:
        try:
            from insightface.app import FaceAnalysis
            _shared_insightface = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            _shared_insightface.prepare(ctx_id=-1, det_size=(640, 640))
            print("Shared InsightFace buffalo_l loaded (single instance)")
        except Exception as e:
            print(f"Failed to load InsightFace: {e}")
    return _shared_insightface


def analyze_face_single_pass(person_crop):
    """Run InsightFace once and return demographics + embedding.

    Returns:
        (age_bracket, gender, embedding, quality) — any can be None/0.
    """
    app = get_shared_insightface()
    if app is None or person_crop is None or person_crop.size == 0:
        return None, None, None, 0.0

    if person_crop.shape[0] < 50 or person_crop.shape[1] < 30:
        return None, None, None, 0.0

    try:
        faces = app.get(person_crop)
        if not faces:
            return None, None, None, 0.0

        best = max(faces, key=lambda f: f.det_score)
        det_score = float(best.det_score)
        if det_score < 0.5:
            return None, None, None, 0.0

        # Demographics
        from demographics import _age_to_bracket, _gender_to_label
        age_bracket = _age_to_bracket(float(best.age))
        gender = _gender_to_label(int(best.gender))

        # Embedding
        embedding = best.normed_embedding.astype(np.float32)

        return age_bracket, gender, embedding, det_score

    except Exception as e:
        print(f"Face analysis error: {e}")
        return None, None, None, 0.0


def get_yolo_model():
    """Get pre-loaded YOLO model or load on demand."""
    global _yolo_model
    if _yolo_model is None:
        load_video_deps()
        from app.video.deps import YOLO as _YOLO
        _yolo_model = _YOLO("yolo11n.pt")
    return _yolo_model


def preload_models():
    """
    Pre-load all ML models on startup to eliminate cold start latency.
    First video will process immediately instead of waiting 30-60s for model loading.
    """
    global _yolo_model

    print("=" * 50)
    print("Pre-loading ML models...")
    print("=" * 50)

    # 1. Load video processing dependencies
    print("[1/6] Loading video dependencies...")
    load_video_deps()
    from app.video.deps import YOLO as _YOLO, reid_module as _reid, behavior_module as _behavior
    print("      OK cv2, YOLO class, yt-dlp, reid, behavior")

    # 2. Load YOLO model weights
    print("[2/6] Loading YOLO model weights...")
    try:
        if _YOLO is not None and _yolo_model is None:
            _yolo_model = _YOLO("yolo11n.pt")
            print("      OK YOLO11n loaded")
    except Exception as e:
        print(f"      FAIL YOLO: {e}")

    # 3. Load shared InsightFace (used by both ReID and demographics)
    print("[3/6] Loading InsightFace (shared: ReID + demographics)...")
    try:
        face_app = get_shared_insightface()
        if face_app is not None:
            print("      OK InsightFace buffalo_l (single instance for ReID + demographics)")
        else:
            print("      FAIL InsightFace not available")
    except Exception as e:
        print(f"      FAIL InsightFace: {e}")

    # 4. Wire shared instance into reid and demographics
    print("[4/6] Wiring shared InsightFace...")
    try:
        if _reid is not None:
            _reid._load_insightface()
        from demographics import _get_face_analyzer
        _get_face_analyzer()
        print("      OK reid + demographics wired to shared instance")
    except Exception as e:
        print(f"      FAIL wiring: {e}")

    # 5. Load behavior/pose model (YOLO11-Pose)
    print("[5/6] Loading behavior model (YOLO11-Pose)...")
    try:
        if _behavior is not None:
            _behavior._load_mediapipe()  # Backward-compatible alias for _load_pose_model()
            print("      OK YOLO11-Pose loaded")
    except Exception as e:
        print(f"      FAIL Behavior: {e}")

    # 6. Warmup inference
    print("[6/6] Warmup inference...")
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        if _yolo_model is not None:
            _yolo_model(dummy, verbose=False)
            print("      OK YOLO warmup complete")
    except Exception as e:
        print(f"      FAIL Warmup: {e}")

    print("=" * 50)
    print("Model pre-loading complete!")
    print("=" * 50)
