"""
ML Model Management
===================
YOLO model cache, preloading, and warmup.
"""

import numpy as np

from app.video.deps import load_video_deps, YOLO, reid_module, behavior_module

# Pre-loaded model cache
_yolo_model = None


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

    # 3. Load ReID models (InsightFace SCRFD + ArcFace)
    print("[3/6] Loading ReID models...")
    try:
        if _reid is not None:
            _reid._load_insightface()
            print("      OK InsightFace (SCRFD + ArcFace) loaded")
    except Exception as e:
        print(f"      FAIL ReID: {e}")

    # 4. Load demographics models (InsightFace age + gender)
    print("[4/6] Loading demographics models...")
    try:
        from demographics import _load_face_detector
        _load_face_detector()
        print("      OK InsightFace demographics loaded")
    except Exception as e:
        print(f"      FAIL Demographics: {e}")

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
