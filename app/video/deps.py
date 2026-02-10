"""
Lazy-Loaded Video Dependencies
===============================
Deferred imports for heavy ML libraries. Only loaded when video processing starts.
"""

# Lazy-loaded modules
cv2 = None
YOLO = None
yt_dlp = None

# ML modules
reid_module = None
behavior_module = None


def load_video_deps():
    """Lazy load video processing dependencies."""
    global cv2, YOLO, yt_dlp, reid_module, behavior_module

    if cv2 is None:
        import cv2 as _cv2
        cv2 = _cv2
    if YOLO is None:
        from ultralytics import YOLO as _YOLO
        YOLO = _YOLO
    if yt_dlp is None:
        import yt_dlp as _yt_dlp
        yt_dlp = _yt_dlp
    if reid_module is None:
        try:
            import reid as _reid
            reid_module = _reid
        except Exception as e:
            print(f"Warning: Could not load ReID module: {e}")
            reid_module = None
    if behavior_module is None:
        try:
            import behavior as _behavior
            behavior_module = _behavior
        except Exception as e:
            print(f"Warning: Could not load Behavior module: {e}")
            behavior_module = None
