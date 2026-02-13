import os

# Video source
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "")  # file path or rtsp:// URL
CAMERA_ID = os.getenv("CAMERA_ID", "camera_01")

# Detection
# Stock COCO (Week 1): "yolo11s.pt"
# Fine-tuned bar model (Week 2+): "yolo11s-bar.pt" (run training/train.py first)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11s.pt")
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
PERSON_TRACKER = "botsort.yaml"   # appearance-based, handles long occlusion
OBJECT_TRACKER = "bytetrack.yaml" # motion-based, fast for drink objects

# COCO class IDs (stock model)
COCO_PERSON_CLASS_ID = 0
COCO_DRINK_CLASS_IDS = [39, 40, 41]  # bottle, wine_glass, cup
COCO_DRINK_CLASS_NAMES = {39: "bottle", 40: "wine_glass", 41: "cup"}

# Bar model class IDs (fine-tuned, from training/config.py FINAL_CLASSES)
BAR_PERSON_CLASS_ID = 0
BAR_DRINK_CLASS_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10]  # all glass + bottle types
BAR_DRINK_CLASS_NAMES = {
    1: "beer_glass", 2: "wine_glass", 3: "rocks_glass",
    4: "shot_glass", 5: "cocktail_glass", 6: "pint_glass",
    8: "liquor_bottle", 9: "beer_bottle", 10: "wine_bottle",
}

# Active config (switches based on YOLO_MODEL)
USE_BAR_MODEL = "bar" in YOLO_MODEL
PERSON_CLASS_ID = BAR_PERSON_CLASS_ID if USE_BAR_MODEL else COCO_PERSON_CLASS_ID
DRINK_CLASS_IDS = BAR_DRINK_CLASS_IDS if USE_BAR_MODEL else COCO_DRINK_CLASS_IDS
DRINK_CLASS_NAMES = BAR_DRINK_CLASS_NAMES if USE_BAR_MODEL else COCO_DRINK_CLASS_NAMES

# Zones
ZONES_FILE = os.getenv("ZONES_FILE", "config/zones.json")

# Interaction detection
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.1"))
TRIGGER_COOLDOWN_SECONDS = float(os.getenv("TRIGGER_COOLDOWN_SECONDS", "10.0"))
OBJECT_VELOCITY_THRESHOLD = float(os.getenv("OBJECT_VELOCITY_THRESHOLD", "2.0"))

# VLM verification
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_VLM_CALLS_PER_HOUR = int(os.getenv("MAX_VLM_CALLS_PER_HOUR", "100"))
CLIP_DURATION_SECONDS = float(os.getenv("CLIP_DURATION_SECONDS", "5.0"))
MAX_FRAMES_PER_CLIP = int(os.getenv("MAX_FRAMES_PER_CLIP", "4"))

# Event deduplication
DEDUP_WINDOW_SECONDS = float(os.getenv("DEDUP_WINDOW_SECONDS", "10.0"))

# Event API
EVENT_API_URL = os.getenv("EVENT_API_URL", "http://localhost:8000/events/batch")
EVENT_API_KEY = os.getenv("EVENT_API_KEY", "")

# Clip storage
CLIP_SAVE_DIR = os.getenv("CLIP_SAVE_DIR", "./confirmed_clips")

# Privacy
STRICT_MODE = os.getenv("STRICT_MODE", "false").lower() == "true"
