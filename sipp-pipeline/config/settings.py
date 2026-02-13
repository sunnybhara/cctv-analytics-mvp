import os

# Video source
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "")  # file path or rtsp:// URL
CAMERA_ID = os.getenv("CAMERA_ID", "camera_01")

# Detection
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11s.pt")  # stock COCO model
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
PERSON_TRACKER = "botsort.yaml"   # appearance-based, handles long occlusion
OBJECT_TRACKER = "bytetrack.yaml" # motion-based, fast for drink objects

# Object classes of interest (COCO class IDs)
# 0=person, 39=bottle, 40=wine_glass, 41=cup, 42=fork, 43=knife
PERSON_CLASS_ID = 0
DRINK_CLASS_IDS = [39, 40, 41]  # bottle, wine_glass, cup
DRINK_CLASS_NAMES = {39: "bottle", 40: "wine_glass", 41: "cup"}

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
