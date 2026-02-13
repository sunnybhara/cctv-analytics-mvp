import os

# Paths
RAW_VIDEO_DIR = os.getenv("RAW_VIDEO_DIR", "./raw_footage")
FRAMES_DIR = os.getenv("FRAMES_DIR", "./frames")
AUTODISTILL_OUTPUT = os.getenv("AUTODISTILL_OUTPUT", "./labels_autodistill")
SAHI_OUTPUT = os.getenv("SAHI_OUTPUT", "./labels_sahi")
MERGED_DATASET = os.getenv("MERGED_DATASET", "./dataset")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "./models")
CONFIRMED_CLIPS_DIR = os.getenv("CONFIRMED_CLIPS_DIR", "../confirmed_clips")

# Frame extraction
EXTRACT_FPS = float(os.getenv("EXTRACT_FPS", "1.0"))  # 1 frame per second
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "10000"))

# Grounding DINO / Autodistill
GDINO_BOX_THRESHOLD = float(os.getenv("GDINO_BOX_THRESHOLD", "0.3"))
GDINO_TEXT_THRESHOLD = float(os.getenv("GDINO_TEXT_THRESHOLD", "0.25"))

# SAHI
SAHI_SLICE_HEIGHT = int(os.getenv("SAHI_SLICE_HEIGHT", "512"))
SAHI_SLICE_WIDTH = int(os.getenv("SAHI_SLICE_WIDTH", "512"))
SAHI_OVERLAP_RATIO = float(os.getenv("SAHI_OVERLAP_RATIO", "0.2"))
SAHI_CONF_THRESHOLD = float(os.getenv("SAHI_CONF_THRESHOLD", "0.35"))

# NMS for label merging
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.5"))

# SAHI sampling: run on low-confidence Autodistill frames + random sample
SAHI_LOW_CONF_THRESHOLD = float(os.getenv("SAHI_LOW_CONF_THRESHOLD", "0.3"))
SAHI_RANDOM_SAMPLE_RATIO = float(os.getenv("SAHI_RANDOM_SAMPLE_RATIO", "0.2"))

# COCO class remapping (from real detection results)
COCO_REMAP = {
    "keyboard": "pos_terminal",
    "dining_table": "bar_counter",
    "tv": "bar_screen",
    # "chair" stays as "bar_stool" for zone context
    "chair": "bar_stool",
    # noise classes to discard
    "handbag": None,   # None = discard
    "book": None,
    "laptop": None,
}

# Bar-specific ontology for Grounding DINO
# Prompt -> Class name mapping
BAR_ONTOLOGY = {
    "beer glass filled with beer": "beer_glass",
    "wine glass with stem on bar": "wine_glass",
    "short rocks glass with drink": "rocks_glass",
    "shot glass on bar counter": "shot_glass",
    "cocktail in martini glass": "cocktail_glass",
    "pint glass on bar": "pint_glass",
    "beer tap handle on bar": "beer_tap",
    "liquor bottle on shelf behind bar": "liquor_bottle",
    "beer bottle on bar counter": "beer_bottle",
    "wine bottle": "wine_bottle",
    "card payment terminal on counter": "pos_terminal",
    "cocktail shaker metal": "shaker",
    "ice bucket on bar": "ice_bucket",
    "bar counter surface": "bar_counter",
    "person standing at bar": "person",
    "person sitting on bar stool": "person",
}

# Unified class list for final model (order matters for YOLO class IDs)
FINAL_CLASSES = [
    "person",          # 0
    "beer_glass",      # 1
    "wine_glass",      # 2
    "rocks_glass",     # 3
    "shot_glass",      # 4
    "cocktail_glass",  # 5
    "pint_glass",      # 6
    "beer_tap",        # 7
    "liquor_bottle",   # 8
    "beer_bottle",     # 9
    "wine_bottle",     # 10
    "pos_terminal",    # 11
    "shaker",          # 12
    "ice_bucket",      # 13
    "bar_counter",     # 14
    "bar_stool",       # 15
    "bar_screen",      # 16
]

# Training hyperparameters
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "100"))
TRAIN_IMGSZ = int(os.getenv("TRAIN_IMGSZ", "640"))
TRAIN_BATCH = int(os.getenv("TRAIN_BATCH", "16"))
TRAIN_BASE_MODEL = os.getenv("TRAIN_BASE_MODEL", "yolo11s.pt")

# Dataset split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.15
TEST_SPLIT = 0.05
