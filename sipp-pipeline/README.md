# SIPP Bar Detection Pipeline

Standalone service that processes bar CCTV footage and emits structured events. Reads video (file or RTSP), runs YOLO11s object detection with dual-tracker (BoT-SORT for persons, ByteTrack for objects), detects person-object interactions in defined bar zones, verifies candidate events via Claude Vision API, deduplicates, and POSTs verified events to the existing analytics API.

## Quickstart

```bash
pip install -r requirements.txt
VIDEO_SOURCE=test.mp4 ANTHROPIC_API_KEY=sk-... python main.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | *(required)* | File path or `rtsp://` URL |
| `CAMERA_ID` | `camera_01` | Camera identifier matching zones.json |
| `YOLO_MODEL` | `yolo11s.pt` | YOLO model file (stock COCO) |
| `YOLO_CONF_THRESHOLD` | `0.25` | Detection confidence threshold |
| `ZONES_FILE` | `config/zones.json` | Zone polygon definitions |
| `IOU_THRESHOLD` | `0.1` | Min IoU for person-object interaction |
| `TRIGGER_COOLDOWN_SECONDS` | `10.0` | Debounce window per person-object pair |
| `OBJECT_VELOCITY_THRESHOLD` | `2.0` | Min object velocity to trigger (pixels/frame) |
| `ANTHROPIC_API_KEY` | *(empty)* | Claude API key for VLM verification |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model for verification |
| `MAX_VLM_CALLS_PER_HOUR` | `100` | Budget cap for API calls |
| `CLIP_DURATION_SECONDS` | `5.0` | Clip window around detected event |
| `MAX_FRAMES_PER_CLIP` | `4` | Frames sent to VLM per verification |
| `DEDUP_WINDOW_SECONDS` | `10.0` | Sliding window for event deduplication |
| `EVENT_API_URL` | `http://localhost:8000/events/batch` | Backend API endpoint |
| `EVENT_API_KEY` | *(empty)* | API key for backend auth |
| `CLIP_SAVE_DIR` | `./confirmed_clips` | Directory for saved training clips |
| `STRICT_MODE` | `false` | Add privacy-aware VLM instructions |

## Architecture

The pipeline runs four steps per frame: (1) **Detect** — YOLO11s detects persons and drink objects, BoT-SORT/ByteTrack assign persistent track IDs with zone classification. (2) **Interact** — Person-object pairs are checked for spatial overlap (IoU) in bar zones; a debouncer filters cooldowns, static objects, and budget limits. (3) **Verify** — Keyframes from a ring buffer are sent to Claude Vision API, which classifies the action (pour, serve, payment, idle) and returns structured JSON. (4) **Emit** — Verified events pass through a sliding-window deduplicator, then batch-POST to the analytics API.

## Training Data

Confirmed event clips are saved to `CLIP_SAVE_DIR` as MP4 files. Each file is named by its event UUID. These clips serve as labeled training data for future model fine-tuning.
