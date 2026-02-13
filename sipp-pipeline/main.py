"""
SIPP Bar Detection Pipeline — Entry Point
==========================================
Video source -> YOLO detection -> interaction trigger -> Claude verification -> events
"""

import asyncio
import logging
import time

import cv2

from config.settings import CAMERA_ID, CLIP_SAVE_DIR, VIDEO_SOURCE, ZONES_FILE
from pipeline.clip_buffer import ClipBuffer
from pipeline.deduplicator import EventDeduplicator
from pipeline.detector import DualTracker
from pipeline.events import BarEvent, post_events
from pipeline.interactions import InteractionDetector
from pipeline.verifier import verify_event
from pipeline.zones import load_zones

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sipp-pipeline")

# Hardcoded for Week 1. Admin panel sets this in Sprint 2.
VENUE_ID = "demo_venue"


async def run_pipeline():
    """Main pipeline loop. Process video frame by frame."""

    logger.info(f"Starting pipeline: source={VIDEO_SOURCE}, camera={CAMERA_ID}")

    zones = load_zones(ZONES_FILE, CAMERA_ID)
    logger.info(f"Loaded {len(zones)} zones: {list(zones.keys())}")

    tracker = DualTracker()
    interaction_detector = InteractionDetector()
    clip_buffer = ClipBuffer()
    deduplicator = EventDeduplicator()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {VIDEO_SOURCE}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_buffer.fps = fps
    frame_count = 0
    events_batch: list[BarEvent] = []

    logger.info(f"Video opened: {fps:.1f} FPS")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            frame_count += 1

            clip_buffer.add_frame(frame, now)

            persons, objects = tracker.process_frame(frame, zones)

            pending_events = interaction_detector.detect(persons, objects)

            for pending in pending_events:
                keyframes = clip_buffer.extract_clip_frames(pending.timestamp)
                if not keyframes:
                    continue

                vlm_result = verify_event(
                    keyframes,
                    {"object_class": pending.object_class, "zone": pending.zone},
                )

                action = vlm_result.get("action", "unknown")
                confidence = vlm_result.get("confidence", 0.0)

                if action in ("unknown", "idle", "cleaning") or confidence < 0.5:
                    logger.debug(f"Skipped: {action} ({confidence:.2f})")
                    continue

                if not deduplicator.should_emit(pending.zone, action, confidence):
                    logger.debug(f"Deduped: {action} in {pending.zone}")
                    continue

                if "pour" in action or "serv" in action:
                    event_type = "serve"
                elif "payment" in action:
                    event_type = "payment"
                else:
                    event_type = "activity"

                event = BarEvent(
                    event_type=event_type,
                    venue_id=VENUE_ID,
                    camera_id=CAMERA_ID,
                    zone=pending.zone,
                    confidence=confidence,
                    action=action,
                    drink_category=vlm_result.get("drink_category", "unknown"),
                    vessel_type=vlm_result.get("vessel_type", "unknown"),
                    description=vlm_result.get("description", ""),
                    person_track_id=pending.person_track_id,
                    object_track_id=pending.object_track_id,
                )

                events_batch.append(event)

                clip_path = clip_buffer.save_clip(pending.timestamp, event.event_id)
                if clip_path:
                    logger.info(f"Saved training clip: {clip_path}")

                logger.info(
                    f"EVENT: {action} | {vlm_result.get('drink_category')} | "
                    f"conf={confidence:.2f} | zone={pending.zone}"
                )

            # Batch post events every 10 seconds
            if events_batch and frame_count % max(1, int(fps * 10)) == 0:
                await post_events(events_batch)
                events_batch = []

            # Log progress every 30 seconds
            if frame_count % max(1, int(fps * 30)) == 0:
                logger.info(
                    f"Processed {frame_count} frames, "
                    f"{len(persons)} persons, {len(objects)} objects in view"
                )

        # Flush remaining events
        if events_batch:
            await post_events(events_batch)

    finally:
        cap.release()

    logger.info(f"Pipeline complete: {frame_count} frames processed")


if __name__ == "__main__":
    if not VIDEO_SOURCE:
        print("Set VIDEO_SOURCE env var to a file path or rtsp:// URL")
        exit(1)
    asyncio.run(run_pipeline())
