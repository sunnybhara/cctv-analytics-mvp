"""Step 3: SAHI sliced inference for small object detection.

Runs on:
  - Frames where Autodistill confidence was low (<0.3)
  - Random 20% sample of remaining frames

Usage:
    python sahi_label.py
    python sahi_label.py --frames ./frames --autodistill-labels ./labels_autodistill
"""
import os
import glob
import random
import argparse

from config import (
    FRAMES_DIR, AUTODISTILL_OUTPUT, SAHI_OUTPUT,
    SAHI_SLICE_HEIGHT, SAHI_SLICE_WIDTH, SAHI_OVERLAP_RATIO,
    SAHI_CONF_THRESHOLD, SAHI_LOW_CONF_THRESHOLD, SAHI_RANDOM_SAMPLE_RATIO,
    NMS_IOU_THRESHOLD, BAR_ONTOLOGY, FINAL_CLASSES,
)


def get_low_confidence_frames(autodistill_dir: str, threshold: float) -> set[str]:
    """Find frames where Autodistill had low average confidence.

    Parse YOLO-format label files to check detection count.
    Returns set of frame basenames (without extension).
    """
    low_conf = set()
    label_dir = os.path.join(autodistill_dir, "train", "labels")
    if not os.path.exists(label_dir):
        label_dir = autodistill_dir  # fallback

    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        basename = os.path.splitext(os.path.basename(label_file))[0]
        with open(label_file) as f:
            lines = f.readlines()

        if not lines:
            low_conf.add(basename)  # no detections
            continue

        # Frames with fewer than 2 detections need SAHI
        if len(lines) < 2:
            low_conf.add(basename)

    return low_conf


def select_frames_for_sahi(frames_dir: str, autodistill_dir: str) -> list[str]:
    """Select frames that need SAHI processing.

    Returns list of frame file paths.
    """
    all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

    low_conf = get_low_confidence_frames(autodistill_dir, SAHI_LOW_CONF_THRESHOLD)

    selected = []
    remaining = []

    for frame_path in all_frames:
        basename = os.path.splitext(os.path.basename(frame_path))[0]
        if basename in low_conf:
            selected.append(frame_path)
        else:
            remaining.append(frame_path)

    # Random sample of remaining
    sample_count = int(len(remaining) * SAHI_RANDOM_SAMPLE_RATIO)
    selected.extend(random.sample(remaining, min(sample_count, len(remaining))))

    return selected


def sahi_detect(image_path: str, detection_model) -> list[dict]:
    """Run SAHI sliced prediction on a single image.

    Returns list of {class_name, bbox_xyxy, confidence} after NMS.
    """
    import torch
    from torchvision.ops import nms
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=SAHI_SLICE_HEIGHT,
        slice_width=SAHI_SLICE_WIDTH,
        overlap_height_ratio=SAHI_OVERLAP_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_RATIO,
    )

    predictions = result.object_prediction_list
    if not predictions:
        return []

    # NMS to remove duplicate boxes from tile overlaps
    boxes = torch.tensor(
        [p.bbox.to_xyxy() for p in predictions], dtype=torch.float32
    )
    scores = torch.tensor(
        [p.score.value for p in predictions], dtype=torch.float32
    )
    keep = nms(boxes, scores, NMS_IOU_THRESHOLD)

    return [
        {
            "class_name": predictions[i].category.name,
            "bbox_xyxy": predictions[i].bbox.to_xyxy(),
            "confidence": predictions[i].score.value,
        }
        for i in keep.tolist()
    ]


def save_yolo_labels(
    detections: list[dict],
    image_path: str,
    output_dir: str,
    img_width: int,
    img_height: int,
):
    """Save detections in YOLO format (class_id cx cy w h)."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{basename}.txt")

    lines = []
    for det in detections:
        class_name = det["class_name"]
        if class_name not in FINAL_CLASSES:
            continue
        class_id = FINAL_CLASSES.index(class_name)

        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="SAHI sliced inference for small bar objects")
    parser.add_argument("--frames", default=FRAMES_DIR)
    parser.add_argument("--autodistill-labels", default=AUTODISTILL_OUTPUT)
    parser.add_argument("--output", default=SAHI_OUTPUT)
    args = parser.parse_args()

    import torch
    from sahi import AutoDetectionModel

    print("Loading Grounding DINO for SAHI...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="grounding_dino",
        confidence_threshold=SAHI_CONF_THRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    frames = select_frames_for_sahi(args.frames, args.autodistill_labels)
    print(f"SAHI processing {len(frames)} frames")

    for i, frame_path in enumerate(frames):
        detections = sahi_detect(frame_path, detection_model)

        from PIL import Image
        img = Image.open(frame_path)
        w, h = img.size

        save_yolo_labels(detections, frame_path, args.output, w, h)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(frames)}] processed")

    print(f"SAHI labeling complete: {args.output}")


if __name__ == "__main__":
    main()
