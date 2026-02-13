"""Step 4: Merge Autodistill + SAHI + COCO remap labels into unified dataset.

Deduplicates overlapping boxes via NMS, remaps class IDs to FINAL_CLASSES,
and splits into train/val/test.

Usage:
    python merge_labels.py
    python merge_labels.py --coco-remap ./labels_coco_remap
"""
import os
import glob
import shutil
import random
import argparse

import torch
from torchvision.ops import nms

from config import (
    FRAMES_DIR, AUTODISTILL_OUTPUT, SAHI_OUTPUT, MERGED_DATASET,
    NMS_IOU_THRESHOLD, FINAL_CLASSES, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
)


def parse_yolo_label(label_path: str) -> list[dict]:
    """Parse a YOLO format label file.

    Returns list of {class_id, cx, cy, w, h}.
    """
    detections = []
    if not os.path.exists(label_path):
        return detections

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            detections.append({
                "class_id": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            })
    return detections


def merge_and_deduplicate(
    autodistill_labels: list[dict],
    sahi_labels: list[dict],
    iou_threshold: float,
) -> list[dict]:
    """Merge two label sets and remove duplicates via NMS.

    Returns deduplicated list.
    """
    all_labels = autodistill_labels + sahi_labels

    if len(all_labels) <= 1:
        return all_labels

    # Convert normalized coords to xyxy for NMS
    boxes = []
    scores = []
    for det in all_labels:
        x1 = det["cx"] - det["w"] / 2
        y1 = det["cy"] - det["h"] / 2
        x2 = det["cx"] + det["w"] / 2
        y2 = det["cy"] + det["h"] / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(1.0)

    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_t, scores_t, iou_threshold)

    return [all_labels[i] for i in keep.tolist()]


def write_yolo_label(detections: list[dict], output_path: str):
    """Write YOLO format label file."""
    lines = []
    for det in detections:
        lines.append(
            f"{det['class_id']} {det['cx']:.6f} {det['cy']:.6f} "
            f"{det['w']:.6f} {det['h']:.6f}"
        )
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def create_data_yaml(output_dir: str) -> str:
    """Create YOLO data.yaml for training."""
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

names:
"""
    for i, name in enumerate(FINAL_CLASSES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Merge and deduplicate label sets")
    parser.add_argument("--frames", default=FRAMES_DIR)
    parser.add_argument("--autodistill", default=AUTODISTILL_OUTPUT)
    parser.add_argument("--sahi", default=SAHI_OUTPUT)
    parser.add_argument("--coco-remap", default="./labels_coco_remap")
    parser.add_argument("--output", default=MERGED_DATASET)
    args = parser.parse_args()

    all_frames = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))

    if not all_frames:
        print(f"No frames found in {args.frames}")
        return

    # Find Autodistill label directory (may be nested under train/labels/)
    autodistill_label_dir = os.path.join(args.autodistill, "train", "labels")
    if not os.path.exists(autodistill_label_dir):
        autodistill_label_dir = args.autodistill

    # Merge labels per frame
    merged = {}
    for frame_path in all_frames:
        basename = os.path.splitext(os.path.basename(frame_path))[0]

        ad_path = os.path.join(autodistill_label_dir, f"{basename}.txt")
        sahi_path = os.path.join(args.sahi, f"{basename}.txt")
        coco_path = os.path.join(args.coco_remap, f"{basename}.txt")

        ad_labels = parse_yolo_label(ad_path)
        sahi_labels = parse_yolo_label(sahi_path)
        coco_labels = parse_yolo_label(coco_path)

        merged_labels = merge_and_deduplicate(
            ad_labels, sahi_labels + coco_labels, NMS_IOU_THRESHOLD
        )
        merged[frame_path] = merged_labels

    # Filter out frames with zero labels
    labeled_frames = [(path, labels) for path, labels in merged.items() if labels]
    print(f"Frames with labels: {len(labeled_frames)} / {len(all_frames)}")

    # Shuffle and split
    random.shuffle(labeled_frames)
    n = len(labeled_frames)
    train_end = int(n * TRAIN_SPLIT)
    val_end = train_end + int(n * VAL_SPLIT)

    splits = {
        "train": labeled_frames[:train_end],
        "val": labeled_frames[train_end:val_end],
        "test": labeled_frames[val_end:],
    }

    # Write dataset
    for split_name, items in splits.items():
        img_dir = os.path.join(args.output, "images", split_name)
        lbl_dir = os.path.join(args.output, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for frame_path, labels in items:
            basename = os.path.basename(frame_path)
            shutil.copy2(frame_path, os.path.join(img_dir, basename))

            label_basename = os.path.splitext(basename)[0] + ".txt"
            write_yolo_label(labels, os.path.join(lbl_dir, label_basename))

        print(f"{split_name}: {len(items)} images")

    yaml_path = create_data_yaml(args.output)
    print(f"\nDataset ready: {args.output}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
