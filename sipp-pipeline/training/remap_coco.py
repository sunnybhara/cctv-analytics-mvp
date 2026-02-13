"""Step 5: Generate supplementary labels by remapping COCO detections.

Runs stock YOLO11s, keeps only classes in COCO_REMAP, remaps to bar classes,
and saves as YOLO labels. These get merged into the dataset alongside
Autodistill and SAHI labels.

Usage:
    python remap_coco.py
    python remap_coco.py --frames ./frames --output ./labels_coco_remap
"""
import os
import glob
import argparse

from ultralytics import YOLO

from config import COCO_REMAP, FINAL_CLASSES, FRAMES_DIR


def remap_frame(
    model, frame_path: str, output_dir: str, conf_threshold: float = 0.3
) -> int:
    """Run YOLO on a frame, remap COCO classes to bar classes, save labels."""
    results = model(frame_path, conf=conf_threshold, verbose=False)

    if not results or not results[0].boxes:
        return 0

    img_h, img_w = results[0].orig_shape
    lines = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        coco_name = model.names[class_id]

        if coco_name not in COCO_REMAP:
            continue

        bar_name = COCO_REMAP[coco_name]
        if bar_name is None:  # discard class
            continue

        if bar_name not in FINAL_CLASSES:
            continue

        new_class_id = FINAL_CLASSES.index(bar_name)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        lines.append(f"{new_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if lines:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(frame_path))[0]
        with open(os.path.join(output_dir, f"{basename}.txt"), "w") as f:
            f.write("\n".join(lines))

    return len(lines)


def main():
    parser = argparse.ArgumentParser(description="Remap COCO detections to bar classes")
    parser.add_argument("--frames", default=FRAMES_DIR)
    parser.add_argument("--output", default="./labels_coco_remap")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    model = YOLO("yolo11s.pt")
    frames = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))

    print(f"Remapping COCO classes on {len(frames)} frames")
    print(f"Remap: {COCO_REMAP}")

    total_labels = 0
    for i, frame_path in enumerate(frames):
        count = remap_frame(model, frame_path, args.output, args.conf)
        total_labels += count

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(frames)}] {total_labels} labels generated")

    print(f"\nCOCO remap complete: {total_labels} labels in {args.output}")


if __name__ == "__main__":
    main()
