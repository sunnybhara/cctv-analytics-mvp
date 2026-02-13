"""Step 7: Compare fine-tuned YOLO11s-bar vs stock COCO YOLO11s.

Runs both models on the test set and prints detection counts per class.

Usage:
    python evaluate.py
    python evaluate.py --test-images ./dataset/images/test
"""
import os
import glob
import argparse
from collections import Counter

from ultralytics import YOLO

from config import MERGED_DATASET, MODEL_OUTPUT_DIR, FINAL_CLASSES


def count_detections(model, image_dir: str, conf: float = 0.25) -> Counter:
    """Run model on all images and count detections per class."""
    counts = Counter()
    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    for img_path in images:
        results = model(img_path, conf=conf, verbose=False)
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names.get(class_id, f"class_{class_id}")
                counts[class_name] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Compare bar model vs COCO baseline")
    parser.add_argument(
        "--test-images",
        default=os.path.join(MERGED_DATASET, "images", "test"),
    )
    parser.add_argument(
        "--bar-model",
        default=os.path.join(MODEL_OUTPUT_DIR, "yolo11s-bar.pt"),
    )
    parser.add_argument("--coco-model", default="yolo11s.pt")
    args = parser.parse_args()

    if not os.path.exists(args.test_images):
        print(f"Test images not found: {args.test_images}")
        return

    images = glob.glob(os.path.join(args.test_images, "*.jpg"))
    print(f"Evaluating on {len(images)} test images\n")

    # Stock COCO
    print("=== Stock COCO YOLO11s ===")
    coco_model = YOLO(args.coco_model)
    coco_counts = count_detections(coco_model, args.test_images)
    for name, count in coco_counts.most_common(20):
        print(f"  {name}: {count}")

    # Fine-tuned bar model
    if os.path.exists(args.bar_model):
        print(f"\n=== Fine-tuned YOLO11s-bar ===")
        bar_model = YOLO(args.bar_model)
        bar_counts = count_detections(bar_model, args.test_images)
        for name, count in bar_counts.most_common(20):
            print(f"  {name}: {count}")

        # Highlight the gap
        bar_objects = [
            "beer_glass", "wine_glass", "rocks_glass", "pint_glass",
            "liquor_bottle", "beer_bottle", "pos_terminal", "beer_tap",
        ]
        print("\n=== Bar Object Detection Gap ===")
        for obj in bar_objects:
            coco = coco_counts.get(obj, 0)
            bar = bar_counts.get(obj, 0)
            delta = bar - coco
            marker = "+" if delta > 0 else ""
            print(f"  {obj}: COCO={coco}, Bar={bar} ({marker}{delta})")
    else:
        print(f"\nBar model not found: {args.bar_model}")
        print("Run train.py first.")


if __name__ == "__main__":
    main()
