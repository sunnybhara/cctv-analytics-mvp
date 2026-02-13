"""Step 6: Fine-tune YOLO11s on the merged bar dataset.

Usage:
    python train.py
    python train.py --data ./dataset/data.yaml --epochs 100
"""
import os
import shutil
import argparse

from ultralytics import YOLO

from config import (
    MERGED_DATASET, MODEL_OUTPUT_DIR, TRAIN_EPOCHS,
    TRAIN_IMGSZ, TRAIN_BATCH, TRAIN_BASE_MODEL,
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11s for bar detection")
    parser.add_argument("--data", default=os.path.join(MERGED_DATASET, "data.yaml"))
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMGSZ)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH)
    parser.add_argument("--base-model", default=TRAIN_BASE_MODEL)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"data.yaml not found: {args.data}")
        print("Run merge_labels.py first.")
        return

    print("Training YOLO11s-bar")
    print(f"  Base model: {args.base_model}")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")

    model = YOLO(args.base_model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=MODEL_OUTPUT_DIR,
        name="yolo11s-bar",
        exist_ok=True,
        verbose=True,
    )

    # Copy best model to a predictable path
    best_path = os.path.join(MODEL_OUTPUT_DIR, "yolo11s-bar", "weights", "best.pt")
    output_path = os.path.join(MODEL_OUTPUT_DIR, "yolo11s-bar.pt")

    if os.path.exists(best_path):
        shutil.copy2(best_path, output_path)
        print(f"\nBest model saved to: {output_path}")
        print(
            f"To deploy: copy {output_path} to sipp-pipeline/ "
            "and update YOLO_MODEL in settings.py"
        )
    else:
        print("\nWarning: best.pt not found. Check training output.")


if __name__ == "__main__":
    main()
