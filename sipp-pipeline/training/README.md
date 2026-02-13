# SIPP Bar Model Training Pipeline

Batch pipeline that produces a fine-tuned `yolo11s-bar.pt` from raw bar CCTV footage. Combines Grounding DINO auto-labeling via Autodistill, SAHI sliced inference for small objects (glasses, tap handles), COCO class remapping (keyboard->pos_terminal, dining_table->bar_counter), NMS-based label merging, and YOLO11s transfer learning.

## Prerequisites

- GPU with 8GB+ VRAM (CUDA)
- ffmpeg installed (`brew install ffmpeg` / `apt install ffmpeg`)
- Python 3.10+

```bash
pip install -r requirements.txt
```

## Workflow

Run steps in order from the `training/` directory:

```bash
# Step 1: Extract 1fps frames from raw footage
python extract_frames.py --source /path/to/bar/videos

# Step 2: Auto-label with Grounding DINO (bar-specific ontology)
python autolabel.py

# Step 3: SAHI pass for small objects (glasses, taps, bottles)
python sahi_label.py

# Step 4: Harvest COCO remaps (keyboard->pos_terminal, dining_table->bar_counter)
python remap_coco.py

# Step 5: Merge all label sources + NMS dedup + train/val/test split
python merge_labels.py

# Step 6: Fine-tune YOLO11s (100 epochs, ~2-4 hours on single GPU)
python train.py

# Step 7: Compare fine-tuned vs stock COCO baseline
python evaluate.py
```

## Deployment

Copy the trained model and update the Week 1 pipeline:

```bash
cp models/yolo11s-bar.pt ../
export YOLO_MODEL=yolo11s-bar.pt
```

## Class List

| ID | Class | Source |
|----|-------|--------|
| 0 | person | COCO + Grounding DINO |
| 1 | beer_glass | Grounding DINO + SAHI |
| 2 | wine_glass | Grounding DINO + SAHI |
| 3 | rocks_glass | Grounding DINO + SAHI |
| 4 | shot_glass | Grounding DINO + SAHI |
| 5 | cocktail_glass | Grounding DINO + SAHI |
| 6 | pint_glass | Grounding DINO + SAHI |
| 7 | beer_tap | Grounding DINO + SAHI |
| 8 | liquor_bottle | Grounding DINO + SAHI |
| 9 | beer_bottle | Grounding DINO + SAHI |
| 10 | wine_bottle | Grounding DINO + SAHI |
| 11 | pos_terminal | COCO remap (keyboard) |
| 12 | shaker | Grounding DINO |
| 13 | ice_bucket | Grounding DINO |
| 14 | bar_counter | COCO remap (dining_table) + GDINO |
| 15 | bar_stool | COCO remap (chair) |
| 16 | bar_screen | COCO remap (tv) |
