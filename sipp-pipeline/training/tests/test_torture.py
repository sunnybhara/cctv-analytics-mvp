"""
Torture Tests — SIPP Training Pipeline
=======================================
Edge cases, malformed labels, empty datasets, split ratios,
NMS boundary conditions, class mapping integrity, and integration flows.
"""

import os
import random
import shutil

import pytest
import torch

from config import (
    BAR_ONTOLOGY, COCO_REMAP, FINAL_CLASSES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
)
from merge_labels import (
    parse_yolo_label, merge_and_deduplicate, write_yolo_label, create_data_yaml,
)
from sahi_label import get_low_confidence_frames, save_yolo_labels


# ── Config Integrity Torture ─────────────────────────────────────────────────

class TestConfigIntegrity:
    def test_splits_sum_to_one(self):
        assert TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT == pytest.approx(1.0)

    def test_ontology_targets_in_final_classes(self):
        """Every BAR_ONTOLOGY value must exist in FINAL_CLASSES."""
        for prompt, class_name in BAR_ONTOLOGY.items():
            assert class_name in FINAL_CLASSES, (
                f"Ontology '{prompt}' -> '{class_name}' not in FINAL_CLASSES"
            )

    def test_coco_remap_targets_in_final_classes(self):
        """Every non-None COCO_REMAP target must exist in FINAL_CLASSES."""
        for src, dst in COCO_REMAP.items():
            if dst is not None:
                assert dst in FINAL_CLASSES, f"COCO_REMAP '{src}' -> '{dst}' not in FINAL_CLASSES"

    def test_final_classes_starts_with_person(self):
        assert FINAL_CLASSES[0] == "person"

    def test_no_duplicate_final_classes(self):
        assert len(FINAL_CLASSES) == len(set(FINAL_CLASSES))

    def test_final_classes_count(self):
        assert len(FINAL_CLASSES) == 17

    def test_ontology_has_person_prompts(self):
        person_prompts = [k for k, v in BAR_ONTOLOGY.items() if v == "person"]
        assert len(person_prompts) >= 1

    def test_drink_classes_are_consecutive(self):
        """Glass classes (1-6) should be consecutive for easy slicing."""
        glass_classes = ["beer_glass", "wine_glass", "rocks_glass",
                         "shot_glass", "cocktail_glass", "pint_glass"]
        indices = [FINAL_CLASSES.index(c) for c in glass_classes]
        assert indices == list(range(1, 7))


# ── Parse YOLO Label Torture ─────────────────────────────────────────────────

class TestParseYoloTorture:
    def test_negative_class_id(self, tmp_path):
        """Negative class ID — parse_yolo_label accepts it (validation later)."""
        label = tmp_path / "neg.txt"
        label.write_text("-1 0.5 0.5 0.2 0.3\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 1
        assert result[0]["class_id"] == -1

    def test_class_id_beyond_range(self, tmp_path):
        """Class ID > len(FINAL_CLASSES) — accepted at parse time."""
        label = tmp_path / "high.txt"
        label.write_text("999 0.5 0.5 0.2 0.3\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 1
        assert result[0]["class_id"] == 999

    def test_coordinates_out_of_range(self, tmp_path):
        """Coords > 1.0 — accepted at parse time (normalization issue upstream)."""
        label = tmp_path / "big.txt"
        label.write_text("0 1.5 1.5 0.2 0.3\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 1
        assert result[0]["cx"] == 1.5

    def test_zero_width_height(self, tmp_path):
        label = tmp_path / "zero.txt"
        label.write_text("0 0.5 0.5 0.0 0.0\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 1
        assert result[0]["w"] == 0.0

    def test_extra_columns_ignored(self, tmp_path):
        """Some label formats have confidence as 6th column."""
        label = tmp_path / "extra.txt"
        label.write_text("0 0.5 0.5 0.2 0.3 0.95\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 1
        assert result[0]["class_id"] == 0

    def test_blank_lines_skipped(self, tmp_path):
        label = tmp_path / "blanks.txt"
        label.write_text("0 0.5 0.5 0.2 0.3\n\n\n1 0.7 0.7 0.1 0.1\n\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 2

    def test_windows_line_endings(self, tmp_path):
        label = tmp_path / "crlf.txt"
        label.write_text("0 0.5 0.5 0.2 0.3\r\n1 0.7 0.7 0.1 0.1\r\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 2


# ── NMS Merge Torture ────────────────────────────────────────────────────────

class TestNMSTorture:
    def test_identical_boxes_collapses_to_one(self):
        labels = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
        ]
        result = merge_and_deduplicate(labels, [], iou_threshold=0.5)
        assert len(result) == 1

    def test_many_scattered_boxes_all_kept(self):
        """100 non-overlapping boxes should all survive NMS."""
        labels = []
        for i in range(10):
            for j in range(10):
                labels.append({
                    "class_id": 0,
                    "cx": 0.05 + i * 0.1,
                    "cy": 0.05 + j * 0.1,
                    "w": 0.05,
                    "h": 0.05,
                })
        result = merge_and_deduplicate(labels, [], iou_threshold=0.5)
        assert len(result) == 100

    def test_threshold_zero_keeps_only_one(self):
        """IoU threshold 0 means all pairs overlap → keep one."""
        labels = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
            {"class_id": 1, "cx": 0.55, "cy": 0.55, "w": 0.2, "h": 0.2},
        ]
        result = merge_and_deduplicate(labels, [], iou_threshold=0.0)
        # NMS with threshold 0 suppresses everything that has any overlap
        assert len(result) == 1

    def test_threshold_one_keeps_all(self):
        """IoU threshold 1.0 means only identical boxes merge."""
        labels = [
            {"class_id": 0, "cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2},
            {"class_id": 1, "cx": 0.35, "cy": 0.35, "w": 0.2, "h": 0.2},
        ]
        result = merge_and_deduplicate(labels, [], iou_threshold=1.0)
        assert len(result) == 2

    def test_mixed_sources_deduped(self):
        """Same box from Autodistill and SAHI — should merge to one."""
        ad = [{"class_id": 1, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}]
        sahi = [{"class_id": 1, "cx": 0.305, "cy": 0.305, "w": 0.1, "h": 0.1}]
        result = merge_and_deduplicate(ad, sahi, iou_threshold=0.5)
        assert len(result) == 1


# ── Save YOLO Labels Torture ─────────────────────────────────────────────────

class TestSaveYoloTorture:
    def test_bbox_at_image_edge(self, tmp_path):
        """Object at image boundary should produce valid normalized coords."""
        detections = [
            {"class_name": "person", "bbox_xyxy": [0, 0, 100, 200], "confidence": 0.9},
        ]
        save_yolo_labels(detections, "edge.jpg", str(tmp_path), 1920, 1080)
        content = (tmp_path / "edge.txt").read_text().strip()
        parts = content.split()
        cx, cy = float(parts[1]), float(parts[2])
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1

    def test_bbox_full_frame(self, tmp_path):
        """Object covering entire frame → cx=0.5, cy=0.5, w=1.0, h=1.0."""
        detections = [
            {"class_name": "bar_counter", "bbox_xyxy": [0, 0, 1920, 1080], "confidence": 0.9},
        ]
        save_yolo_labels(detections, "full.jpg", str(tmp_path), 1920, 1080)
        parts = (tmp_path / "full.txt").read_text().strip().split()
        assert float(parts[1]) == pytest.approx(0.5)
        assert float(parts[2]) == pytest.approx(0.5)
        assert float(parts[3]) == pytest.approx(1.0)
        assert float(parts[4]) == pytest.approx(1.0)

    def test_tiny_object(self, tmp_path):
        """1x1 pixel object at center."""
        detections = [
            {"class_name": "shot_glass", "bbox_xyxy": [960, 540, 961, 541], "confidence": 0.9},
        ]
        save_yolo_labels(detections, "tiny.jpg", str(tmp_path), 1920, 1080)
        parts = (tmp_path / "tiny.txt").read_text().strip().split()
        w, h = float(parts[3]), float(parts[4])
        assert w == pytest.approx(1 / 1920, abs=1e-4)
        assert h == pytest.approx(1 / 1080, abs=1e-4)

    def test_multiple_classes_in_one_image(self, tmp_path):
        detections = [
            {"class_name": "person", "bbox_xyxy": [100, 100, 300, 400], "confidence": 0.9},
            {"class_name": "beer_glass", "bbox_xyxy": [350, 200, 400, 280], "confidence": 0.8},
            {"class_name": "pos_terminal", "bbox_xyxy": [500, 100, 600, 200], "confidence": 0.7},
        ]
        save_yolo_labels(detections, "multi.jpg", str(tmp_path), 1920, 1080)
        lines = (tmp_path / "multi.txt").read_text().strip().split("\n")
        assert len(lines) == 3
        class_ids = [int(l.split()[0]) for l in lines]
        assert 0 in class_ids   # person
        assert 1 in class_ids   # beer_glass
        assert 11 in class_ids  # pos_terminal


# ── Data YAML Torture ────────────────────────────────────────────────────────

class TestDataYamlTorture:
    def test_yaml_has_absolute_path(self, tmp_path):
        yaml_path = create_data_yaml(str(tmp_path))
        with open(yaml_path) as f:
            content = f.read()
        assert "path:" in content
        assert str(tmp_path) in content

    def test_yaml_is_parseable(self, tmp_path):
        """data.yaml should be valid YAML."""
        import yaml
        yaml_path = create_data_yaml(str(tmp_path))
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert "names" in data
        assert data["names"][0] == "person"
        assert len(data["names"]) == 17


# ── Write+Read Label Roundtrip ───────────────────────────────────────────────

class TestLabelRoundtrip:
    def test_write_then_parse_matches(self, tmp_path):
        original = [
            {"class_id": 0, "cx": 0.123456, "cy": 0.654321, "w": 0.111111, "h": 0.222222},
            {"class_id": 5, "cx": 0.900000, "cy": 0.100000, "w": 0.050000, "h": 0.080000},
        ]
        path = str(tmp_path / "roundtrip.txt")
        write_yolo_label(original, path)
        parsed = parse_yolo_label(path)

        assert len(parsed) == 2
        for orig, read in zip(original, parsed):
            assert orig["class_id"] == read["class_id"]
            assert orig["cx"] == pytest.approx(read["cx"], abs=1e-5)
            assert orig["cy"] == pytest.approx(read["cy"], abs=1e-5)
            assert orig["w"] == pytest.approx(read["w"], abs=1e-5)
            assert orig["h"] == pytest.approx(read["h"], abs=1e-5)

    def test_large_dataset_roundtrip(self, tmp_path):
        """1000 labels roundtrip without loss."""
        random.seed(42)
        labels = [
            {
                "class_id": random.randint(0, 16),
                "cx": random.random(),
                "cy": random.random(),
                "w": random.random() * 0.5,
                "h": random.random() * 0.5,
            }
            for _ in range(1000)
        ]
        path = str(tmp_path / "big.txt")
        write_yolo_label(labels, path)
        parsed = parse_yolo_label(path)
        assert len(parsed) == 1000


# ── Low Confidence Frames Torture ────────────────────────────────────────────

class TestLowConfTorture:
    def test_mixed_confidence_frames(self, tmp_path):
        """Mix of empty, 1-det, and 3-det files."""
        label_dir = tmp_path / "train" / "labels"
        label_dir.mkdir(parents=True)
        (label_dir / "empty.txt").write_text("")
        (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        (label_dir / "three.txt").write_text(
            "0 0.1 0.1 0.1 0.1\n1 0.5 0.5 0.1 0.1\n2 0.9 0.9 0.1 0.1\n"
        )

        result = get_low_confidence_frames(str(tmp_path), 0.3)
        assert "empty" in result
        assert "one" in result
        assert "three" not in result

    def test_all_frames_low_confidence(self, tmp_path):
        label_dir = tmp_path / "train" / "labels"
        label_dir.mkdir(parents=True)
        for i in range(10):
            (label_dir / f"f{i}.txt").write_text("")

        result = get_low_confidence_frames(str(tmp_path), 0.3)
        assert len(result) == 10

    def test_fallback_to_flat_dir(self, tmp_path):
        """When train/labels/ doesn't exist, use the dir directly."""
        (tmp_path / "frame.txt").write_text("")
        result = get_low_confidence_frames(str(tmp_path), 0.3)
        assert "frame" in result


# ── Integration: Merge Pipeline ──────────────────────────────────────────────

class TestMergeIntegration:
    def test_full_merge_creates_dataset_structure(self, tmp_path):
        """Simulate a small merge pipeline end-to-end."""
        # Create fake frames
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(20):
            (frames_dir / f"f{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xe0")

        # Create Autodistill labels for all frames
        ad_dir = tmp_path / "ad"
        ad_dir.mkdir()
        for i in range(20):
            (ad_dir / f"f{i:03d}.txt").write_text(
                f"0 {0.1 + i * 0.04:.6f} 0.500000 0.100000 0.200000\n"
            )

        # Create SAHI labels for first 5 frames
        sahi_dir = tmp_path / "sahi"
        sahi_dir.mkdir()
        for i in range(5):
            (sahi_dir / f"f{i:03d}.txt").write_text(
                f"1 {0.8 - i * 0.05:.6f} 0.300000 0.050000 0.080000\n"
            )

        # Run merge
        from merge_labels import main as merge_main
        import sys

        output_dir = tmp_path / "dataset"
        old_argv = sys.argv
        sys.argv = [
            "merge_labels.py",
            "--frames", str(frames_dir),
            "--autodistill", str(ad_dir),
            "--sahi", str(sahi_dir),
            "--coco-remap", str(tmp_path / "nonexistent"),
            "--output", str(output_dir),
        ]
        try:
            merge_main()
        finally:
            sys.argv = old_argv

        # Verify structure
        assert (output_dir / "data.yaml").exists()
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()
        assert (output_dir / "images" / "test").exists()
        assert (output_dir / "labels" / "train").exists()

        # Count files across splits
        train_imgs = list((output_dir / "images" / "train").glob("*.jpg"))
        val_imgs = list((output_dir / "images" / "val").glob("*.jpg"))
        test_imgs = list((output_dir / "images" / "test").glob("*.jpg"))
        total = len(train_imgs) + len(val_imgs) + len(test_imgs)
        assert total == 20

        # Check split ratios (approximate)
        assert len(train_imgs) >= 14  # 80% of 20 = 16, allow some variance
        assert len(val_imgs) >= 1
