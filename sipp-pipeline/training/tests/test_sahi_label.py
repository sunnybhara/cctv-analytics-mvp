"""Tests for training/sahi_label.py — tests frame selection and label saving logic."""

import os
import pytest
from sahi_label import get_low_confidence_frames, select_frames_for_sahi, save_yolo_labels
from config import FINAL_CLASSES


class TestGetLowConfidenceFrames:
    def test_empty_label_file_is_low_conf(self, tmp_path):
        label_dir = tmp_path / "train" / "labels"
        label_dir.mkdir(parents=True)
        (label_dir / "frame001.txt").write_text("")
        result = get_low_confidence_frames(str(tmp_path), 0.3)
        assert "frame001" in result

    def test_single_detection_is_low_conf(self, tmp_path):
        label_dir = tmp_path / "train" / "labels"
        label_dir.mkdir(parents=True)
        (label_dir / "frame002.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        result = get_low_confidence_frames(str(tmp_path), 0.3)
        assert "frame002" in result

    def test_two_detections_not_low_conf(self, tmp_path):
        label_dir = tmp_path / "train" / "labels"
        label_dir.mkdir(parents=True)
        (label_dir / "frame003.txt").write_text(
            "0 0.5 0.5 0.2 0.3\n1 0.7 0.7 0.1 0.1\n"
        )
        result = get_low_confidence_frames(str(tmp_path), 0.3)
        assert "frame003" not in result

    def test_no_label_dir_returns_empty(self, tmp_path):
        result = get_low_confidence_frames(str(tmp_path / "nonexistent"), 0.3)
        assert result == set()


class TestSelectFramesForSahi:
    def test_includes_all_low_conf_plus_sample(self, tmp_path):
        # Create frame files
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        for i in range(20):
            (frames_dir / f"frame{i:03d}.jpg").write_bytes(b"\xff\xd8")

        # Create autodistill labels — frames 0-4 have < 2 detections (low conf)
        labels_dir = tmp_path / "labels" / "train" / "labels"
        labels_dir.mkdir(parents=True)
        for i in range(5):
            (labels_dir / f"frame{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        for i in range(5, 20):
            (labels_dir / f"frame{i:03d}.txt").write_text(
                "0 0.5 0.5 0.2 0.3\n1 0.7 0.7 0.1 0.1\n"
            )

        selected = select_frames_for_sahi(str(frames_dir), str(tmp_path / "labels"))
        # Should have all 5 low-conf + ~20% of 15 remaining = ~3
        assert len(selected) >= 5
        assert len(selected) <= 20


class TestSaveYoloLabels:
    def test_correct_yolo_format(self, tmp_path):
        detections = [
            {"class_name": "beer_glass", "bbox_xyxy": [100, 200, 200, 400], "confidence": 0.9},
            {"class_name": "person", "bbox_xyxy": [300, 100, 500, 600], "confidence": 0.85},
        ]
        save_yolo_labels(detections, "/path/to/frame001.jpg", str(tmp_path), 1920, 1080)
        label_path = tmp_path / "frame001.txt"
        assert label_path.exists()

        lines = label_path.read_text().strip().split("\n")
        assert len(lines) == 2

        # beer_glass = class 1
        parts = lines[0].split()
        assert parts[0] == "1"
        cx = float(parts[1])
        assert 0 < cx < 1  # normalized

    def test_skips_unknown_classes(self, tmp_path):
        detections = [
            {"class_name": "unknown_thing", "bbox_xyxy": [100, 200, 200, 400], "confidence": 0.9},
        ]
        save_yolo_labels(detections, "/path/to/frame002.jpg", str(tmp_path), 1920, 1080)
        label_path = tmp_path / "frame002.txt"
        content = label_path.read_text().strip()
        assert content == ""

    def test_correct_normalization(self, tmp_path):
        """Verify coordinates are normalized correctly."""
        # bbox at center of 1920x1080 frame
        detections = [
            {"class_name": "person", "bbox_xyxy": [860, 440, 1060, 640], "confidence": 0.9},
        ]
        save_yolo_labels(detections, "/path/to/center.jpg", str(tmp_path), 1920, 1080)
        label_path = tmp_path / "center.txt"
        parts = label_path.read_text().strip().split()
        cx = float(parts[1])
        cy = float(parts[2])
        assert cx == pytest.approx(0.5, abs=0.01)
        assert cy == pytest.approx(0.5, abs=0.01)

    def test_empty_detections_writes_empty(self, tmp_path):
        save_yolo_labels([], "/path/to/empty.jpg", str(tmp_path), 1920, 1080)
        label_path = tmp_path / "empty.txt"
        assert label_path.read_text() == ""
