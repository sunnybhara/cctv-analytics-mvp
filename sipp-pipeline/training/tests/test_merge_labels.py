"""Tests for training/merge_labels.py"""

import os
import pytest
from merge_labels import (
    parse_yolo_label, merge_and_deduplicate, write_yolo_label, create_data_yaml,
)
from config import FINAL_CLASSES


class TestParseYoloLabel:
    def test_reads_standard_format(self, tmp_path):
        label = tmp_path / "frame.txt"
        label.write_text("0 0.500000 0.500000 0.200000 0.300000\n"
                         "1 0.700000 0.400000 0.100000 0.150000\n")
        result = parse_yolo_label(str(label))
        assert len(result) == 2
        assert result[0]["class_id"] == 0
        assert result[0]["cx"] == pytest.approx(0.5)
        assert result[0]["cy"] == pytest.approx(0.5)
        assert result[1]["class_id"] == 1

    def test_handles_empty_file(self, tmp_path):
        label = tmp_path / "empty.txt"
        label.write_text("")
        result = parse_yolo_label(str(label))
        assert result == []

    def test_handles_missing_file(self):
        result = parse_yolo_label("/nonexistent/path/label.txt")
        assert result == []

    def test_skips_malformed_lines(self, tmp_path):
        label = tmp_path / "bad.txt"
        label.write_text("0 0.5 0.5\n"                          # too few fields
                         "1 0.7 0.4 0.1 0.15\n"                 # valid
                         "not a label\n"                          # garbage
                         "2 0.3 0.3 0.05 0.05\n")                # valid
        result = parse_yolo_label(str(label))
        assert len(result) == 2

    def test_handles_extra_whitespace(self, tmp_path):
        label = tmp_path / "ws.txt"
        label.write_text("  0  0.500000  0.500000  0.200000  0.300000  \n")
        result = parse_yolo_label(str(label))
        assert len(result) == 1
        assert result[0]["class_id"] == 0


class TestMergeAndDeduplicate:
    def test_keeps_non_overlapping_boxes(self):
        ad = [{"class_id": 0, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1}]
        sahi = [{"class_id": 1, "cx": 0.8, "cy": 0.8, "w": 0.1, "h": 0.1}]
        result = merge_and_deduplicate(ad, sahi, iou_threshold=0.5)
        assert len(result) == 2

    def test_removes_duplicate_boxes(self):
        # Two nearly identical boxes
        ad = [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
        sahi = [{"class_id": 0, "cx": 0.505, "cy": 0.505, "w": 0.2, "h": 0.2}]
        result = merge_and_deduplicate(ad, sahi, iou_threshold=0.5)
        assert len(result) == 1

    def test_handles_empty_both(self):
        result = merge_and_deduplicate([], [], iou_threshold=0.5)
        assert result == []

    def test_handles_empty_one_source(self):
        ad = [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
        result = merge_and_deduplicate(ad, [], iou_threshold=0.5)
        assert len(result) == 1

    def test_single_detection_passthrough(self):
        ad = [{"class_id": 3, "cx": 0.1, "cy": 0.1, "w": 0.05, "h": 0.05}]
        result = merge_and_deduplicate(ad, [], iou_threshold=0.5)
        assert len(result) == 1
        assert result[0]["class_id"] == 3


class TestWriteYoloLabel:
    def test_produces_correct_format(self, tmp_path):
        detections = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3},
            {"class_id": 1, "cx": 0.7, "cy": 0.4, "w": 0.1, "h": 0.15},
        ]
        path = str(tmp_path / "out.txt")
        write_yolo_label(detections, path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parts = lines[0].strip().split()
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.5)

    def test_empty_detections_writes_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.txt")
        write_yolo_label([], path)
        with open(path) as f:
            content = f.read()
        assert content == ""


class TestCreateDataYaml:
    def test_includes_all_classes(self, tmp_path):
        yaml_path = create_data_yaml(str(tmp_path))
        with open(yaml_path) as f:
            content = f.read()
        for i, name in enumerate(FINAL_CLASSES):
            assert f"{i}: {name}" in content

    def test_includes_split_paths(self, tmp_path):
        yaml_path = create_data_yaml(str(tmp_path))
        with open(yaml_path) as f:
            content = f.read()
        assert "train: images/train" in content
        assert "val: images/val" in content
        assert "test: images/test" in content

    def test_class_count_matches_final_classes(self, tmp_path):
        yaml_path = create_data_yaml(str(tmp_path))
        with open(yaml_path) as f:
            lines = f.readlines()
        # Count lines that match "  N: class_name" pattern
        class_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
        assert len(class_lines) == len(FINAL_CLASSES)
