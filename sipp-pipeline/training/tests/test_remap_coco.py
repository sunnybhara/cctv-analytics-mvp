"""Tests for training/remap_coco.py — tests the mapping logic, not YOLO inference."""

import pytest
from config import COCO_REMAP, FINAL_CLASSES


class TestCocoRemapConfig:
    def test_keyboard_maps_to_pos_terminal(self):
        assert COCO_REMAP["keyboard"] == "pos_terminal"
        assert "pos_terminal" in FINAL_CLASSES
        assert FINAL_CLASSES.index("pos_terminal") == 11

    def test_dining_table_maps_to_bar_counter(self):
        assert COCO_REMAP["dining_table"] == "bar_counter"
        assert "bar_counter" in FINAL_CLASSES
        assert FINAL_CLASSES.index("bar_counter") == 14

    def test_chair_maps_to_bar_stool(self):
        assert COCO_REMAP["chair"] == "bar_stool"
        assert "bar_stool" in FINAL_CLASSES

    def test_tv_maps_to_bar_screen(self):
        assert COCO_REMAP["tv"] == "bar_screen"
        assert "bar_screen" in FINAL_CLASSES

    def test_handbag_discarded(self):
        assert COCO_REMAP["handbag"] is None

    def test_book_discarded(self):
        assert COCO_REMAP["book"] is None

    def test_laptop_discarded(self):
        assert COCO_REMAP["laptop"] is None

    def test_classes_not_in_remap_are_skipped(self):
        # "person" is not in COCO_REMAP — should not be remapped
        assert "person" not in COCO_REMAP

    def test_all_remap_targets_in_final_classes(self):
        """Every non-None remap target must exist in FINAL_CLASSES."""
        for coco_name, bar_name in COCO_REMAP.items():
            if bar_name is not None:
                assert bar_name in FINAL_CLASSES, f"{coco_name}->{bar_name} not in FINAL_CLASSES"


class TestFinalClasses:
    def test_person_is_class_zero(self):
        assert FINAL_CLASSES[0] == "person"

    def test_no_duplicates(self):
        assert len(FINAL_CLASSES) == len(set(FINAL_CLASSES))

    def test_class_count(self):
        assert len(FINAL_CLASSES) == 17
