"""Tests for pipeline/zones.py"""

import json
import os
import tempfile

import pytest
from shapely.geometry import Polygon

from pipeline.zones import get_zone_for_bbox, get_zone_for_point, load_zones


@pytest.fixture
def zones_file(tmp_path):
    """Create a temporary zones JSON file."""
    data = {
        "camera_01": {
            "frame_width": 1920,
            "frame_height": 1080,
            "bar_zone": [[100, 100], [500, 100], [500, 400], [100, 400]],
            "service_rail": [[100, 350], [500, 350], [500, 420], [100, 420]],
            "queue_zone": [[550, 100], [800, 100], [800, 400], [550, 400]],
        }
    }
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(data))
    return str(path)


class TestLoadZones:
    def test_loads_correct_polygon_count(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert len(zones) == 3
        assert set(zones.keys()) == {"bar_zone", "service_rail", "queue_zone"}

    def test_all_values_are_polygons(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        for name, poly in zones.items():
            assert isinstance(poly, Polygon), f"{name} is not a Polygon"

    def test_missing_camera_raises(self, zones_file):
        with pytest.raises(KeyError, match="camera_99"):
            load_zones(zones_file, "camera_99")

    def test_scaling_doubles_coordinates(self, zones_file):
        zones = load_zones(zones_file, "camera_01", actual_width=3840, actual_height=2160)
        # bar_zone original top-left is (100, 100), scaled should be (200, 200)
        bar = zones["bar_zone"]
        coords = list(bar.exterior.coords)
        assert coords[0] == (200.0, 200.0)

    def test_no_scaling_at_native_resolution(self, zones_file):
        zones = load_zones(zones_file, "camera_01", actual_width=1920, actual_height=1080)
        bar = zones["bar_zone"]
        coords = list(bar.exterior.coords)
        assert coords[0] == (100.0, 100.0)


class TestGetZoneForPoint:
    def test_point_inside_bar_zone(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_point(300, 250, zones) == "bar_zone"

    def test_point_inside_queue_zone(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_point(675, 250, zones) == "queue_zone"

    def test_point_outside_all_zones(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_point(1500, 900, zones) is None

    def test_point_at_origin_outside(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_point(0, 0, zones) is None


class TestGetZoneForBbox:
    def test_bbox_center_in_bar_zone(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        # bbox center = (300, 250) -> bar_zone
        assert get_zone_for_bbox(200, 200, 400, 300, zones) == "bar_zone"

    def test_bbox_center_calculation(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        # bbox center = (675, 250) -> queue_zone
        assert get_zone_for_bbox(600, 200, 750, 300, zones) == "queue_zone"

    def test_bbox_center_outside_all_zones(self, zones_file):
        zones = load_zones(zones_file, "camera_01")
        assert get_zone_for_bbox(1400, 800, 1600, 1000, zones) is None
