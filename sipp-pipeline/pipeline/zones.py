"""
Zone Logic
==========
Load zone polygons from JSON, point-in-polygon checks via Shapely.
"""

import json
from shapely.geometry import Point, Polygon


def load_zones(
    zones_file: str,
    camera_id: str,
    actual_width: int = 1920,
    actual_height: int = 1080,
) -> dict[str, Polygon]:
    """Load zone polygons from JSON file for a specific camera.

    Scales coordinates if camera resolution differs from the reference frame
    stored in the JSON (defaults to 1920x1080).

    Returns: {"bar_zone": Polygon(...), "queue_zone": Polygon(...), ...}
    """
    with open(zones_file) as f:
        data = json.load(f)

    if camera_id not in data:
        raise KeyError(f"Camera '{camera_id}' not found in zones file. Available: {list(data.keys())}")

    camera_data = data[camera_id]
    ref_width = camera_data.get("frame_width", 1920)
    ref_height = camera_data.get("frame_height", 1080)

    scale_x = actual_width / ref_width
    scale_y = actual_height / ref_height

    zones: dict[str, Polygon] = {}
    for key, coords in camera_data.items():
        if key in ("frame_width", "frame_height"):
            continue
        scaled = [(x * scale_x, y * scale_y) for x, y in coords]
        zones[key] = Polygon(scaled)

    return zones


def get_zone_for_point(x: float, y: float, zones: dict[str, Polygon]) -> str | None:
    """Return the zone name containing this point, or None."""
    point = Point(x, y)
    for name, polygon in zones.items():
        if polygon.contains(point):
            return name
    return None


def get_zone_for_bbox(
    x1: float, y1: float, x2: float, y2: float, zones: dict[str, Polygon]
) -> str | None:
    """Return the zone name containing the center of this bounding box."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return get_zone_for_point(cx, cy, zones)
