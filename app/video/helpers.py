"""
Video Processing Helpers
========================
Utility functions for video processing pipeline.
"""

import hashlib
from typing import Dict, List


def generate_pseudo_id(track_id: int, date_str: str) -> str:
    """Generate pseudonymized ID from track ID and date."""
    raw = f"{track_id}_{date_str}_salt_xyz"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def lat_long_to_h3(lat: float, lng: float, resolution: int = 9) -> str:
    """Convert lat/long to H3 hexagon index."""
    try:
        import h3
        return h3.latlng_to_cell(lat, lng, resolution)
    except ImportError:
        return None
    except Exception:
        return None


def get_zone(x: float, y: float, frame_width: int, frame_height: int) -> str:
    """Divide frame into 3x3 grid zones."""
    col = int(x / frame_width * 3)
    row = int(y / frame_height * 3)
    zones = [
        ["entrance", "center-front", "exit"],
        ["bar-left", "center", "bar-right"],
        ["seating-left", "back", "seating-right"]
    ]
    row = min(row, 2)
    col = min(col, 2)
    return zones[row][col]


def estimate_demographics_from_crop(person_crop, box_height: float = 0, frame_height: float = 0) -> tuple:
    """
    Estimate age bracket and gender from person crop using real ML models.
    """
    try:
        from demographics import estimate_demographics
        return estimate_demographics(person_crop, box_height, frame_height)
    except ImportError:
        return None, None
    except Exception as e:
        print(f"Demographics error: {e}")
        return None, None


def estimate_demographics_simple(box_height: float, frame_height: float) -> tuple:
    """Fallback when no crop available - returns unknown."""
    return None, None


class SimpleCentroidTracker:
    """Simple centroid-based object tracker with adaptive distance threshold."""

    def __init__(self, max_disappeared: int = 50, max_distance: int = 300):
        self.next_id = 0
        self.objects: Dict[int, tuple] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: tuple) -> int:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids: List[tuple]) -> Dict[int, tuple]:
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            used_centroids = set()
            used_objects = set()

            matches = []
            for oid, obj_centroid in zip(object_ids, object_centroids):
                for idx, centroid in enumerate(centroids):
                    dist = ((obj_centroid[0] - centroid[0])**2 + (obj_centroid[1] - centroid[1])**2)**0.5
                    if dist < self.max_distance:
                        matches.append((dist, oid, idx))

            matches.sort(key=lambda x: x[0])
            for dist, oid, idx in matches:
                if oid in used_objects or idx in used_centroids:
                    continue
                self.objects[oid] = centroids[idx]
                self.disappeared[oid] = 0
                used_objects.add(oid)
                used_centroids.add(idx)

            for oid in object_ids:
                if oid not in used_objects:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)

            for idx, centroid in enumerate(centroids):
                if idx not in used_centroids:
                    self.register(centroid)

        return self.objects
