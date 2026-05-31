"""
Return Visitor Re-Identification Module
=======================================
Uses face embeddings to track visitors across sessions.

How it works:
1. Detect face in person crop using SCRFD (via InsightFace buffalo_l)
2. Generate 512-dim embedding using ArcFace (via InsightFace buffalo_l)
3. Compare against stored embeddings using cosine similarity
4. If similarity > threshold (0.68), it's a return visitor
5. Store new embeddings for new visitors
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import hashlib
import pickle

# Lazy-loaded InsightFace app
_insightface_app = None


def _load_insightface():
    """Get the shared InsightFace singleton from models.py."""
    global _insightface_app
    if _insightface_app is None:
        try:
            from app.video.models import get_shared_insightface
            _insightface_app = get_shared_insightface()
        except Exception as e:
            print(f"Failed to load InsightFace: {e}")
            _insightface_app = None
    return _insightface_app


def extract_face_embedding(person_crop: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Extract face embedding from a person crop.

    Args:
        person_crop: BGR image of the person (numpy array)

    Returns:
        Tuple of (embedding, quality_score) or (None, 0) if no face found
        - embedding: 512-dimensional numpy array (ArcFace, L2-normalized)
        - quality_score: 0-1 detection confidence from SCRFD
    """
    app = _load_insightface()
    if app is None:
        return None, 0.0

    try:
        # InsightFace expects BGR numpy array directly
        faces = app.get(person_crop)

        if not faces:
            return None, 0.0

        # Pick the face with the highest detection score
        best_face = max(faces, key=lambda f: f.det_score)

        det_score = float(best_face.det_score)
        if det_score < 0.5:
            return None, 0.0

        embedding = best_face.normed_embedding.astype(np.float32)

        return embedding, det_score

    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None, 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Serialize embedding for database storage."""
    return pickle.dumps(embedding.astype(np.float32))


def deserialize_embedding(data: bytes) -> np.ndarray:
    """Deserialize embedding from database."""
    return pickle.loads(data)


def generate_visitor_id(embedding: np.ndarray) -> str:
    """Generate stable visitor ID from embedding."""
    # Use hash of embedding for stable ID
    embedding_bytes = embedding.tobytes()
    return hashlib.md5(embedding_bytes).hexdigest()[:16]


class VisitorMatcher:
    """
    Matches visitors against stored embeddings.
    Handles caching and batch operations for efficiency.
    """

    def __init__(self, venue_id: str, similarity_threshold: float = 0.68):
        """
        Initialize matcher for a venue.

        Args:
            venue_id: The venue to match against
            similarity_threshold: Minimum similarity to consider a match (0.68 = 68%)
        """
        self.venue_id = venue_id
        self.threshold = similarity_threshold
        self.cached_embeddings: Dict[str, np.ndarray] = {}
        self.cached_visitors: Dict[str, dict] = {}
        self._loaded = False

    def load_embeddings(self, db_rows: List[dict]):
        """
        Load embeddings from database rows into memory.

        Args:
            db_rows: List of rows from visitor_embeddings table
        """
        self.cached_embeddings = {}
        self.cached_visitors = {}

        for row in db_rows:
            try:
                visitor_id = row["visitor_id"]
                embedding = deserialize_embedding(row["embedding"])
                self.cached_embeddings[visitor_id] = embedding
                self.cached_visitors[visitor_id] = {
                    "visitor_id": visitor_id,
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "visit_count": row["visit_count"],
                    "age_bracket": row["age_bracket"],
                    "gender": row["gender"]
                }
            except Exception as e:
                print(f"Error loading embedding for {row.get('visitor_id')}: {e}")

        self._loaded = True
        print(f"Loaded {len(self.cached_embeddings)} visitor embeddings for {self.venue_id}")

    def find_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find matching visitor for an embedding.

        Args:
            embedding: 512-dim face embedding

        Returns:
            Tuple of (visitor_id, similarity) or (None, 0) if no match
        """
        if not self.cached_embeddings:
            return None, 0.0

        best_match = None
        best_similarity = 0.0

        for visitor_id, stored_embedding in self.cached_embeddings.items():
            similarity = cosine_similarity(embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = visitor_id

        if best_similarity >= self.threshold:
            return best_match, best_similarity

        return None, best_similarity

    def add_visitor(self, visitor_id: str, embedding: np.ndarray, visitor_info: dict):
        """Add a new visitor to the cache."""
        self.cached_embeddings[visitor_id] = embedding
        self.cached_visitors[visitor_id] = visitor_info

    def get_visitor_info(self, visitor_id: str) -> Optional[dict]:
        """Get cached visitor info."""
        return self.cached_visitors.get(visitor_id)

    @property
    def visitor_count(self) -> int:
        """Number of known visitors."""
        return len(self.cached_embeddings)


# Convenience functions for integration

def process_person_for_reid(
    person_crop: np.ndarray,
    matcher: VisitorMatcher,
    timestamp: datetime,
    age_bracket: Optional[str] = None,
    gender: Optional[str] = None
) -> Tuple[str, bool, float]:
    """
    Process a person crop for re-identification.

    Args:
        person_crop: BGR image of the person
        matcher: VisitorMatcher instance with loaded embeddings
        timestamp: When the person was seen
        age_bracket: Detected age bracket (optional)
        gender: Detected gender (optional)

    Returns:
        Tuple of (visitor_id, is_return_visitor, confidence)
    """
    # Extract embedding
    embedding, quality = extract_face_embedding(person_crop)

    if embedding is None:
        # No face found, generate random ID (can't track)
        random_id = hashlib.md5(str(timestamp.timestamp()).encode()).hexdigest()[:16]
        return random_id, False, 0.0

    # Try to match
    matched_id, similarity = matcher.find_match(embedding)

    if matched_id:
        # Return visitor found!
        return matched_id, True, similarity
    else:
        # New visitor
        visitor_id = generate_visitor_id(embedding)

        # Add to cache
        matcher.add_visitor(visitor_id, embedding, {
            "visitor_id": visitor_id,
            "first_seen": timestamp,
            "last_seen": timestamp,
            "visit_count": 1,
            "age_bracket": age_bracket,
            "gender": gender
        })

        return visitor_id, False, quality


def test_reid():
    """Test the re-identification module."""
    print("Testing Re-ID module...")

    # Test model loading
    print("\n1. Loading InsightFace (SCRFD + ArcFace)...")
    app = _load_insightface()
    print(f"   InsightFace loaded: {app is not None}")

    # Test with dummy image
    print("\n2. Testing embedding extraction...")
    dummy = np.zeros((200, 100, 3), dtype=np.uint8)
    dummy[:] = (128, 128, 128)

    embedding, quality = extract_face_embedding(dummy)
    print(f"   Embedding from dummy (no face): {embedding is None} (expected: True)")

    # Test matcher
    print("\n3. Testing VisitorMatcher...")
    matcher = VisitorMatcher("test_venue", similarity_threshold=0.68)
    print(f"   Matcher created, threshold: {matcher.threshold}")
    print(f"   Embedding model: arcface")

    # Test cosine similarity with known vectors
    print("\n4. Testing cosine similarity...")
    a = np.random.randn(512).astype(np.float32)
    a = a / np.linalg.norm(a)
    b = a.copy()
    sim = cosine_similarity(a, b)
    print(f"   Identical vectors similarity: {sim:.4f} (expected: 1.0)")

    c = np.random.randn(512).astype(np.float32)
    c = c / np.linalg.norm(c)
    sim2 = cosine_similarity(a, c)
    print(f"   Random vectors similarity: {sim2:.4f} (expected: ~0.0)")

    # Test serialize/deserialize round-trip
    print("\n5. Testing embedding serialization...")
    data = serialize_embedding(a)
    recovered = deserialize_embedding(data)
    assert np.allclose(a, recovered), "Round-trip serialization failed"
    print(f"   Serialize/deserialize round-trip: OK ({len(data)} bytes)")

    print("\nRe-ID module ready!")
    print("Note: Real testing requires images with faces")


if __name__ == "__main__":
    test_reid()
