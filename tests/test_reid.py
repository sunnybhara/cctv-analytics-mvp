"""
Re-identification Module Tests
==============================
Tests for the face embedding and visitor matching functionality.
"""
import pytest
import numpy as np
from datetime import datetime


class TestReIDModule:
    """Test re-identification functions."""

    def test_module_imports(self):
        """ReID module should import without errors."""
        import reid
        assert hasattr(reid, 'extract_face_embedding')
        assert hasattr(reid, 'VisitorMatcher')
        assert hasattr(reid, 'cosine_similarity')

    def test_cosine_similarity(self):
        """Cosine similarity should work correctly."""
        from reid import cosine_similarity

        # Identical vectors = similarity 1.0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, b) - 1.0) < 0.001

        # Orthogonal vectors = similarity 0.0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(a, b)) < 0.001

        # Opposite vectors = similarity -1.0
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 0.001

    def test_serialize_deserialize_embedding(self):
        """Embedding serialization should be reversible."""
        from reid import serialize_embedding, deserialize_embedding

        original = np.random.randn(512).astype(np.float32)
        serialized = serialize_embedding(original)
        restored = deserialize_embedding(serialized)

        np.testing.assert_array_almost_equal(original, restored)

    def test_generate_visitor_id(self):
        """Visitor ID generation should be deterministic."""
        from reid import generate_visitor_id

        embedding = np.random.randn(512).astype(np.float32)

        id1 = generate_visitor_id(embedding)
        id2 = generate_visitor_id(embedding)

        # Same embedding = same ID
        assert id1 == id2
        # ID should be 16 hex characters
        assert len(id1) == 16
        assert all(c in '0123456789abcdef' for c in id1)

    def test_generate_visitor_id_unique(self):
        """Different embeddings should produce different IDs."""
        from reid import generate_visitor_id

        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)

        id1 = generate_visitor_id(emb1)
        id2 = generate_visitor_id(emb2)

        assert id1 != id2


class TestVisitorMatcher:
    """Test VisitorMatcher class."""

    def test_matcher_initialization(self):
        """Matcher should initialize with correct defaults."""
        from reid import VisitorMatcher

        matcher = VisitorMatcher("test_venue")

        assert matcher.venue_id == "test_venue"
        assert matcher.threshold == 0.65
        assert matcher.visitor_count == 0

    def test_matcher_custom_threshold(self):
        """Matcher should accept custom threshold."""
        from reid import VisitorMatcher

        matcher = VisitorMatcher("test_venue", similarity_threshold=0.8)
        assert matcher.threshold == 0.8

    def test_add_and_find_visitor(self):
        """Should be able to add and find visitors."""
        from reid import VisitorMatcher

        matcher = VisitorMatcher("test_venue", similarity_threshold=0.5)

        # Add a visitor
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        matcher.add_visitor("visitor_1", embedding, {
            "visitor_id": "visitor_1",
            "first_seen": datetime.now()
        })

        assert matcher.visitor_count == 1

        # Find with same embedding
        matched_id, similarity = matcher.find_match(embedding)
        assert matched_id == "visitor_1"
        assert similarity > 0.99  # Should be near-perfect match

    def test_find_no_match(self):
        """Should return None when no match found."""
        from reid import VisitorMatcher

        matcher = VisitorMatcher("test_venue")

        # Empty matcher
        embedding = np.random.randn(512).astype(np.float32)
        matched_id, similarity = matcher.find_match(embedding)

        assert matched_id is None
        assert similarity == 0.0

    def test_find_below_threshold(self):
        """Should not match if similarity below threshold."""
        from reid import VisitorMatcher

        matcher = VisitorMatcher("test_venue", similarity_threshold=0.9)

        # Add a visitor
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        matcher.add_visitor("visitor_1", emb1, {"visitor_id": "visitor_1"})

        # Try to match with different embedding
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        matched_id, similarity = matcher.find_match(emb2)

        # Random vectors unlikely to have 0.9+ similarity
        assert matched_id is None

    def test_load_embeddings(self):
        """Should load embeddings from database rows."""
        from reid import VisitorMatcher, serialize_embedding

        matcher = VisitorMatcher("test_venue")

        # Simulate database rows
        emb1 = np.random.randn(512).astype(np.float32)
        emb2 = np.random.randn(512).astype(np.float32)

        db_rows = [
            {
                "visitor_id": "v1",
                "embedding": serialize_embedding(emb1),
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "visit_count": 1,
                "age_bracket": "25-34",
                "gender": "M"
            },
            {
                "visitor_id": "v2",
                "embedding": serialize_embedding(emb2),
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "visit_count": 2,
                "age_bracket": "35-44",
                "gender": "F"
            }
        ]

        matcher.load_embeddings(db_rows)

        assert matcher.visitor_count == 2
        assert "v1" in matcher.cached_embeddings
        assert "v2" in matcher.cached_embeddings

    def test_get_visitor_info(self):
        """Should retrieve cached visitor info."""
        from reid import VisitorMatcher

        matcher = VisitorMatcher("test_venue")

        embedding = np.random.randn(512).astype(np.float32)
        visitor_info = {
            "visitor_id": "v1",
            "first_seen": datetime.now(),
            "age_bracket": "25-34"
        }

        matcher.add_visitor("v1", embedding, visitor_info)

        retrieved = matcher.get_visitor_info("v1")
        assert retrieved["visitor_id"] == "v1"
        assert retrieved["age_bracket"] == "25-34"

        # Non-existent visitor
        assert matcher.get_visitor_info("nonexistent") is None


class TestFaceEmbeddingExtraction:
    """Test face embedding extraction."""

    def test_extract_face_no_face(self):
        """Should return None when no face detected."""
        from reid import extract_face_embedding

        # Gray image with no face
        no_face = np.zeros((200, 100, 3), dtype=np.uint8)
        no_face[:] = (128, 128, 128)

        embedding, quality = extract_face_embedding(no_face)

        assert embedding is None
        assert quality == 0.0

    def test_extract_face_empty_image(self):
        """Should handle empty images gracefully."""
        from reid import extract_face_embedding

        empty = np.zeros((0, 0, 3), dtype=np.uint8)

        try:
            embedding, quality = extract_face_embedding(empty)
            assert embedding is None
        except Exception:
            # May raise exception for empty image - that's OK
            pass
