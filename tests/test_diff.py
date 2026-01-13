"""Tests for diff modules."""

import pytest
from promptdiff.diff.text import text_diff, text_diff_stats
from promptdiff.diff.semantic import semantic_diff, semantic_diff_detailed


class TestTextDiff:
    """Tests for text diff functions."""
    
    def test_identical_texts(self):
        """Test diff of identical texts."""
        text = "Hello world"
        diff = text_diff(text, text)
        # Should be empty or minimal for identical texts
        assert isinstance(diff, str)
    
    def test_different_texts(self):
        """Test diff of different texts."""
        a = "Hello world"
        b = "Hello Python"
        diff = text_diff(a, b)
        assert isinstance(diff, str)
        assert len(diff) > 0
    
    def test_text_diff_stats(self):
        """Test text diff statistics."""
        a = "Line 1\nLine 2\nLine 3"
        b = "Line 1\nLine 2 modified\nLine 3\nLine 4"
        stats = text_diff_stats(a, b)
        
        assert "similarity" in stats
        assert "added_lines" in stats
        assert "removed_lines" in stats
        assert "baseline_length" in stats
        assert "candidate_length" in stats
        assert "length_diff" in stats
        
        assert 0 <= stats["similarity"] <= 1
        assert stats["added_lines"] >= 0
        assert stats["removed_lines"] >= 0


class TestSemanticDiff:
    """Tests for semantic diff functions."""
    
    def test_semantic_diff_identical(self):
        """Test semantic diff of identical texts."""
        text = "The quick brown fox jumps over the lazy dog"
        similarity = semantic_diff(text, text)
        assert 0 <= similarity <= 1
        # Identical texts should have high similarity
        assert similarity > 0.9
    
    def test_semantic_diff_similar(self):
        """Test semantic diff of similar texts."""
        a = "The cat sat on the mat"
        b = "A cat was sitting on a mat"
        similarity = semantic_diff(a, b)
        assert 0 <= similarity <= 1
        # Similar meanings should have decent similarity
        assert similarity > 0.5
    
    def test_semantic_diff_different(self):
        """Test semantic diff of different texts."""
        a = "The weather is sunny today"
        b = "Python is a programming language"
        similarity = semantic_diff(a, b)
        assert 0 <= similarity <= 1
        # Different topics should have lower similarity
        assert similarity < 0.8
    
    def test_semantic_diff_detailed(self):
        """Test detailed semantic diff."""
        a = "Hello world"
        b = "Hi there"
        stats = semantic_diff_detailed(a, b)
        
        assert "similarity" in stats
        assert "baseline_length" in stats
        assert "candidate_length" in stats
        assert "is_similar" in stats
        assert "is_very_similar" in stats
        
        assert isinstance(stats["is_similar"], bool)
        assert isinstance(stats["is_very_similar"], bool)
