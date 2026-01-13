"""Tests for validation module."""

import pytest
from pathlib import Path

from promptdiff.validation import (
    validate_model_name,
    safe_path,
    validate_prompt_id,
    mask_api_key
)
from promptdiff.errors import ValidationError


class TestValidateModelName:
    """Tests for model name validation."""
    
    def test_valid_name(self):
        """Test valid model names."""
        assert validate_model_name("gpt-4") == "gpt-4"
        assert validate_model_name("claude-3-opus") == "claude-3-opus"
        assert validate_model_name("llama3:latest") == "llama3:latest"
        assert validate_model_name("model_123") == "model_123"
    
    def test_empty_name(self):
        """Test empty model name raises error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_model_name("")
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_model_name("   ")
    
    def test_too_long(self):
        """Test model name too long."""
        long_name = "a" * 101
        with pytest.raises(ValidationError, match="too long"):
            validate_model_name(long_name)
    
    def test_invalid_characters(self):
        """Test invalid characters in model name."""
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_model_name("model@name")
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_model_name("model name")
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_model_name("model#name")


class TestSafePath:
    """Tests for safe path resolution."""
    
    def test_basic_path(self):
        """Test basic path resolution."""
        path = safe_path("test.json")
        assert isinstance(path, Path)
    
    def test_path_traversal_detection(self, tmp_path):
        """Test path traversal detection."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        
        # This should work
        safe_path("file.json", base_dir)
        
        # This should fail
        with pytest.raises(ValidationError, match="Path traversal detected"):
            safe_path("../../etc/passwd", base_dir)


class TestMaskApiKey:
    """Tests for API key masking."""
    
    def test_mask_key(self):
        """Test API key masking."""
        key = "sk-1234567890abcdef"
        masked = mask_api_key(key)
        assert masked.startswith("sk-1")
        assert masked.endswith("cdef")
        assert "..." in masked
        assert len(masked) < len(key)
    
    def test_short_key(self):
        """Test masking of short keys."""
        assert mask_api_key("short") == "****"
        assert mask_api_key("ab") == "****"
    
    def test_none_key(self):
        """Test masking of None key."""
        assert mask_api_key(None) == "not set"
        assert mask_api_key("") == "not set"
