"""Tests for config module."""

import pytest
import os
from promptdiff.config import ModelConfig, ModelProvider
from promptdiff.errors import ConfigurationError


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_from_string_openai(self):
        """Test parsing OpenAI model string."""
        config = ModelConfig.from_string("gpt-4")
        assert config.provider == ModelProvider.OPENAI
        assert config.name == "gpt-4"
    
    def test_from_string_with_provider(self):
        """Test parsing model string with provider."""
        config = ModelConfig.from_string("ollama:llama3")
        assert config.provider == ModelProvider.OLLAMA
        assert config.name == "llama3"
    
    def test_from_string_empty(self):
        """Test error with empty model string."""
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            ModelConfig.from_string("")
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            ModelConfig.from_string("   ")
    
    def test_from_string_invalid_provider(self):
        """Test error with invalid provider."""
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            ModelConfig.from_string("invalid:model")
    
    def test_validation_in_model_name(self):
        """Test that model name is validated."""
        with pytest.raises(ConfigurationError, match="Invalid model name"):
            ModelConfig.from_string("model@invalid")
