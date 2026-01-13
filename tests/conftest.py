"""Pytest configuration and fixtures."""

import pytest
import json
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def sample_prompts(tmp_path) -> str:
    """Create a sample prompts file for testing."""
    prompts_file = tmp_path / "prompts.json"
    prompts = [
        {
            "id": "test1",
            "prompt": "Say hello",
            "vars": {}
        },
        {
            "id": "test2",
            "prompt": "Hello {{name}}",
            "vars": {"name": "World"}
        }
    ]
    with open(prompts_file, "w", encoding="utf-8") as f:
        json.dump(prompts, f)
    return str(prompts_file)


@pytest.fixture
def mock_model_config():
    """Create a mock model config for testing."""
    from promptdiff.config import ModelConfig, ModelProvider
    return ModelConfig(
        name="test-model",
        provider=ModelProvider.LOCAL
    )
