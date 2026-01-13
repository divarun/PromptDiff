"""Tests for runner module."""

import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from promptdiff.runner import render_prompt
from promptdiff.errors import ValidationError


class TestRenderPrompt:
    """Tests for prompt rendering."""
    
    def test_simple_substitution(self):
        """Test simple variable substitution."""
        template = "Hello {{name}}"
        vars = {"name": "World"}
        result = render_prompt(template, vars)
        assert result == "Hello World"
    
    def test_multiple_vars(self):
        """Test multiple variable substitution."""
        template = "{{greeting}} {{name}}"
        vars = {"greeting": "Hi", "name": "Alice"}
        result = render_prompt(template, vars)
        assert result == "Hi Alice"
    
    def test_missing_variable(self):
        """Test error when variable is missing."""
        template = "Hello {{name}}"
        vars = {}
        with pytest.raises(KeyError, match="Variable 'name' not found"):
            render_prompt(template, vars)
    
    def test_nested_braces(self):
        """Test that nested braces don't break substitution."""
        template = "Value: {{value}}"
        vars = {"value": "{{not_substituted}}"}
        result = render_prompt(template, vars)
        assert result == "Value: {{not_substituted}}"
    
    def test_special_characters(self):
        """Test substitution with special characters."""
        template = "Text: {{text}}"
        vars = {"text": "Hello\nWorld\tTest"}
        result = render_prompt(template, vars)
        assert result == "Text: Hello\nWorld\tTest"


class TestRunDiff:
    """Tests for run_diff function."""
    
    @pytest.fixture
    def sample_prompts_file(self, tmp_path):
        """Create a sample prompts file."""
        prompts_file = tmp_path / "prompts.json"
        prompts = [
            {
                "id": "test1",
                "prompt": "Say hello",
                "vars": {}
            }
        ]
        with open(prompts_file, "w", encoding="utf-8") as f:
            json.dump(prompts, f)
        return str(prompts_file)
    
    def test_prompts_file_not_found(self):
        """Test error when prompts file doesn't exist."""
        from promptdiff.runner import run_diff
        
        with pytest.raises(FileNotFoundError):
            run_diff("nonexistent.json", "test:model1", "test:model2")
    
    def test_invalid_json(self, tmp_path):
        """Test error with invalid JSON."""
        from promptdiff.runner import run_diff
        
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not json")
        
        with pytest.raises(ValidationError, match="Invalid JSON"):
            run_diff(str(invalid_file), "test:model1", "test:model2")
    
    def test_not_array(self, tmp_path):
        """Test error when JSON is not an array."""
        from promptdiff.runner import run_diff
        
        invalid_file = tmp_path / "not_array.json"
        with open(invalid_file, "w") as f:
            json.dump({"not": "array"}, f)
        
        with pytest.raises(ValidationError, match="must contain a JSON array"):
            run_diff(str(invalid_file), "test:model1", "test:model2")
