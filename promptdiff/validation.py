"""Input validation utilities."""

import re
from pathlib import Path
from typing import Optional

from promptdiff.errors import ValidationError


def validate_model_name(name: str) -> str:
    """
    Validate model name format.
    
    Args:
        name: Model name to validate
    
    Returns:
        Validated model name
    
    Raises:
        ValidationError: If model name is invalid
    """
    if not name or not name.strip():
        raise ValidationError("Model name cannot be empty")
    
    name = name.strip()
    
    if len(name) > 100:
        raise ValidationError(f"Model name too long (max 100 characters, got {len(name)})")
    
    # Allow alphanumeric, dots, dashes, underscores, and colons (for tags)
    if not re.match(r'^[a-zA-Z0-9._:-]+$', name):
        raise ValidationError(
            f"Model name contains invalid characters. "
            f"Allowed: letters, numbers, dots, dashes, underscores, colons"
        )
    
    return name


def safe_path(file_path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Safely resolve a file path, preventing path traversal attacks.
    
    Args:
        file_path: Path to resolve
        base_dir: Optional base directory to restrict paths to
    
    Returns:
        Resolved Path object
    
    Raises:
        ValidationError: If path traversal is detected
    """
    path = Path(file_path).resolve()
    
    if base_dir:
        base_dir = base_dir.resolve()
        # Ensure the resolved path is within base_dir
        try:
            path.relative_to(base_dir)
        except ValueError:
            raise ValidationError(
                f"Path traversal detected: {file_path} resolves outside {base_dir}"
            )
    
    return path


def validate_prompt_id(prompt_id: str) -> str:
    """
    Validate prompt ID format.
    
    Args:
        prompt_id: Prompt ID to validate
    
    Returns:
        Validated prompt ID
    
    Raises:
        ValidationError: If prompt ID is invalid
    """
    if not prompt_id or not prompt_id.strip():
        raise ValidationError("Prompt ID cannot be empty")
    
    prompt_id = prompt_id.strip()
    
    if len(prompt_id) > 100:
        raise ValidationError(f"Prompt ID too long (max 100 characters, got {len(prompt_id)})")
    
    # Allow alphanumeric, dots, dashes, underscores
    if not re.match(r'^[a-zA-Z0-9._-]+$', prompt_id):
        raise ValidationError(
            f"Prompt ID contains invalid characters. "
            f"Allowed: letters, numbers, dots, dashes, underscores"
        )
    
    return prompt_id


def mask_api_key(api_key: Optional[str]) -> str:
    """
    Mask API key for safe display in logs/errors.
    
    Args:
        api_key: API key to mask
    
    Returns:
        Masked API key string
    """
    if not api_key:
        return "not set"
    
    if len(api_key) <= 8:
        return "****"
    
    return f"{api_key[:4]}...{api_key[-4:]}"
