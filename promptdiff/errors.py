"""Custom exceptions for PromptDiff."""

from typing import Optional, Dict, Any


class PromptDiffError(Exception):
    """Base exception for PromptDiff errors."""
    
    def __init__(
        self,
        message: str,
        prompt_id: Optional[str] = None,
        model: Optional[str] = None,
        **context: Any
    ):
        """
        Initialize PromptDiff error.
        
        Args:
            message: Error message
            prompt_id: Optional prompt ID where error occurred
            model: Optional model name where error occurred
            **context: Additional context information
        """
        self.message = message
        self.prompt_id = prompt_id
        self.model = model
        self.context = context
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with context."""
        parts = [self.message]
        if self.prompt_id:
            parts.append(f"Prompt: {self.prompt_id}")
        if self.model:
            parts.append(f"Model: {self.model}")
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        return " | ".join(parts)


class ModelError(PromptDiffError):
    """Error related to model calls."""
    pass


class ValidationError(PromptDiffError):
    """Error related to input validation."""
    pass


class ConfigurationError(PromptDiffError):
    """Error related to configuration."""
    pass
