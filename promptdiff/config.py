"""Configuration and model integration."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from promptdiff.logging_config import get_logger
from promptdiff.errors import ModelError, ConfigurationError
from promptdiff.validation import validate_model_name, mask_api_key

logger = get_logger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root or current directory
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LOCAL = "local"


class ModelConfig:
    """Configuration for a model."""
    
    def __init__(
        self,
        name: str,
        provider: ModelProvider = ModelProvider.OPENAI,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.name = name
        self.provider = provider
        self.api_key = api_key or self._get_api_key(provider)
        self.base_url = base_url
        self.extra_params = kwargs
    
    @staticmethod
    def _get_api_key(provider: ModelProvider) -> Optional[str]:
        """Get API key from environment variables."""
        env_map = {
            ModelProvider.OPENAI: "OPENAI_API_KEY",
            ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }
        env_var = env_map.get(provider)
        return os.getenv(env_var) if env_var else None
    
    @classmethod
    def from_string(cls, model_str: str) -> "ModelConfig":
        """
        Parse model string like 'gpt-4', 'openai:gpt-4', 'anthropic:claude-3-opus'.
        
        Format: [provider:]model-name
        
        Args:
            model_str: Model identifier string
        
        Returns:
            ModelConfig instance
        
        Raises:
            ConfigurationError: If model string is invalid
        """
        if not model_str or not model_str.strip():
            raise ConfigurationError("Model string cannot be empty")
        
        model_str = model_str.strip()
        
        if ":" in model_str:
            provider_str, model_name = model_str.split(":", 1)
            try:
                provider = ModelProvider(provider_str.lower())
            except ValueError:
                raise ConfigurationError(f"Unknown provider: {provider_str}")
        else:
            # Default to OpenAI for backward compatibility
            provider = ModelProvider.OPENAI
            model_name = model_str
        
        # Validate model name
        try:
            model_name = validate_model_name(model_name)
        except Exception as e:
            raise ConfigurationError(f"Invalid model name: {e}")
        
        return cls(name=model_name, provider=provider)


def call_model(config: ModelConfig, prompt: str, **kwargs) -> str:
    """
    Call a model with the given prompt.
    
    Args:
        config: Model configuration
        prompt: The prompt to send
        **kwargs: Additional parameters (temperature, max_tokens, etc.)
    
    Returns:
        Model output as string
    """
    if config.provider == ModelProvider.OPENAI:
        return _call_openai(config, prompt, **kwargs)
    elif config.provider == ModelProvider.ANTHROPIC:
        return _call_anthropic(config, prompt, **kwargs)
    elif config.provider == ModelProvider.OLLAMA:
        return _call_ollama(config, prompt, **kwargs)
    elif config.provider == ModelProvider.LOCAL:
        return _call_local(config, prompt, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


def _call_openai(config: ModelConfig, prompt: str, **kwargs) -> str:
    """Call OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "OpenAI SDK not installed. Install with: pip install openai"
        )
    
    if not config.api_key:
        masked_key = mask_api_key(config.api_key)
        raise ModelError(
            f"OpenAI API key not found. Set OPENAI_API_KEY environment variable.",
            model=config.name
        )
    
    try:
        client = openai.OpenAI(api_key=config.api_key, base_url=config.base_url)
        
        response = client.chat.completions.create(
            model=config.name,
            messages=[{"role": "user", "content": prompt}],
            **{**config.extra_params, **kwargs}
        )
        
        return response.choices[0].message.content
    except openai.AuthenticationError as e:
        masked_key = mask_api_key(config.api_key)
        raise ModelError(
            f"OpenAI authentication failed. API key: {masked_key}",
            model=config.name
        ) from e
    except Exception as e:
        raise ModelError(f"OpenAI API error: {e}", model=config.name) from e


def _call_anthropic(config: ModelConfig, prompt: str, **kwargs) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK not installed. Install with: pip install anthropic"
        )
    
    if not config.api_key:
        masked_key = mask_api_key(config.api_key)
        raise ModelError(
            f"Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.",
            model=config.name
        )
    
    try:
        client = anthropic.Anthropic(api_key=config.api_key)
        
        response = client.messages.create(
            model=config.name,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
            **{k: v for k, v in {**config.extra_params, **kwargs}.items() 
               if k != "max_tokens"}
        )
        
        return response.content[0].text
    except anthropic.AuthenticationError as e:
        masked_key = mask_api_key(config.api_key)
        raise ModelError(
            f"Anthropic authentication failed. API key: {masked_key}",
            model=config.name
        ) from e
    except Exception as e:
        raise ModelError(f"Anthropic API error: {e}", model=config.name) from e


def _call_ollama(config: ModelConfig, prompt: str, **kwargs) -> str:
    """Call Ollama API."""
    import requests
    
    # Default Ollama base URL
    base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Ollama API endpoint
    api_url = f"{base_url}/api/chat"
    
    # Prepare request
    payload = {
        "model": config.name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    }
    
    # Add optional parameters
    options = {}
    if "temperature" in kwargs:
        options["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs or "num_predict" in kwargs:
        options["num_predict"] = kwargs.get("max_tokens") or kwargs.get("num_predict", 512)
    
    # Merge any extra params from config
    if config.extra_params:
        options.update(config.extra_params)
    
    if options:
        payload["options"] = options
    
    try:
        # Try the request
        response = requests.post(api_url, json=payload, timeout=kwargs.get("timeout", 300))
        
        # If 500 error and no tag in model name, try with :latest
        if response.status_code == 500 and ":" not in config.name:
            # Retry with :latest tag
            payload["model"] = f"{config.name}:latest"
            try:
                response = requests.post(api_url, json=payload, timeout=kwargs.get("timeout", 300))
            except requests.exceptions.RequestException:
                pass  # Fall through to error handling
        
        # Check for errors and provide detailed messages
        if response.status_code != 200:
            error_msg = f"Ollama API error: {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f" - {error_data['error']}"
                # Also check for common issues
                if response.status_code == 404:
                    available_models = _get_ollama_models(base_url)
                    error_msg += f"\nModel '{config.name}' not found."
                    if available_models != "unable to fetch":
                        error_msg += f"\nAvailable models: {available_models}"
                    error_msg += f"\nTry: ollama pull {config.name}"
                elif response.status_code == 500:
                    available_models = _get_ollama_models(base_url)
                    error_msg += f"\nServer error. This might indicate:"
                    error_msg += f"\n  1. Model '{config.name}' is loading (first use can be slow)"
                    error_msg += f"\n  2. Model name might need a tag (try '{config.name}:latest')"
                    if available_models != "unable to fetch":
                        error_msg += f"\n  3. Available models: {available_models}"
                    error_msg += f"\n  4. Check with: ollama list"
                    error_msg += f"\n  5. Try pulling: ollama pull {config.name}"
                    if ":" not in config.name:
                        error_msg += f"\n  6. Or try using: ollama:{config.name}:latest"
            except:
                error_msg += f" - {response.text[:200]}"
            raise ModelError(error_msg, model=config.name)
        
        # Success - parse response
        result = response.json()
        
        # Validate response structure
        if "message" not in result:
            raise ModelError(
                f"Unexpected Ollama response format: {result}",
                model=config.name
            )
        if "content" not in result["message"]:
            raise ModelError(
                f"Ollama response missing content: {result}",
                model=config.name
            )
        
        return result["message"]["content"]
        
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Could not connect to Ollama at {base_url}. "
            "Make sure Ollama is running: ollama serve"
        ) from e
    except requests.exceptions.Timeout as e:
        raise TimeoutError(
            f"Ollama request timed out after {kwargs.get('timeout', 300)} seconds. "
            "The model might be loading or the request is too large."
        ) from e
    except ModelError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        raise ModelError(
            f"Unexpected error calling Ollama: {e}",
            model=config.name
        ) from e


def _get_ollama_models(base_url: str) -> str:
    """Get list of available Ollama models for error messages."""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model.get("name", "unknown") for model in data.get("models", [])]
            return ", ".join(models[:5]) if models else "none found"
    except:
        pass
    return "unable to fetch"


def _call_local(config: ModelConfig, prompt: str, **kwargs) -> str:
    """
    Call a local model (stub implementation).
    
    For local models, you would integrate with:
    - vLLM
    - HuggingFace transformers
    - etc.
    """
    # Stub implementation
    return f"[Local model {config.name} output for prompt]"
