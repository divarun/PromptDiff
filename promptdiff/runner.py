"""Core runner for PromptDiff."""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from promptdiff.config import ModelConfig, call_model
from promptdiff.diff.text import text_diff, text_diff_stats
from promptdiff.diff.semantic import semantic_diff, semantic_diff_detailed
from promptdiff.logging_config import get_logger
from promptdiff.errors import PromptDiffError, ModelError, ValidationError
from promptdiff.validation import safe_path, validate_prompt_id

logger = get_logger(__name__)


def render_prompt(template: str, vars: Dict[str, Any]) -> str:
    """
    Render a prompt template with variable substitution.
    
    Supports double-brace syntax: {{variable_name}}
    
    Args:
        template: Prompt template string with {{variable}} placeholders
        vars: Dictionary mapping variable names to values
    
    Returns:
        Rendered prompt string with variables substituted
    
    Raises:
        KeyError: If template references a variable not in vars
    
    Example:
        >>> render_prompt("Hello {{name}}", {"name": "World"})
        'Hello World'
        >>> render_prompt("{{greeting}} {{name}}", {"greeting": "Hi", "name": "Alice"})
        'Hi Alice'
    """
    pattern = r'\{\{(\w+)\}\}'
    
    def replacer(match):
        key = match.group(1)
        if key not in vars:
            raise KeyError(f"Variable '{key}' not found in vars. Available: {list(vars.keys())}")
        return str(vars[key])
    
    return re.sub(pattern, replacer, template)


def run_diff(
    prompts_file: str,
    baseline: str,
    candidate: str,
    output_file: Optional[str] = None,
    embedding_model: Optional[str] = None,
    **model_kwargs
) -> List[Dict[str, Any]]:
    """
    Run PromptDiff comparison between baseline and candidate models.
    
    Args:
        prompts_file: Path to JSON file containing prompts
        baseline: Baseline model identifier (e.g., "gpt-4" or "openai:gpt-4")
        candidate: Candidate model identifier
        output_file: Optional path to save results JSON
        embedding_model: Optional embedding model for semantic diff
        **model_kwargs: Additional parameters to pass to model calls
    
    Returns:
        List of diff results
    """
    # Load prompts with path validation
    prompts_path = safe_path(prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    try:
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in prompts file: {e}")
    except Exception as e:
        raise PromptDiffError(f"Error reading prompts file: {e}")
    
    if not isinstance(prompts, list):
        raise ValidationError("Prompts file must contain a JSON array")
    
    # Parse model configs
    baseline_config = ModelConfig.from_string(baseline)
    candidate_config = ModelConfig.from_string(candidate)
    
    results = []
    
    logger.info(f"Running PromptDiff with {len(prompts)} prompt(s)...")
    logger.info(f"Baseline: {baseline_config.provider.value}:{baseline_config.name}")
    logger.info(f"Candidate: {candidate_config.provider.value}:{candidate_config.name}")
    
    for i, prompt_data in enumerate(prompts, 1):
        prompt_id = prompt_data.get("id", f"prompt_{i}")
        prompt_template = prompt_data.get("prompt", "")
        prompt_vars = prompt_data.get("vars", {})
        
        # Validate prompt ID
        try:
            prompt_id = validate_prompt_id(prompt_id)
        except ValidationError as e:
            logger.warning(f"Invalid prompt ID '{prompt_id}': {e}, using default")
            prompt_id = f"prompt_{i}"
        
        if not prompt_template:
            logger.warning(f"Prompt {prompt_id} has no template, skipping")
            continue
        
        # Render prompt
        try:
            rendered_prompt = render_prompt(prompt_template, prompt_vars)
        except KeyError as e:
            error_msg = f"Prompt {prompt_id}: Missing variable in template - {e}"
            logger.error(error_msg)
            results.append({
                "id": prompt_id,
                "error": error_msg,
                "error_type": "ValidationError"
            })
            continue
        
        logger.info(f"[{i}/{len(prompts)}] Processing: {prompt_id}")
        
        try:
            # Call models
            logger.debug(f"Calling baseline model for prompt {prompt_id}")
            baseline_output = call_model(baseline_config, rendered_prompt, **model_kwargs)
            
            logger.debug(f"Calling candidate model for prompt {prompt_id}")
            candidate_output = call_model(candidate_config, rendered_prompt, **model_kwargs)
            
            # Compute diffs
            logger.debug(f"Computing diffs for prompt {prompt_id}")
            td = text_diff(baseline_output, candidate_output)
            text_stats = text_diff_stats(baseline_output, candidate_output)
            semantic_sim = semantic_diff(baseline_output, candidate_output, embedding_model)
            semantic_stats = semantic_diff_detailed(
                baseline_output, candidate_output, embedding_model
            )
            
            result = {
                "id": prompt_id,
                "prompt": rendered_prompt,
                "baseline_output": baseline_output,
                "candidate_output": candidate_output,
                "text_diff": td,
                "text_stats": text_stats,
                "semantic_similarity": semantic_sim,
                "semantic_stats": semantic_stats,
            }
            
            results.append(result)
            
            logger.info(f"Prompt {prompt_id} completed - Similarity: {semantic_sim:.3f}")
            
        except (ConnectionError, TimeoutError) as e:
            error_msg = f"Connection/timeout error for prompt {prompt_id}: {e}"
            logger.error(error_msg, exc_info=True)
            results.append({
                "id": prompt_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
        except (ValueError, ValidationError) as e:
            error_msg = f"Validation error for prompt {prompt_id}: {e}"
            logger.error(error_msg, exc_info=True)
            results.append({
                "id": prompt_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
        except ModelError as e:
            error_msg = f"Model error for prompt {prompt_id}: {e}"
            logger.error(error_msg, exc_info=True)
            results.append({
                "id": prompt_id,
                "error": str(e),
                "error_type": "ModelError"
            })
        except Exception as e:
            # Unexpected errors should be logged and re-raised
            error_msg = f"Unexpected error for prompt {prompt_id}: {e}"
            logger.critical(error_msg, exc_info=True)
            results.append({
                "id": prompt_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            # Optionally re-raise for critical failures
            # raise PromptDiffError(error_msg, prompt_id=prompt_id) from e
    
    # Save results if output file specified
    if output_file:
        try:
            output_path = safe_path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}", exc_info=True)
            raise PromptDiffError(f"Failed to save results: {e}")
    
    return results
