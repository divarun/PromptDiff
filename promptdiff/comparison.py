"""Multi-model comparison utilities."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from promptdiff.runner import run_diff
from promptdiff.report.markdown import generate as generate_markdown
from promptdiff.logging_config import get_logger
from promptdiff.errors import PromptDiffError

logger = get_logger(__name__)


def compare_multiple_models(
    prompts_file: str,
    models: List[tuple[str, str]],
    output_dir: Optional[str] = None,
    embedding_model: Optional[str] = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Compare multiple models pairwise.
    
    Args:
        prompts_file: Path to JSON file containing prompts
        models: List of (name, model_identifier) tuples, e.g., [("llama3", "ollama:llama3"), ...]
        output_dir: Optional directory to save results and reports
        embedding_model: Optional embedding model for semantic diff
        **model_kwargs: Additional parameters to pass to model calls
    
    Returns:
        Dictionary with:
            - results: List of all comparison results
            - report: Generated markdown report
            - summary: Summary statistics
    """
    all_results = []
    output_path = Path(output_dir) if output_dir else None
    
    logger.info("=" * 60)
    logger.info(f"Comparing {len(models)} models pairwise")
    logger.info("=" * 60)
    
    # Compare each model pair
    for i, (name1, model1) in enumerate(models):
        for name2, model2 in models[i+1:]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Comparison: {name1} (baseline) vs {name2} (candidate)")
            logger.info(f"{'='*60}")
            
            try:
                results = run_diff(
                    prompts_file=prompts_file,
                    baseline=model1,
                    candidate=model2,
                    output_file=None,  # We'll create a combined report
                    embedding_model=embedding_model,
                    **model_kwargs
                )
                
                # Add model names to results
                for result in results:
                    result["baseline_model"] = name1
                    result["candidate_model"] = name2
                    result["comparison"] = f"{name1}_vs_{name2}"
                
                all_results.extend(results)
                logger.info(f"✓ Completed comparison: {name1} vs {name2}")
                
            except PromptDiffError as e:
                logger.error(f"Error comparing {name1} vs {name2}: {e}", exc_info=True)
                # Continue with other comparisons
                continue
            except Exception as e:
                logger.critical(
                    f"Unexpected error comparing {name1} vs {name2}: {e}",
                    exc_info=True
                )
                # Continue with other comparisons
                continue
    
    # Generate report
    report = None
    if all_results:
        report = generate_markdown(all_results)
    
    # Save files if output directory specified
    if output_path and all_results:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save report
        report_file = output_path / "comparison_report.md"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"\n{'='*60}")
            logger.info(f"Report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}", exc_info=True)
            raise PromptDiffError(f"Failed to save report: {e}")
        
        # Save results JSON
        results_file = output_path / "comparison_results.json"
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results JSON saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results JSON: {e}", exc_info=True)
            raise PromptDiffError(f"Failed to save results JSON: {e}")
    
    # Calculate summary
    summary = _calculate_summary(all_results)
    
    return {
        "results": all_results,
        "report": report,
        "summary": summary
    }


def _calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from results."""
    if not results:
        return {"total_comparisons": 0}
    
    similarities = [r.get("semantic_similarity", 0.0) for r in results if "semantic_similarity" in r]
    
    return {
        "total_comparisons": len(results),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
        "min_similarity": min(similarities) if similarities else 0.0,
        "max_similarity": max(similarities) if similarities else 0.0,
    }
