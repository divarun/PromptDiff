"""Markdown report generation."""

from typing import List, Dict, Any
from datetime import datetime


def generate(results: List[Dict[str, Any]], format: str = "markdown") -> str:
    """
    Generate a markdown report from diff results.
    
    Args:
        results: List of diff results, each containing:
            - id: Prompt ID
            - baseline_output: Baseline model output
            - candidate_output: Candidate model output
            - text_diff: Text diff string
            - semantic_similarity: Semantic similarity score
            - text_stats: Text diff statistics
            - semantic_stats: Semantic diff statistics
        format: Report format (currently only "markdown" supported)
    
    Returns:
        Markdown report as string
    """
    md = "# PromptDiff Report\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += f"Total prompts evaluated: {len(results)}\n\n"
    md += "---\n\n"
    
    for r in results:
        prompt_id = r.get("id", "unknown")
        md += f"## Prompt: {prompt_id}\n\n"
        
        # Similarity score
        similarity = r.get("semantic_similarity", 0.0)
        md += f"**Similarity Score:** {similarity:.3f}\n\n"
        
        # Status indicators
        md += _generate_status_indicators(r)
        
        # Text stats
        text_stats = r.get("text_stats", {})
        if text_stats:
            md += "### Text Statistics\n\n"
            md += f"- Added lines: {text_stats.get('added_lines', 0)}\n"
            md += f"- Removed lines: {text_stats.get('removed_lines', 0)}\n"
            md += f"- Text similarity: {text_stats.get('similarity', 0.0):.3f}\n"
            md += f"- Length difference: {text_stats.get('length_diff', 0)} chars\n\n"
        
        # Semantic stats
        semantic_stats = r.get("semantic_stats", {})
        if semantic_stats:
            md += "### Semantic Analysis\n\n"
            md += f"- Similar: {'Yes' if semantic_stats.get('is_similar', False) else 'No'}\n"
            md += f"- Very similar: {'Yes' if semantic_stats.get('is_very_similar', False) else 'No'}\n\n"
        
        # Text diff preview
        text_diff = r.get("text_diff", "")
        if text_diff:
            md += "### Text Diff\n\n"
            md += "```diff\n"
            # Limit diff preview to first 50 lines
            diff_lines = text_diff.split("\n")
            preview_lines = diff_lines[:50]
            md += "\n".join(preview_lines)
            if len(diff_lines) > 50:
                md += f"\n... ({len(diff_lines) - 50} more lines)"
            md += "\n```\n\n"
        
        # Output previews
        baseline_output = r.get("baseline_output", "")
        candidate_output = r.get("candidate_output", "")
        
        if baseline_output or candidate_output:
            md += "### Outputs\n\n"
            
            if baseline_output:
                md += "**Baseline:**\n```\n"
                preview = baseline_output[:200] + "..." if len(baseline_output) > 200 else baseline_output
                md += preview + "\n```\n\n"
            
            if candidate_output:
                md += "**Candidate:**\n```\n"
                preview = candidate_output[:200] + "..." if len(candidate_output) > 200 else candidate_output
                md += preview + "\n```\n\n"
        
        md += "---\n\n"
    
    # Summary
    md += _generate_summary(results)
    
    return md


def _generate_status_indicators(result: Dict[str, Any]) -> str:
    """Generate status indicators (✔, ✖, ⚠) based on diff results."""
    indicators = []
    
    similarity = result.get("semantic_similarity", 0.0)
    text_stats = result.get("text_stats", {})
    semantic_stats = result.get("semantic_stats", {})
    
    # Similarity check
    if similarity > 0.95:
        indicators.append("✔ High similarity")
    elif similarity < 0.7:
        indicators.append("✖ Low similarity")
    else:
        indicators.append("⚠ Moderate similarity")
    
    # Length change
    length_diff = text_stats.get("length_diff", 0)
    if abs(length_diff) > 100:
        if length_diff > 0:
            indicators.append("⚠ Verbosity increased")
        else:
            indicators.append("⚠ Verbosity decreased")
    
    # Semantic similarity
    if semantic_stats.get("is_very_similar", False):
        indicators.append("✔ Semantic match")
    elif not semantic_stats.get("is_similar", False):
        indicators.append("✖ Semantic divergence")
    
    if not indicators:
        return ""
    
    return "**Status:** " + " | ".join(indicators) + "\n\n"


def _generate_summary(results: List[Dict[str, Any]]) -> str:
    """Generate summary statistics."""
    md = "## Summary\n\n"
    
    if not results:
        return md + "No results to summarize.\n"
    
    similarities = [r.get("semantic_similarity", 0.0) for r in results]
    avg_similarity = sum(similarities) / len(similarities)
    
    high_similarity = sum(1 for s in similarities if s > 0.95)
    low_similarity = sum(1 for s in similarities if s < 0.7)
    
    md += f"- Average similarity: {avg_similarity:.3f}\n"
    md += f"- High similarity (>0.95): {high_similarity}/{len(results)}\n"
    md += f"- Low similarity (<0.7): {low_similarity}/{len(results)}\n"
    
    return md
