"""Text-based diff using difflib."""

import difflib
from typing import Dict, Any


def text_diff(a: str, b: str) -> str:
    """
    Generate a unified diff between two texts.
    
    Args:
        a: Baseline text
        b: Candidate text
    
    Returns:
        Unified diff as string
    """
    return "\n".join(
        difflib.unified_diff(
            a.splitlines(keepends=False),
            b.splitlines(keepends=False),
            lineterm="",
            fromfile="baseline",
            tofile="candidate"
        )
    )


def text_diff_stats(a: str, b: str) -> Dict[str, Any]:
    """
    Calculate statistics about text differences.
    
    Args:
        a: Baseline text
        b: Candidate text
    
    Returns:
        Dictionary with diff statistics
    """
    diff_lines = list(
        difflib.unified_diff(
            a.splitlines(keepends=False),
            b.splitlines(keepends=False),
            lineterm=""
        )
    )
    
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, a, b).ratio()
    
    return {
        "added_lines": added,
        "removed_lines": removed,
        "similarity": similarity,
        "baseline_length": len(a),
        "candidate_length": len(b),
        "length_diff": len(b) - len(a)
    }
