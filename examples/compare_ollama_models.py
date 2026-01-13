"""Compare Ollama models: llama3, granite4, and qwen2.5

This is an example script showing how to use PromptDiff to compare multiple models.
You can run this script directly, or use the CLI command instead:

    promptdiff compare \
        --prompts examples/prompts.json \
        --models ollama:llama3,ollama:granite4,ollama:qwen2.5 \
        --names llama3,granite4,qwen2.5 \
        --output-dir results
"""

from pathlib import Path

# Import from the installed promptdiff package
from promptdiff.comparison import compare_multiple_models


def main():
    """Compare llama3, granite4, and qwen2.5 models."""
    
    # Get prompts file path
    prompts_file = Path(__file__).parent / "prompts.json"
    
    # Define models to compare
    models = [
        ("llama3", "ollama:llama3"),
        ("granite4", "ollama:granite4"),
        ("qwen2.5", "ollama:qwen2.5"),
    ]
    
    # Output directory
    output_dir = Path(__file__).parent.parent / "results"
    
    # Run comparison
    result = compare_multiple_models(
        prompts_file=str(prompts_file),
        models=models,
        output_dir=str(output_dir)
    )
    
    if result["results"]:
        print(f"\n✓ Comparison complete!")
        print(f"  Total comparisons: {result['summary']['total_comparisons']}")
        print(f"  Successful: {result['summary']['successful']}")
        if result['summary']['failed'] > 0:
            print(f"  Failed: {result['summary']['failed']}")
        if result['summary']['avg_similarity'] > 0:
            print(f"  Average similarity: {result['summary']['avg_similarity']:.3f}")
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
