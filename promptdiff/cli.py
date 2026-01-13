"""CLI interface for PromptDiff."""

import json
import typer
from pathlib import Path
from typing import Optional, Dict, Any

from promptdiff.runner import run_diff
from promptdiff.report.markdown import generate as generate_markdown
from promptdiff.logging_config import get_logger
from promptdiff.errors import PromptDiffError
from promptdiff.validation import safe_path

logger = get_logger(__name__)

app = typer.Typer(help="PromptDiff - Git-style diffs for LLM outputs")


def build_model_kwargs(temperature: Optional[float], max_tokens: Optional[int]) -> Dict[str, Any]:
    """
    Build model kwargs from CLI arguments.
    
    Args:
        temperature: Optional temperature value
        max_tokens: Optional max tokens value
    
    Returns:
        Dictionary of model kwargs
    """
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


@app.command()
def run(
    prompts: str = typer.Option(..., "--prompts", "-p", help="Path to prompts JSON file"),
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline model (e.g., 'gpt-4' or 'openai:gpt-4')"),
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate model (e.g., 'gpt-4.1' or 'anthropic:claude-3-opus')"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results JSON"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Embedding model for semantic diff"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Temperature for model calls"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max tokens for model calls"),
):
    """Run PromptDiff comparison between baseline and candidate models."""
    model_kwargs = build_model_kwargs(temperature, max_tokens)
    
    try:
        results = run_diff(
            prompts_file=prompts,
            baseline=baseline,
            candidate=candidate,
            output_file=output,
            embedding_model=embedding_model,
            **model_kwargs
        )
        
        successful = len([r for r in results if "error" not in r])
        failed = len([r for r in results if "error" in r])
        
        typer.echo(f"\n✓ Completed: {len(results)} prompt(s) evaluated")
        typer.echo(f"  Successful: {successful}")
        if failed > 0:
            typer.echo(f"  Failed: {failed}")
        
    except PromptDiffError as e:
        logger.error(f"PromptDiff error: {e}", exc_info=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def report(
    results_file: str = typer.Option(..., "--results", "-r", help="Path to results JSON file"),
    format: str = typer.Option("markdown", "--format", "-f", help="Report format (markdown)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for report"),
):
    """Generate a report from PromptDiff results."""
    try:
        results_path = safe_path(results_file)
        if not results_path.exists():
            typer.echo(f"Error: Results file not found: {results_file}", err=True)
            raise typer.Exit(1)
        
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: Invalid JSON in results file: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error reading results file: {e}", exc_info=True)
        typer.echo(f"Error reading results file: {e}", err=True)
        raise typer.Exit(1)
    
    if format == "markdown":
        report_content = generate_markdown(results)
    else:
        typer.echo(f"Error: Unsupported format: {format}", err=True)
        raise typer.Exit(1)
    
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        typer.echo(f"Report saved to: {output}")
    else:
        typer.echo(report_content)


@app.command()
def compare(
    prompts: str = typer.Option(..., "--prompts", "-p", help="Path to prompts JSON file"),
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated model identifiers (e.g., 'ollama:llama3,ollama:granite4,ollama:qwen2.5')"),
    names: Optional[str] = typer.Option(None, "--names", "-n", help="Comma-separated model names (e.g., 'llama3,granite4,qwen2.5')"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory for results and report"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", help="Embedding model for semantic diff"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Temperature for model calls"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max tokens for model calls"),
):
    """Compare multiple models pairwise."""
    from promptdiff.comparison import compare_multiple_models
    
    # Parse models
    model_list = [m.strip() for m in models.split(",")]
    
    # Parse names if provided, otherwise use model identifiers
    if names:
        name_list = [n.strip() for n in names.split(",")]
        if len(name_list) != len(model_list):
            typer.echo("Error: Number of names must match number of models", err=True)
            raise typer.Exit(1)
    else:
        # Extract names from model identifiers
        name_list = []
        for model_id in model_list:
            if ":" in model_id:
                name_list.append(model_id.split(":")[-1])
            else:
                name_list.append(model_id)
    
    # Create model tuples
    model_tuples = list(zip(name_list, model_list))
    
    model_kwargs = build_model_kwargs(temperature, max_tokens)
    
    try:
        result = compare_multiple_models(
            prompts_file=prompts,
            models=model_tuples,
            output_dir=output_dir,
            embedding_model=embedding_model,
            **model_kwargs
        )
        
        typer.echo(f"\n✓ Completed: {result['summary']['total_comparisons']} comparison(s)")
        typer.echo(f"  Successful: {result['summary']['successful']}")
        if result['summary']['failed'] > 0:
            typer.echo(f"  Failed: {result['summary']['failed']}")
        if result['summary']['avg_similarity'] > 0:
            typer.echo(f"  Average similarity: {result['summary']['avg_similarity']:.3f}")
        
    except PromptDiffError as e:
        logger.error(f"PromptDiff error: {e}", exc_info=True)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def ui(
    host: str = typer.Option("localhost", "--host", help="Host to bind the server to"),
    port: int = typer.Option(8501, "--port", help="Port to bind the server to"),
):
    """Launch the Streamlit web UI."""
    import subprocess
    import sys
    from pathlib import Path
    
    ui_file = Path(__file__).parent / "ui.py"
    
    if not ui_file.exists():
        typer.echo(f"Error: UI file not found at {ui_file}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"🚀 Starting PromptDiff UI at http://{host}:{port}")
    typer.echo("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(ui_file),
            "--server.address", host,
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        typer.echo("\n👋 Shutting down UI server...")


@app.command()
def version():
    """Show PromptDiff version."""
    from promptdiff import __version__
    typer.echo(f"PromptDiff {__version__}")


if __name__ == "__main__":
    app()
