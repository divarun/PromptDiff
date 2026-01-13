# Ollama Model Comparison Guide

This guide shows how to compare Ollama models (llama3, granite4, qwen2.5) using PromptDiff.

## Prerequisites

1. **Install Ollama:**
   - Download from https://ollama.ai
   - Start Ollama: `ollama serve`

2. **Pull the models:**
   ```bash
   ollama pull llama3
   ollama pull granite4
   ollama pull qwen2.5
   ```

3. **Install PromptDiff:**
   ```bash
   pip install -e ".[all]"
   ```

## Quick Start

### Option 1: Using the CLI Compare Command (Recommended)

Use the built-in `compare` command to compare multiple models:

```bash
promptdiff compare \
  --prompts examples/prompts.json \
  --models ollama:llama3,ollama:granite4,ollama:qwen2.5 \
  --names llama3,granite4,qwen2.5 \
  --output-dir results
```

This will:
- Compare llama3 vs granite4
- Compare llama3 vs qwen2.5
- Compare granite4 vs qwen2.5
- Generate a combined report: `results/comparison_report.md`
- Save results: `results/comparison_results.json`

### Option 2: Using the Python Script

Run the example script:

```bash
python examples/compare_ollama_models.py
```

### Option 3: Using Individual CLI Commands

Compare models pairwise using the CLI:

**Windows (PowerShell):**
```powershell
.\examples\compare_ollama_cli.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x examples/compare_ollama_cli.sh
./examples/compare_ollama_cli.sh
```

### Option 3: Manual CLI Commands

Compare two models manually:

```bash
# Compare llama3 vs granite4
promptdiff run \
  --prompts examples/prompts.json \
  --baseline ollama:llama3 \
  --candidate ollama:granite4 \
  --output results_llama3_vs_granite4.json

# Generate report
promptdiff report \
  --results results_llama3_vs_granite4.json \
  --format markdown \
  --output report_llama3_vs_granite4.md
```

## Model Identifiers

Use the `ollama:` prefix when specifying models:

- `ollama:llama3`
- `ollama:granite4`
- `ollama:qwen2.5`

## Custom Ollama Server

If Ollama is running on a different host/port:

```bash
export OLLAMA_BASE_URL="http://your-server:11434"
```

Or specify in code:
```python
from promptdiff.config import ModelConfig, ModelProvider

config = ModelConfig(
    name="llama3",
    provider=ModelProvider.OLLAMA,
    base_url="http://your-server:11434"
)
```

## Troubleshooting

**Error: "Could not connect to Ollama"**
- Make sure Ollama is running: `ollama serve`
- Check the base URL (default: http://localhost:11434)

**Error: "Model 'llama3' not found"**
- Pull the model: `ollama pull llama3`
- Verify with: `ollama list`

**Slow responses**
- First run may be slow as models load into memory
- Subsequent runs should be faster

## Example Output

The comparison report includes:
- Similarity scores between models
- Text differences
- Semantic analysis
- Output previews
- Summary statistics
