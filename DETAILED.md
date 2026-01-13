# PromptDiff - Detailed Documentation

This document provides comprehensive information about PromptDiff, including advanced usage, configuration options, and detailed explanations.

---

## 🔧 How It Works

1. **Run prompts** against two model configs
2. **Normalize outputs**
3. **Compare using:**
   - Text diff
   - Embedding similarity
   - (Optional) LLM judge
4. **Generate a report**

---

## 🧩 Use Cases

- Model upgrades
- Prompt tuning
- Regression testing
- AI QA pipelines

---

## 📖 Usage

### Comparison Modes

PromptDiff supports two comparison modes:

#### Single Comparison
Compare **two models** (baseline vs candidate) in a one-to-one comparison.

**Use when:**
- Testing a new model version against a baseline
- Quick iteration between two specific models
- CI/CD regression testing
- Focused, detailed analysis

**Example:**
```bash
promptdiff run \
  --prompts prompts.json \
  --baseline ollama:llama3 \
  --candidate ollama:granite4
```

**Output:** One comparison report showing detailed differences between the two models.

#### Multi-Model Comparison
Compare **multiple models** pairwise (each model compared with every other model).

**Use when:**
- Evaluating 3+ model options simultaneously
- Choosing between multiple models
- Comprehensive model benchmarking
- Research and analysis

**Example:**
```bash
promptdiff compare \
  --prompts prompts.json \
  --models ollama:llama3,ollama:granite4,ollama:qwen2.5 \
  --names llama3,granite4,qwen2.5 \
  --output-dir results
```

**Output:** Combined report with all pairwise comparisons:
- llama3 vs granite4
- llama3 vs qwen2.5
- granite4 vs qwen2.5

**Performance Note:** Multi-model comparison performs N×(N-1)/2 comparisons:
- 3 models = 3 comparisons
- 4 models = 6 comparisons
- 5 models = 10 comparisons

### Model Identifiers

PromptDiff supports multiple model providers:

```bash
# OpenAI (default)
--baseline gpt-4
--baseline openai:gpt-4

# Anthropic
--candidate anthropic:claude-3-opus
--candidate anthropic:claude-3-sonnet

# Ollama (local models)
--baseline ollama:llama3
--candidate ollama:granite4
--candidate ollama:qwen2.5

# Local (stub implementation)
--baseline local:my-model
```

### Environment Variables

PromptDiff supports loading configuration from a `.env` file for convenience.

#### Using .env File (Recommended)

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   OLLAMA_BASE_URL=http://localhost:11434
   ```

3. **The `.env` file is automatically loaded** when you run PromptDiff.

#### Using Environment Variables Directly

You can also set environment variables directly:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OLLAMA_BASE_URL="http://localhost:11434"
```

**Note:** Environment variables take precedence over `.env` file values.

### Ollama Setup

Before using Ollama models, make sure:

1. **Ollama is installed and running:**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ```

2. **Pull the models you want to compare:**
   ```bash
   ollama pull llama3
   ollama pull granite4
   ollama pull qwen2.5
   ```

3. **Run comparisons:**
   ```bash
   # Compare two Ollama models
   promptdiff run \
     --prompts examples/prompts.json \
     --baseline ollama:llama3 \
     --candidate ollama:granite4
   
   # Or use the provided comparison script
   python examples/compare_ollama_models.py
   ```

### Advanced Options

```bash
promptdiff run \
  --prompts prompts.json \
  --baseline gpt-4 \
  --candidate gpt-4.1 \
  --output results.json \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --temperature 0.7 \
  --max-tokens 1000
```

---

## 🎨 Web UI

PromptDiff includes a modern web interface for interactive comparisons:

```bash
# Launch the UI
promptdiff ui

# Or specify host and port
promptdiff ui --host 0.0.0.0 --port 8501
```

The UI provides:
- 📝 **Interactive prompt loading** - Upload JSON files, type/paste JSON, use examples, or build manually
- 🤖 **Model configuration** - Always-visible model selection in sidebar
- 🔍 **Single comparison mode** - Compare two models with detailed side-by-side diffs
- 🔬 **Multi-model comparison mode** - Compare multiple models pairwise with combined reports
- 📊 **Visual results** - Charts, metrics, and similarity scores
- 📈 **Summary statistics** - Aggregate metrics across all prompts
- 💾 **Download reports** - Export markdown reports and JSON results
- ✅ **Selective examples** - Choose specific example prompts instead of loading all

Then open your browser to `http://localhost:8501` for the interactive interface!
