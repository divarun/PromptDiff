#!/bin/bash
# Compare Ollama models using CLI commands

# Models to compare
BASELINE="ollama:llama3"
CANDIDATE1="ollama:granite4"
CANDIDATE2="ollama:qwen2.5"

PROMPTS_FILE="examples/prompts.json"

echo "=========================================="
echo "Comparing Ollama Models"
echo "=========================================="
echo ""

# Compare llama3 vs granite4
echo "1. Comparing llama3 (baseline) vs granite4 (candidate)..."
promptdiff run \
  --prompts "$PROMPTS_FILE" \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE1" \
  --output results_llama3_vs_granite4.json

promptdiff report \
  --results results_llama3_vs_granite4.json \
  --format markdown \
  --output report_llama3_vs_granite4.md

echo ""

# Compare llama3 vs qwen2.5
echo "2. Comparing llama3 (baseline) vs qwen2.5 (candidate)..."
promptdiff run \
  --prompts "$PROMPTS_FILE" \
  --baseline "$BASELINE" \
  --candidate "$CANDIDATE2" \
  --output results_llama3_vs_qwen2.5.json

promptdiff report \
  --results results_llama3_vs_qwen2.5.json \
  --format markdown \
  --output report_llama3_vs_qwen2.5.md

echo ""

# Compare granite4 vs qwen2.5
echo "3. Comparing granite4 (baseline) vs qwen2.5 (candidate)..."
promptdiff run \
  --prompts "$PROMPTS_FILE" \
  --baseline "$CANDIDATE1" \
  --candidate "$CANDIDATE2" \
  --output results_granite4_vs_qwen2.5.json

promptdiff report \
  --results results_granite4_vs_qwen2.5.json \
  --format markdown \
  --output report_granite4_vs_qwen2.5.md

echo ""
echo "=========================================="
echo "All comparisons complete!"
echo "=========================================="
