# Contributing to PromptDiff

Thank you for your interest in contributing to PromptDiff! 🎉

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/promptdiff.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
5. Install in development mode: `pip install -e ".[all]"`
6. Install pre-commit hooks: `pre-commit install` (if available)

## Development

### Running Tests

```bash
pytest
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions and classes

### Making Changes

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test your changes
4. Commit: `git commit -m "Add feature: your feature"`
5. Push: `git push origin feature/your-feature-name`
6. Open a Pull Request

## Areas for Contribution

- Additional model providers (Ollama, vLLM, etc.)
- More diff algorithms
- LLM judge integration
- Performance improvements
- Documentation improvements
- Bug fixes

## Questions?

Open an issue for discussion!
