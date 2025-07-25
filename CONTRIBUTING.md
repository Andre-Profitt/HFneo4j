# Contributing to Hedge Fund Graph Stack

Thank you for your interest in contributing to the Hedge Fund Graph Stack project!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/HFneo4j.git
   cd HFneo4j
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black ruff mypy
   ```

## Code Style

- Use Black for code formatting: `black src/`
- Use Ruff for linting: `ruff src/`
- Type hints are encouraged where applicable

## Testing

- Write tests for new features in the `tests/` directory
- Run tests with: `pytest tests/ -v`
- Ensure coverage remains above 80%: `pytest --cov=src --cov-report=html`

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes and commit with descriptive messages
3. Push to your fork: `git push origin feature/your-feature-name`
4. Open a Pull Request with:
   - Clear description of changes
   - Any relevant issue numbers
   - Performance impact (if applicable)

## Areas for Contribution

- **Performance Optimization**: Improve algorithm efficiency
- **New Graph Algorithms**: Add novel graph-based trading strategies
- **Documentation**: Enhance examples and tutorials
- **Testing**: Increase test coverage
- **Visualization**: Create better graph visualization tools
- **Integration**: Add support for more data sources

## Questions?

Open an issue for discussion before making major changes.