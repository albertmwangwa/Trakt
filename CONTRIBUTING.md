# Contributing to Trakt

Thank you for considering contributing to Trakt! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How Can I Contribute?

### Reporting Bugs

Before submitting a bug report:
1. Check existing issues to avoid duplicates
2. Collect relevant information (OS, Python version, error messages)
3. Try to reproduce the issue with minimal configuration

Submit bug reports with:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Environment details

### Suggesting Features

Feature suggestions are welcome! Please:
1. Check if the feature already exists or is planned
2. Clearly describe the use case
3. Explain why this would be useful
4. Provide examples if possible

### Pull Requests

#### Before Starting

1. Open an issue to discuss major changes
2. Fork the repository
3. Create a feature branch from `main`

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Trakt.git
cd Trakt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

#### Coding Standards

**Python Style**
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public methods
- Maximum line length: 100 characters

**Code Formatting**
```bash
# Format code with black
black src/ tests/ main.py

# Check with flake8
flake8 src/ tests/ main.py

# Type checking
mypy src/
```

**Naming Conventions**
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

#### Writing Tests

All new features should include tests:

```python
# tests/test_new_feature.py
import unittest
from src.new_module import NewClass

class TestNewFeature(unittest.TestCase):
    def test_basic_functionality(self):
        obj = NewClass()
        result = obj.some_method()
        self.assertEqual(result, expected_value)
```

Run tests:
```bash
python -m unittest discover tests -v
```

#### Commit Messages

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123"

Examples:
```
Add license plate detection example

- Create new example for license plate recognition
- Add pattern matching for common formats
- Update documentation

Fixes #45
```

#### Pull Request Process

1. Update documentation for any changed functionality
2. Add tests for new features
3. Ensure all tests pass
4. Update README.md if needed
5. Create pull request with clear description

**PR Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

## Project Structure

```
Trakt/
├── src/                    # Source code
│   ├── camera_handler.py   # ONVIF camera handling
│   └── ocr_engine.py       # OCR implementations
├── tests/                  # Unit tests
├── examples/               # Example scripts
├── models/                 # TensorFlow models
└── main.py                 # Application entry point
```

## Adding New Features

### Adding a New OCR Engine

1. Extend `OCREngine` class in `src/ocr_engine.py`
2. Implement required methods:
   - `_initialize_engine()`
   - `detect_text()`
3. Add configuration options to `config.yaml`
4. Write tests
5. Update documentation

### Adding Camera Support

1. Extend `CameraHandler` in `src/camera_handler.py`
2. Implement connection and streaming methods
3. Add configuration options
4. Test with actual hardware
5. Document compatibility

### Adding Examples

1. Create new file in `examples/`
2. Include inline documentation
3. Add to README examples section
4. Keep dependencies minimal

## Documentation

- Use clear, concise language
- Include code examples
- Add screenshots for visual features
- Keep README updated
- Document configuration options

## Release Process

Releases are managed by maintainers:

1. Version bump in `src/__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Publish to PyPI (if applicable)

## Getting Help

- Open an issue for questions
- Join discussions in issues
- Check existing documentation
- Review closed issues for solutions

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant code comments

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions make Trakt better for everyone. We appreciate your time and effort! 🙏
