# Contributing to Auto-Deployment ML Models

Thank you for your interest in contributing to this MLOps project! This document provides guidelines and instructions for contributing.

## 🎯 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Docker (optional, for containerization)
- Basic understanding of ML and DevOps concepts

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Aegis_Code_AI.git
   cd Aegis_Code_AI
   ```

3. **Set up the development environment**
   ```bash
   # Windows
   .\ci-cd\setup.ps1

   # Linux/Mac
   chmod +x ci-cd/setup.sh
   ./ci-cd/setup.sh
   ```

4. **Create a branch for your feature**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details** (OS, Python version, etc.)
- **Logs or screenshots** if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- **Clear description** of the enhancement
- **Use case** explaining why it would be valuable
- **Proposed implementation** (if you have ideas)

### Pull Requests

1. **Ensure your code follows the project style**
   - Use meaningful variable names
   - Add docstrings to functions
   - Follow PEP 8 for Python code

2. **Write or update tests**
   ```bash
   pytest tests/
   ```

3. **Update documentation** if needed
   - Update README.md for new features
   - Add docstrings to new functions/classes
   - Update relevant docs in `/docs`

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template with details

## 🧪 Testing Guidelines

- Write tests for new features
- Ensure all existing tests pass
- Aim for at least 80% code coverage
- Test both success and failure scenarios

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📋 Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited in releases

## 🎨 Coding Standards

### Python Code Style
- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable/function names

### Documentation
- Add docstrings to all public functions/classes
- Include parameter descriptions and return types
- Update README.md for significant changes

### Example Function Documentation
```python
def train_model(data: pd.DataFrame, params: dict) -> Model:
    """
    Train a machine learning model with given data and parameters.
    
    Args:
        data (pd.DataFrame): Training dataset
        params (dict): Model hyperparameters
        
    Returns:
        Model: Trained model instance
        
    Raises:
        ValueError: If data is empty or invalid
    """
    pass
```

## 🏗️ Project Structure

```
Devops_Project/
├── src/              # Source code
├── tests/            # Test files
├── models/           # Trained models (gitignored)
├── data/             # Data files (gitignored)
├── docker/           # Docker configurations
├── ci-cd/            # CI/CD scripts
├── docs/             # Documentation
└── config/           # Configuration files
```

## 🤔 Questions?

- Open an issue for questions
- Check existing issues/PRs first
- Contact project maintainers

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions make this project better for everyone. Thank you for taking the time to contribute!
