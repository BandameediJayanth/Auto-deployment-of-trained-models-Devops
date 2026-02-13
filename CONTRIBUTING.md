# Contributing to ML API Deployment

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Describe the bug in detail
- Include steps to reproduce
- Provide system information (OS, Python version, Docker version)

### Suggesting Features
- Open an issue with the "enhancement" label
- Clearly describe the feature and its benefits
- Provide use cases and examples

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add comments where necessary
   - Update documentation

4. **Test your changes**
   ```bash
   python src/validate_model.py
   docker-compose -f docker/docker-compose.yml up -d
   ```

5. **Commit your changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```
   
   Use conventional commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `style:` code style changes
   - `refactor:` code refactoring
   - `test:` test additions/changes
   - `chore:` maintenance tasks

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description
   - Reference related issues
   - Include screenshots if applicable

## 📝 Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and small

### Docker
- Use multi-stage builds
- Minimize layer count
- Don't run as root
- Include health checks

### Documentation
- Update README.md for major changes
- Add inline comments for complex logic
- Update API documentation
- Keep CHANGELOG.md current

## 🧪 Testing

- Ensure all existing tests pass
- Add tests for new features
- Test Docker builds locally
- Verify API endpoints work correctly

## 📋 Checklist

Before submitting a PR, ensure:
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No sensitive data is committed
- [ ] Docker builds successfully
- [ ] API endpoints are tested

## 🎯 Priority Areas

We especially welcome contributions in:
- Additional ML models
- Performance optimizations
- Security enhancements
- Documentation improvements
- Test coverage
- UI/UX improvements

## 📧 Questions?

Feel free to open an issue for any questions or reach out to the maintainers.

Thank you for contributing! 🙏
