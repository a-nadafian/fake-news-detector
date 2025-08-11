# 🤝 Contributing to Fake News Detector

Thank you for your interest in contributing to the Fake News Detector project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic knowledge of Python, PyTorch, and NLP

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**
   ```bash
   pip install -e .
   ```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose

### Testing
- Write tests for new features
- Ensure all tests pass before submitting
- Test with different Python versions if possible

### Documentation
- Update README.md if adding new features
- Add inline comments for complex logic
- Update docstrings when modifying functions

## 🔧 Project Structure

```
fake-news-detector/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   └── models/            # Model definitions and training
├── notebooks/             # Jupyter notebooks
├── data/                  # Data storage
├── models/                # Trained models
├── docs/                  # Documentation
├── tests/                 # Test files
├── app.py                 # Streamlit web app
└── requirements.txt       # Dependencies
```

## 🚀 Adding New Features

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Implement Your Feature
- Follow the existing code structure
- Add appropriate error handling
- Include tests for new functionality

### 3. Test Your Changes
```bash
python -m pytest tests/
python app.py  # Test the Streamlit app
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📊 Model Improvements

### Adding New Models
1. Create a new model class in `src/models/`
2. Implement the required interface methods
3. Add training scripts
4. Update the main training pipeline
5. Add model evaluation metrics

### Improving Existing Models
1. Identify areas for improvement
2. Implement changes incrementally
3. Test performance improvements
4. Document changes and rationale

## 🐛 Reporting Issues

### Bug Reports
- Use the issue template
- Provide clear reproduction steps
- Include error messages and stack traces
- Specify your environment (OS, Python version, etc.)

### Feature Requests
- Describe the desired functionality
- Explain the use case
- Suggest implementation approach if possible

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)

## 🎯 Contribution Areas

We welcome contributions in these areas:
- **Model Improvements**: Better architectures, hyperparameter optimization
- **Data Processing**: Enhanced preprocessing, new features
- **Web Interface**: UI/UX improvements, new features
- **Documentation**: Better explanations, examples, tutorials
- **Testing**: More comprehensive test coverage
- **Performance**: Optimization, faster inference

## 📞 Getting Help

- Open an issue for questions or problems
- Join our discussions in GitHub Discussions
- Check existing issues and pull requests

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to making fake news detection more accessible and accurate! 🎉
