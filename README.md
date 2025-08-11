# 🔍 Fake News Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI-powered fake news detection system that combines advanced NLP techniques with a hybrid BERT-based model to identify and classify fake news articles with high accuracy.

## 🚀 Features

### 🤖 AI Model
- **Hybrid Architecture**: Combines BERT transformers with engineered features
- **High Accuracy**: Trained on multiple datasets for robust performance
- **Real-time Processing**: Fast inference for instant results
- **Confidence Scoring**: Provides prediction confidence levels

### 🌐 Web Application
- **Beautiful UI**: Modern, responsive design with Streamlit
- **Multiple Analysis Modes**: Single text, batch processing, and examples
- **Interactive Visualizations**: Charts and graphs using Plotly
- **User-friendly Interface**: Intuitive navigation and clear results

### 🔧 Technical Features
- **Advanced Preprocessing**: Sophisticated text cleaning pipeline
- **Feature Engineering**: Linguistic and statistical feature extraction
- **Model Persistence**: Save and load trained models
- **Scalable Architecture**: Handle both individual and batch requests

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Web Application](#-web-application)
- [API Usage](#-api-usage)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **Git**
- **Sufficient disk space** (at least 2GB for models and dependencies)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### Step 2: Install Dependencies

#### Option A: Full Installation (Recommended for Development)
```bash
pip install -r requirements.txt
```

#### Option B: Streamlit App Only
```bash
pip install -r requirements_streamlit.txt
```

### Step 3: Verify Installation

```bash
python test_environment.py
```

## 🚀 Quick Start

### 1. Run the Web Application

```bash
# Using the smart launcher (recommended)
python run_app.py

# Or directly with Streamlit
streamlit run app.py

# On Windows, you can also double-click:
run_app.bat
```

### 2. Open Your Browser

Navigate to `http://localhost:8501` to access the application.

### 3. Start Analyzing

- **Single Text**: Paste any news article or headline
- **Batch Analysis**: Upload CSV files with multiple texts
- **Test Examples**: Try pre-defined examples

## 📁 Project Structure

```
fake-news-detector/
├── 📁 data/                          # Data storage
│   ├── 📁 raw/                       # Raw datasets
│   └── 📁 processed/                 # Processed datasets
├── 📁 models/                        # Model storage
│   └── 📁 fake_news_detector/       # Main model directory
│       └── 📁 final_model/           # Trained model files
├── 📁 src/                           # Source code
│   ├── 📁 data/                      # Data processing
│   │   └── advanced_preprocessing.py # Advanced text preprocessing
│   ├── 📁 features/                  # Feature engineering
│   │   └── build_features.py         # Feature extraction
│   └── 📁 models/                    # Model code
│       ├── train_model.py            # Training script
│       └── predict_model.py          # Prediction script
├── 📁 notebooks/                     # Jupyter notebooks
│   └── eda-and-preprocessing.ipynb   # EDA and preprocessing
├── 📁 docs/                          # Documentation
├── 📁 reports/                       # Generated reports
├── 📁 mlruns/                        # MLflow tracking
├── app.py                            # Streamlit web application
├── run_app.py                        # Smart app launcher
├── test_app.py                       # App testing script
├── requirements.txt                  # Full dependencies
├── requirements_streamlit.txt        # Streamlit dependencies
└── README.md                         # This file
```

## 📱 Usage

### Web Application

The Streamlit app provides several ways to analyze text:

#### 🏠 Home Page
- **Welcome screen** with model statistics
- **Quick demo** for instant testing
- **Feature overview** and usage instructions

#### 🔍 Single Text Analysis
- **Input**: Paste any text, article, or headline
- **Output**: Real/Fake classification with confidence
- **Features**: Probability charts, insights, recommendations

#### 📊 Batch Analysis
- **Input**: Upload CSV files with 'text' column
- **Output**: Comprehensive analysis of multiple texts
- **Features**: Progress tracking, visualizations, CSV download

#### 🧪 Test Examples
- **Pre-defined examples** for testing
- **Categories**: Fake news, real news, mixed examples
- **Instant results** with confidence scores

#### 📈 Model Information
- **Architecture details** and configuration
- **Training parameters** and performance metrics
- **Technical explanations** of how it works

#### ⚙️ Settings
- **Customizable parameters** for analysis
- **Display preferences** and themes
- **Performance tuning** options

### Command Line Interface

#### Training the Model

```bash
# Train the final model
python src/models/train_model.py

# Or use the Makefile
make train
```

#### Making Predictions

```bash
# Single prediction
python src/models/predict_model.py

# Test preprocessing
python simple_test.py
```

## 🎯 Model Training

### Architecture

The system uses a **hybrid model** that combines:

1. **BERT Transformer**: Contextual text understanding
2. **Engineered Features**: Linguistic and statistical features
3. **Neural Classifier**: Final prediction layer

### Training Process

1. **Data Preprocessing**: Advanced text cleaning and feature extraction
2. **Model Initialization**: BERT base model with custom classifier
3. **Training Loop**: Optimized with AdamW optimizer
4. **Validation**: Regular evaluation on validation set
5. **Model Saving**: Automatic saving of best performing model

### Configuration

Key training parameters in `src/models/train_model.py`:

```python
config = {
    "model_name": "distilbert-base-uncased",
    "learning_rate": 2.77e-05,
    "batch_size": 16,
    "dropout": 0.209,
    "feature_hidden_size": 512,
    "num_epochs": 15,
    "max_token_length": 256
}
```

## 🌐 Web Application

### Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Processing**: Instant results with progress indicators
- **Beautiful Visualizations**: Interactive charts and graphs
- **User Experience**: Intuitive navigation and clear feedback

### Deployment Options

#### Local Development
```bash
streamlit run app.py
```

#### Streamlit Cloud
1. Push code to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with `app.py` as main file

<div align="center">
  <p>Made with ❤️ by the Fake News Detector Team</p>
  <p>
    <a href="https://github.com/a-nadafian/fake-news-detector">GitHub</a> •

  </p>
</div>

