# 🔍 Fake News Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Transformers-4.54+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered fake news detection system built with BERT transformers and deployed using Streamlit. This project provides a complete pipeline from data preprocessing to model training and real-time prediction.

## 🌟 Features

### 🤖 AI Model
- **BERT-based Transformer**: Fine-tuned on ISOT Fake News Dataset
- **High Accuracy**: Achieves excellent performance on test data
- **Consistent Preprocessing**: Fixed preprocessing pipeline for reliable predictions
- **Real-time Inference**: Fast prediction with confidence scores

### 🖥️ Web Application
- **Beautiful UI**: Modern, responsive design with custom styling
- **Single Text Analysis**: Analyze individual news articles or headlines
- **Batch Processing**: Upload CSV files for bulk analysis
- **Test Examples**: Try pre-defined fake and real news examples
- **Visual Results**: Interactive charts showing prediction confidence
- **Preprocessing Transparency**: View cleaned text for debugging

### 🛠️ Development Tools
- **Modular Architecture**: Clean separation of data, features, and models
- **Comprehensive Testing**: Preprocessing pipeline verification
- **Easy Deployment**: Multiple deployment options (Streamlit Cloud, Heroku, Docker)
- **Documentation**: Complete guides and examples

## 📊 Model Performance

The model has been trained on the ISOT Fake News Dataset and achieves:
- **High accuracy** on test data
- **Robust preprocessing** pipeline for external data
- **Consistent predictions** across different text formats
- **Confidence scoring** for prediction reliability

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- At least 4GB RAM (8GB recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python simple_test.py
   ```

## 📁 Project Structure

```
fake-news-detector/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── DEPLOYMENT.md                   # Deployment guide
├── simple_test.py                  # Quick preprocessing test
├── test_preprocessing.py           # Comprehensive preprocessing tests
├── data/                           # Data directory
│   ├── raw/                        # Raw dataset files
│   │   └── WELFake_Dataset.csv     # Your dataset here
│   └── processed/                  # Processed dataset files
├── models/                         # Trained models
│   └── fake-news-detector/
│       └── final_model/            # Your trained model
├── src/                            # Source code
│   ├── data/
│   │   └── make_dataset.py         # Data processing pipeline
│   ├── features/
│   │   └── build_features.py       # Preprocessing utilities
│   └── models/
│       ├── train_model.py          # Model training script
│       └── predict_model.py        # Prediction script
├── notebooks/                      # Jupyter notebooks
├── reports/                        # Evaluation reports
└── docs/                          # Documentation
```

## 📋 Complete Workflow Guide

### Step 1: Data Preparation

#### 1.1 Prepare Your Dataset

Your dataset should be in CSV format with the following structure:

```csv
title,text,label
"News Title 1","Full news text content...",0
"News Title 2","Another news article...",1
```

**Label Mapping:**
- `0` = Real News
- `1` = Fake News

#### 1.2 Place Your Dataset

Put your dataset in the `data/raw/` directory:
```bash
# Copy your dataset to the raw data directory
cp your_dataset.csv data/raw/WELFake_Dataset.csv
```

### Step 2: Data Preprocessing

#### 2.1 Understanding the Preprocessing Pipeline

The preprocessing pipeline is centralized in `src/features/build_features.py`, which contains the `preprocess_text()` function. This function is used throughout the project to ensure consistent text preprocessing.

**Key Components of `build_features.py`:**
- **`preprocess_text(text)`**: Main preprocessing function that:
  - Removes URLs and special characters
  - Converts text to lowercase
  - Uses BERT tokenizer for consistent tokenization
  - Removes stopwords and applies lemmatization
- **`preprocess_batch(texts)`**: Batch processing for multiple texts
- **NLTK resource management**: Downloads and manages required NLP resources

#### 2.2 Where `build_features.py` is Used

**1. Data Processing (`src/data/make_dataset.py`):**
```python
from src.features.build_features import preprocess_text
# Used to preprocess all training data
df['cleaned_content'] = df['content'].apply(preprocess_text)
```

**2. Model Training (`src/models/train_model.py`):**
```python
# The training script uses preprocessed data from make_dataset.py
# which was processed using build_features.py
```

**3. Model Prediction (`src/models/predict_model.py`):**
```python
from src.features.build_features import preprocess_text
# Used to preprocess new input text before prediction
preprocessed_text = preprocess_text(text)
```

**4. Web Application (`app.py`):**
```python
from src.features.build_features import preprocess_text
# Used to preprocess user input in real-time
preprocessed_text = preprocess_text(text_input)
```

**5. Testing (`test_preprocessing.py` and `simple_test.py`):**
```python
from src.features.build_features import preprocess_text
# Used to verify preprocessing works correctly
```

#### 2.3 Run the Preprocessing Pipeline

```bash
# Process your dataset
python src/data/make_dataset.py
```

This will:
- Load your raw dataset from `data/raw/WELFake_Dataset.csv`
- Use `build_features.py` to apply text preprocessing
- Split data into train (80%), validation (10%), and test (10%) sets
- Save processed data to `data/processed/`

#### 2.4 Verify Preprocessing

```bash
# Test the preprocessing pipeline
python test_preprocessing.py
```

This will show you:
- Sample preprocessed texts
- Tokenization results
- Preprocessing consistency

### Step 3: Model Training

#### 3.1 Train the Model

```bash
# Train the BERT model
python src/models/train_model.py
```

**Training Configuration:**
- **Model**: BERT (bert-base-uncased)
- **Epochs**: 3
- **Batch Size**: 16
- **Max Sequence Length**: 512
- **Learning Rate**: 2e-5

**Training Output:**
- Model checkpoints saved to `models/fake-news-detector/`
- Final model saved to `models/fake-news-detector/final_model/`
- Training reports saved to `reports/`

#### 3.2 Monitor Training Progress

The training script will show:
- Sample training data
- Preprocessing verification
- Training progress and metrics
- Final evaluation results
- Sample predictions

### Step 4: Model Evaluation

#### 4.1 Evaluate on Test Set

```bash
# Evaluate the trained model
python src/models/predict_model.py
```

This will:
- Load your trained model
- Evaluate on the test set
- Generate classification reports
- Create confusion matrix
- Test with sample fake and real news

#### 4.2 Review Results

Check the generated reports in `reports/`:
- `classification_report.txt` - Detailed metrics
- `confusion_matrix.png` - Visual confusion matrix

### Step 5: Web Application

#### 5.1 Start the Streamlit App

```bash
# Launch the web application
streamlit run app.py
```

#### 5.2 Use the Application

1. **Single Text Analysis**:
   - Paste news text in the text area
   - Click "Analyze Text"
   - View prediction and confidence

2. **Batch Analysis**:
   - Upload a CSV file with a 'text' column
   - Click "Analyze All Texts"
   - View results table and statistics

3. **Test Examples**:
   - Try pre-defined fake and real news examples
   - See how the model performs

## 🔧 Configuration Options

### Model Configuration

Edit `src/models/train_model.py` to modify training parameters:

```python
config = {
    "model_name": "bert-base-uncased",  # Change model
    "num_epochs": 3,                    # Training epochs
    "batch_size": 16,                   # Batch size
    "max_token_length": 512,            # Max sequence length
    # ... other parameters
}
```

### Preprocessing Configuration

The preprocessing pipeline is centralized in `src/features/build_features.py`. This ensures consistency across all parts of the project.

**Key Functions:**
```python
def preprocess_text(text):
    # Main preprocessing function used throughout the project
    # 1. URL removal
    # 2. Special character cleaning
    # 3. Lowercase conversion
    # 4. BERT tokenization
    # 5. Stopword removal
    # 6. Lemmatization

def preprocess_batch(texts):
    # Batch processing for multiple texts
    return [preprocess_text(text) for text in texts]
```

**Why Centralized Preprocessing?**
- **Consistency**: Same preprocessing for training and prediction
- **Maintainability**: Single place to update preprocessing logic
- **Reliability**: Ensures model works on new data
- **Debugging**: Easy to test and verify preprocessing

## 🧪 Testing Your Pipeline

### Test Preprocessing

```bash
python test_preprocessing.py
```

**Expected Output:**
```
✅ Successfully imported preprocess_text function
✅ Preprocessing successful!
Original: BREAKING: Scientists discover that drinking hot water with lemon cures all diseases!
Preprocessed: scientist discover drink hot water lemon cure disease
```

### Test Model Predictions

```bash
python src/models/predict_model.py
```

**Expected Output:**
```
Testing fake news sample:
Original text: BREAKING: Scientists discover that drinking hot water with lemon cures all diseases instantly!
Prediction: Fake (Confidence: 0.892)
```

## 📊 Understanding the Results

### Model Metrics

- **Accuracy**: Overall correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Prediction Confidence

- **High Confidence (>0.8)**: Very confident prediction
- **Medium Confidence (0.6-0.8)**: Moderately confident
- **Low Confidence (<0.6)**: Less confident, may need review

### Interpreting Predictions

- **Real News (0)**: Reliable, factual information
- **Fake News (1)**: Misleading, false, or unreliable information

## 🚀 Deployment

### Local Development

```bash
streamlit run app.py
```

### Streamlit Cloud (Recommended)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with main file: `app.py`


## 🔍 Troubleshooting

### Common Issues

1. **Model not found error**
   ```bash
   # Ensure model is trained first
   python src/models/train_model.py
   ```

2. **Dataset not found**
   ```bash
   # Place your dataset in data/raw/
   cp your_dataset.csv data/raw/WELFake_Dataset.csv
   ```

3. **Preprocessing errors**
   ```bash
   # Test preprocessing
   python test_preprocessing.py
   ```

4. **Memory issues**
   - Reduce batch size in training
   - Use smaller model (e.g., distilbert-base-uncased)

### Performance Tips

- **GPU Training**: Use CUDA for faster training
- **Batch Processing**: Use larger batch sizes if memory allows
- **Model Caching**: First run downloads models, subsequent runs are faster

## 📚 Dataset Information

### ISOT Fake News Dataset

- **Real News**: Articles from reliable sources (Reuters, BBC, etc.)
- **Fake News**: Articles from unreliable sources
- **Size**: ~45,000 articles
- **Balance**: Equal distribution of real and fake news
- **Topics**: Politics, technology, health, entertainment

### Custom Datasets

To use your own dataset:

1. **Format**: CSV with 'title', 'text', 'label' columns
2. **Labels**: 0 for real, 1 for fake
3. **Place**: `data/raw/WELFake_Dataset.csv`
4. **Process**: Run the preprocessing pipeline

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library
- **Streamlit**: For the web framework
- **ISOT Dataset**: For the training data
- **BERT**: For the base model architecture

---

⭐ **Star this repository if you find it helpful!**

