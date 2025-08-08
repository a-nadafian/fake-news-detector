# Fake News Detector - Deployment Guide

## Overview

This guide explains how to deploy and use the Fake News Detector application built with Streamlit. The app provides a user-friendly interface for detecting fake news using a pre-trained BERT-based model.

## Features

- 🔍 **Single Text Analysis**: Analyze individual news articles or headlines
- 📊 **Batch Analysis**: Process multiple texts from CSV files
- 🧪 **Test Examples**: Try the model with pre-defined examples
- 📈 **Visual Results**: Interactive charts showing prediction confidence
- 🔧 **Preprocessing Pipeline**: Consistent text preprocessing for accurate predictions

## Prerequisites

1. **Python 3.8+** installed
2. **Trained Model**: Ensure your model is saved at `models/fake-news-detector/final_model/`
3. **Dependencies**: Install required packages

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <your-repo-url>
   cd fake-news-detector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model exists**:
   ```bash
   ls models/fake-news-detector/final_model/
   ```

## Running the Application

### Local Development

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

### Production Deployment

#### Option 1: Streamlit Cloud (Recommended)

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repository**
4. **Deploy** with the following settings:
   - Main file path: `app.py`
   - Python version: 3.8+

#### Option 2: Heroku

1. **Create a `Procfile`**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create `setup.sh`**:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### Option 3: Docker

1. **Create `Dockerfile`**:
   ```dockerfile
   FROM python:3.8-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t fake-news-detector .
   docker run -p 8501:8501 fake-news-detector
   ```

## Usage Guide

### Single Text Analysis

1. **Navigate to the "Single Text Analysis" tab**
2. **Enter news text** in the text area
3. **Click "Analyze Text"**
4. **View results**:
   - Prediction (Real/Fake)
   - Confidence score
   - Probability chart
   - Preprocessed text (expandable)

### Batch Analysis

1. **Navigate to the "Batch Analysis" tab**
2. **Upload a CSV file** with a 'text' column
3. **Click "Analyze All Texts"**
4. **View results**:
   - Summary table
   - Statistics (total, fake, real counts)

### Test Examples

1. **Navigate to the "Test Examples" tab**
2. **Select example category** (Fake/Real news)
3. **Choose specific example**
4. **Click "Test This Example"**
5. **View prediction results**

## Testing the Preprocessing Fix

To verify that the preprocessing pipeline works correctly:

```bash
python test_preprocessing.py
```

This will test the preprocessing function with various inputs and show you the cleaned text output.

## Model Testing

Test the model with external data:

```bash
python src/models/predict_model.py
```

This will:
- Load your trained model
- Test it with sample fake and real news
- Show predictions and confidence scores

## Troubleshooting

### Common Issues

1. **Model not found error**:
   - Ensure the model is saved at `models/fake-news-detector/final_model/`
   - Check file permissions

2. **Import errors**:
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes the `src` directory

3. **CUDA/GPU issues**:
   - The app automatically falls back to CPU if CUDA is not available
   - Check `nvidia-smi` for GPU status

4. **Memory issues**:
   - Reduce batch size in batch analysis
   - Close other applications using GPU memory

### Performance Tips

1. **First run**: The app downloads NLTK resources on first use
2. **Model loading**: Uses Streamlit caching for faster subsequent loads
3. **Batch processing**: Progress bar shows processing status
4. **Memory management**: Large files are processed in chunks

## API Usage

For programmatic access, you can import the prediction function:

```python
from src.models.predict_model import predict_sentence
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained('path/to/model')
tokenizer = AutoTokenizer.from_pretrained('path/to/model')

# Predict
result = predict_sentence("Your news text here", model, tokenizer)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub
4. Contact the development team 