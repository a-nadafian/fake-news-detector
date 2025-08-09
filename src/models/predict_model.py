# src/models/predict_model.py

import os
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix
from src.features.build_features import preprocess_text


def predict_sentence(text, model, tokenizer, device="cpu"):
    """
    Predict whether a given text is fake or real news.
    
    Args:
        text (str): Raw text input to classify
        model: Loaded transformer model
        tokenizer: Loaded tokenizer
        device (str): Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        dict: Prediction results with label and confidence
    """
    # Preprocess the text using the same pipeline as training
    preprocessed_text = preprocess_text(text)
    
    # Tokenize the preprocessed text
    inputs = tokenizer(
        preprocessed_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Map prediction to label using model config (0: Fake, 1: Real)
    id2label = getattr(model.config, 'id2label', {0: 'Fake', 1: 'Real'})
    label2id = getattr(model.config, 'label2id', {'Fake': 0, 'Real': 1})
    predicted_label = id2label.get(predicted_class, str(predicted_class))
    
    return {
        "text": text,
        "preprocessed_text": preprocessed_text,
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": {
            "Fake": probabilities[0][label2id['Fake']].item(),
            "Real": probabilities[0][label2id['Real']].item()
        }
    }


def main(config):
    """
    Main function to evaluate the fine-tuned model on the unseen test set.

    Args:
        config (dict): A dictionary containing configuration parameters.
    """
    print("--- Starting model evaluation pipeline ---")

    # 1. Load Fine-Tuned Model and Tokenizer
    print(f"Loading model and tokenizer from: {config['model_path']}")
    if not os.path.exists(config['model_path']):
        print(f"Error: Model not found at {config['model_path']}")
        print("Please run the training script first (`make train` or `python src/models/train_model.py`).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    model = AutoModelForSequenceClassification.from_pretrained(config['model_path']).to(device)
    print(f"Model loaded successfully on device: {device}")

    # 2. Load and Prepare Test Data
    print(f"Loading dedicated test dataset from: {config['data_path']}")
    if not os.path.exists(config['data_path']):
        print(f"Error: Test data file not found at {config['data_path']}")
        print(
            "Please run the data processing script first (`make data` or `python src/data/make_dataset.py`) to generate the test set.")
        return

    test_df = pd.read_csv(config['data_path'])
    test_dataset = Dataset.from_pandas(test_df)

    # --- FIX: Ensure all 'text' entries are strings to prevent tokenizer errors ---
    # This handles any potential null/NaN values loaded from the CSV.
    def sanitize_text(example):
        example['text'] = '' if example['text'] is None else str(example['text'])
        return example

    test_dataset = test_dataset.map(sanitize_text)
    print(f"Unseen test set loaded and sanitized with {len(test_dataset)} samples.")

    # 3. Tokenize the Test Data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    print("Tokenizing test data...")
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 4. Generate Predictions
    print("Generating predictions...")
    eval_args = TrainingArguments(
        output_dir=os.path.join(config['report_dir'], 'eval_temp'),
        report_to="none",
    )

    trainer = Trainer(model=model, args=eval_args)
    predictions = trainer.predict(tokenized_test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # 5. Calculate Metrics and Generate Reports
    print("Calculating metrics and generating reports...")
    os.makedirs(config['report_dir'], exist_ok=True)

    # a) Classification Report
    report = classification_report(true_labels, predicted_labels, target_names=['Fake (0)', 'Real (1)'])
    report_path = os.path.join(config['report_dir'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nClassification Report:\n\n{report}")
    print(f"Classification report saved to: {report_path}")

    # b) Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Unseen Test Data')

    cm_path = os.path.join(config['report_dir'], 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix visualization saved to: {cm_path}")

    print("\n--- Evaluation complete! ✅ ---")
    
    # Demonstrate the new predict_sentence function
    print("\n--- Testing predict_sentence function ---")
    
    # Sample fake news text for testing
    sample_fake_news = "BREAKING: Scientists discover that drinking hot water with lemon cures all diseases instantly! This revolutionary treatment has been hidden by big pharma for decades."
    
    # Sample real news text for testing
    sample_real_news = "The World Health Organization released new guidelines for COVID-19 prevention measures based on recent scientific studies."
    
    # Test predictions
    print(f"\nTesting fake news sample:")
    fake_result = predict_sentence(sample_fake_news, model, tokenizer, device)
    print(f"Original text: {fake_result['text']}")
    print(f"Preprocessed: {fake_result['preprocessed_text']}")
    print(f"Prediction: {fake_result['prediction']} (Confidence: {fake_result['confidence']:.3f})")
    print(f"Probabilities - Real: {fake_result['probabilities']['Real']:.3f}, Fake: {fake_result['probabilities']['Fake']:.3f}")
    
    print(f"\nTesting real news sample:")
    real_result = predict_sentence(sample_real_news, model, tokenizer, device)
    print(f"Original text: {real_result['text']}")
    print(f"Preprocessed: {real_result['preprocessed_text']}")
    print(f"Prediction: {real_result['prediction']} (Confidence: {real_result['confidence']:.3f})")
    print(f"Probabilities - Real: {real_result['probabilities']['Real']:.3f}, Fake: {real_result['probabilities']['Fake']:.3f}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    config = {
        "model_path": os.path.join(project_root, 'models', 'fake-news-detector', 'final_model'),
        "data_path": os.path.join(project_root, 'data', 'processed', 'test.csv'),
        "report_dir": os.path.join(project_root, 'reports')
    }

    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned fake news classification model.")
    parser.add_argument('--model_path', type=str, default=config['model_path'])
    parser.add_argument('--data_path', type=str, default=config['data_path'])
    parser.add_argument('--report_dir', type=str, default=config['report_dir'])

    args = parser.parse_args()
    config.update(vars(args))

    main(config)
