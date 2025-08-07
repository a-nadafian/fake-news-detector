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
    report = classification_report(true_labels, predicted_labels, target_names=['Real (0)', 'Fake (1)'])
    report_path = os.path.join(config['report_dir'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nClassification Report:\n\n{report}")
    print(f"Classification report saved to: {report_path}")

    # b) Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Unseen Test Data')

    cm_path = os.path.join(config['report_dir'], 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix visualization saved to: {cm_path}")

    print("\n--- Evaluation complete! ✅ ---")


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
