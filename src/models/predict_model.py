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
    TrainingArguments  # <-- Import TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main(config):
    """
    Main function to evaluate the fine-tuned model.

    Args:
        config (dict): A dictionary containing configuration parameters.
    """
    print("--- Starting model evaluation pipeline ---")

    # 1. Load Fine-Tuned Model and Tokenizer
    print(f"Loading model and tokenizer from: {config['model_path']}")
    if not os.path.exists(config['model_path']):
        print(f"Error: Model not found at {config['model_path']}")
        print("Please run the training script first (`make train`).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    model = AutoModelForSequenceClassification.from_pretrained(config['model_path']).to(device)
    print(f"Model loaded successfully on device: {device}")

    # 2. Load and Prepare Test Data
    print(f"Loading dataset from: {config['data_path']}")
    df = pd.read_csv(config['data_path'])
    df['content'] = df['title'].fillna('') + " - " + df['text'].fillna('')

    # IMPORTANT: Split the data in the exact same way as in the training script
    # to ensure we are evaluating on the correct, unseen test set.
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    test_dataset = Dataset.from_pandas(test_df)
    print(f"Test set prepared with {len(test_dataset)} samples.")

    # 3. Tokenize the Test Data
    def tokenize_function(examples):
        return tokenizer(examples['content'], padding='max_length', truncation=True, max_length=512)

    print("Tokenizing test data...")
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 4. Generate Predictions
    print("Generating predictions...")

    # --- FIX ---
    # The Trainer object automatically tries to connect to services like wandb.
    # To prevent this, we must pass it TrainingArguments with reporting disabled.
    eval_args = TrainingArguments(
        output_dir=os.path.join(config['report_dir'], 'eval_temp'),  # A temporary output dir is required
        report_to="none",
    )

    trainer = Trainer(model=model, args=eval_args)
    predictions = trainer.predict(tokenized_test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # 5. Calculate Metrics and Generate Reports
    print("Calculating metrics and generating reports...")

    # Ensure the reports directory exists
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
    plt.title('Confusion Matrix')

    cm_path = os.path.join(config['report_dir'], 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix visualization saved to: {cm_path}")

    print("\n--- Evaluation complete! ✅ ---")


if __name__ == '__main__':
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # --- Configuration Dictionary ---
    config = {
        "model_path": os.path.join(project_root, 'models', 'fake-news-detector', 'final_model'),
        "data_path": os.path.join(project_root, 'data', 'processed', 'processed_news.csv'),
        "report_dir": os.path.join(project_root, 'reports')
    }

    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned fake news classification model.")

    parser.add_argument('--model_path', type=str, default=config['model_path'],
                        help='Directory of the saved fine-tuned model and tokenizer.')
    parser.add_argument('--data_path', type=str, default=config['data_path'],
                        help='Path to the processed news CSV file.')
    parser.add_argument('--report_dir', type=str, default=config['report_dir'],
                        help='Directory to save the evaluation reports.')

    args = parser.parse_args()
    config.update(vars(args))

    main(config)
