# src/models/evaluate_model.py

import os
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings

# Add the 'src' directory to the Python path
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from src.models.train_model import HybridFakeNewsClassifier, HybridDataset, collate_fn, get_engineered_feature_columns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')


def evaluate_model(model_path, test_data_path, reports_dir):
    """
    Loads the final model, evaluates it on the test set, and saves a detailed report.
    """
    print("--- Starting Final Model Evaluation on Test Set ---")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(reports_dir, exist_ok=True)

    # --- Load Model and Configuration ---
    print(f"Loading model from: {model_path}")
    config_path = os.path.join(model_path, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = HybridFakeNewsClassifier(
        bert_model_name=config['model_name'],
        num_labels=config['num_labels'],
        num_engineered_features=len(config['feature_columns']),
        feature_hidden_size=config['feature_hidden_size'],
        dropout=config['dropout']
    )
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # --- Load Test Data ---
    print(f"Loading test data from: {test_data_path}")
    test_dataset = pd.read_csv(test_data_path)
    # Convert to Hugging Face Dataset object for consistency
    from datasets import Dataset
    test_hf_dataset = Dataset.from_pandas(test_dataset)

    feature_columns = get_engineered_feature_columns(test_hf_dataset)

    test_hybrid = HybridDataset(test_hf_dataset, tokenizer, config['max_token_length'], feature_columns)
    test_dataloader = DataLoader(test_hybrid, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    # --- Make Predictions ---
    print("Making predictions on the test set...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            labels = batch.pop('labels').cpu()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            preds = torch.argmax(outputs['logits'], dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # --- Generate and Save Evaluation Report ---
    print("Generating evaluation report...")

    # Classification Report
    report_dict = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=['Fake', 'Real'])

    final_accuracy = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average='weighted')

    report_path = os.path.join(reports_dir, 'final_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("--- Final Model Evaluation Report ---\n\n")
        f.write(f"Accuracy: {final_accuracy:.4f}\n")
        f.write(f"Weighted F1-Score: {final_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)

    print(f"\n✅ Detailed evaluation report saved to: {report_path}")
    print(report_str)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Confusion matrix saved to: {cm_path}")

    print("\n--- Evaluation Complete ---")


if __name__ == '__main__':
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'final_model')
    test_file = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test.csv')
    reports_output_dir = os.path.join(PROJECT_ROOT, 'reports')

    if not os.path.exists(model_dir):
        print(f"Error: Model not found at {model_dir}")
        print("Please train a model first by running 'src/models/train_model.py'.")
    elif not os.path.exists(test_file):
        print(f"Error: Test data not found at {test_file}")
        print("Please process your data first by running 'src/data/advanced_preprocessing.py'.")
    else:
        evaluate_model(model_dir, test_file, reports_output_dir)
