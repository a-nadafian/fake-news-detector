# src/models/train_model.py

import os
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    logging as hf_logging
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set verbosity for the transformers library to provide informative logs.
hf_logging.set_verbosity_info()


def verify_preprocessing_consistency(dataset, tokenizer, max_length=512):
    """
    Verify that the preprocessing pipeline is working correctly.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        bool: True if preprocessing looks correct
    """
    print("\n--- Verifying Preprocessing Consistency ---")
    
    # Check a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]['text']
        print(f"Sample {i+1}:")
        print(f"  Original length: {len(sample)} characters")
        print(f"  Text preview: {sample[:100]}...")
        
        # Tokenize and check
        tokens = tokenizer.tokenize(sample)
        print(f"  Token count: {len(tokens)}")
        print(f"  First 10 tokens: {tokens[:10]}")
        
        # Check if tokens are reasonable
        if len(tokens) == 0:
            print(f"  ⚠️  WARNING: Sample {i+1} has no tokens after tokenization!")
            return False
        
        print()
    
    print("✅ Preprocessing verification complete")
    return True


def compute_metrics(p):
    """
    Computes accuracy and F1 score for evaluation during training.

    Args:
        p (EvalPrediction): A tuple containing model predictions and true labels.

    Returns:
        dict: A dictionary with 'accuracy' and 'f1' scores.
    """
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted'),
        'precision': f1_score(p.label_ids, preds, average='weighted', zero_division=0),
        'recall': f1_score(p.label_ids, preds, average='weighted', zero_division=0)
    }


def main(config):
    """
    Main function to execute the model training pipeline.
    This function now loads the pre-split train and validation sets.

    Args:
        config (dict): A dictionary containing configuration parameters.
    """
    print("--- Starting model training pipeline ---")

    # 1. Load Pre-split Datasets
    processed_data_dir = os.path.dirname(config['train_data_path'])
    print(f"Loading pre-split datasets from: {processed_data_dir}")

    if not os.path.exists(config['train_data_path']) or not os.path.exists(config['val_data_path']):
        print("Error: train.csv or val.csv not found.")
        print("Please run the data processing script first (`make data` or `python src/data/make_dataset.py`).")
        return

    # Load the datasets using the Hugging Face datasets library
    train_dataset = Dataset.from_csv(config['train_data_path'])
    val_dataset = Dataset.from_csv(config['val_data_path'])

    # --- FIX 1: Ensure all 'text' entries are strings to prevent tokenizer errors ---
    # This handles any potential null/NaN values loaded from the CSV.
    def sanitize_text(example):
        example['text'] = '' if example['text'] is None else str(example['text'])
        return example

    train_dataset = train_dataset.map(sanitize_text)
    val_dataset = val_dataset.map(sanitize_text)
    
    # Print some sample data to verify preprocessing
    print("\n--- Sample Training Data ---")
    for i in range(min(3, len(train_dataset))):
        print(f"Sample {i+1}:")
        print(f"  Text: {train_dataset[i]['text'][:100]}...")
        print(f"  Label: {train_dataset[i]['label']}")
        print()


    # Combine them into a DatasetDict for the Trainer
    datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    print("Datasets loaded and sanitized successfully.")
    print(f"Training samples: {len(datasets['train'])}, Validation samples: {len(datasets['validation'])}")

    # 2. Load Tokenizer
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # 3. Tokenize Data
    def tokenize_function(examples):
        """Tokenizes the 'text' column of the dataset."""
        return tokenizer(examples['text'], padding='max_length', truncation=True,
                         max_length=config['max_token_length'])

    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    print("Tokenization complete.")
    
    # Verify preprocessing consistency
    verify_preprocessing_consistency(train_dataset, tokenizer, config['max_token_length'])

    # 4. Load Pre-trained Model
    print(f"Loading model: {config['model_name']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=config['num_labels']
    ).to(device)
    print(f"Model loaded on device: {device}")

    # 5. Define Training Arguments
    print("Setting up training arguments...")
    # --- FIX 2: Use `evaluation_strategy` instead of the outdated `eval_strategy` ---
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    # 6. Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 7. Start Training
    print("--- Starting model fine-tuning ---")
    trainer.train()
    print("--- Training complete! ✅ ---")

    # 8. Evaluate on validation set and generate detailed metrics
    print("--- Evaluating final model on validation set ---")
    val_results = trainer.evaluate()
    print(f"Final validation accuracy: {val_results['eval_accuracy']:.4f}")
    print(f"Final validation F1 score: {val_results['eval_f1']:.4f}")
    
    # Generate predictions for detailed analysis
    val_predictions = trainer.predict(tokenized_datasets['validation'])
    val_preds = np.argmax(val_predictions.predictions, axis=1)
    val_true = val_predictions.label_ids
    
    # Create detailed classification report
    report = classification_report(val_true, val_preds, target_names=['Real', 'Fake'])
    print("\n--- Detailed Classification Report ---")
    print(report)
    
    # Save classification report
    os.makedirs(os.path.join(project_root, 'reports'), exist_ok=True)
    report_path = os.path.join(project_root, 'reports', 'training_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # Create and save confusion matrix
    cm = confusion_matrix(val_true, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Training Validation')
    
    cm_path = os.path.join(project_root, 'reports', 'training_confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # 9. Save the final model and tokenizer
    final_model_path = os.path.join(config['output_dir'], 'final_model')
    print(f"\nSaving the best fine-tuned model to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("Model saved successfully.")
    
    # 10. Test the model with sample predictions
    print("\n--- Testing model with sample predictions ---")
    test_samples = [
        "BREAKING: Scientists discover that drinking hot water with lemon cures all diseases instantly!",
        "The World Health Organization released new guidelines for COVID-19 prevention measures."
    ]
    
    for i, sample in enumerate(test_samples):
        inputs = tokenizer(sample, padding='max_length', truncation=True, 
                          max_length=config['max_token_length'], return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        label_map = {0: "Real", 1: "Fake"}
        print(f"Sample {i+1}: {sample[:50]}...")
        print(f"  Prediction: {label_map[predicted_class]} (Confidence: {confidence:.3f})")
        print()


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    config = {
        "model_name": "bert-base-uncased",  # Changed to bert-base-uncased for consistency with preprocessing
        "train_data_path": os.path.join(project_root, 'data', 'processed', 'train.csv'),
        "val_data_path": os.path.join(project_root, 'data', 'processed', 'val.csv'),
        "output_dir": os.path.join(project_root, 'models', 'fake-news-detector'),
        "num_labels": 2,
        "num_epochs": 3,
        "batch_size": 16,
        "max_token_length": 512
    }

    parser = argparse.ArgumentParser(description="Train a transformer model for fake news classification.")
    parser.add_argument('--model_name', type=str, default=config['model_name'])
    parser.add_argument('--train_data_path', type=str, default=config['train_data_path'])
    parser.add_argument('--val_data_path', type=str, default=config['val_data_path'])
    parser.add_argument('--output_dir', type=str, default=config['output_dir'])
    parser.add_argument('--num_epochs', type=int, default=config['num_epochs'])
    parser.add_argument('--batch_size', type=int, default=config['batch_size'])
    parser.add_argument('--max_token_length', type=int, default=config['max_token_length'])

    args = parser.parse_args()
    config.update(vars(args))

    main(config)
