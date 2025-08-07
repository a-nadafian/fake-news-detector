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
from sklearn.metrics import accuracy_score, f1_score

# Set verbosity for the transformers library to provide informative logs.
hf_logging.set_verbosity_info()


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
        'f1': f1_score(p.label_ids, preds, average='weighted')
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

    # 8. Save the final model and tokenizer
    final_model_path = os.path.join(config['output_dir'], 'final_model')
    print(f"Saving the best fine-tuned model to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("Model saved successfully.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    config = {
        "model_name": "distilbert-base-uncased",
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
