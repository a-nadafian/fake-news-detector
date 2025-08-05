# src/models/train_model.py

import os
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def compute_metrics(p):
    """
    Computes accuracy and F1 score for evaluation.

    Args:
        p (EvalPrediction): A tuple containing predictions and labels.

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

    Args:
        config (dict): A dictionary containing configuration parameters.
    """
    print("--- Starting model training pipeline ---")

    # 1. Load Processed Dataset
    print(f"Loading processed dataset from: {config['data_path']}")
    if not os.path.exists(config['data_path']):
        print(f"Error: Data file not found at {config['data_path']}")
        print("Please run the data processing script first (`make data`).")
        return

    # Load data with pandas and then convert to a Hugging Face Dataset object
    df = pd.read_csv(config['data_path'])

    # Combine title and text for a richer input, handling potential NaN values
    df['content'] = df['title'].fillna('') + " - " + df['text'].fillna('')

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("Dataset loaded and split successfully.")
    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    # 2. Load Tokenizer
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # 3. Tokenize Data
    def tokenize_function(examples):
        """Tokenizes the 'content' column of the dataset."""
        return tokenizer(examples['content'], padding='max_length', truncation=True,
                         max_length=config['max_token_length'])

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
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

    # --- FIX ---
    # The error indicates a very old version of the `transformers` library.
    # The best solution is to update your libraries (`pip install --upgrade transformers datasets`).
    # As a fallback, this version removes evaluation during training to ensure compatibility.
    # The model will still be evaluated once after training is complete.
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        report_to="none"  # Disables integration with wandb, etc.
    )

    # 6. Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,  # The test set is still passed for the final evaluation
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 7. Start Training
    print("--- Starting model fine-tuning ---")
    trainer.train()
    print("--- Training complete! ✅ ---")

    # 8. Evaluate the final model
    print("--- Evaluating final model on the test set ---")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # 9. Save the final model and tokenizer
    final_model_path = os.path.join(config['output_dir'], 'final_model')
    print(f"Saving the fine-tuned model to: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print("Model saved successfully.")


if __name__ == '__main__':
    # Get the directory of the current script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get the project root (from src/models -> src -> root).
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # --- Configuration Dictionary ---
    # Centralizes all settings for easy access and modification.
    config = {
        "model_name": "distilbert-base-uncased",  # A lighter version of BERT, good for faster training
        "data_path": os.path.join(project_root, 'data', 'processed', 'processed_news.csv'),
        "output_dir": os.path.join(project_root, 'models', 'fake-news-detector'),
        "num_labels": 2,
        "num_epochs": 1,  # Start with 1 epoch for a quick initial run
        "batch_size": 16,  # Adjust based on your GPU memory
        "max_token_length": 512  # Max length for BERT-like models
    }

    # This block allows the script to be run from the command line,
    # potentially overriding values from the config dictionary.
    parser = argparse.ArgumentParser(description="Train a transformer model for fake news classification.")

    parser.add_argument('--model_name', type=str, default=config['model_name'],
                        help='Name of the pre-trained model from Hugging Face Hub.')
    parser.add_argument('--data_path', type=str, default=config['data_path'],
                        help='Path to the processed news CSV file.')
    parser.add_argument('--output_dir', type=str, default=config['output_dir'],
                        help='Directory to save the trained model.')
    parser.add_argument('--num_epochs', type=int, default=config['num_epochs'], help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=config['batch_size'],
                        help='Training and evaluation batch size.')
    parser.add_argument('--max_token_length', type=int, default=config['max_token_length'],
                        help='Maximum token length for the tokenizer.')

    args = parser.parse_args()

    # Update config with any command-line arguments
    config.update(vars(args))

    # Run the main function with the final configuration
    main(config)
