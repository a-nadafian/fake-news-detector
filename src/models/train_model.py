# src/models/train_model.py

import os
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    logging as hf_logging
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
import warnings
import json
import mlflow
import mlflow.pytorch
import optuna

warnings.filterwarnings('ignore')
hf_logging.set_verbosity_info()

# Define project root at the top level
# This assumes the script is in /path/to/project/src/models/train_model.py
# and calculates the path to /path/to/project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HybridFakeNewsClassifier(nn.Module):
    """
    Hybrid model that combines transformer text embeddings with engineered features.
    Dynamically handles different transformer model architectures.
    """

    def __init__(self, bert_model_name, num_labels, num_engineered_features,
                 feature_hidden_size=256, dropout=0.3):
        super(HybridFakeNewsClassifier, self).__init__()

        self.num_labels = num_labels

        # Load the base transformer model and its configuration
        self.bert = AutoModel.from_pretrained(
            bert_model_name,
            output_hidden_states=True,
            return_dict=True
        )
        bert_hidden_size = self.bert.config.hidden_size

        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(num_engineered_features, feature_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden_size, feature_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer to combine BERT and engineered features
        combined_size = bert_hidden_size + feature_hidden_size // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_size // 2, combined_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classification layer
        self.classifier = nn.Linear(combined_size // 4, num_labels)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in [self.feature_processor, self.fusion_layer, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                engineered_features=None, labels=None):

        # Create a dictionary for the arguments to the BERT model
        bert_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        # Only add token_type_ids if the model is NOT DistilBERT
        if self.bert.config.model_type != 'distilbert':
            bert_inputs['token_type_ids'] = token_type_ids

        # Pass the conditional inputs to the BERT model
        bert_outputs = self.bert(**bert_inputs)

        bert_cls_output = bert_outputs.last_hidden_state[:, 0, :]

        if engineered_features is not None:
            feature_output = self.feature_processor(engineered_features)
        else:
            # If no features are provided, create a zero tensor
            feature_output = torch.zeros(bert_cls_output.size(0),
                                         self.feature_processor[3].out_features,
                                         device=bert_cls_output.device)

        combined_features = torch.cat([bert_cls_output, feature_output], dim=1)
        fused_features = self.fusion_layer(combined_features)
        logits = self.classifier(fused_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}


class HybridDataset(TorchDataset):
    """Custom dataset for text and engineered features."""

    def __init__(self, dataset, tokenizer, max_length, feature_columns):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = str(item.get('text', ''))

        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        engineered_features = [float(item.get(col, 0.0)) for col in self.feature_columns]
        engineered_features = torch.tensor(engineered_features, dtype=torch.float32)

        if 'token_type_ids' not in tokenized:
            tokenized['token_type_ids'] = torch.zeros_like(tokenized['input_ids'])

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'token_type_ids': tokenized['token_type_ids'],
            'engineered_features': engineered_features,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def get_engineered_feature_columns(dataset):
    """Extracts engineered feature column names."""
    return sorted([col for col in dataset.column_names if col.startswith('feature_') or col.startswith('tfidf_')])


def run_training(config, trial=None):
    """
    Encapsulates a single training and evaluation run.
    Integrates with MLflow for tracking and Optuna for hyperparameter tuning.
    """
    with mlflow.start_run():
        # --- Log parameters ---
        mlflow.log_params(config)
        print("--- Starting a new training run ---")
        print(f"Config: {json.dumps(config, indent=2)}")

        # --- Setup ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dir = os.path.join(PROJECT_ROOT, 'models', 'fake-news-detector', mlflow.active_run().info.run_id)
        os.makedirs(output_dir, exist_ok=True)

        # --- Load Data ---
        train_dataset = Dataset.from_csv(config['train_data_path'])
        val_dataset = Dataset.from_csv(config['val_data_path'])
        feature_columns = get_engineered_feature_columns(train_dataset)

        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        train_hybrid = HybridDataset(train_dataset, tokenizer, config['max_token_length'], feature_columns)
        val_hybrid = HybridDataset(val_dataset, tokenizer, config['max_token_length'], feature_columns)

        train_dataloader = DataLoader(train_hybrid, batch_size=config['batch_size'], shuffle=True,
                                      collate_fn=collate_fn, num_workers=2)
        val_dataloader = DataLoader(val_hybrid, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn,
                                    num_workers=2)

        # --- Initialize Model ---
        model = HybridFakeNewsClassifier(
            bert_model_name=config['model_name'],
            num_labels=config['num_labels'],
            num_engineered_features=len(feature_columns),
            feature_hidden_size=config['feature_hidden_size'],
            dropout=config['dropout']
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

        # --- Training Loop ---
        best_val_f1 = 0.0
        for epoch in range(config['num_epochs']):
            model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()

            # --- Validation ---
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch.pop('labels')
                    outputs = model(**batch)
                    preds = torch.argmax(outputs['logits'], dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            val_accuracy = accuracy_score(all_labels, all_preds)

            print(f"Epoch {epoch + 1}/{config['num_epochs']} | Val F1: {val_f1:.4f} | Val Acc: {val_accuracy:.4f}")
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                mlflow.pytorch.log_model(model, "best_model")
                print(f"  -> New best model saved with F1: {best_val_f1:.4f}")

            if trial:
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        print(f"--- Run finished. Best Val F1: {best_val_f1:.4f} ---")
        return best_val_f1


def objective(trial):
    """Optuna objective function for hyperparameter search."""
    config = {
        "model_name": trial.suggest_categorical("model_name",
                                                ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "feature_hidden_size": trial.suggest_categorical("feature_hidden_size", [128, 256, 512]),
        "train_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'train.csv'),
        "val_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'val.csv'),
        "num_labels": 2,
        "num_epochs": 5,  # Keep epochs low for HPO, train the best model for longer
        "max_token_length": 256,
    }

    return run_training(config, trial)


def perform_error_analysis(model, dataloader, device, output_path):
    """Performs error analysis and saves incorrect predictions to a CSV."""
    print("\n--- Performing Error Analysis ---")
    model.eval()
    incorrect_predictions = []

    # We need to access the original text, which is not in the dataloader.
    # We'll iterate through the dataset and dataloader in parallel.
    dataset = dataloader.dataset.dataset

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            original_items = dataset[i * dataloader.batch_size: (i + 1) * dataloader.batch_size]

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            outputs = model(**batch)
            preds = torch.argmax(outputs['logits'], dim=1)

            for j in range(len(labels)):
                if preds[j] != labels[j]:
                    incorrect_predictions.append({
                        'text': original_items[j]['text'],
                        'true_label': labels[j].item(),
                        'predicted_label': preds[j].item()
                    })

    if incorrect_predictions:
        df_errors = pd.DataFrame(incorrect_predictions)
        df_errors.to_csv(output_path, index=False)
        print(f"✅ Error analysis saved to {output_path} ({len(df_errors)} incorrect predictions)")
    else:
        print("✅ No incorrect predictions found on the validation set!")


def main(args):
    """Main function to drive training or hyperparameter tuning."""
    if args.tune_hyperparameters:
        print("--- Starting Hyperparameter Tuning with Optuna ---")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=args.n_trials)

        print("\n--- Hyperparameter Tuning Complete ---")
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value (F1 Score): {trial.value:.4f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train the final best model with more epochs
        print("\n--- Training final model with best hyperparameters ---")
        best_config = trial.params
        best_config.update({
            "train_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'train.csv'),
            "val_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'val.csv'),
            "num_labels": 2,
            "num_epochs": 15,  # Train for longer
            "max_token_length": 256,
            "batch_size": best_config.get('batch_size', 16)  # Ensure batch_size is set
        })
        run_training(best_config)

    else:
        print("--- Starting a Single Training Run ---")
        config = {
            "model_name": args.model_name,
            "train_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'train.csv'),
            "val_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'val.csv'),
            "num_labels": 2,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "max_token_length": 512,
            "learning_rate": 2e-5,
            "feature_hidden_size": 256,
            "dropout": 0.3,
        }
        run_training(config)

    # --- Final Error Analysis on the Best Model ---
    # Find the best run from MLflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_f1 DESC"],
        max_results=1
    )[0]

    best_model_uri = f"runs:/{best_run.info.run_id}/best_model"
    best_model = mlflow.pytorch.load_model(best_model_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    # Create dataloader for error analysis
    val_dataset = Dataset.from_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'val.csv'))
    feature_columns = get_engineered_feature_columns(val_dataset)
    tokenizer = AutoTokenizer.from_pretrained(best_run.data.params['model_name'])
    val_hybrid = HybridDataset(val_dataset, tokenizer, 256, feature_columns)
    val_dataloader = DataLoader(val_hybrid, batch_size=16, shuffle=False, collate_fn=collate_fn)

    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    error_analysis_path = os.path.join(reports_dir, 'error_analysis.csv')
    perform_error_analysis(best_model, val_dataloader, device, error_analysis_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or tune a hybrid model for fake news classification.")
    parser.add_argument('--tune_hyperparameters', action='store_true', help="Enable hyperparameter tuning with Optuna.")
    parser.add_argument('--n_trials', type=int, default=20, help="Number of trials for Optuna hyperparameter tuning.")
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help="Default model name for a single run.")
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of epochs for a single run.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for a single run.")

    args = parser.parse_args()
    main(args)
