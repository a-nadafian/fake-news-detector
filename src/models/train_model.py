# src/models/train_model.py

import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    logging as hf_logging
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
import warnings
import json
import mlflow
import mlflow.pytorch

# Add the 'src' directory to the Python path to allow for package-like imports
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from src.data.advanced_preprocessing import EnhancedFakeNewsPreprocessor

warnings.filterwarnings('ignore')
hf_logging.set_verbosity_info()


class HybridFakeNewsClassifier(nn.Module):
    """
    Hybrid model that combines transformer text embeddings with engineered features.
    """

    def __init__(self, bert_model_name, num_labels, num_engineered_features,
                 feature_hidden_size=256, dropout=0.3):
        super(HybridFakeNewsClassifier, self).__init__()

        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(bert_model_name, return_dict=True)
        bert_hidden_size = self.bert.config.hidden_size

        self.feature_processor = nn.Sequential(
            nn.Linear(num_engineered_features, feature_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_hidden_size, feature_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        combined_size = bert_hidden_size + feature_hidden_size // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(combined_size // 2, num_labels)
        self._init_weights()

    def _init_weights(self):
        for module in [self.feature_processor, self.fusion_layer, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                engineered_features=None, labels=None):

        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if self.bert.config.model_type != 'distilbert':
            bert_inputs['token_type_ids'] = token_type_ids

        bert_outputs = self.bert(**bert_inputs)
        bert_cls_output = bert_outputs.last_hidden_state[:, 0, :]

        feature_output = self.feature_processor(engineered_features)
        combined_features = torch.cat([bert_cls_output, feature_output], dim=1)
        fused_features = self.fusion_layer(combined_features)
        logits = self.classifier(fused_features)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

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

        tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                   return_tensors='pt')
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        engineered_features = torch.tensor([float(item.get(col, 0.0)) for col in self.feature_columns],
                                           dtype=torch.float32)

        if 'token_type_ids' not in tokenized:
            tokenized['token_type_ids'] = torch.zeros_like(tokenized['input_ids'])

        return {**tokenized, 'engineered_features': engineered_features,
                'labels': torch.tensor(item['label'], dtype=torch.long)}


def collate_fn(batch):
    """Custom collate function for batching."""
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


def get_engineered_feature_columns(dataset):
    """Extracts engineered feature column names."""
    return sorted([col for col in dataset.column_names if col.startswith('feature_') or col.startswith('tfidf_')])


def main():
    """
    Main function to train the final model with the best parameters.
    """
    # --- Best Parameters Identified from Tuning ---
    config = {
        "model_name": "distilbert-base-uncased",
        "learning_rate": 2.770747124561795e-05,
        "batch_size": 16,
        "dropout": 0.20949130557990628,
        "feature_hidden_size": 512,
        "num_epochs": 15,  # Train for more epochs for the final model
        "max_token_length": 256,
        "num_labels": 2,
        "train_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'train.csv'),
        "val_data_path": os.path.join(PROJECT_ROOT, 'data', 'processed', 'val.csv'),
        "output_dir": os.path.join(PROJECT_ROOT, 'models', 'final_model')
    }

    print("--- Starting Final Model Training ---")
    print(f"Using configuration: {json.dumps(config, indent=2)}")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config['output_dir'], exist_ok=True)

    # --- Load Data ---
    train_dataset = Dataset.from_csv(config['train_data_path'])
    val_dataset = Dataset.from_csv(config['val_data_path'])
    feature_columns = get_engineered_feature_columns(train_dataset)
    config['feature_columns'] = feature_columns  # Save for prediction

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    train_hybrid = HybridDataset(train_dataset, tokenizer, config['max_token_length'], feature_columns)
    val_hybrid = HybridDataset(val_dataset, tokenizer, config['max_token_length'], feature_columns)

    train_dataloader = DataLoader(train_hybrid, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn,
                                  num_workers=2)
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

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save the best model directly
            torch.save(model.state_dict(), os.path.join(config['output_dir'], 'model.pth'))
            tokenizer.save_pretrained(config['output_dir'])
            with open(os.path.join(config['output_dir'], 'model_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  -> New best model saved to {config['output_dir']} with F1: {best_val_f1:.4f}")

    print("\n--- Final Model Training Complete ---")
    print(f"Best validation F1 score: {best_val_f1:.4f}")
    print(f"Final model saved in: {config['output_dir']}")


if __name__ == '__main__':
    main()
