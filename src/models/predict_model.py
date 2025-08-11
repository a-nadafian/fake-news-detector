# src/models/predict_model.py

import os
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch.nn.functional as F
import json
import mlflow
import warnings

# Import the necessary classes from your other scripts
# This requires your src directory to be in the Python path
import sys

# Add the 'src' directory to the Python path to allow for package-like imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from data.advanced_preprocessing import EnhancedFakeNewsPreprocessor
from models.train_model import HybridFakeNewsClassifier

warnings.filterwarnings('ignore')


class Predictor:
    """
    A class to load a trained model and make predictions on new text.
    """

    def __init__(self, model_uri):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model configuration
        config_path = os.path.join(model_uri.replace("runs:", "/mlruns/0").replace("/best_model", ""),
                                   "artifacts/best_model/data/model_config.json")

        # A fallback for older MLflow versions
        if not os.path.exists(config_path):
            config_path = os.path.join(model_uri.replace("runs:", "mlruns/0").replace("/best_model", ""),
                                       "artifacts/model/data/model_config.json")

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load the model from MLflow
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.to(self.device)
        self.model.eval()

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['bert_model_name'])

        # Initialize the preprocessor used during training
        self.preprocessor = EnhancedFakeNewsPreprocessor()

    def predict(self, text):
        """
        Makes a prediction on a single piece of text.
        """
        print(f"\n--- Predicting for text: '{text[:50]}...' ---")

        # 1. Preprocess the text to generate engineered features
        print("Step 1: Cleaning text and extracting engineered features...")
        cleaned_text = self.preprocessor.clean_text_enhanced(text)
        features_dict = self.preprocessor.extract_advanced_features(cleaned_text)

        # Ensure the order of features matches the training configuration
        engineered_features = [features_dict[col.replace('feature_', '')] for col in self.config['feature_columns'] if
                               col.startswith('feature_')]

        # Note: TF-IDF features cannot be generated for a single text instance in the same way.
        # For prediction, we will pass zeros for the TF-IDF part. The model's primary strength
        # will come from the text content (BERT) and the other engineered features.
        num_tfidf_features = len([col for col in self.config['feature_columns'] if col.startswith('tfidf_')])
        engineered_features.extend([0.0] * num_tfidf_features)

        features_tensor = torch.tensor([engineered_features], dtype=torch.float32).to(self.device)
        print(f"Step 2: Generated {len(engineered_features)} features.")

        # 2. Tokenize the text for the transformer model
        print("Step 3: Tokenizing text for the model...")
        inputs = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_token_length'],
            return_tensors='pt'
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        print("Step 4: Text tokenized.")

        # 3. Make the prediction
        print("Step 5: Making prediction...")
        with torch.no_grad():
            model_inputs = {**inputs, 'engineered_features': features_tensor}

            # Handle token_type_ids for different model types
            if self.model.bert.config.model_type == 'distilbert':
                model_inputs.pop('token_type_ids', None)

            outputs = self.model(**model_inputs)

            probabilities = F.softmax(outputs['logits'], dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        predicted_label = "Real" if predicted_class == 1 else "Fake"

        print("--- Prediction Complete ---")
        return predicted_label, confidence


def main(args):
    """
    Main function to drive the prediction process.
    """
    if args.model_uri:
        model_uri = args.model_uri
    else:
        # Find the best run from MLflow automatically
        print("--- No model URI provided. Finding the best model from MLflow... ---")
        client = mlflow.tracking.MlflowClient()
        try:
            experiment = client.get_experiment_by_name("Default")
            if not experiment:
                print("Error: Could not find the 'Default' MLflow experiment.")
                return

            best_run = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.val_f1 DESC"],
                max_results=1
            )[0]
            model_uri = f"runs:/{best_run.info.run_id}/best_model"
            print(f"✅ Found best model from run ID: {best_run.info.run_id}")
        except (IndexError, mlflow.exceptions.RestException):
            print("Error: No runs found in MLflow. Please train a model first using 'train_model.py'.")
            return

    predictor = Predictor(model_uri)

    if args.text:
        label, confidence = predictor.predict(args.text)
        print(f"\nInput Text: {args.text}")
        print(f"Predicted Label: {label}")
        print(f"Confidence: {confidence:.2%}")
    elif args.csv_file:
        print(f"\n--- Predicting for CSV file: {args.csv_file} ---")
        df = pd.read_csv(args.csv_file)
        if 'text' not in df.columns:
            print("Error: CSV file must contain a 'text' column.")
            return

        predictions = []
        for text in df['text']:
            label, _ = predictor.predict(text)
            predictions.append(label)

        df['predicted_label'] = predictions
        output_path = 'predictions.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions with a trained fake news detector.")
    parser.add_argument('--text', type=str, help="A single string of text to classify.")
    parser.add_argument('--csv_file', type=str, help="Path to a CSV file with a 'text' column to classify.")
    parser.add_argument('--model_uri', type=str,
                        help="Optional: MLflow URI of a specific model to use (e.g., 'runs:/<run_id>/best_model').")

    args = parser.parse_args()

    if not args.text and not args.csv_file:
        # Provide example usage if no arguments are given
        print("--- Example Usage ---")
        print("To predict a single piece of text:")
        print('python src/models/predict_model.py --text "This is some news text to classify."')
        print("\nTo predict for a CSV file:")
        print('python src/models/predict_model.py --csv_file "path/to/your/file.csv"')
    else:
        main(args)

