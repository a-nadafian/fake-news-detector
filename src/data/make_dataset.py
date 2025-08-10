# src/data/make_dataset.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from src.features.build_features import preprocess_text

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(input_dir, output_dir):
    """
    Main function to execute the data processing pipeline for the final_training_corpus.csv.
    """
    logging.info("--- Starting data processing pipeline for final_training_corpus.csv ---")

    # 1. Define file paths
    input_filepath = os.path.join(input_dir, 'final_training_corpus.csv')
    if not os.path.exists(input_filepath):
        logging.error(f"Input file not found at: {input_filepath}")
        return

    # 2. Load Data
    logging.info(f"Loading raw data from: {input_filepath}")
    df = pd.read_csv(input_filepath)
    logging.info(f"Columns found in the dataset: {df.columns.tolist()}")

    # 3. Initial Cleaning & Feature Engineering
    logging.info("Performing initial cleaning and feature engineering...")
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # --- Check for 'text' column ---
    if 'text' not in df.columns:
        logging.error("Column 'text' not found in the dataframe. Please check your CSV file.")
        return

    df['text'] = df['text'].fillna('')
    # The 'content' is now just the 'text' since there is no 'title'
    df['content'] = df['text']
    logging.info("Dataset labels: 0 = Fake, 1 = Real")

    # 4. Apply Advanced Text Cleaning
    logging.info("Applying advanced text cleaning to the 'content' column. This may take a while...")
    df['cleaned_content'] = df['content'].apply(preprocess_text)

    df_processed = df[['cleaned_content', 'label']].copy()
    df_processed.rename(columns={'cleaned_content': 'text'}, inplace=True)
    df_processed.dropna(subset=['text'], inplace=True)

    # 5. Split Data
    logging.info("Splitting data into training (80%), validation (10%), and test (10%) sets...")
    train_df, temp_df = train_test_split(df_processed, test_size=0.2, random_state=42, stratify=df_processed['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    logging.info(
        f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}, Test set size: {len(test_df)}")

    # 6. Save Processed Data
    os.makedirs(output_dir, exist_ok=True)
    train_filepath = os.path.join(output_dir, 'train.csv')
    val_filepath = os.path.join(output_dir, 'val.csv')
    test_filepath = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_filepath, index=False)
    val_df.to_csv(val_filepath, index=False)
    test_df.to_csv(test_filepath, index=False)
    logging.info(f"Processed data saved to: {output_dir}")

    logging.info("--- Data processing complete! ✅ ---")


if __name__ == '__main__':
    # Define project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    parser = argparse.ArgumentParser(
        description="Process the news dataset into training, validation, and test sets.")

    default_input_path = os.path.join(project_root, 'data', 'raw')
    default_output_path = os.path.join(project_root, 'data', 'processed')

    parser.add_argument('--input_dir', type=str, default=default_input_path,
                        help="Directory where the raw 'final_training_corpus.csv' file is located.")
    parser.add_argument('--output_dir', type=str, default=default_output_path,
                        help="Directory where the final 'train.csv', 'val.csv', and 'test.csv' files will be saved.")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
