# src/data/make_dataset.py

import os
import argparse
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import logging
from transformers import BertTokenizer  # <-- FIX: Import a reliable tokenizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLTK Data Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
LOCAL_NLTK_DATA_PATH = os.path.join(project_root, '.nltk_data')
if LOCAL_NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, LOCAL_NLTK_DATA_PATH)


def download_nltk_resources():
    """
    Downloads necessary NLTK resources (stopwords and wordnet).
    The problematic 'punkt' tokenizer is no longer needed.
    """
    os.makedirs(LOCAL_NLTK_DATA_PATH, exist_ok=True)

    # We only need stopwords and wordnet now
    resources_to_download = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4'
    }
    for resource_name, resource_path in resources_to_download.items():
        try:
            nltk.data.find(resource_path, paths=[LOCAL_NLTK_DATA_PATH])
            logging.info(f"NLTK resource '{resource_name}' already present.")
        except LookupError:
            logging.info(f"Downloading NLTK '{resource_name}' to '{LOCAL_NLTK_DATA_PATH}'...")
            nltk.download(resource_name, download_dir=LOCAL_NLTK_DATA_PATH)


# --- Initialize NLP resources once for performance ---
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def clean_and_prepare_text(text):
    """
    Cleans and preprocesses a single text entry.
    Uses BertTokenizer instead of NLTK's word_tokenize.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 2. Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Convert to lowercase
    text = text.lower()

    # 4. Tokenize using the much more reliable BertTokenizer
    tokens = TOKENIZER.tokenize(text)

    # 5. Remove stopwords and 6. Lemmatize
    cleaned_tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOP_WORDS and len(word) > 2]

    return " ".join(cleaned_tokens)


def main(input_dir, output_dir):
    """
    Main function to execute the data processing pipeline for the WELFake dataset.
    """
    logging.info("--- Starting data processing pipeline for WELFake dataset ---")

    # Download NLTK resources (stopwords, wordnet)
    download_nltk_resources()

    # 1. Define file paths
    input_filepath = os.path.join(input_dir, 'WELFake_Dataset.csv')
    if not os.path.exists(input_filepath):
        logging.error(f"Input file not found at: {input_filepath}")
        return

    # 2. Load Data
    logging.info(f"Loading raw data from: {input_filepath}")
    df = pd.read_csv(input_filepath)

    # 3. Initial Cleaning & Feature Engineering
    logging.info("Performing initial cleaning and feature engineering...")
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + " " + df['text']
    logging.info("Dataset labels: 0 = Fake, 1 = Real")

    # 4. Apply Advanced Text Cleaning
    logging.info("Applying advanced text cleaning to the 'content' column. This may take a while...")
    df['cleaned_content'] = df['content'].apply(clean_and_prepare_text)

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
    parser = argparse.ArgumentParser(
        description="Process the WELFake news dataset into training, validation, and test sets.")

    default_input_path = os.path.join(project_root, 'data', 'raw')
    default_output_path = os.path.join(project_root, 'data', 'processed')

    parser.add_argument('--input_dir', type=str, default=default_input_path,
                        help="Directory where the raw 'WELFake_Dataset.csv' file is located.")
    parser.add_argument('--output_dir', type=str, default=default_output_path,
                        help="Directory where the final 'train.csv', 'val.csv', and 'test.csv' files will be saved.")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
