# src/features/build_features.py

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
import logging

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
def _initialize_nlp_resources():
    """Initialize NLP resources (tokenizer, lemmatizer, stopwords)"""
    download_nltk_resources()
    return (
        BertTokenizer.from_pretrained('bert-base-uncased'),
        WordNetLemmatizer(),
        set(stopwords.words('english'))
    )


# Global variables for NLP resources
TOKENIZER, LEMMATIZER, STOP_WORDS = _initialize_nlp_resources()


def preprocess_text(text):
    """
    Cleans and preprocesses a single text entry.
    Uses BertTokenizer instead of NLTK's word_tokenize.
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Preprocessed text
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


def preprocess_batch(texts):
    """
    Preprocess a batch of texts efficiently.
    
    Args:
        texts (list): List of raw text strings
        
    Returns:
        list: List of preprocessed text strings
    """
    return [preprocess_text(text) for text in texts] 