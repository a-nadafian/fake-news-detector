#!/usr/bin/env python3
"""
Enhanced Advanced Data Preprocessing Pipeline for Fake News Detection
Includes additional techniques: TF-IDF features, sentiment analysis, 
readability scores, and advanced augmentation
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class EnhancedFakeNewsPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Enhanced custom stop words for news articles
        self.custom_stops = {
            'said', 'says', 'according', 'reported', 'news', 'story', 'article',
            'read', 'click', 'share', 'comment', 'like', 'follow', 'subscribe',
            'breaking', 'exclusive', 'update', 'latest', 'developing', 'just',
            'now', 'today', 'yesterday', 'tomorrow', 'week', 'month', 'year',
            'time', 'times', 'daily', 'weekly', 'monthly', 'yearly', 'annual',
            'source', 'sources', 'official', 'officials', 'government', 'gov',
            'president', 'minister', 'spokesperson', 'spokesman', 'spokeswoman'
        }
        self.stop_words.update(self.custom_stops)

        # Fake news indicators
        self.fake_indicators = {
            'shocking', 'amazing', 'incredible', 'unbelievable', 'mind-blowing',
            'you_wont_believe', 'doctors_hate', 'secret_trick', 'miracle_cure',
            'conspiracy', 'cover_up', 'hidden_truth', 'exposed', 'revealed',
            'breaking_now', 'just_in', 'urgent', 'warning', 'alert', 'scam'
        }

        # Real news indicators
        self.real_indicators = {
            'study', 'research', 'analysis', 'report', 'survey', 'poll',
            'official', 'announcement', 'statement', 'press_release',
            'interview', 'expert', 'professor', 'scientist', 'doctor',
            'government', 'federal', 'state', 'local', 'authority'
        }

    def clean_text_enhanced(self, text):
        """
        Enhanced text cleaning with additional techniques
        """
        if pd.isna(text) or text == '':
            return ''

        text = str(text)

        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)

        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)

        # Remove social media elements
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'RT\s+@\w+', '', text)
        text = re.sub(r'via\s+@\w+', '', text)

        # Remove special characters but preserve sentence structure
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def extract_advanced_features(self, text):
        """
        Extract comprehensive text features including sentiment and readability
        """
        if not text:
            return self._empty_features()

        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        # Use simple sentence splitting to avoid NLTK issues
        sentence_count = len([s.strip() for s in text.split('.') if s.strip()])

        # Word analysis
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_words = len(set(words))
        word_diversity = unique_words / word_count if word_count > 0 else 0

        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        quote_count = text.count('"') + text.count("'")

        # Character analysis
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0

        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / char_count if char_count > 0 else 0

        punctuation_count = sum(1 for c in text if c in string.punctuation)
        punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0

        # Sentiment analysis
        try:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            sentiment_compound = sentiment_scores['compound']
            sentiment_positive = sentiment_scores['pos']
            sentiment_negative = sentiment_scores['neg']
            sentiment_neutral = sentiment_scores['neu']
        except:
            sentiment_compound = sentiment_positive = sentiment_negative = sentiment_neutral = 0

        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
        except:
            textblob_polarity = textblob_subjectivity = 0

        # Readability features
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Fake news indicator features
        fake_indicator_count = sum(1 for indicator in self.fake_indicators if indicator in text.lower())
        real_indicator_count = sum(1 for indicator in self.real_indicators if indicator in text.lower())

        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'unique_words': unique_words,
            'word_diversity': word_diversity,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'quote_count': quote_count,
            'uppercase_ratio': uppercase_ratio,
            'digit_ratio': digit_ratio,
            'punctuation_ratio': punctuation_ratio,
            'sentiment_compound': sentiment_compound,
            'sentiment_positive': sentiment_positive,
            'sentiment_negative': sentiment_negative,
            'sentiment_neutral': sentiment_neutral,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'avg_sentence_length': avg_sentence_length,
            'fake_indicator_count': fake_indicator_count,
            'real_indicator_count': real_indicator_count
        }

    def _empty_features(self):
        """Return empty feature set"""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'unique_words': 0, 'word_diversity': 0,
            'exclamation_count': 0, 'question_count': 0, 'quote_count': 0,
            'uppercase_ratio': 0, 'digit_ratio': 0, 'punctuation_ratio': 0,
            'sentiment_compound': 0, 'sentiment_positive': 0, 'sentiment_negative': 0,
            'sentiment_neutral': 0, 'textblob_polarity': 0, 'textblob_subjectivity': 0,
            'avg_sentence_length': 0, 'fake_indicator_count': 0, 'real_indicator_count': 0
        }

    def extract_tfidf_features(self, texts, max_features=1000):
        """
        Extract TF-IDF features from text corpus
        """
        print("Extracting TF-IDF features...")

        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7,
            stop_words='english'
        )

        # Fit and transform
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Reduce dimensionality using SVD
        svd = TruncatedSVD(n_components=100, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)

        # Create feature names
        feature_names = [f'tfidf_{i}' for i in range(tfidf_reduced.shape[1])]

        return pd.DataFrame(tfidf_reduced, columns=feature_names)

    def advanced_tokenization(self, text):
        """
        Advanced tokenization with POS tagging and named entity recognition
        """
        if not text:
            return []

        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())

        # Remove stop words and short tokens
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Lemmatize
                lemmatized = self.lemmatizer.lemmatize(token)
                # Stem
                stemmed = self.stemmer.stem(lemmatized)
                processed_tokens.append(stemmed)

        return processed_tokens

    def advanced_augmentation(self, text, augmentation_factor=3):
        """
        Advanced text augmentation using multiple techniques
        """
        if not text or augmentation_factor <= 1:
            return [text]

        augmented_texts = [text]

        # Enhanced synonym replacement
        enhanced_synonyms = {
            'said': ['stated', 'reported', 'announced', 'declared', 'mentioned', 'noted'],
            'big': ['large', 'huge', 'enormous', 'massive', 'substantial', 'considerable'],
            'good': ['great', 'excellent', 'wonderful', 'fantastic', 'outstanding', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'atrocious', 'appalling'],
            'important': ['crucial', 'essential', 'vital', 'critical', 'key', 'significant'],
            'new': ['recent', 'fresh', 'modern', 'contemporary', 'current', 'latest'],
            'show': ['demonstrate', 'reveal', 'indicate', 'suggest', 'prove', 'establish'],
            'find': ['discover', 'uncover', 'reveal', 'identify', 'locate', 'detect'],
            'think': ['believe', 'consider', 'suppose', 'assume', 'presume', 'conclude'],
            'know': ['understand', 'realize', 'recognize', 'comprehend', 'grasp', 'perceive']
        }

        # Back-translation simulation (word replacement)
        for _ in range(augmentation_factor - 1):
            augmented_text = text

            # Apply synonym replacement
            for word, syns in enhanced_synonyms.items():
                if word in augmented_text.lower():
                    replacement = np.random.choice(syns)
                    augmented_text = re.sub(r'\b' + word + r'\b', replacement, augmented_text, flags=re.IGNORECASE)

            # Random word insertion (add common words)
            if np.random.random() < 0.3:
                common_words = ['also', 'furthermore', 'moreover', 'additionally', 'besides']
                words = augmented_text.split()
                if len(words) > 5:
                    insert_pos = np.random.randint(1, len(words))
                    insert_word = np.random.choice(common_words)
                    words.insert(insert_pos, insert_word)
                    augmented_text = ' '.join(words)

            # Random word deletion (remove some words)
            if np.random.random() < 0.2:
                words = augmented_text.split()
                if len(words) > 10:
                    delete_pos = np.random.randint(0, len(words))
                    if delete_pos < len(words):
                        words.pop(delete_pos)
                        augmented_text = ' '.join(words)

            if augmented_text != text:
                augmented_texts.append(augmented_text)

        return augmented_texts[:augmentation_factor]


def load_datasets_enhanced():
    """
    Enhanced dataset loading with better error handling and validation
    """
    print("Loading and combining datasets with enhanced processing...")

    combined_data = []

    # Load BuzzFeed datasets
    try:
        print("Loading BuzzFeed datasets...")
        buzzfeed_fake = pd.read_csv('data/raw/BuzzFeed_fake_news_content.csv')
        buzzfeed_real = pd.read_csv('data/raw/BuzzFeed_real_news_content.csv')

        # Validate and clean
        for df, name in [(buzzfeed_fake, 'BuzzFeed Fake'), (buzzfeed_real, 'BuzzFeed Real')]:
            if 'title' in df.columns and 'text' in df.columns:
                df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
                df['label'] = 1 if 'real' in name.lower() else 0
                df['source_dataset'] = 'BuzzFeed'
                combined_data.append(df)
                print(f"{name}: {len(df)} articles")
            else:
                print(f"Warning: {name} missing required columns")

    except Exception as e:
        print(f"Error loading BuzzFeed datasets: {e}")

    # Load PolitiFact datasets
    try:
        print("Loading PolitiFact datasets...")
        politifact_fake = pd.read_csv('data/raw/PolitiFact_fake_news_content.csv')
        politifact_real = pd.read_csv('data/raw/PolitiFact_real_news_content.csv')

        for df, name in [(politifact_fake, 'PolitiFact Fake'), (politifact_real, 'PolitiFact Real')]:
            if 'title' in df.columns and 'text' in df.columns:
                df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
                df['label'] = 1 if 'real' in name.lower() else 0
                df['source_dataset'] = 'PolitiFact'
                combined_data.append(df)
                print(f"{name}: {len(df)} articles")
            else:
                print(f"Warning: {name} missing required columns")

    except Exception as e:
        print(f"Error loading PolitiFact datasets: {e}")

    # Load LIAR dataset
    try:
        print("Loading LIAR dataset...")
        train_data = pd.read_csv('data/raw/train.tsv', sep='\t', header=None,
                                 names=['id', 'label', 'statement', 'subject', 'speaker',
                                        'job_title', 'state', 'party', 'barely_true', 'false',
                                        'half_true', 'mostly_true', 'pants_on_fire', 'context'])

        test_data = pd.read_csv('data/raw/test.tsv', sep='\t', header=None,
                                names=['id', 'label', 'statement', 'subject', 'speaker',
                                       'job_title', 'state', 'party', 'barely_true', 'false',
                                       'half_true', 'mostly_true', 'pants_on_fire', 'context'])

        valid_data = pd.read_csv('data/raw/valid.tsv', sep='\t', header=None,
                                 names=['id', 'label', 'statement', 'subject', 'speaker',
                                        'job_title', 'state', 'party', 'barely_true', 'false',
                                        'half_true', 'mostly_true', 'pants_on_fire', 'context'])

        liar_data = pd.concat([train_data, test_data, valid_data], ignore_index=True)

        # Convert labels to binary
        fake_labels = ['pants-on-fire', 'false', 'barely-true']
        liar_data['label'] = liar_data['label'].apply(lambda x: 0 if x in fake_labels else 1)
        liar_data['text'] = liar_data['statement']
        liar_data['source_dataset'] = 'LIAR'

        combined_data.append(liar_data)
        print(f"LIAR: {len(liar_data)} total articles")

    except Exception as e:
        print(f"Error loading LIAR dataset: {e}")

    if combined_data:
        final_dataset = pd.concat(combined_data, ignore_index=True)
        print(f"\nTotal combined dataset: {len(final_dataset)} articles")
        print(f"Label distribution: {final_dataset['label'].value_counts()}")
        return final_dataset
    else:
        raise ValueError("No datasets could be loaded")


def apply_enhanced_preprocessing(dataset):
    """
    Apply enhanced preprocessing techniques
    """
    print("\nApplying enhanced preprocessing...")

    preprocessor = EnhancedFakeNewsPreprocessor()

    # 1. Enhanced text cleaning
    print("1. Enhanced text cleaning...")
    dataset['cleaned_text'] = dataset['text'].apply(preprocessor.clean_text_enhanced)

    # 2. Extract comprehensive features
    print("2. Extracting comprehensive features...")
    text_features = dataset['cleaned_text'].apply(preprocessor.extract_advanced_features)
    feature_df = pd.DataFrame(text_features.tolist())

    # Add features to dataset
    for col in feature_df.columns:
        dataset[f'feature_{col}'] = feature_df[col]

    # 3. Advanced tokenization
    print("3. Advanced tokenization...")
    dataset['processed_tokens'] = dataset['cleaned_text'].apply(preprocessor.advanced_tokenization)
    dataset['processed_text'] = dataset['processed_tokens'].apply(lambda x: ' '.join(x))

    # 4. Extract TF-IDF features
    print("4. Extracting TF-IDF features...")
    tfidf_features = preprocessor.extract_tfidf_features(dataset['cleaned_text'].fillna(''))

    # Combine TF-IDF features
    for col in tfidf_features.columns:
        dataset[col] = tfidf_features[col]

    # 5. Remove low-quality samples (lowered threshold)
    initial_count = len(dataset)
    dataset = dataset[dataset['cleaned_text'].str.len() > 20]  # Minimum 20 characters
    print(f"Removed {initial_count - len(dataset)} rows with insufficient text")

    return dataset


def enhanced_balancing(dataset):
    """
    Enhanced dataset balancing with stratification
    """
    print("\nApplying enhanced balancing...")

    label_counts = dataset['label'].value_counts()
    print(f"Current label distribution: {label_counts}")

    # Separate classes
    fake_articles = dataset[dataset['label'] == 0]
    real_articles = dataset[dataset['label'] == 1]

    # Use the larger class as target
    target_size = max(len(fake_articles), len(real_articles))

    # Upsample minority class with stratification
    if len(fake_articles) < target_size:
        fake_articles = resample(fake_articles,
                                 replace=True,
                                 n_samples=target_size,
                                 random_state=42)
        print(f"Upsampled fake articles to {len(fake_articles)}")

    if len(real_articles) < target_size:
        real_articles = resample(real_articles,
                                 replace=True,
                                 n_samples=target_size,
                                 random_state=42)
        print(f"Upsampled real articles to {len(real_articles)}")

    # Combine and shuffle
    balanced_dataset = pd.concat([fake_articles, real_articles], ignore_index=True)
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Balanced dataset size: {len(balanced_dataset)}")
    print(f"Final label distribution: {balanced_dataset['label'].value_counts()}")

    return balanced_dataset


def enhanced_augmentation(dataset, augmentation_factor=1.5):
    """
    Enhanced data augmentation
    """
    print(f"\nApplying enhanced data augmentation (factor: {augmentation_factor})...")

    preprocessor = EnhancedFakeNewsPreprocessor()

    # Identify minority class
    label_counts = dataset['label'].value_counts()
    minority_class = label_counts.idxmin()
    minority_articles = dataset[dataset['label'] == minority_class]

    # Calculate augmentation needs
    target_count = int(len(dataset[dataset['label'] != minority_class]) * augmentation_factor)
    samples_needed = target_count - len(minority_articles)

    if samples_needed > 0:
        augmented_samples = []

        for idx, row in minority_articles.iterrows():
            if len(augmented_samples) >= samples_needed:
                break

            augmented_texts = preprocessor.advanced_augmentation(row['cleaned_text'], 3)

            for aug_text in augmented_texts[1:]:
                if len(augmented_samples) >= samples_needed:
                    break

                # Create augmented row
                aug_row = row.copy()
                aug_row['cleaned_text'] = aug_text
                aug_row['text'] = aug_text
                aug_row['processed_text'] = ' '.join(preprocessor.advanced_tokenization(aug_text))
                aug_row['processed_tokens'] = preprocessor.advanced_tokenization(aug_text)
                aug_row['source_dataset'] = f"{row['source_dataset']}_augmented"

                # Recalculate features for augmented text
                aug_features = preprocessor.extract_advanced_features(aug_text)
                for feature_name, feature_value in aug_features.items():
                    aug_row[f'feature_{feature_name}'] = feature_value

                augmented_samples.append(aug_row)

        # Add augmented samples
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            dataset = pd.concat([dataset, augmented_df], ignore_index=True)
            print(f"Added {len(augmented_samples)} enhanced augmented samples")

    return dataset


def create_enhanced_final_dataset(dataset):
    """
    Create enhanced final dataset
    """
    print("\nCreating enhanced final dataset...")

    # Select all feature columns
    feature_columns = [col for col in dataset.columns if col.startswith('feature_') or col.startswith('tfidf_')]

    final_columns = ['label', 'cleaned_text', 'processed_text', 'source_dataset'] + feature_columns

    # Filter available columns
    available_columns = [col for col in final_columns if col in dataset.columns]
    final_dataset = dataset[available_columns].copy()

    # Rename for clarity
    column_mapping = {
        'cleaned_text': 'text',
        'processed_text': 'processed_text'
    }

    final_dataset = final_dataset.rename(columns=column_mapping)

    # Quality filtering (lowered threshold)
    final_dataset = final_dataset[final_dataset['text'].str.len() >= 20]  # Minimum 20 characters

    print(f"Enhanced final dataset shape: {final_dataset.shape}")
    print(
        f"Feature columns: {len([col for col in final_dataset.columns if col.startswith('feature_') or col.startswith('tfidf_')])}")

    return final_dataset


def main():
    """
    Enhanced main preprocessing pipeline
    """
    print("=== Enhanced Advanced Fake News Dataset Preprocessing Pipeline ===\n")

    try:
        # 1. Load and combine datasets
        dataset = load_datasets_enhanced()

        # 2. Apply enhanced preprocessing
        dataset = apply_enhanced_preprocessing(dataset)

        # 3. Enhanced balancing
        dataset = enhanced_balancing(dataset)

        # 4. Enhanced augmentation
        dataset = enhanced_augmentation(dataset, augmentation_factor=1.3)

        # 5. Create enhanced final dataset
        final_dataset = create_enhanced_final_dataset(dataset)

        # 6. Save datasets
        output_dir = 'data/processed'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, 'enhanced_preprocessed_fake_news_dataset.csv')
        final_dataset.to_csv(output_file, index=False)
        print(f"\n✅ Enhanced preprocessed dataset saved to: {output_file}")

        # 7. Create train/validation/test split (70% train, 15% validation, 15% test)
        train_data, temp_data = train_test_split(
            final_dataset,
            test_size=0.3,
            random_state=42,
            stratify=final_dataset['label']
        )

        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,  # 50% of 30% is 15%
            random_state=42,
            stratify=temp_data['label']
        )

        train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_data.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

        print(f"✅ Training set saved to data/processed/train.csv: {len(train_data)} samples")
        print(f"✅ Validation set saved to data/processed/val.csv: {len(val_data)} samples")
        print(f"✅ Test set saved to data/processed/test.csv: {len(test_data)} samples")

        # 8. Enhanced statistics
        print("\n=== Enhanced Dataset Statistics ===")
        print(f"Total samples: {len(final_dataset)}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        print(f"Label distribution: {final_dataset['label'].value_counts().to_dict()}")
        print(
            f"Total features: {len([col for col in final_dataset.columns if col.startswith('feature_') or col.startswith('tfidf_')])}")
        print(f"Average text length: {final_dataset['text'].str.len().mean():.1f} characters")

        return final_dataset

    except Exception as e:
        print(f"❌ Error in enhanced preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
