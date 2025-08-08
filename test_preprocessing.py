#!/usr/bin/env python3
"""
Test script to verify the preprocessing pipeline works correctly
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.features.build_features import preprocess_text

def test_preprocessing():
    """Test the preprocessing function with various inputs"""
    
    test_cases = [
        {
            "input": "BREAKING: Scientists discover that drinking hot water with lemon cures all diseases instantly!",
            "expected_contains": ["scientist", "discover", "drink", "hot", "water", "lemon", "cure", "disease"]
        },
        {
            "input": "The World Health Organization released new guidelines for COVID-19 prevention measures.",
            "expected_contains": ["world", "health", "organization", "release", "guideline", "prevention", "measure"]
        },
        {
            "input": "SHOCKING: 5G towers are actually mind control devices!",
            "expected_contains": ["tower", "mind", "control", "device"]
        },
        {
            "input": "NASA's Perseverance rover successfully landed on Mars.",
            "expected_contains": ["nasa", "perseverance", "rover", "successfully", "land", "mars"]
        }
    ]
    
    print("Testing preprocessing pipeline...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input: {test_case['input']}")
        
        # Apply preprocessing
        preprocessed = preprocess_text(test_case['input'])
        print(f"Preprocessed: {preprocessed}")
        
        # Check if expected words are present
        missing_words = []
        for word in test_case['expected_contains']:
            if word not in preprocessed:
                missing_words.append(word)
        
        if missing_words:
            print(f"❌ Missing expected words: {missing_words}")
        else:
            print("✅ All expected words found")
        
        print("-" * 30)
    
    print("\nPreprocessing test completed!")

if __name__ == "__main__":
    test_preprocessing() 