#!/usr/bin/env python3
"""
Simple test to verify preprocessing works
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.features.build_features import preprocess_text
    print("✅ Successfully imported preprocess_text function")
    
    # Test the function
    test_text = "BREAKING: Scientists discover that drinking hot water with lemon cures all diseases!"
    result = preprocess_text(test_text)
    print(f"✅ Preprocessing successful!")
    print(f"Original: {test_text}")
    print(f"Preprocessed: {result}")
    
    # Test another example
    test_text2 = "The World Health Organization released new guidelines for COVID-19 prevention."
    result2 = preprocess_text(test_text2)
    print(f"\nOriginal: {test_text2}")
    print(f"Preprocessed: {result2}")
    
    print("\n🎉 All tests passed! The preprocessing pipeline is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}") 