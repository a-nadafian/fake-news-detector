# src/data/make_dataset.py

import os
import pandas as pd
import argparse
from sklearn.utils import shuffle


def load_and_label(file_path, label):
    """
    Loads a CSV file and adds a label column.

    Args:
        file_path (str): The path to the CSV file.
        label (int): The label to assign to the data (e.g., 1 for fake, 0 for real).

    Returns:
        pandas.DataFrame: The loaded and labeled DataFrame, or None if the file is not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    df = pd.read_csv(file_path)
    df['label'] = label
    return df


def main(input_dir, output_dir):
    """
    Main function to execute the data processing pipeline.
    Loads raw data, combines it, shuffles it, and saves the processed file.

    Args:
        input_dir (str): Directory containing the raw data files ('True.csv', 'Fake.csv').
        output_dir (str): Directory where the processed data file will be saved.
    """
    print("--- Starting data processing pipeline ---")

    # 1. Define file paths
    true_file_path = os.path.join(input_dir, 'True.csv')
    fake_file_path = os.path.join(input_dir, 'Fake.csv')

    # 2. Load and label data
    print(f"Loading real news from: {true_file_path}")
    df_true = load_and_label(true_file_path, 0)  # 0 for real news

    print(f"Loading fake news from: {fake_file_path}")
    df_fake = load_and_label(fake_file_path, 1)  # 1 for fake news

    if df_true is None or df_fake is None:
        print("Halting execution due to missing file(s).")
        return

    # 3. Combine and shuffle
    print("Combining and shuffling datasets...")
    combined_df = pd.concat([df_true, df_fake], ignore_index=True)
    shuffled_df = shuffle(combined_df, random_state=42).reset_index(drop=True)

    # 4. Save processed data
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, 'processed_news.csv')
    print(f"Saving processed data to: {output_filepath}")
    shuffled_df.to_csv(output_filepath, index=False)

    print("--- Data processing complete! ✅ ---")


if __name__ == '__main__':
    # This block allows the script to be run from the command line.
    parser = argparse.ArgumentParser(description="Process raw news data into a single labeled CSV file.")

    # Make paths relative to the project root, not the current working directory.
    # This makes the script runnable from anywhere.

    # Get the directory of the current script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get the project root (from src/data -> src -> root).
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Define default paths relative to the calculated project root.
    default_input_path = os.path.join(project_root, 'data', 'raw')
    default_output_path = os.path.join(project_root, 'data', 'processed')

    parser.add_argument(
        '--input_dir',
        type=str,
        default=default_input_path,
        help='Directory where the raw True.csv and Fake.csv files are located.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=default_output_path,
        help='Directory where the final processed CSV file will be saved.'
    )

    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.input_dir, args.output_dir)
