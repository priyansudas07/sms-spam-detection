"""
Data Loader Module for SMS Spam Detection.

This module handles:
1. Downloading the official UCI SMS Spam Collection dataset.
2. Loading the raw tab-delimited text data into a pandas DataFrame.
3. Inspecting dataset properties (samples, missing values, duplicates, class distribution).
4. Cleaning data by handling duplicates and standardizing labels.
"""

import os
import zipfile
import urllib.request
import pandas as pd


UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_uci_dataset(dest_dir="data"):
    """
    Downloads and extracts the official UCI SMS Spam Collection dataset
    if it does not already exist in the destination directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    raw_file = os.path.join(dest_dir, "SMSSpamCollection")
    zip_path = os.path.join(dest_dir, "smsspamcollection.zip")

    if os.path.exists(raw_file):
        print(f"[INFO] Dataset already exists at: {raw_file}")
        return raw_file

    print(f"[INFO] Downloading UCI SMS Spam Collection dataset from {UCI_DATASET_URL}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(UCI_DATASET_URL, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())

    print("[INFO] Extracting dataset archive...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    # Clean up zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)

    if os.path.exists(raw_file):
        print(f"[INFO] Dataset successfully extracted to: {raw_file}")
        return raw_file
    else:
        raise FileNotFoundError(f"Expected file 'SMSSpamCollection' not found in extracted archive.")


def load_raw_data(filepath="data/SMSSpamCollection"):
    """
    Loads the raw SMS Spam Collection dataset.
    The file is tab-separated with two columns:
    - label: 'ham' or 'spam'
    - message: SMS text content
    """
    if not os.path.exists(filepath):
        filepath = download_uci_dataset(os.path.dirname(filepath) or "data")

    # Read tab-delimited file, handling any multi-line or encoding nuances
    df = pd.read_csv(
        filepath,
        sep='\t',
        names=['label', 'message'],
        header=None,
        encoding='utf-8',
        quoting=3  # csv.QUOTE_NONE to prevent quote issues in SMS text
    )
    return df


def inspect_data(df):
    """
    Computes and returns summary inspection metrics for the dataset:
    - total samples
    - missing values per column
    - duplicate message count
    - class counts and proportions
    """
    total_samples = len(df)
    missing_counts = df.isnull().sum().to_dict()
    duplicate_count = int(df.duplicated(subset=['message']).sum())
    class_counts = df['label'].value_counts().to_dict()
    class_proportions = (df['label'].value_counts(normalize=True) * 100).round(2).to_dict()

    summary = {
        "total_samples": total_samples,
        "missing_counts": missing_counts,
        "duplicate_count": duplicate_count,
        "class_counts": class_counts,
        "class_proportions": class_proportions
    }
    return summary


def clean_data(df):
    """
    Cleans the raw SMS Spam dataset:
    1. Removes duplicate messages to prevent data leakage and bias.
    2. Drops any potential null/empty entries.
    3. Strips whitespace from labels and messages.
    4. Adds binary numeric label ('label_num': 0 for ham, 1 for spam).
    """
    df_clean = df.copy()

    # Drop nulls if any
    df_clean = df_clean.dropna(subset=['label', 'message'])

    # Strip whitespace
    df_clean['label'] = df_clean['label'].astype(str).str.strip().str.lower()
    df_clean['message'] = df_clean['message'].astype(str).str.strip()

    # Drop duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['message']).reset_index(drop=True)
    duplicates_dropped = initial_count - len(df_clean)

    # Encode binary target: ham -> 0, spam -> 1
    df_clean['label_num'] = df_clean['label'].map({'ham': 0, 'spam': 1})

    return df_clean, duplicates_dropped


if __name__ == "__main__":
    raw_df = load_raw_data()
    print("--- Top 10 Messages ---")
    print(raw_df.head(10))
    print("\n--- Dataset Inspection Summary ---")
    print(inspect_data(raw_df))
    clean_df, dropped = clean_data(raw_df)
    print(f"\n[INFO] Cleaned dataset shape: {clean_df.shape} (Dropped {dropped} duplicates)")
    print("--- Cleaned Class Distribution ---")
    print(clean_df['label'].value_counts())
