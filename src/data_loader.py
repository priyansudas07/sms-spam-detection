"""
Data Loader Module for SMS Spam Detection.

This module handles:
1. Downloading the official UCI SMS Spam Collection dataset.
2. Loading raw tab-delimited text data into a pandas DataFrame.
3. Inspecting dataset properties (samples, missing values, duplicates, class distribution).
4. Verifying duplicate messages for conflicting labels.
5. Cleaning data by handling duplicates, stripping whitespace, and standardizing binary labels.
"""

import os
import zipfile
import urllib.request
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DATA_FILE = os.path.join(DEFAULT_DATA_DIR, "SMSSpamCollection")
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_uci_dataset(dest_dir=None):
    """
    Downloads and extracts the official UCI SMS Spam Collection dataset
    if it does not already exist in the destination directory.
    """
    if dest_dir is None:
        dest_dir = DEFAULT_DATA_DIR
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

    if os.path.exists(zip_path):
        os.remove(zip_path)

    if os.path.exists(raw_file):
        print(f"[INFO] Dataset successfully extracted to: {raw_file}")
        return raw_file
    else:
        raise FileNotFoundError("Expected file 'SMSSpamCollection' not found in extracted archive.")


def load_raw_data(filepath=None):
    """
    Loads the raw SMS Spam Collection dataset.
    The file is tab-separated with two columns:
    - label: 'ham' or 'spam'
    - message: SMS text content
    """
    if filepath is None:
        if os.path.exists(DEFAULT_DATA_FILE):
            filepath = DEFAULT_DATA_FILE
        elif os.path.exists("data/SMSSpamCollection"):
            filepath = "data/SMSSpamCollection"
        else:
            filepath = DEFAULT_DATA_FILE

    if not os.path.exists(filepath):
        filepath = download_uci_dataset(os.path.dirname(filepath) or DEFAULT_DATA_DIR)

    df = pd.read_csv(
        filepath,
        sep='\t',
        names=['label', 'message'],
        header=None,
        encoding='utf-8',
        quoting=3  # csv.QUOTE_NONE to prevent quote issues in SMS text
    )
    return df


def check_conflicting_labels(df, text_col='message', label_col='label'):
    """
    Audits the dataset for duplicate messages that carry conflicting classification labels
    (e.g., identical message text labeled as both 'ham' and 'spam').
    
    Returns:
    - num_conflicts (int): Count of messages with ambiguous/conflicting labels.
    - conflict_df (pd.DataFrame): DataFrame detailing the conflicting messages if any.
    """
    grouped = df.groupby(text_col)[label_col].nunique()
    conflicting_texts = grouped[grouped > 1].index.tolist()
    
    if conflicting_texts:
        conflict_df = df[df[text_col].isin(conflicting_texts)].sort_values(by=text_col)
        return len(conflicting_texts), conflict_df
    return 0, pd.DataFrame()


def inspect_data(df):
    """
    Computes summary inspection metrics for the raw dataset:
    - total samples
    - missing values per column
    - duplicate message count (exact and after whitespace normalization)
    - conflicting label occurrences
    - class counts and natural proportions
    """
    total_samples = len(df)
    missing_counts = df.isnull().sum().to_dict()
    exact_duplicates = int(df.duplicated(subset=['message']).sum())
    
    # Strip whitespace to check normalized duplicates
    msg_stripped = df['message'].astype(str).str.strip()
    stripped_duplicates = int(msg_stripped.duplicated().sum())

    conflicts_count, _ = check_conflicting_labels(df)
    class_counts = df['label'].value_counts().to_dict()
    class_proportions = (df['label'].value_counts(normalize=True) * 100).round(2).to_dict()

    summary = {
        "total_samples": total_samples,
        "missing_counts": missing_counts,
        "exact_duplicates": exact_duplicates,
        "whitespace_stripped_duplicates": stripped_duplicates,
        "conflicting_label_messages": conflicts_count,
        "class_counts": class_counts,
        "class_proportions": class_proportions
    }
    return summary


def clean_data(df):
    """
    Cleans the raw SMS Spam dataset under strict methodology:
    1. Validates and drops null/empty records if present.
    2. Strips leading/trailing whitespace from labels and messages.
    3. Audits and asserts zero conflicting labels.
    4. Removes duplicate messages (keeping first occurrence) to prevent data leakage.
    5. Preserves the authentic natural class distribution (~87.6% Ham : ~12.4% Spam).
    6. Encodes binary numeric target ('label_num': 0 for ham, 1 for spam).
    
    Returns:
    - df_clean (pd.DataFrame): Cleaned unique DataFrame.
    - cleaning_report (dict): Comprehensive audit report of cleaning operations.
    """
    df_clean = df.copy()

    initial_count = len(df_clean)

    # 1. Drop nulls if any
    missing_count = int(df_clean.isnull().sum().sum())
    df_clean = df_clean.dropna(subset=['label', 'message'])

    # 2. Strip whitespace and standardize casing
    df_clean['label'] = df_clean['label'].astype(str).str.strip().str.lower()
    df_clean['message'] = df_clean['message'].astype(str).str.strip()

    # 3. Check for conflicting labels
    conflicts_count, conflict_records = check_conflicting_labels(df_clean)
    if conflicts_count > 0:
        print(f"[WARNING] Detected {conflicts_count} messages with conflicting labels! Resolving by removing ambiguous records.")
        ambiguous_messages = conflict_records['message'].unique()
        df_clean = df_clean[~df_clean['message'].isin(ambiguous_messages)].reset_index(drop=True)

    # 4. Drop duplicates
    count_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['message']).reset_index(drop=True)
    duplicates_dropped = count_before_dedup - len(df_clean)

    # 5. Encode binary target: ham -> 0, spam -> 1
    df_clean['label_num'] = df_clean['label'].map({'ham': 0, 'spam': 1})

    # Validate mapping
    if df_clean['label_num'].isnull().any():
        unknown_labels = df_clean[df_clean['label_num'].isnull()]['label'].unique()
        raise ValueError(f"Unknown labels found in dataset: {unknown_labels}")

    ham_count = int((df_clean['label_num'] == 0).sum())
    spam_count = int((df_clean['label_num'] == 1).sum())
    total_clean = len(df_clean)

    cleaning_report = {
        "raw_samples": initial_count,
        "missing_dropped": missing_count,
        "conflicting_messages_found": conflicts_count,
        "duplicates_dropped": duplicates_dropped,
        "final_unique_samples": total_clean,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_proportion_pct": round((ham_count / total_clean) * 100, 2),
        "spam_proportion_pct": round((spam_count / total_clean) * 100, 2)
    }

    return df_clean, cleaning_report


if __name__ == "__main__":
    raw_df = load_raw_data()
    print("=== DATASET INSPECTION ===")
    inspection = inspect_data(raw_df)
    for k, v in inspection.items():
        print(f"  {k}: {v}")

    clean_df, report = clean_data(raw_df)
    print("\n=== CLEANING REPORT ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
