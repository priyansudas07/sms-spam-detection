"""
Exploratory Data Analysis (EDA) Script for SMS Spam Detection.

Generates and saves visual plots for:
1. Raw vs. Cleaned class distributions (Ham vs. Spam).
2. Message length distribution comparison between Ham and Spam.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_raw_data, clean_data, inspect_data


def generate_eda_figures(output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    raw_df = load_raw_data()
    clean_df, duplicates_dropped = clean_data(raw_df)

    # Set aesthetic style
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --- Plot 1: Class Distribution (Raw vs Cleaned) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    raw_counts = raw_df['label'].value_counts()
    sns.barplot(
        x=raw_counts.index,
        y=raw_counts.values,
        palette=['#2b5c8f', '#d9534f'],
        hue=raw_counts.index,
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title(f"Raw Dataset Distribution\n(Total: {len(raw_df):,} messages)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Class", fontweight='semibold')
    axes[0].set_ylabel("Count", fontweight='semibold')
    for p in axes[0].patches:
        height = p.get_height()
        pct = (height / len(raw_df)) * 100
        axes[0].annotate(f"{int(height):,}\n({pct:.1f}%)",
                         (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    # Cleaned counts
    clean_counts = clean_df['label'].value_counts()
    sns.barplot(
        x=clean_counts.index,
        y=clean_counts.values,
        palette=['#2b5c8f', '#d9534f'],
        hue=clean_counts.index,
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title(f"Cleaned Dataset Distribution (Duplicates Dropped: {duplicates_dropped})\n(Total: {len(clean_df):,} messages)",
                      fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Class", fontweight='semibold')
    axes[1].set_ylabel("Count", fontweight='semibold')
    for p in axes[1].patches:
        height = p.get_height()
        pct = (height / len(clean_df)) * 100
        axes[1].annotate(f"{int(height):,}\n({pct:.1f}%)",
                         (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Class distribution plot saved to: {dist_plot_path}")

    # --- Plot 2: Message Length Distribution ---
    clean_df['char_length'] = clean_df['message'].apply(len)
    clean_df['word_count'] = clean_df['message'].apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data=clean_df, x='char_length', hue='label', bins=50, kde=True,
                 palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[0])
    axes[0].set_title("Character Length Distribution", fontweight='bold')
    axes[0].set_xlabel("Character Count")
    axes[0].set_xlim(0, 300)

    sns.histplot(data=clean_df, x='word_count', hue='label', bins=40, kde=True,
                 palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[1])
    axes[1].set_title("Word Count Distribution", fontweight='bold')
    axes[1].set_xlabel("Word Count")
    axes[1].set_xlim(0, 60)

    plt.tight_layout()
    length_plot_path = os.path.join(output_dir, "message_length_distribution.png")
    plt.savefig(length_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Message length distribution plot saved to: {length_plot_path}")

    return {
        "class_distribution_plot": dist_plot_path,
        "message_length_plot": length_plot_path,
        "raw_samples": len(raw_df),
        "clean_samples": len(clean_df),
        "duplicates_dropped": duplicates_dropped,
        "ham_count": int(clean_counts['ham']),
        "spam_count": int(clean_counts['spam'])
    }


if __name__ == "__main__":
    generate_eda_figures()
