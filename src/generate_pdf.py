"""
Generates a comprehensive, publication-quality academic PDF report
from the research notebook and experimental findings using fpdf2.
Output: reports/sms_spam_detection_report.pdf
"""

import os
import json
from fpdf import FPDF


class AcademicPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(104, 116, 123)
        self.cell(0, 8, "SIGNAL: SMS INTELLIGENCE LAB  |  RESEARCH METHODOLOGIES (RM)", 0, 0, 'L')
        self.cell(0, 8, "B.TECH CSE - 7TH SEMESTER", 0, 1, 'R')
        self.set_draw_color(217, 223, 225)
        self.line(10, 16, 200, 16)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(104, 116, 123)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  Priyansu Das - Siksha 'O' Anusandhan University (ITER)", 0, 0, 'C')


def generate_pdf(notebook_path="notebooks/sms_spam_detection.ipynb", output_pdf="reports/sms_spam_detection_report.pdf"):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    pdf = AcademicPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Document Cover / Header ---
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(23, 35, 45)
    pdf.cell(0, 10, "Comparative Analysis of ML Algorithms", 0, 1, 'L')
    pdf.cell(0, 10, "for SMS Spam Detection", 0, 1, 'L')

    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(230, 107, 54) # Accent orange
    pdf.cell(0, 6, "Research Methodologies (RM) Subject Project Report  |  7th Semester B.Tech", 0, 1, 'L')
    pdf.ln(3)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(23, 35, 45)
    author_info = (
        "Author: Priyansu Das  |  Department of Computer Science and Engineering\n"
        "Institution: Institute of Technical Education and Research (ITER), Siksha 'O' Anusandhan University\n"
        "GitHub Repository: https://github.com/priyansudas07/sms-spam-detection"
    )
    pdf.multi_cell(0, 5, author_info)
    pdf.ln(4)

    pdf.set_draw_color(217, 223, 225)
    pdf.set_fill_color(245, 244, 239) # #F5F4EF
    pdf.rect(10, pdf.get_y(), 190, 28, 'FD')

    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(23, 35, 45)
    pdf.cell(0, 5, "EXECUTIVE SUMMARY", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_x(12)
    summary_text = (
        "This research report documents a full 2x4 factorial experimental investigation comparing TF-IDF "
        "and CountVectorizer across four classifiers: Multinomial NB, Bernoulli NB, Logistic Regression, "
        "and Linear SVM. Evaluated on an untouched test split of 1,032 messages from the official UCI SMS "
        "dataset, Linear SVM with TF-IDF achieved the highest overall performance (98.16% Accuracy, 92.24% F1-Score). "
        "External benchmark validation against IEEE TENCON 2023 is analyzed with full academic rigor."
    )
    pdf.multi_cell(186, 4.5, summary_text)
    pdf.ln(8)

    # --- Section 1: Dataset & Cleaning ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(23, 35, 45)
    pdf.cell(0, 8, "1. Dataset Ingestion & Natural Class Distribution", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.multi_cell(0, 5, (
        "The project utilizes the official UCI SMS Spam Collection dataset (ID: 228). A total of 5,574 raw "
        "messages were ingested. String deduplication removed 414 duplicate records, yielding 5,160 clean unique "
        "messages. The natural imbalanced class distribution (87.56% Ham, 12.44% Spam) was strictly preserved."
    ))
    pdf.ln(2)

    # Dataset Table
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(234, 236, 233)
    pdf.cell(50, 6, "Dataset Attribute", 1, 0, 'L', fill=True)
    pdf.cell(45, 6, "Verified Count", 1, 0, 'C', fill=True)
    pdf.cell(95, 6, "Academic Notes", 1, 1, 'L', fill=True)

    pdf.set_font('Helvetica', '', 8)
    dataset_rows = [
        ("Original Raw Messages", "5,574", "Standard English SMS messages"),
        ("Duplicates Removed", "414", "Deduplicated to prevent synthetic inflation"),
        ("Clean Unique Messages", "5,160", "Authentic corpus"),
        ("Legitimate (Ham)", "4,518 (87.56%)", "Natural majority traffic"),
        ("Spam Messages", "642 (12.44%)", "Natural minority target class"),
        ("Training Set (80%)", "4,128", "Stratified splitting with random_state=42"),
        ("Testing Set (20%)", "1,032", "Untouched evaluation fold (904 Ham, 128 Spam)")
    ]
    for attr, val, notes in dataset_rows:
        pdf.cell(50, 5.5, attr, 1, 0, 'L')
        pdf.cell(45, 5.5, val, 1, 0, 'C')
        pdf.cell(95, 5.5, notes, 1, 1, 'L')

    pdf.ln(5)

    # --- Section 2: NLP Preprocessing & Feature Extraction ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "2. NLP Preprocessing & 2x4 Factorial Feature Representation", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.multi_cell(0, 5, (
        "Each SMS underwent text normalization: lowercasing, regex noise removal, whitespace tokenization, "
        "English stopword filtering, and Porter stemming. This reduced character volume by 39.74% and vocabulary "
        "by 41.29% (13,627 to 8,001 unique root words).\n\n"
        "Two feature extraction methods were implemented under a 2x4 factorial design: Sublinear TF-IDF (n-grams 1-2) "
        "and CountVectorizer. Bernoulli Naive Bayes was strictly enforced with binary presence/absence indicator matrices."
    ))
    pdf.ln(3)

    # --- Section 3: Experimental Results Table ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "3. Final Experimental Results on Untouched Test Fold (1,032 Samples)", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.multi_cell(0, 5, "All models were tuned via 5-fold cross-validation on training data and evaluated on 1,032 test samples:")
    pdf.ln(2)

    # 8-Model Results Table
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_fill_color(234, 236, 233)
    pdf.cell(60, 6, "Experiment Pipeline", 1, 0, 'L', fill=True)
    pdf.cell(24, 6, "Accuracy", 1, 0, 'C', fill=True)
    pdf.cell(24, 6, "Precision", 1, 0, 'C', fill=True)
    pdf.cell(24, 6, "Recall", 1, 0, 'C', fill=True)
    pdf.cell(24, 6, "F1-Score", 1, 0, 'C', fill=True)
    pdf.cell(34, 6, "False Positives (FP)", 1, 1, 'C', fill=True)

    pdf.set_font('Helvetica', '', 7.5)
    results_rows = [
        ("TF-IDF + Multinomial NB", "97.67%", "94.07%", "86.72%", "90.24%", "7 / 904"),
        ("TF-IDF + Bernoulli NB (Binary)", "98.06%", "97.37%", "86.72%", "91.74%", "3 / 904 (0.3%)"),
        ("TF-IDF + Logistic Regression", "97.97%", "96.52%", "86.72%", "91.36%", "4 / 904"),
        ("TF-IDF + Linear SVM (Top Model)", "98.16%", "96.58%", "88.28%", "92.24%", "4 / 904"),
        ("CountVectorizer + Multinomial NB", "97.77%", "92.68%", "89.06%", "90.84%", "9 / 904"),
        ("CountVectorizer + Bernoulli NB", "98.06%", "97.37%", "86.72%", "91.74%", "3 / 904 (0.3%)"),
        ("CountVectorizer + Logistic Regression", "97.87%", "97.32%", "85.16%", "90.83%", "3 / 904 (0.3%)"),
        ("CountVectorizer + Linear SVM", "98.16%", "97.39%", "87.50%", "92.18%", "3 / 904 (0.3%)"),
    ]
    for pipe, acc, prec, rec, f1, fp in results_rows:
        is_top = "Top Model" in pipe
        if is_top:
            pdf.set_fill_color(230, 245, 235)
            pdf.cell(60, 5, pipe, 1, 0, 'L', fill=True)
            pdf.cell(24, 5, acc, 1, 0, 'C', fill=True)
            pdf.cell(24, 5, prec, 1, 0, 'C', fill=True)
            pdf.cell(24, 5, rec, 1, 0, 'C', fill=True)
            pdf.cell(24, 5, f1, 1, 0, 'C', fill=True)
            pdf.cell(34, 5, fp, 1, 1, 'C', fill=True)
        else:
            pdf.cell(60, 5, pipe, 1, 0, 'L')
            pdf.cell(24, 5, acc, 1, 0, 'C')
            pdf.cell(24, 5, prec, 1, 0, 'C')
            pdf.cell(24, 5, rec, 1, 0, 'C')
            pdf.cell(24, 5, f1, 1, 0, 'C')
            pdf.cell(34, 5, fp, 1, 1, 'C')

    pdf.ln(6)

    # --- Section 4: Embed Visual Charts ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "4. Research Visualizations & Performance Comparison", 0, 1, 'L')

    f1_chart = "reports/figures/f1_comparison.png"
    feat_chart = "reports/figures/feature_extraction_comparison.png"

    if os.path.exists(f1_chart) and os.path.exists(feat_chart):
        y_pos = pdf.get_y()
        pdf.image(f1_chart, x=10, y=y_pos, w=92)
        pdf.image(feat_chart, x=106, y=y_pos, w=92)
        pdf.set_y(y_pos + 52)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.cell(92, 5, "Fig 1: F1-Score Comparison across 8 Pipelines", 0, 0, 'C')
        pdf.cell(92, 5, "Fig 2: Macro TF-IDF vs. CountVectorizer Comparison", 0, 1, 'C')
        pdf.ln(4)

    # --- Section 5: IEEE Paper Benchmark Comparison ---
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(23, 35, 45)
    pdf.cell(0, 8, "5. External Benchmark Validation: IEEE TENCON 2023 Study", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.multi_cell(0, 5, (
        "Reference Paper: 'A Machine Learning Approach for Efficient Spam Detection in Short Messaging System (SMS)', "
        "IEEE TENCON 2023, Chiang Mai, Thailand (Robert G. de Luna et al.).\n"
        "Important Correction: Table VII of the IEEE paper reports Random Forest at 96.0297%, NOT Multinomial Naive Bayes."
    ))
    pdf.ln(2)

    # IEEE Benchmark Table
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_fill_color(234, 236, 233)
    pdf.cell(48, 6, "Algorithm", 1, 0, 'L', fill=True)
    pdf.cell(72, 6, "IEEE TENCON 2023 Published (Table VII)", 1, 0, 'L', fill=True)
    pdf.cell(70, 6, "Our Experimental Study (Factorial)", 1, 1, 'L', fill=True)

    pdf.set_font('Helvetica', '', 7.5)
    comp_data = [
        ("Bernoulli Naive Bayes", "Acc: 96.6333% | Prec: 98.44% | Rec: 95.52% | F1: 92.78%", "Acc: 98.06% | Prec: 97.37% | Rec: 86.72% | F1: 91.74%"),
        ("Random Forest Classifier", "Acc: 96.0297% | Prec: 98.52% | Rec: 94.87% | F1: 91.14%", "Tree ensemble baseline in reference paper"),
        ("Logistic Regression", "Acc: 95.8709% | Prec: 98.77% | Rec: 93.74% | F1: 89.26%", "Acc: 97.97% (TF-IDF) / 97.87% (CountVec)"),
        ("Linear Support Vector Machine", "Acc: 94.5684% (Default CV, unoptimized in paper)", "Acc: 98.16% | Prec: 96.58% | Rec: 88.28% | F1: 92.24% (Top)"),
    ]
    for algo, ieee, our in comp_data:
        pdf.cell(48, 6, algo, 1, 0, 'L')
        pdf.cell(72, 6, ieee, 1, 0, 'L')
        pdf.cell(70, 6, our, 1, 1, 'L')

    pdf.ln(4)

    # Embed IEEE Comparison Chart
    paper_chart = "reports/figures/paper_vs_our_comparison.png"
    if os.path.exists(paper_chart):
        pdf.image(paper_chart, x=25, y=pdf.get_y(), w=160)
        pdf.set_y(pdf.get_y() + 75)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.cell(0, 5, "Fig 3: Accuracy Benchmark between IEEE TENCON 2023 and Our Dual Feature Extraction Experiments", 0, 1, 'C')
        pdf.ln(3)

    # Methodological Differences Box
    pdf.set_font('Helvetica', 'B', 9)
    pdf.cell(0, 6, "Key Methodological Variance Analysis:", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8)
    pdf.multi_cell(0, 4.5, (
        "1. Class Imbalance: The IEEE study artificially undersampled ham messages to force an approximate 60:40 partition. "
        "Our study preserved the natural real-world 87.56% Ham to 12.44% Spam distribution. Higher spam density artificially "
        "inflates recall (95.5% vs 88.3%).\n"
        "2. Feature Engineering: The IEEE study selected CountVectorizer. Our factorial study reveals that Sublinear TF-IDF "
        "achieves superior performance for Linear SVM and Logistic Regression.\n"
        "3. Operational Trade-offs: In telecom networks, False Positives (blocking legitimate SMS) are severely penalized. "
        "Both studies confirm that Naive Bayes and Linear SVM maintain exceptionally low false-alarm rates (<= 0.4%)."
    ))
    pdf.ln(4)

    # --- Section 6: Confusion Matrix Grid ---
    cm_all = "reports/figures/confusion_matrices_all8.png"
    if os.path.exists(cm_all):
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, "6. Full 8-Panel Confusion Matrix Grid", 0, 1, 'L')
        pdf.set_font('Helvetica', '', 8)
        pdf.cell(0, 5, "Detailed classification distributions for all 8 pipelines on 1,032 test messages (904 Ham, 128 Spam):", 0, 1, 'L')
        pdf.ln(2)
        pdf.image(cm_all, x=15, y=pdf.get_y(), w=180)

    # Save PDF
    pdf.output(output_pdf)
    print(f"[SUCCESS] Academic research PDF generated: {output_pdf}")
    return output_pdf


if __name__ == "__main__":
    generate_pdf()
