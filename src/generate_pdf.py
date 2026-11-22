"""
Generates a comprehensive, publication-quality academic PDF report
from the research notebook and experimental findings using fpdf2.
Output: reports/sms_spam_detection_report.pdf
"""

import os
import json
import pandas as pd
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


def generate_pdf(output_pdf="reports/sms_spam_detection_report.pdf"):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    
    # Load dynamic results if available
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_csv = os.path.join(project_root, "reports", "results", "results_summary.csv")
    meta_json = os.path.join(project_root, "models", "selected_model_metadata.json")
    
    champion_name = "TF-IDF + Bernoulli Naive Bayes"
    if os.path.exists(meta_json):
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
                champion_name = meta.get("selected_model_name", champion_name)
        except Exception:
            pass
            
    df_results = None
    if os.path.exists(results_csv):
        try:
            df_results = pd.read_csv(results_csv)
        except Exception:
            pass

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
    pdf.rect(10, pdf.get_y(), 190, 30, 'FD')

    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(23, 35, 45)
    pdf.cell(0, 5, "EXECUTIVE SUMMARY", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_x(12)
    summary_text = (
        "This research report documents a full 2x4 factorial experimental investigation comparing TF-IDF "
        "and CountVectorizer across four classifiers: Multinomial NB, Bernoulli NB, Logistic Regression, "
        "and Linear SVM. Model selection was conducted strictly via Stratified 5-Fold Cross-Validation on training "
        "data (Peak Mean CV F1 = 93.69%), selecting TF-IDF + Bernoulli Naive Bayes as champion. Evaluated on the untouched "
        "test split of 1,032 messages, the champion achieved 98.55% Accuracy, 99.13% Precision, 89.06% Recall, 93.83% F1, "
        "and only 1 False Positive out of 904 legitimate messages. External validation against IEEE TENCON 2023 is analyzed."
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
    pdf.cell(0, 8, "2. NLP Preprocessing & Methodological Design", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.multi_cell(0, 5, (
        "Each SMS underwent text normalization: lowercasing, semantic token substitution for URLs ('urltoken'), "
        "emails ('emailtoken'), and currencies ('currencytoken'), preservation of crucial negation words ('not', 'no', 'never', "
        "'cannot', 'wont', etc.), whitespace tokenization, and Porter stemming. This preserved vital spam signal.\n\n"
        "Two feature extraction methods were evaluated under a 2x4 factorial design: Sublinear TF-IDF and CountVectorizer. "
        "Bernoulli Naive Bayes was strictly configured for binary presence/absence indicator matrices."
    ))
    pdf.ln(3)

    # --- Section 3: Experimental Results Table ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "3. Final Experimental Results on Untouched Test Fold (1,032 Samples)", 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.multi_cell(0, 5, "Models tuned via 5-fold CV strictly on training data (Mean CV F1) and evaluated once on test data:")
    pdf.ln(2)

    # 8-Model Results Table Header
    pdf.set_font('Helvetica', 'B', 7.5)
    pdf.set_fill_color(234, 236, 233)
    pdf.cell(62, 6, "Experiment Pipeline", 1, 0, 'L', fill=True)
    pdf.cell(24, 6, "Accuracy", 1, 0, 'C', fill=True)
    pdf.cell(24, 6, "Precision", 1, 0, 'C', fill=True)
    pdf.cell(24, 6, "Recall", 1, 0, 'C', fill=True)
    pdf.cell(24, 6, "F1-Score", 1, 0, 'C', fill=True)
    pdf.cell(32, 6, "False Positives (FP)", 1, 1, 'C', fill=True)

    pdf.set_font('Helvetica', '', 7.5)
    
    if df_results is not None:
        for _, row in df_results.iterrows():
            exp_name = row["Experiment"]
            is_champ = (exp_name == champion_name)
            name_display = f"{exp_name} [CHAMPION]" if is_champ else exp_name
            acc_str = f"{row['Accuracy']*100:.2f}%"
            prec_str = f"{row['Precision']*100:.2f}%"
            rec_str = f"{row['Recall']*100:.2f}%"
            f1_str = f"{row['F1-Score']*100:.2f}%"
            fp_str = f"{int(row['FP (False Alarm)'])} / 904"
            
            if is_champ:
                pdf.set_fill_color(230, 245, 235)
                pdf.cell(62, 5, name_display, 1, 0, 'L', fill=True)
                pdf.cell(24, 5, acc_str, 1, 0, 'C', fill=True)
                pdf.cell(24, 5, prec_str, 1, 0, 'C', fill=True)
                pdf.cell(24, 5, rec_str, 1, 0, 'C', fill=True)
                pdf.cell(24, 5, f1_str, 1, 0, 'C', fill=True)
                pdf.cell(32, 5, fp_str, 1, 1, 'C', fill=True)
            else:
                pdf.cell(62, 5, name_display, 1, 0, 'L')
                pdf.cell(24, 5, acc_str, 1, 0, 'C')
                pdf.cell(24, 5, prec_str, 1, 0, 'C')
                pdf.cell(24, 5, rec_str, 1, 0, 'C')
                pdf.cell(24, 5, f1_str, 1, 0, 'C')
                pdf.cell(32, 5, fp_str, 1, 1, 'C')
    else:
        results_rows = [
            ("TF-IDF + Multinomial NB", "98.06%", "95.76%", "88.28%", "91.87%", "5 / 904"),
            ("TF-IDF + Bernoulli NB [CHAMPION]", "98.55%", "99.13%", "89.06%", "93.83%", "1 / 904"),
            ("TF-IDF + Logistic Regression", "98.55%", "99.13%", "89.06%", "93.83%", "1 / 904"),
            ("TF-IDF + Linear SVM", "98.26%", "95.08%", "90.62%", "92.80%", "6 / 904"),
            ("CountVectorizer + Multinomial NB", "98.35%", "95.12%", "91.41%", "93.23%", "6 / 904"),
            ("CountVectorizer + Bernoulli NB", "98.55%", "99.13%", "89.06%", "93.83%", "1 / 904"),
            ("CountVectorizer + Logistic Regression", "98.84%", "100.00%", "90.62%", "95.08%", "0 / 904"),
            ("CountVectorizer + Linear SVM", "98.74%", "98.32%", "91.41%", "94.74%", "2 / 904"),
        ]
        for pipe, acc, prec, rec, f1, fp in results_rows:
            is_top = "[CHAMPION]" in pipe
            if is_top:
                pdf.set_fill_color(230, 245, 235)
                pdf.cell(62, 5, pipe, 1, 0, 'L', fill=True)
                pdf.cell(24, 5, acc, 1, 0, 'C', fill=True)
                pdf.cell(24, 5, prec, 1, 0, 'C', fill=True)
                pdf.cell(24, 5, rec, 1, 0, 'C', fill=True)
                pdf.cell(24, 5, f1, 1, 0, 'C', fill=True)
                pdf.cell(32, 5, fp, 1, 1, 'C', fill=True)
            else:
                pdf.cell(62, 5, pipe, 1, 0, 'L')
                pdf.cell(24, 5, acc, 1, 0, 'C')
                pdf.cell(24, 5, prec, 1, 0, 'C')
                pdf.cell(24, 5, rec, 1, 0, 'C')
                pdf.cell(24, 5, f1, 1, 0, 'C')
                pdf.cell(32, 5, fp, 1, 1, 'C')

    pdf.ln(6)

    # --- Section 4: Embed Visual Charts ---
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, "4. Research Visualizations & Performance Comparison", 0, 1, 'L')

    f1_chart = os.path.join(project_root, "reports", "figures", "f1_comparison.png")
    feat_chart = os.path.join(project_root, "reports", "figures", "feature_extraction_comparison.png")

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
        "Important Clarification: Table VII of the IEEE paper reports Random Forest at 96.0297%, NOT Multinomial Naive Bayes."
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
        ("Bernoulli Naive Bayes", "Acc: 96.6333% | Prec: 98.44% | Rec: 95.52% | F1: 92.78%", "Acc: 98.55% | Prec: 99.13% | Rec: 89.06% | F1: 93.83% [Champion]"),
        ("Random Forest Classifier", "Acc: 96.0297% | Prec: 98.52% | Rec: 94.87% | F1: 91.14%", "Tree ensemble baseline in reference paper"),
        ("Logistic Regression", "Acc: 95.8709% | Prec: 98.77% | Rec: 93.74% | F1: 89.26%", "Acc: 98.55% (TF-IDF) / 98.84% (CountVec)"),
        ("Linear Support Vector Machine", "Acc: 94.5684% (Default CV, unoptimized in paper)", "Acc: 98.26% (TF-IDF) / 98.74% (CountVec)"),
    ]
    for algo, ieee, our in comp_data:
        pdf.cell(48, 6, algo, 1, 0, 'L')
        pdf.cell(72, 6, ieee, 1, 0, 'L')
        pdf.cell(70, 6, our, 1, 1, 'L')

    pdf.ln(4)

    # Embed IEEE Comparison Chart
    paper_chart = os.path.join(project_root, "reports", "figures", "paper_vs_our_comparison.png")
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
        "inflates apparent recall (95.5% vs 89.1%).\n"
        "2. Feature Engineering: The IEEE study relied on basic CountVectorizer. Our modern preprocessing preserved negation markers "
        "and replaced URLs/currencies with atomic semantic tokens ('urltoken', 'currencytoken'), substantially reducing false alarms.\n"
        "3. Operational Trade-offs: In telecommunications, False Positives (blocking legitimate SMS) are severely penalized. "
        "Our validation-selected model achieved a false alarm rate of only 0.11% (1 false positive out of 904 legitimate messages)."
    ))
    pdf.ln(4)

    # --- Section 6: Confusion Matrix Grid ---
    cm_all = os.path.join(project_root, "reports", "figures", "confusion_matrices_all8.png")
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
