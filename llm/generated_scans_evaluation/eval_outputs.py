#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os

DEFAULT_INPUT_CSV = "llm_eval_outputs/gpt-4o/outputs3.csv" 

def get_ground_truth(path):
    """
    Extracts the true label from the file path.
    """
    path_lower = str(path).lower()
    if "invalid" in path_lower:
        return "invalid"
    elif "valid" in path_lower:
        return "valid"
    else:
        return "unknown"

def normalize_prediction(val):
    """
    Standardizes the model's output to 'valid' or 'invalid'.
    """
    val_str = str(val).lower().strip()
    if val_str in ['true', 'yes', 'valid', '1']:
        return "valid"
    elif val_str in ['false', 'no', 'invalid', '0']:
        return "invalid"
    else:
        return "error"

input_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_CSV

print(f"Loading results from: {input_csv}")

if not os.path.exists(input_csv):
    print(f"Error: File {input_csv} does not exist.")
    sys.exit(1)

df = pd.read_csv(input_csv)

df['y_true'] = df['image_path'].apply(get_ground_truth)
df['y_pred'] = df['valid'].apply(normalize_prediction)

df_clean = df[
    (df['y_true'] != 'unknown') & 
    (df['y_pred'] != 'error')
]

y_true = df_clean['y_true']
y_pred = df_clean['y_pred']
labels = ["valid", "invalid"]

cm = confusion_matrix(y_true, y_pred, labels=labels)
report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

stats_text = (
    f"Accuracy: {report_dict['accuracy']:.2f}\n\n"
    f"Class 'valid':\n"
    f"  Precision: {report_dict['valid']['precision']:.2f}\n"
    f"  Recall:    {report_dict['valid']['recall']:.2f}\n"
    f"  F1-Score:  {report_dict['valid']['f1-score']:.2f}\n\n"
    f"Class 'invalid':\n"
    f"  Precision: {report_dict['invalid']['precision']:.2f}\n"
    f"  Recall:    {report_dict['invalid']['recall']:.2f}\n"
    f"  F1-Score:  {report_dict['invalid']['f1-score']:.2f}"
)

try:
    plt.figure(figsize=(8, 8))
    
    plt.subplots_adjust(bottom=0.25)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label (from path)')
    plt.title(f'Confusion Matrix: {os.path.basename(input_csv)}')
    
    plt.figtext(0.5, 0.02, stats_text, ha="center", fontsize=10, 
                bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    #output_img = "confusion_matrix_with_metrics.png"
    #plt.savefig(output_img)
    #print(f"Success! Plot saved to: {output_img}")
    plt.show()
    
except Exception as e:
    print(f"Error generating plot: {e}")

print("\n" + "="*60)
print(classification_report(y_true, y_pred, target_names=labels))
print("="*60)