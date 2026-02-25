#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import sys
import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv('env.local')

DEFAULT_INPUT_CSV = os.getenv("INPUT_CSV", "llm_eval_outputs/gpt-4o/outputs5.csv")
def get_ground_truth(path):
    path_lower = str(path).lower()
    if "invalid" in path_lower:
        return 0
    elif "valid" in path_lower:
        return 1
    else:
        return None

def normalize_prediction(val):
    val_str = str(val).lower().strip()
    if val_str in ['true', 'yes', 'valid', '1']:
        return 1
    elif val_str in ['false', 'no', 'invalid', '0']:
        return 0
    else:
        return None

input_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_CSV

if not os.path.exists(input_csv):
    print(f"Error: File {input_csv} does not exist.")
    sys.exit(1)

df = pd.read_csv(input_csv)

df['y_true'] = df['image_path'].apply(get_ground_truth)
df['valid_clean'] = df['valid'].apply(normalize_prediction)

df_clean = df.dropna(subset=['y_true', 'valid_clean'])

y_true = df_clean['y_true'].astype(int)
y_pred = df_clean['valid_clean'].astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

stats_text = (
    f"--- OVERALL BINARY METRICS ---\n"
    f"Total Samples: {len(df_clean)}\n"
    f"Accuracy:      {acc:.2%}\n"
    f"F1-Score:      {f1:.4f}\n\n"
    f"--- CONFUSION MATRIX DETAILS ---\n"
    f"TP (True Valid):    {tp}\n"
    f"TN (True Invalid):  {tn}\n"
    f"FP (False Valid):   {fp}  <-- Type I Error\n"
    f"FN (False Invalid): {fn}  <-- Type II Error\n\n"
    f"--- QUALITY RATIOS ---\n"
    f"Precision:     {prec:.2%}\n"
    f"Recall:        {rec:.2%}\n"
    f"Specificity:   {tn/(tn+fp):.2%}"
)

print("\n" + "="*50)
print(f"RESULTS FOR: {os.path.basename(input_csv)}")
print("="*50)
print(stats_text)
print("="*50)

try:
    plt.figure(figsize=(10, 7))
    plt.subplots_adjust(right=0.6)
    
    cm_data = [[tn, fp], [fn, tp]]
    
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred INVALID', 'Pred VALID'], 
                yticklabels=['Actual INVALID', 'Actual VALID'],
                cbar=False)
    
    plt.title(f'Binary Classification: {os.path.basename(input_csv)}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')
    
    plt.figtext(0.65, 0.5, stats_text, ha="left", va="center", fontsize=11, 
                family='monospace', bbox={"facecolor":"#f8f9fa", "alpha":0.9, "pad":10})
    
    plt.show()
    
except Exception as e:
    print(f"Error generating plot: {e}")