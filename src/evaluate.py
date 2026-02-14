import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

input_path = 'results/predictions.csv'
output_file = 'results/metrics.txt'

df = pd.read_csv(input_path)
y_true = df['Actual']
y_pred = df['Predicted']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

metrics_text = f"""Model Evaluation Metrics:
-------------------------
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}
"""

with open(output_file, 'w') as f:
    f.write(metrics_text)

print(metrics_text)
print(f"Metrics saved to {output_file}")