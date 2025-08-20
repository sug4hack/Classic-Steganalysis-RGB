import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Path ke hasil deteksi chi-square
input_csv = "/home/dandi/smt8/paperstegano/results/chi_square/results_chisquare_detection.csv"
output_csv = input_csv.replace(".csv", "_summary.csv")

# Load data
df = pd.read_csv(input_csv)

# Ringkasan hasil per channel dan rate
summary = []

for ch in df['channel'].unique():
    for rate in sorted(df['rate'].unique()):
        subset = df[(df['channel'] == ch) & (df['rate'] == rate)]
        if subset.empty:
            continue

        y_true = subset['label']
        y_pred = subset['detected']
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        avg_time = subset['time_sec'].mean()

        summary.append({
            'channel': ch,
            'rate': rate,
            'accuracy': acc,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn,
            'avg_time_sec': avg_time,
            'total_images': len(subset)
        })

# Simpan ringkasan
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_csv, index=False)
print(f"[✓] Ringkasan evaluasi chi-square disimpan ke: {output_csv}")
