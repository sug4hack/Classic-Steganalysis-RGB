import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

# Path ke file hasil eksperimen
csv_path = "/home/dandi/smt8/paperstegano/results/histogram/results_histogram_standard.csv"

# Load data
df = pd.read_csv(csv_path)

# Inisialisasi hasil
summary = []

# Loop tiap channel & embedding rate
for ch in df['channel'].unique():
    print(f"\n=== [Channel: {ch}] ===")
    for rate in sorted(df['rate'].unique()):
        subset = df[(df['channel'] == ch) & (df['rate'] == rate)]

        if subset.empty:
            continue

        y_true = subset['label']
        y_pred = subset['predicted']
        times = subset['time_sec']

        # Akurasi
        acc = accuracy_score(y_true, y_pred)

        # Confusion Matrix: [TN, FP, FN, TP]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Waktu rata-rata pemrosesan
        avg_time = times.mean()

        # Tambahkan ke summary
        summary.append({
            'channel': ch,
            'rate': rate,
            'accuracy': acc,
            'avg_time_sec': avg_time,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn,
            'total_images': len(subset)
        })

        # Tampilkan ringkasan
        print(f"Rate {rate:>3}% | Acc: {acc:.3f} | Time: {avg_time*1000:.1f} ms | FP: {fp} | FN: {fn}")

# Simpan hasil ringkasan
summary_df = pd.DataFrame(summary)
summary_csv = csv_path.replace(".csv", "_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\n[✓] Ringkasan disimpan ke: {summary_csv}")
