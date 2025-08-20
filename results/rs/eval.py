# import pandas as pd
# from sklearn.metrics import accuracy_score, confusion_matrix

# input_csv = "/home/dandi/smt8/paperstegano/results/rs/results_rs_estimation.csv"
# output_csv = input_csv.replace(".csv", "_summary.csv")

# df = pd.read_csv(input_csv)

# summary = []

# for ch in df['channel'].unique():
#     for rate in sorted(df['rate'].unique()):
#         subset = df[(df['channel'] == ch) & (df['rate'] == rate)]
#         if subset.empty: continue

#         y_true = subset['label']
#         y_pred = subset['detected']
#         acc = accuracy_score(y_true, y_pred)
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#         avg_time = subset['time_sec'].mean()

#         summary.append({
#             'channel': ch,
#             'rate': rate,
#             'accuracy': acc,
#             'false_positives': fp,
#             'false_negatives': fn,
#             'true_positives': tp,
#             'true_negatives': tn,
#             'avg_time_sec': avg_time,
#             'total_images': len(subset)
#         })

# pd.DataFrame(summary).to_csv(output_csv, index=False)
# print(f"[✓] Ringkasan evaluasi RS disimpan ke: {output_csv}")

# import pandas as pd
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Konfigurasi path dan threshold
# input_csv = "/home/dandi/smt8/paperstegano/results/rs/results_rs_estimation.csv"
# output_csv = input_csv.replace(".csv", "_summary.csv")
# threshold_e = 0.02  # Threshold untuk deteksi (bisa disesuaikan)

# # Load data
# df = pd.read_csv(input_csv)
# df['detected'] = (df['estimated_e'] >= threshold_e).astype(int)

# # Evaluasi per channel dan rate
# summary = []
# for ch in df['channel'].unique():
#     for rate in sorted(df['rate'].unique()):
#         subset = df[(df['channel'] == ch) & (df['rate'] == rate)]
#         if subset.empty:
#             continue

#         y_true = subset['label']
#         y_pred = subset['detected']
#         acc = accuracy_score(y_true, y_pred)
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#         avg_time = subset['time_sec'].mean()
#         avg_e = subset['estimated_e'].mean()

#         summary.append({
#             'channel': ch,
#             'rate': rate,
#             'accuracy': acc,
#             'false_positives': fp,
#             'false_negatives': fn,
#             'true_positives': tp,
#             'true_negatives': tn,
#             'avg_estimated_e': avg_e,
#             'avg_time_sec': avg_time,
#             'total_images': len(subset)
#         })

# pd.DataFrame(summary).to_csv(output_csv, index=False)
# print(f"[✓] Ringkasan evaluasi RS estimation disimpan ke: {output_csv}")

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Path ke hasil deteksi RS Analysis
input_csv = "/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_classic.csv"
output_csv = input_csv.replace(".csv", "_summary.csv")

# Load hasil deteksi
df = pd.read_csv(input_csv)

# Ringkasan per channel & embedding rate
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

# Simpan hasil rekap evaluasi
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_csv, index=False)
print(f"[✓] Ringkasan evaluasi RS Analysis disimpan ke: {output_csv}")
