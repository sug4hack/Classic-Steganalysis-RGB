import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path ke ringkasan hasil evaluasi
summary_csv = "/home/dandi/smt8/paperstegano/results/chi_square/results_chisquare_detection_summary.csv"
df = pd.read_csv(summary_csv)

sns.set(style="whitegrid")
channels = df['channel'].unique()

# === 1. Akurasi vs Embedding Rate ===
plt.figure(figsize=(10, 6))
for ch in channels:
    subset = df[df['channel'] == ch]
    plt.plot(subset['rate'], subset['accuracy'] * 100, marker='o', label=f"{ch}")
plt.title("Chi-Square Detection Accuracy vs Embedding Rate")
plt.xlabel("Embedding Rate (%)")
plt.ylabel("Accuracy (%)")
plt.legend(title="Channel")
plt.tight_layout()
plt.savefig("plot_chisquare_accuracy_vs_rate.png")
plt.show()

# === 2. Waktu Pemrosesan vs Embedding Rate ===
plt.figure(figsize=(10, 6))
for ch in channels:
    subset = df[df['channel'] == ch]
    plt.plot(subset['rate'], subset['avg_time_sec'] * 1000, marker='s', label=f"{ch}")
plt.title("Chi-Square Average Processing Time vs Embedding Rate")
plt.xlabel("Embedding Rate (%)")
plt.ylabel("Time (ms)")
plt.legend(title="Channel")
plt.tight_layout()
plt.savefig("plot_chisquare_time_vs_rate.png")
plt.show()

# === 3. False Positives & Negatives vs Rate ===
plt.figure(figsize=(12, 6))
for ch in channels:
    subset = df[df['channel'] == ch]
    plt.plot(subset['rate'], subset['false_positives'], marker='^', linestyle='--', label=f"{ch} - FP")
    plt.plot(subset['rate'], subset['false_negatives'], marker='v', linestyle='-', label=f"{ch} - FN")
plt.title("Chi-Square FP & FN vs Embedding Rate")
plt.xlabel("Embedding Rate (%)")
plt.ylabel("Jumlah Kasus")
plt.legend(title="Channel - Type")
plt.tight_layout()
plt.savefig("plot_chisquare_fp_fn_vs_rate.png")
plt.show()
