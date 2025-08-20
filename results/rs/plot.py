# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_csv("/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_summary.csv")

# sns.set(style="whitegrid")
# channels = df['channel'].unique()

# # Akurasi
# plt.figure(figsize=(10, 6))
# for ch in channels:
#     subset = df[df['channel'] == ch]
#     plt.plot(subset['rate'], subset['accuracy'] * 100, marker='o', label=f"{ch}")
# plt.title("RS Detection Accuracy vs Embedding Rate")
# plt.xlabel("Embedding Rate (%)")
# plt.ylabel("Accuracy (%)")
# plt.legend(title="Channel")
# plt.tight_layout()
# plt.savefig("plot_rs_accuracy_vs_rate.png")
# plt.show()

# # Waktu
# plt.figure(figsize=(10, 6))
# for ch in channels:
#     subset = df[df['channel'] == ch]
#     plt.plot(subset['rate'], subset['avg_time_sec'] * 1000, marker='s', label=f"{ch}")
# plt.title("RS Avg Processing Time vs Embedding Rate")
# plt.xlabel("Embedding Rate (%)")
# plt.ylabel("Time (ms)")
# plt.legend(title="Channel")
# plt.tight_layout()
# plt.savefig("plot_rs_time_vs_rate.png")
# plt.show()

# # FP & FN
# plt.figure(figsize=(12, 6))
# for ch in channels:
#     subset = df[df['channel'] == ch]
#     plt.plot(subset['rate'], subset['false_positives'], marker='^', linestyle='--', label=f"{ch} - FP")
#     plt.plot(subset['rate'], subset['false_negatives'], marker='v', linestyle='-', label=f"{ch} - FN")
# plt.title("RS False Positives & False Negatives vs Embedding Rate")
# plt.xlabel("Embedding Rate (%)")
# plt.ylabel("Jumlah Kasus")
# plt.legend(title="Channel - Type")
# plt.tight_layout()
# plt.savefig("plot_rs_fp_fn_vs_rate.png")
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load hasil ringkasan evaluasi
# summary_csv = "/home/dandi/smt8/paperstegano/results/rs/results_rs_estimation_summary.csv"
# df = pd.read_csv(summary_csv)

# sns.set(style="whitegrid")
# channels = df['channel'].unique()

# # === 1. Akurasi Deteksi ===
# for ch in channels:
#     subset = df[df['channel'] == ch]
#     plt.figure(figsize=(10, 6))
#     plt.plot(subset['rate'], subset['accuracy'] * 100, marker='o')
#     plt.title(f"RS Estimation Accuracy vs Embedding Rate [Channel: {ch}]")
#     plt.xlabel("Embedding Rate (%)")
#     plt.ylabel("Accuracy (%)")
#     plt.ylim(0, 105)
#     plt.tight_layout()
#     plt.savefig(f"plot_rs_estimation_accuracy_{ch}.png")
#     plt.show()

# # === 2. Estimasi e rata-rata ===
# for ch in channels:
#     subset = df[df['channel'] == ch]
#     plt.figure(figsize=(10, 6))
#     plt.plot(subset['rate'], subset['avg_estimated_e'], marker='s', label="Avg e")
#     plt.axhline(0.02, color='r', linestyle='--', label='Threshold')
#     plt.title(f"RS Estimated e vs Embedding Rate [Channel: {ch}]")
#     plt.xlabel("Embedding Rate (%)")
#     plt.ylabel("Estimated e")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"plot_rs_estimated_e_{ch}.png")
#     plt.show()

# # === 3. False Positives dan Negatives ===
# for ch in channels:
#     subset = df[df['channel'] == ch]
#     plt.figure(figsize=(12, 6))
#     plt.plot(subset['rate'], subset['false_positives'], marker='^', linestyle='--', label="False Positives")
#     plt.plot(subset['rate'], subset['false_negatives'], marker='v', linestyle='-', label="False Negatives")
#     plt.title(f"RS FP & FN vs Embedding Rate [Channel: {ch}]")
#     plt.xlabel("Embedding Rate (%)")
#     plt.ylabel("Jumlah Kasus")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"plot_rs_fp_fn_{ch}.png")
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path ke hasil rekap evaluasi RS Analysis
summary_csv = "/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_classic_summary.csv"
df = pd.read_csv(summary_csv)

sns.set(style="whitegrid")
channels = df['channel'].unique()

# === 1. Akurasi vs Embedding Rate ===
plt.figure(figsize=(10, 6))
for ch in channels:
    subset = df[df['channel'] == ch]
    plt.plot(subset['rate'], subset['accuracy'] * 100, marker='o', label=f"{ch}")
plt.title("RS Analysis Accuracy vs Embedding Rate")
plt.xlabel("Embedding Rate (%)")
plt.ylabel("Accuracy (%)")
plt.legend(title="Channel")
plt.tight_layout()
plt.savefig("plot_rs_accuracy_vs_rate.png")
plt.show()

# === 2. Waktu Pemrosesan vs Embedding Rate ===
plt.figure(figsize=(10, 6))
for ch in channels:
    subset = df[df['channel'] == ch]
    plt.plot(subset['rate'], subset['avg_time_sec'] * 1000, marker='s', label=f"{ch}")
plt.title("RS Analysis Processing Time vs Embedding Rate")
plt.xlabel("Embedding Rate (%)")
plt.ylabel("Time (ms)")
plt.legend(title="Channel")
plt.tight_layout()
plt.savefig("plot_rs_time_vs_rate.png")
plt.show()

# === 3. FP dan FN vs Embedding Rate ===
plt.figure(figsize=(12, 6))
for ch in channels:
    subset = df[df['channel'] == ch]
    plt.plot(subset['rate'], subset['false_positives'], marker='^', linestyle='--', label=f"{ch} - FP")
    plt.plot(subset['rate'], subset['false_negatives'], marker='v', linestyle='-', label=f"{ch} - FN")
plt.title("RS Analysis False Positives & False Negatives vs Embedding Rate")
plt.xlabel("Embedding Rate (%)")
plt.ylabel("Jumlah Kasus")
plt.legend(title="Channel - Type")
plt.tight_layout()
plt.savefig("plot_rs_fp_fn_vs_rate.png")
plt.show()
