import pandas as pd
import matplotlib.pyplot as plt

# Load all results
df_hist = pd.read_csv('/home/dandi/smt8/paperstegano/results/histogram/results_histogram_standard.csv')
df_chi2 = pd.read_csv('/home/dandi/smt8/paperstegano/results/chi_square/results_chisquare_detection.csv')
df_rs = pd.read_csv('/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_classic.csv')

def compute_accuracy(df):
    return df.groupby('rate').apply(lambda x: (x['label'] == x['detected']).mean()).reset_index(name='accuracy')

# Hitung akurasi
acc_hist = compute_accuracy(df_hist)
acc_chi2 = compute_accuracy(df_chi2)
acc_rs = compute_accuracy(df_rs)

# Plot
plt.figure(figsize=(10,6))
plt.plot(acc_hist['rate'], acc_hist['accuracy'], label='Histogram', marker='o')
plt.plot(acc_chi2['rate'], acc_chi2['accuracy'], label='Chi-Square', marker='s')
plt.plot(acc_rs['rate'], acc_rs['accuracy'], label='RS Analysis', marker='^')

plt.xlabel('Embedding Rate (%)')
plt.ylabel('Akurasi Deteksi')
plt.title('Perbandingan Akurasi Deteksi')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_comparison_plot.png')
plt.show()
