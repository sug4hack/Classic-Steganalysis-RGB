import pandas as pd
import matplotlib.pyplot as plt

# Load results
df_hist = pd.read_csv('/home/dandi/smt8/paperstegano/results/histogram/results_histogram_standard.csv')
df_chi2 = pd.read_csv('/home/dandi/smt8/paperstegano/results/chi_square/results_chisquare_detection.csv')
df_rs = pd.read_csv('/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_classic.csv')


def compute_time(df):
    return df.groupby('rate')['time_sec'].mean().reset_index()

time_hist = compute_time(df_hist)
time_chi2 = compute_time(df_chi2)
time_rs = compute_time(df_rs)

# Plot
plt.figure(figsize=(10,6))
plt.plot(time_hist['rate'], time_hist['time_sec'], label='Histogram', marker='o')
plt.plot(time_chi2['rate'], time_chi2['time_sec'], label='Chi-Square', marker='s')
plt.plot(time_rs['rate'], time_rs['time_sec'], label='RS Analysis', marker='^')

plt.xlabel('Embedding Rate (%)')
plt.ylabel('Waktu Pemrosesan per Gambar (detik)')
plt.title('Perbandingan Waktu Pemrosesan')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('time_comparison_plot.png')
plt.show()
