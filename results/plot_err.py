import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_hist = pd.read_csv('/home/dandi/smt8/paperstegano/results/histogram/results_histogram_standard.csv')
df_chi2 = pd.read_csv('/home/dandi/smt8/paperstegano/results/chi_square/results_chisquare_detection.csv')
df_rs = pd.read_csv('/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_classic.csv')


def compute_error(df):
    grouped = df.groupby('rate')
    fp = grouped.apply(lambda x: ((x['label'] == 0) & (x['detected'] == 1)).sum() / len(x)).reset_index(name='FP')
    fn = grouped.apply(lambda x: ((x['label'] == 1) & (x['detected'] == 0)).sum() / len(x)).reset_index(name='FN')
    return pd.merge(fp, fn, on='rate')

err_hist = compute_error(df_hist)
err_chi2 = compute_error(df_chi2)
err_rs = compute_error(df_rs)

# Plot
plt.figure(figsize=(10,6))
plt.plot(err_hist['rate'], err_hist['FP'], label='FP - Histogram', linestyle='--', marker='o')
plt.plot(err_hist['rate'], err_hist['FN'], label='FN - Histogram', linestyle='-', marker='o')

plt.plot(err_chi2['rate'], err_chi2['FP'], label='FP - Chi-Square', linestyle='--', marker='s')
plt.plot(err_chi2['rate'], err_chi2['FN'], label='FN - Chi-Square', linestyle='-', marker='s')

plt.plot(err_rs['rate'], err_rs['FP'], label='FP - RS', linestyle='--', marker='^')
plt.plot(err_rs['rate'], err_rs['FN'], label='FN - RS', linestyle='-', marker='^')

plt.xlabel('Embedding Rate (%)')
plt.ylabel('Rasio Kesalahan')
plt.title('False Positive & False Negative per Metode')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('error_comparison_plot.png')
plt.show()
