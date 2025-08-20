import pandas as pd
import matplotlib.pyplot as plt

# Path file (harap sesuaikan jika berbeda)
file_path = "/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_classic.csv"

# Coba baca file jika tersedia
try:
    df = pd.read_csv(file_path)

    # Group by embedding rate dan channel, hitung rata-rata jumlah blok R dan S
    grouped = df.groupby(['rate', 'channel'])[['R_blocks', 'S_blocks']].mean().reset_index()

    # Inisialisasi plot
    plt.figure(figsize=(10, 6))

    # Warna dan marker per channel
    style_map = {
        'R': {'color': 'red', 'marker': 'o'},
        'G': {'color': 'green', 'marker': 's'},
        'B': {'color': 'blue', 'marker': '^'},
        'RGB': {'color': 'purple', 'marker': 'D'}
    }

    # Plot semua channel dalam satu grafik
    for ch in grouped['channel'].unique():
        subset = grouped[grouped['channel'] == ch]
        plt.plot(subset['rate'], subset['R_blocks'], label=f'R - {ch}', linestyle='-', **style_map[ch])
        plt.plot(subset['rate'], subset['S_blocks'], label=f'S - {ch}', linestyle='--', **style_map[ch])

    # Format plot
    plt.title('Perubahan Jumlah Blok Regular dan Singular terhadap Embedding Rate')
    plt.xlabel('Embedding Rate (%)')
    plt.ylabel('Jumlah Rata-rata Blok')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("File results_rs_detection.csv tidak ditemukan. Silakan unggah terlebih dahulu.")
