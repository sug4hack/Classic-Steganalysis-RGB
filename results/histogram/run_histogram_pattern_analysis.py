import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time

# ==== Konfigurasi direktori ====
original_dir = "/home/dandi/smt8/paperstegano/data/original_resized"
embedded_base_dir = "/home/dandi/smt8/paperstegano/data/embedded"
output_csv = "/home/dandi/smt8/paperstegano/results/histogram/results_histogram_pattern.csv"

# ==== Parameter eksperimen ====
channels = ['R', 'G', 'B', 'RGB']
rates = [0, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]

# Threshold jumlah pasangan (x, -x) untuk mendeteksi stego
default_threshold = 25  # Sesuaikan berdasarkan eksperimen

# ==== Fungsi utama analisis pola histogram ====
def compute_histogram_pattern_score(img1, img2, channel):
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    ch_idx = {'R': 0, 'G': 1, 'B': 2}
    channels_to_check = [ch_idx[c] for c in channel] if channel != 'RGB' else [0, 1, 2]

    pattern_count = 0

    for ch in channels_to_check:
        hist1 = np.histogram(arr1[:, :, ch].flatten(), bins=256, range=(0, 256))[0]
        hist2 = np.histogram(arr2[:, :, ch].flatten(), bins=256, range=(0, 256))[0]

        diff = hist2 - hist1  # selisih bisa negatif
        for i in range(len(diff) - 1):
            if diff[i] != 0 and diff[i] == -diff[i + 1]:  # contoh: -3, 3 atau 25, -25
                pattern_count += 1

    return pattern_count

# ==== Jalankan eksperimen ====
results = []
image_files = sorted(os.listdir(original_dir))

for ch in channels:
    for rate in rates:
        print(f"[→] Memproses channel={ch} | rate={rate}%")
        stego_dir = os.path.join(embedded_base_dir, ch, str(rate))
        for fname in tqdm(image_files):
            cover_path = os.path.join(original_dir, fname)
            stego_path = os.path.join(stego_dir, fname)
            if not os.path.exists(stego_path): continue

            try:
                img_cover = Image.open(cover_path).convert('RGB')
                img_stego = Image.open(stego_path).convert('RGB')

                start = time.time()
                score = compute_histogram_pattern_score(img_cover, img_stego, ch)
                elapsed = time.time() - start

                label = 1 if rate > 0 else 0
                predicted = 1 if score >= default_threshold else 0

                results.append({
                    'filename': fname,
                    'channel': ch,
                    'rate': rate,
                    'label': label,
                    'score': score,
                    'predicted': predicted,
                    'time_sec': elapsed
                })
            except Exception as e:
                print(f"[!] Error {fname}: {e}")

# ==== Simpan hasil ====
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"[✓] Hasil disimpan: {output_csv}")
