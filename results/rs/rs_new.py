import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === Konfigurasi ===
embedded_base_dir = "/home/dandi/smt8/paperstegano/data/embedded"
channels = ['R', 'G', 'B', 'RGB']
rates = [0, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]
output_csv = "/home/dandi/smt8/paperstegano/results/rs/results_rs_detection_classic.csv"

# === Fungsi pembalik LSB (flipping) ===
def flip_lsb(array, mask):
    return array ^ mask  # XOR dengan mask (1010...)

# === Fungsi diskriminan (total perbedaan absolut antar elemen) ===
def discriminant(block):
    return np.sum(np.abs(np.diff(block)))

# === RS analysis untuk 1D blok di 1 channel ===
def rs_analysis_channel(flat_channel):
    block_size = 4
    num_blocks = len(flat_channel) // block_size
    R = S = U = 0

    # Gunakan masker flipping 1010
    mask = np.array([1, 0, 1, 0], dtype=np.uint8)

    for i in range(num_blocks):
        block = flat_channel[i * block_size : (i + 1) * block_size]
        if len(block) < 4:
            continue

        f_original = discriminant(block)
        f_flipped = discriminant(flip_lsb(block, mask))

        if f_flipped > f_original:
            R += 1
        elif f_flipped < f_original:
            S += 1
        else:
            U += 1

    return R, S, U

# === Proses semua citra ===
results = []

for ch in channels:
    for rate in rates:
        img_dir = os.path.join(embedded_base_dir, ch, str(rate))
        print(f"[→] Processing channel={ch}, rate={rate}%")
        for fname in tqdm(sorted(os.listdir(img_dir))):
            img_path = os.path.join(img_dir, fname)
            try:
                start = time.time()
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)

                R_total = S_total = U_total = 0

                if ch == 'RGB':
                    for i in range(3):
                        flat = arr[:, :, i].flatten()
                        R, S, U = rs_analysis_channel(flat)
                        R_total += R
                        S_total += S
                        U_total += U
                else:
                    idx = {'R': 0, 'G': 1, 'B': 2}[ch]
                    flat = arr[:, :, idx].flatten()
                    R_total, S_total, U_total = rs_analysis_channel(flat)

                # Deteksi: jika S > R, kemungkinan stego
                detected = 1 if S_total > R_total else 0
                label = 1 if rate > 0 else 0
                elapsed = time.time() - start

                results.append({
                    'filename': fname,
                    'channel': ch,
                    'rate': rate,
                    'label': label,
                    'detected': detected,
                    'R_blocks': R_total,
                    'S_blocks': S_total,
                    'U_blocks': U_total,
                    'time_sec': elapsed
                })

            except Exception as e:
                print(f"❌ Error {fname}: {e}")

# Simpan hasil
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"[✓] Hasil RS Analysis disimpan ke: {output_csv}")
