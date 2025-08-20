import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2
from tqdm import tqdm

# Konfigurasi
embedded_base_dir = "/home/dandi/smt8/paperstegano/data/embedded"
channels = ['R', 'G', 'B', 'RGB']
rates = [0, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]
threshold_p = 0.5  
output_csv = "/home/dandi/smt8/paperstegano/results/chi_square/results_chisquare_detection.csv"

# Fungsi chi-square untuk satu channel
def chi_square_from_channel(flat_channel):
    hist = np.bincount(flat_channel, minlength=256)
    chi_val = 0
    dof = 0
    for i in range(0, 256, 2):
        x = hist[i]
        y = hist[i + 1]
        n = x + y
        if n <= 4:
            continue
        z = n / 2
        chi_val += ((x - z) ** 2) / z + ((y - z) ** 2) / z
        dof += 1
    return chi_val, dof

# Jalankan eksperimen
results = []

for ch in channels:
    for rate in rates:
        img_dir = os.path.join(embedded_base_dir, ch, str(rate))
        print(f"[→] Processing channel={ch}, rate={rate}%")
        for fname in tqdm(sorted(os.listdir(img_dir))):
            img_path = os.path.join(img_dir, fname)
            try:
                start_time = time.time()

                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)

                chi_total = 0
                dof_total = 0

                if ch == 'RGB':
                    for c in range(3):
                        chi_val, dof = chi_square_from_channel(arr[:, :, c].flatten())
                        chi_total += chi_val
                        dof_total += dof
                else:
                    idx = {'R': 0, 'G': 1, 'B': 2}[ch]
                    chi_total, dof_total = chi_square_from_channel(arr[:, :, idx].flatten())

                p_value = chi2.sf(chi_total, dof_total) if dof_total > 0 else 0
                detected = 1 if p_value >= threshold_p else 0
                label = 1 if rate > 0 else 0

                elapsed = time.time() - start_time

                results.append({
                    'filename': fname,
                    'channel': ch,
                    'rate': rate,
                    'label': label,
                    'detected': detected,
                    'p_value': p_value,
                    'chi2': chi_total,
                    'dof': dof_total,
                    'time_sec': elapsed
                })

            except Exception as e:
                print(f"❌ Error {fname}: {e}")

# Simpan ke CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"[✓] Hasil chi-square + waktu disimpan ke: {output_csv}")
