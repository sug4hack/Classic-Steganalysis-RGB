import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

original_dir = "/home/dandi/smt8/paperstegano/data/original_resized"
output_base_dir = "/home/dandi/smt8/paperstegano/data/embedded"

channels = ['R', 'G', 'B', 'RGB']
rates = [0, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100]

def embed_lsb(image_array, channel, rate):
    h, w, _ = image_array.shape
    total_pixels = h * w

    channels_map = {'R': 0, 'G': 1, 'B': 2}
    affected_channels = [channels_map[c] for c in channel] if channel != 'RGB' else [0, 1, 2]

    modified = image_array.copy()
    for ch in affected_channels:
        num_bits = int(total_pixels * rate / 100)
        indices = random.sample(range(total_pixels), num_bits)
        for idx in indices:
            y, x = divmod(idx, w)
            modified[y, x, ch] = (modified[y, x, ch] & ~1) | random.randint(0, 1)
    return modified

for ch in channels:
    for rate in rates:
        out_dir = os.path.join(output_base_dir, ch, str(rate))
        os.makedirs(out_dir, exist_ok=True)

image_files = sorted(os.listdir(original_dir))
for ch in channels:
    for rate in rates:
        print(f"[→] Embedding channel={ch} | rate={rate}%")
        out_dir = os.path.join(output_base_dir, ch, str(rate))
        for fname in tqdm(image_files):
            img_path = os.path.join(original_dir, fname)
            img = Image.open(img_path)
            arr = np.array(img)
            modified_arr = embed_lsb(arr, ch, rate)
            out_img = Image.fromarray(modified_arr)
            out_img.save(os.path.join(out_dir, fname))

print("[✓] Selesai melakukan embedding pada semua channel dan rate.")




