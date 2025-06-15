
import os
import numpy as np
from scipy.io.wavfile import read, write
import random

SOURCE_DIR = "data/train"
OUT_WAV_DIR = "data_neural/wavs"
OUT_META = "data_neural/metadata.csv"
SAMPLE_RATE = 16000

os.makedirs(OUT_WAV_DIR, exist_ok=True)

kalimat_list = [
    ["saya", "belajar"],
    ["saya", "mandi"],
    ["saya", "makan"],
    ["saya", "tidur"],
    ["saya", "olahraga"],
    ["saya", "masak"],
    ["belajar", "mandi", "tidur"]
]

def get_random_file(kata):
    folder = os.path.join(SOURCE_DIR, kata)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    return os.path.join(folder, random.choice(files)) if files else None

def gabungkan_audio(kata_list, output_path):
    full_audio = np.array([], dtype=np.int16)
    for kata in kata_list:
        if kata == "saya":
            silence = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.int16)
            full_audio = np.concatenate((full_audio, silence))
            continue
        path = get_random_file(kata)
        if not path: continue
        sr, audio = read(path)
        full_audio = np.concatenate((full_audio, audio, np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.int16)))
    write(output_path, SAMPLE_RATE, full_audio)

metadata = []
for i, kata_list in enumerate(kalimat_list, start=1):
    fname = f"{i:03d}.wav"
    gabungkan_audio(kata_list, os.path.join(OUT_WAV_DIR, fname))
    metadata.append(f"{fname.replace('.wav','')}|{' '.join(kata_list)}")

with open(OUT_META, "w") as f:
    f.write("\n".join(metadata))

print("✅ Dataset SPSS (gabungan) selesai dibuat.")
