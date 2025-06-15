
import os
import numpy as np
from scipy.io import wavfile
import random

DATA_DIR = "data/train"
KATA_UNIK = ["makan", "tidur", "mandi", "masak", "olahraga", "belajar"]
SAMPLE_RATE = 16000
JEDA_MS = 300

def text_to_speech(input_text, output_file="output_concat.wav"):
    words = input_text.lower().split()
    combined_audio = np.array([], dtype=np.int16)

    for word in words:
        if word not in KATA_UNIK:
            print(f"[!] Kata '{word}' tidak dikenali.")
            continue

        folder = os.path.join(DATA_DIR, word)
        samples = [f for f in os.listdir(folder) if f.endswith('.wav')]
        if not samples:
            continue

        chosen = random.choice(samples)
        path = os.path.join(folder, chosen)
        sr, audio = wavfile.read(path)
        if sr != SAMPLE_RATE:
            raise ValueError("Sample rate tidak cocok.")
        combined_audio = np.concatenate((combined_audio, audio, np.zeros(int(SAMPLE_RATE * JEDA_MS / 1000), dtype=np.int16)))

    wavfile.write(output_file, SAMPLE_RATE, combined_audio)
    print(f"[✓] Output disimpan: {output_file}")

if __name__ == "__main__":
    kalimat = input("Ketik kalimat:")
    text_to_speech(kalimat)
