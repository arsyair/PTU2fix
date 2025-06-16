import os
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data/train")
FEATURE_DIR = os.path.join(BASE_DIR, "../features")
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_and_save_mfcc(word_dir, word):
    output_path = os.path.join(FEATURE_DIR, word)
    os.makedirs(output_path, exist_ok=True)

    for fname in os.listdir(word_dir):
        if fname.endswith(".wav"):
            path = os.path.join(word_dir, fname)
            y, sr = librosa.load(path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            np.save(os.path.join(output_path, fname.replace(".wav", ".npy")), mfcc.T)

def process_all_words():
    for word in os.listdir(DATA_DIR):
        word_dir = os.path.join(DATA_DIR, word)
        if os.path.isdir(word_dir):
            extract_and_save_mfcc(word_dir, word)

if __name__ == "__main__":
    process_all_words()
