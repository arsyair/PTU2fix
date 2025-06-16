import os
import numpy as np
import pickle
from hmmlearn import hmm

# ABSOLUTE path dari posisi file ini (SPSS_HMM/scripts/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(BASE_DIR, "../features")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_hmm_for_word(word):
    word_dir = os.path.join(FEATURE_DIR, word)
    X = []
    lengths = []

    for fname in os.listdir(word_dir):
        if fname.endswith(".npy"):
            mfcc = np.load(os.path.join(word_dir, fname))
            X.append(mfcc)
            lengths.append(len(mfcc))

    if not X:
        print(f"⚠️ Tidak ada fitur untuk kata: {word}")
        return

    X_concat = np.concatenate(X)
    model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
    model.fit(X_concat, lengths)

    with open(os.path.join(MODEL_DIR, f"{word}.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Trained model saved: {word}.pkl")

def train_all():
    if not os.path.exists(FEATURE_DIR):
        print(f"❌ Folder fitur tidak ditemukan: {FEATURE_DIR}")
        return

    for word in os.listdir(FEATURE_DIR):
        train_hmm_for_word(word)

if __name__ == "__main__":
    train_all()
