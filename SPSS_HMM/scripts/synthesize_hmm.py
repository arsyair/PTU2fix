import os
import pickle
import numpy as np
import librosa
import soundfile as sf
from hmmlearn import hmm

MODEL_DIR = "../models"
SAMPLE_RATE = 22050

def generate_mfcc_from_hmm(model, n_frames=50):
    X, _ = model.sample(n_frames)
    return X.T

def synthesize_word(word):
    model_path = os.path.join(MODEL_DIR, f"{word}.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"Model untuk kata '{word}' tidak ditemukan.")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Generate MFCC from HMM
    mfcc = generate_mfcc_from_hmm(model, n_frames=30)

    # Approximate inversion (bukan 100% akurat)
    # Buat spectrogram dummy dari MFCC
    # 1. Normalisasi (optional)
    mfcc = mfcc.T  # [n_mfcc x time]
    mfcc = mfcc[:13, :]  # pastikan 13 koefisien

    # 2. Gunakan pseudo-inverse: MFCC → log-mel → linear → waveform (pakai Griffin-Lim)
    mel_spec = librosa.feature.inverse.mfcc_to_mel(mfcc)
    S = librosa.feature.inverse.mel_to_stft(mel_spec)
    y = librosa.griffinlim(S, hop_length=256, win_length=1024)

    return y


def synthesize_sentence(text, output_file="output_hmm.wav"):
    words = text.strip().lower().split()
    final_audio = np.array([])

    for word in words:
        try:
            audio = synthesize_word(word)
            final_audio = np.concatenate((final_audio, audio, np.zeros(int(0.15 * SAMPLE_RATE))))
        except Exception as e:
            print(f"Gagal menghasilkan kata '{word}': {e}")

    if len(final_audio) > 0:
        sf.write(output_file, final_audio, SAMPLE_RATE)
        print(f"✅ Output disimpan di: {output_file}")
    else:
        print("❌ Tidak ada audio yang dihasilkan.")

if __name__ == "__main__":
    text = input("Masukkan kalimat:\n> ")
    synthesize_sentence(text)