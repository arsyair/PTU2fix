import os
import pickle
import numpy as np
import soundfile as sf

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'output_hmm.wav')

def synthesize_word(word):
    model_path = os.path.join(MODEL_DIR, f"{word}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model untuk kata '{word}' tidak ditemukan.")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Akses parameter HMM (jika pakai hmmlearn)
    means = model.means_
    covars = model.covars_
    transmat = model.transmat_
    startprob = model.startprob_

    num_states = len(startprob)
    mfcc_dim = means.shape[1]

    # Pilih urutan state secara acak berdasarkan startprob dan transmat
    sequence = [np.random.choice(num_states, p=startprob)]
    for _ in range(29):
        prev = sequence[-1]
        sequence.append(np.random.choice(num_states, p=transmat[prev]))

    # Sampel MFCC dari distribusi Gaussian tiap state
    features = []
    for state in sequence:
        mean = means[state]
        cov = np.diag(covars[state]) if covars.ndim == 2 else covars[state]
        sample = np.random.multivariate_normal(mean, cov)
        features.append(sample)

    features = np.array(features)
    audio = mfcc_to_audio(features)
    return audio

def mfcc_to_audio(mfcc_features, sr=16000):
    # Versi dummy: sinyal sinusoidal dimodulasi oleh MFCC energi (koefisien pertama)
    t = np.linspace(0, len(mfcc_features)/100, len(mfcc_features) * int(sr / 100), endpoint=False)
    signal = np.sin(2 * np.pi * 220 * t) * np.repeat(mfcc_features[:, 0], int(sr / 100))
    return signal.astype(np.float32)

def synthesize_sentence(text, output_file=OUTPUT_FILE):
    words = text.strip().lower().split()
    final_audio = np.array([], dtype=np.float32)

    for word in words:
        try:
            audio = synthesize_word(word)
            final_audio = np.concatenate([final_audio, audio])
        except Exception as e:
            print(f"Gagal menghasilkan kata '{word}': {e}")

    if len(final_audio) > 0:
        sf.write(output_file, final_audio, 16000)
        print(f"✅ Output disimpan di: {output_file}")
    else:
        print("❌ Tidak ada audio yang dihasilkan.")

if __name__ == "__main__":
    text = input("Masukkan kalimat:\n> ")
    synthesize_sentence(text)
