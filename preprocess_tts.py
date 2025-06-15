
import os
import numpy as np
import librosa

DATA_DIR = "data_neural"
WAV_DIR = os.path.join(DATA_DIR, "wavs")
OUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def extract(wav_path):
    y, _ = librosa.load(wav_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=80, hop_length=256)
    mel_db = librosa.power_to_db(mel).T
    f0, _, _ = librosa.pyin(y, sr=16000, fmin=75, fmax=300, hop_length=256)
    pitch = np.nan_to_num(f0)[:mel_db.shape[0]]
    energy = np.array([np.sqrt(np.mean(y[i*256:(i+1)*256]**2)) for i in range(mel_db.shape[0])])
    return mel_db, pitch, energy

with open(os.path.join(DATA_DIR, "metadata.csv")) as f:
    lines = f.readlines()

for line in lines:
    uid, text = line.strip().split("|")
    mel, pitch, energy = extract(os.path.join(WAV_DIR, f"{uid}.wav"))
    np.save(os.path.join(OUT_DIR, f"{uid}.npy"), {
        "mel": mel,
        "pitch": pitch,
        "energy": energy,
        "duration": np.ones(len(text.split()), dtype=np.int32),
        "text": text
    })

print("✅ Preprocessing selesai.")
