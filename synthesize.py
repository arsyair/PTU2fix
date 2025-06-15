import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from train_fastspeech import MiniFastSpeech, SimpleTokenizer

# ==== Load model ====
model = MiniFastSpeech(28, mel_dim=80)
model.load_state_dict(torch.load("mini_fastspeech.pth", map_location="cpu"))
model.eval()

tokenizer = SimpleTokenizer()

# ==== Synthesize ====
def synthesize(text, output_path="output.wav"):
    tokens = tokenizer.encode(text)
    x = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        mel = model(x).squeeze(0).T  # (80, T)
        mel = mel.numpy()

        # FIX: amplifikasi output model
        mel = mel * 2.5
        mel = np.clip(mel, -30, 0)


        # Konversi ke power dan waveform
        mel_power = librosa.db_to_power(mel)
        audio = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=16000,
            hop_length=256,
            n_fft=1024,
            win_length=1024,
            window="hann",
            n_iter=200
        )

        # Normalisasi audio
        audio = audio / np.max(np.abs(audio))
        sf.write(output_path, audio, 16000)

        print(f"[✓] Suara disimpan: {output_path}")
        print(f"Mel shape: {mel.shape}, Mean: {mel.mean():.4f}")

        # Visualisasi
        plt.imshow(mel, origin="lower", aspect="auto")
        plt.title("Mel Output dari Model (Amplified)")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

# ==== Main ====
if __name__ == "__main__":
    teks = input("Masukkan kalimat:\n> ")
    synthesize(teks)
