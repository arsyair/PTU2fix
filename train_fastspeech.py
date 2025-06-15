import os, torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==== Tokenizer ====
class SimpleTokenizer:
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz ")
        self.token2id = {c: i + 1 for i, c in enumerate(self.chars)}
        self.token2id["<pad>"] = 0
    def encode(self, text):
        return [self.token2id.get(c, 0) for c in text.lower()]

# ==== Dataset ====
class TTSDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")]
        self.tokenizer = tokenizer
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        data = np.load(self.files[i], allow_pickle=True).item()
        x = torch.tensor(self.tokenizer.encode(data['text']), dtype=torch.long)
        y = torch.tensor(data['mel'], dtype=torch.float32)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    return pad_sequence(xs, batch_first=True).long(), pad_sequence(ys, batch_first=True).float()

# ==== FastSpeech Mini with Duration Predictor ====
class MiniFastSpeech(nn.Module):
    def __init__(self, vocab_size, mel_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        self.duration_proj = nn.Linear(128, 1)
        self.lstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(512, mel_dim)

    def forward(self, x):
        x_embed = self.embed(x)                    # (B, T, 128)
        durations = torch.relu(self.duration_proj(x_embed)).squeeze(-1).int() + 1  # (B, T)

        expanded = []
        for b in range(x.size(0)):
            tokens = []
            for t in range(x.size(1)):
                repeat = durations[b, t].item()
                tokens.append(x_embed[b, t].unsqueeze(0).repeat(repeat, 1))  # (repeat, 128)
            expanded.append(torch.cat(tokens, dim=0))  # (total_repeat, 128)

        x_padded = pad_sequence(expanded, batch_first=True)  # (B, T', 128)
        out, _ = self.lstm(x_padded)
        return self.linear(out)  # (B, T', 80)

# ==== Training ====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer()
    ds = TTSDataset("data_neural/processed", tokenizer)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = MiniFastSpeech(len(tokenizer.token2id), mel_dim=80).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            if y_pred.size(1) > y.size(1):
                y_pred = y_pred[:, :y.size(1), :]
            else:
                y = y[:, :y_pred.size(1), :]
            # crop target agar panjang cocok
            loss = loss_fn(y_pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "mini_fastspeech.pth")
    print("✅ Model disimpan.")
