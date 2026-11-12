"""Sıfırdan CNN için log-mel spektrogram çıkarımı (önceden eğitilmiş model YOK).

Giriş: 16 kHz mono 30 sn'lik pencereler (n, örnek). Çıktı: (n, N_MELS, kare) float16 log-mel dB,
yani pencere başına 64 x 938. pipeline.py ile aynı arayüz (sr, windows) kullanılır, böylece
indirme/çözme hattı paylaşılır:

    python src/pipeline.py --models mel --max-windows 3 --out-dir /kaggle/working/emb_mel ...
"""
import numpy as np
import torch

N_MELS = 64
N_FFT = 1024
HOP = 512


class MelEmbedder:
    sr = 16000
    dim = N_MELS

    def __init__(self, device: str, batch: int = 16):
        import torchaudio

        self.device = device
        self.batch = batch
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, n_fft=N_FFT, hop_length=HOP, f_min=20.0, f_max=8000.0,
            n_mels=N_MELS, power=2.0,
        ).to(device)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(device)

    @torch.no_grad()
    def windows(self, windows16k: np.ndarray) -> np.ndarray:
        """(n, örnek) -> (n, 64, kare) float16 log-mel (dB, pencere başına tepe değerinden en fazla 80 dB aşağı)."""
        parts = []
        for i in range(0, len(windows16k), self.batch):
            x = torch.from_numpy(np.ascontiguousarray(windows16k[i:i + self.batch])).to(self.device)
            m = self.db(self.mel(x))
            parts.append(m.cpu().numpy().astype(np.float16))
        return np.concatenate(parts)

    def __call__(self, windows16k: np.ndarray) -> np.ndarray:
        return self.windows(windows16k).mean(0)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = MelEmbedder(dev).windows(np.random.randn(2, 30 * 16000).astype(np.float32) * 0.1)
    print("mel boyutu:", out.shape, out.dtype)
