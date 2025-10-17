import os, torch, torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model.behavior import pack_features

class DeepSonarDataset(Dataset):
    def __init__(self, fake_dir, real_dir, bb, max_length=10.0, sr=16000, device="cuda"):
        self.files = []
        self.labels = []
        self.bb = bb
        self.max_length = max_length
        self.sr = sr
        self.device = device

        for f in os.listdir(fake_dir):
            if f.endswith(".wav"):
                self.files.append(os.path.join(fake_dir, f))
                self.labels.append(1)
        for f in os.listdir(real_dir):
            if f.endswith(".wav"):
                self.files.append(os.path.join(real_dir, f))
                self.labels.append(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        try:
            wav, sr_ = torchaudio.load(file_path)
            if sr_ != self.sr:
                wav = torchaudio.functional.resample(wav, sr_, self.sr)
            if wav.ndim == 2:
                if wav.size(0) > 1:
                    wav = wav.mean(0)
                else:
                    wav = wav.squeeze(0)
            wav = wav.squeeze()
            T = int(self.max_length * self.sr)
            if wav.numel() > T:
                wav = wav[:T]
            else:
                wav = torch.nn.functional.pad(wav, (0, T - wav.numel()))

            # forward pass backbone
            feats = self.bb.forward_with_hooks(wav.to(self.device))
            fvec = pack_features(feats)

            if fvec is None:
                raise RuntimeError("No features extracted")

            return fvec.cpu().float(), torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            # return dummy
            return torch.zeros(1,3), torch.tensor(label, dtype=torch.long)
