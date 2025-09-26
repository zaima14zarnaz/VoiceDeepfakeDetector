from tqdm import tqdm
import os, numpy as np, torch, torchaudio
from tqdm import tqdm

from backbone import SRBackbone

def extract_features_deepsonar(fake_root_dir, real_root_dir, feat_save_dir,
                               max_length=10.0, sr=16000, device="cuda", dataset_size=None, 
                               extract_feats=False):
    os.makedirs(feat_save_dir, exist_ok=True)
    feat_file = os.path.join(feat_save_dir, "features_labels.npz")

    # ✅ If already saved, just load
    if not extract_feats and os.path.exists(feat_file):
        print(f"Loading cached features from {feat_file}")
        data = np.load(feat_file, allow_pickle=True)
        return data["features"], data["labels"]

    # otherwise extract
    bb = SRBackbone(device=device)
    features, labels = [], []

    def process_dir(root_dir, label):
        files = [f for f in os.listdir(root_dir) if f.endswith(".wav")]
        if dataset_size is not None:
            files = files[:dataset_size]
        for file in tqdm(files, desc=f"Processing {root_dir}", unit="file"):
            file_path = os.path.join(root_dir, file)
            try:
                wav, sr_ = torchaudio.load(file_path)
                if sr_ != sr:
                    wav = torchaudio.functional.resample(wav, sr_, sr)
                if wav.ndim == 2:
                    if wav.size(0) > 1:
                        wav = wav.mean(0)
                    else:
                        wav = wav.squeeze(0)
                wav = wav.squeeze()
                T = int(max_length * sr)
                if wav.numel() > T:
                    wav = wav[:T]
                else:
                    wav = torch.nn.functional.pad(wav, (0, T - wav.numel()))

                feats = bb.forward_with_hooks(wav.to(device))
                fvec = pack_features(feats)
                if fvec is not None:
                    features.append(fvec.cpu().numpy())
                    labels.append(label)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    print('Extracting features from fake samples')
    process_dir(fake_root_dir, 1)
    print('Extracting features from real samples')
    process_dir(real_root_dir, 0)

    features = np.array(features)
    labels = np.array(labels)

    # ✅ Save once
    np.savez_compressed(feat_file, features=features, labels=labels)
    print(f"Saved features to {feat_file}")

    return features, labels