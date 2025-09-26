from tqdm import tqdm
import os, numpy as np, torch, torchaudio
from tqdm import tqdm

from backbone import SRBackbone
from behavior import pack_features
import librosa

def extract_audio_features(wav, sr=16000, n_mfcc=40, max_len=500):
    """
    Extract MFCC features from raw waveform tensor.
    Input:
        wav: torch.Tensor, shape [time]
        sr: sample rate
        n_mfcc: number of MFCC coefficients
        max_len: pad/trim length along time dimension
    Returns:
        torch.Tensor of shape [n_mfcc, max_len]
    """
    wav_np = wav.cpu().numpy()
    mfccs = librosa.feature.mfcc(y=wav_np, sr=sr, n_mfcc=n_mfcc)

    # pad or trim to fixed length
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode="constant")
    else:
        mfccs = mfccs[:, :max_len]

    return torch.tensor(mfccs, dtype=torch.float32)



def extract_features(fake_root_dir, real_root_dir, feat_save_dir,
                               max_length=10.0, sr=16000, device="cuda", dataset_size=None, 
                               extract_feats=False):
    os.makedirs(feat_save_dir, exist_ok=True)
    feat_file = os.path.join(feat_save_dir, "deepsonar_features_labels.npz")

    # ✅ If already saved, just load
    if not extract_feats and os.path.exists(feat_file):
        print(f"Loading cached deepsonar_features from {feat_file}")
        data = np.load(feat_file, allow_pickle=True)
        return data["deepsonar_features"], data["audio_features"],  data["labels"]

    # otherwise extract
    bb = SRBackbone(device=device)
    deepsonar_features, audio_features, labels = [], [], []

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

                audio_feats = extract_audio_features(wav, sr=sr)

                if fvec is not None:
                    deepsonar_features.append(fvec.cpu().numpy())
                    audio_features.append(audio_feats.cpu().numpy())
                    labels.append(label)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    print("Extracting features from fake samples")
    process_dir(fake_root_dir, 1)
    print("Extracting features from real samples")
    process_dir(real_root_dir, 0)

    deepsonar_features = np.array(deepsonar_features)
    audio_features = np.array(audio_features)
    labels = np.array(labels)

    # ✅ Save
    np.savez_compressed(
        feat_file,
        deepsonar_features=deepsonar_features,
        audio_features=audio_features,
        labels=labels
    )
    print(f"Saved features to {feat_file}")

    return deepsonar_features, audio_features, labels