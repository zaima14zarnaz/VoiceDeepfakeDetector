from tqdm import tqdm
import os, numpy as np, torch, torchaudio
from tqdm import tqdm
import librosa

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backbone import SRBackbone
from behavior import pack_features


def extract_audio_features(wav, sr=16000, n_mfcc=40, max_len=500):
    """
    Extracts Mel-Frequency Cepstral Coefficient (MFCC) features from a raw waveform.

    Parameters:
        wav (torch.Tensor): 1D waveform tensor, shape [time].
        sr (int): Sampling rate of the waveform (default: 16000 Hz).
        n_mfcc (int): Number of MFCC coefficients to extract (default: 40).
        max_len (int): Maximum number of time frames to keep (pad or trim if needed).

    Returns:
        torch.Tensor: MFCC feature tensor of shape [n_mfcc, max_len].
    """
    wav_np = wav.cpu().numpy()  # Convert to NumPy array for librosa processing

    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=wav_np, sr=sr, n_mfcc=n_mfcc)

    # Pad or trim MFCCs to fixed length along time axis
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode="constant")
    else:
        mfccs = mfccs[:, :max_len]

    return torch.tensor(mfccs, dtype=torch.float32)  # Convert back to PyTorch tensor


def extract_features(fake_root_dir, real_root_dir, feat_save_dir,
                     max_length=10.0, sr=16000, device="cuda", dataset_size=None, 
                     extract_feats=False):
    """
    Extracts both DeepSonar (Wav2Vec2-based) and audio (MFCC) features from fake and real audio files.
    Combines both feature types into aligned NumPy arrays and saves them to disk.

    Parameters:
        fake_root_dir (str): Directory containing fake (synthetic) .wav files.
        real_root_dir (str): Directory containing real (authentic) .wav files.
        feat_save_dir (str): Directory to save the extracted features.
        max_length (float): Maximum duration (in seconds) for each audio clip (default: 10.0).
        sr (int): Target sample rate for resampling audio (default: 16000 Hz).
        device (str): Device for running DeepSonar (default: "cuda").
        dataset_size (int or None): Optional limit on number of files to process per class.
        extract_feats (bool): If False and saved features exist, loads from cache instead.

    Returns:
        tuple:
            deepsonar_features (np.ndarray): DeepSonar feature vectors, shape [N, D1].
            audio_features (np.ndarray): MFCC feature arrays, shape [N, n_mfcc, max_len].
            labels (np.ndarray): Binary labels (1=fake, 0=real), shape [N].
    """
    if feat_save_dir is not None:
        os.makedirs(feat_save_dir, exist_ok=True)  # Create save directory if missing
        feat_file = os.path.join(feat_save_dir, "features_multi.npz")
    else:
        feat_file = None

    # ✅ Load pre-extracted features if available and extract_feats=False
    if not extract_feats and os.path.exists(feat_file):
        print(f"Loading cached deepsonar_features from {feat_file}")
        data = np.load(feat_file, allow_pickle=True)
        return data["deepsonar_features"], data["audio_features"], data["labels"]

    # Initialize DeepSonar backbone (Wav2Vec2 model)
    bb = SRBackbone(device=device)
    deepsonar_features, audio_features, labels = [], [], []

    def process_dir(root_dir, label):
        """Extracts DeepSonar + MFCC features for all .wav files in a directory."""
        files = [f for f in os.listdir(root_dir) if f.endswith(".wav")]
        if dataset_size is not None:
            files = files[:dataset_size]

        for file in tqdm(files, desc=f"Processing {root_dir}", unit="file"):
            file_path = os.path.join(root_dir, file)
            try:
                wav, sr_ = torchaudio.load(file_path)  # Load audio waveform and sample rate
                if sr_ != sr:
                    wav = torchaudio.functional.resample(wav, sr_, sr)  # Resample if needed

                # Convert to mono if stereo
                if wav.ndim == 2:
                    if wav.size(0) > 1:
                        wav = wav.mean(0)
                    else:
                        wav = wav.squeeze(0)
                wav = wav.squeeze()

                # Trim or pad to fixed max length
                T = int(max_length * sr)
                if wav.numel() > T:
                    wav = wav[:T]
                else:
                    wav = torch.nn.functional.pad(wav, (0, T - wav.numel()))

                # DeepSonar features (via model hooks)
                feats = bb.forward_with_hooks(wav.to(device))
                fvec = pack_features(feats)

                # Audio (MFCC) features
                audio_feats = extract_audio_features(wav, sr=sr)

                # Append only if valid feature extracted
                if fvec is not None:
                    deepsonar_features.append(fvec.cpu().numpy())
                    audio_features.append(audio_feats.cpu().numpy())
                    labels.append(label)

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    # Process both fake and real directories
    print("Extracting features from fake samples")
    process_dir(fake_root_dir, 1)
    print("Extracting features from real samples")
    process_dir(real_root_dir, 0)

    # Convert to NumPy arrays
    deepsonar_features = np.array(deepsonar_features)
    audio_features = np.array(audio_features)
    labels = np.array(labels)

    # Save features for reuse
    if feat_file is not None:
        np.savez_compressed(
            feat_file,
            deepsonar_features=deepsonar_features,
            audio_features=audio_features,
            labels=labels
        )
        print(f"Saved features to {feat_file}")

    return deepsonar_features, audio_features, labels
