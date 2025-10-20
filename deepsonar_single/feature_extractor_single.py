from tqdm import tqdm
import os, numpy as np, torch, torchaudio

try:
    from .backbone import SRBackbone
    from .behavior import pack_features
except ImportError:  # pragma: no cover - fallback for standalone execution
    from backbone import SRBackbone
    from behavior import pack_features
    
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def extract_features(fake_root_dir, real_root_dir, feat_save_dir,
                     max_length=10.0, sr=16000, device="cuda", dataset_size=None, 
                     extract_feats=False):
    """
    Extracts DeepSonar (Wav2Vec2-based) features from fake and real audio files, 
    processes them using SRBackbone hooks, aggregates layer statistics, and 
    saves them as a compressed .npz file.

    Parameters:
        fake_root_dir (str): Directory containing fake (synthetic) audio files.
        real_root_dir (str): Directory containing real (genuine) audio files.
        feat_save_dir (str): Directory where extracted features will be saved.
        max_length (float): Maximum audio duration (in seconds) per sample (default: 10.0).
        sr (int): Target sample rate for resampling (default: 16000 Hz).
        device (str): Device for model inference ("cuda" or "cpu").
        dataset_size (int or None): Optional limit on number of files per class.
        extract_feats (bool): If False and cached features exist, loads from file instead of recomputing.

    Returns:
        tuple: (deepsonar_features, labels)
            deepsonar_features (np.ndarray): Extracted feature vectors, shape [N, T].
            labels (np.ndarray): Corresponding binary labels (1=fake, 0=real), shape [N].
    """
    if feat_save_dir is not None:
        os.makedirs(feat_save_dir, exist_ok=True)  # Ensure output directory exists
        feat_file = os.path.join(feat_save_dir, "features_single.npz")
    else:
        feat_file = None

    # ✅ Load precomputed features if available
    if not extract_feats and os.path.exists(feat_file):
        print(f"Loading cached deepsonar_features from {feat_file}")
        data = np.load(feat_file, allow_pickle=True)
        return data["deepsonar_features"], data["labels"]

    # Initialize pretrained backbone for DeepSonar feature extraction
    bb = SRBackbone(device=device)
    deepsonar_features, labels = [], []

    def process_dir(root_dir, label):
        """Process all .wav files in the directory and extract DeepSonar features."""
        files = [f for f in os.listdir(root_dir) if f.endswith(".wav")]
        if dataset_size is not None:
            files = files[:dataset_size]  # Limit dataset size if specified

        for file in tqdm(files, desc=f"Processing {root_dir}", unit="file"):
            file_path = os.path.join(root_dir, file)
            try:
                wav, sr_ = torchaudio.load(file_path)  # Load waveform and sample rate
                if sr_ != sr:
                    wav = torchaudio.functional.resample(wav, sr_, sr)  # Resample if needed

                if wav.ndim == 2:
                    wav = wav.mean(0)  # Convert stereo → mono
                wav = wav.squeeze()

                T = int(max_length * sr)  # Desired total samples per clip
                if wav.numel() > T:
                    wav = wav[:T]  # Trim
                else:
                    wav = torch.nn.functional.pad(wav, (0, T - wav.numel()))  # Pad to fixed length

                # Extract intermediate layer outputs and aggregate features
                feats = bb.forward_with_hooks(wav.to(device))
                fvec = pack_features(feats)

                # Store features and labels if extraction succeeded
                if fvec is not None:
                    deepsonar_features.append(fvec.cpu().numpy())
                    labels.append(label)

            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    # Process fake and real directories separately
    print("Extracting features from fake samples")
    process_dir(fake_root_dir, 1)
    print("Extracting features from real samples")
    process_dir(real_root_dir, 0)

    # Convert to NumPy arrays for saving
    deepsonar_features = np.array(deepsonar_features)  # shape: [N, T]
    labels = np.array(labels)  # shape: [N]

    # Save all extracted features and labels in compressed format
    if feat_file is not None:
        np.savez_compressed(
            feat_file,
            deepsonar_features=deepsonar_features,
            labels=labels
        )
        print(f"Saved features to {feat_file}")

    return deepsonar_features, labels  # Return extracted features and labels
