"""Shared inference utilities for DeepSonar detectors.

This module consolidates the logic used by the CLI and the Streamlit app so
that both interfaces produce consistent predictions when loading the pretrained
checkpoints shipped in `ckpt/`.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(BASE_DIR / ".torch_cache"))

# Cache torchaudio weights locally and relax SSL verification for downloads.
ssl._create_default_https_context = ssl._create_unverified_context


from deepsonar_multi.backbone import SRBackbone
from deepsonar_multi.behavior import pack_features as pack_features_multi
from deepsonar_multi.feature_extractor_multi import extract_audio_features
from deepsonar_multi.model import Detector_Multi_Feat

from deepsonar_single.behavior import pack_features as pack_features_single
from deepsonar_single.model import Detector_Single_Feat


DEFAULT_MULTI_CKPT = BASE_DIR / "ckpt" / "best_multi.pth"
DEFAULT_SINGLE_CKPT = BASE_DIR / "ckpt" / "best_single.pth"
DEFAULT_SAMPLE_RATE = 16000

LabelMap = Dict[int, str]
LABEL_MAP: LabelMap = {0: "real", 1: "fake"}


@dataclass
class Prediction:
    model_name: str
    label_map: LabelMap
    probabilities: np.ndarray

    @property
    def predicted_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def predicted_label(self) -> str:
        return self.label_map[self.predicted_index]


def resolve_device(user_choice: str) -> torch.device:
    """Return best-fit torch device based on the user's preference."""
    if user_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if user_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    return torch.device(user_choice)


def load_waveform(
    audio_path: Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    max_length_sec: float = 10.0,
) -> torch.Tensor:
    """Load, resample, and pad an audio file to the model's expected format."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except (ImportError, OSError) as err:
        if isinstance(err, ImportError) and "torchcodec" not in str(err).lower():
            raise
        # Fall back to librosa when torchaudio backend is unavailable.
        waveform_np, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        waveform = torch.from_numpy(waveform_np)

    if waveform.ndim == 2:
        waveform = waveform.mean(0)
    waveform = waveform.squeeze().float()

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if waveform.numel() == 0:
        raise ValueError("Audio file is empty after loading.")

    max_length_samples = int(max_length_sec * target_sr)
    if waveform.numel() > max_length_samples:
        waveform = waveform[:max_length_samples]
    else:
        waveform = F.pad(waveform, (0, max_length_samples - waveform.numel()))

    return waveform


def compute_deepsonar_features(
    backbone: SRBackbone,
    waveform: torch.Tensor,
    pack_fn,
    device: torch.device,
) -> torch.Tensor:
    """Extract DeepSonar backbone activations and flatten them into feature vectors."""
    backbone.reset_hooks()
    with torch.no_grad():
        feats = backbone.forward_with_hooks(waveform.to(device))
        packed = pack_fn(feats)
    if packed is None:
        raise RuntimeError("Failed to generate DeepSonar features from backbone.")
    # Flatten batch features to match classifier expectations.
    return packed.view(packed.size(0), -1).cpu()


def _align_feature_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Trim or zero-pad features so they match the checkpoint's expected dimension."""
    current_dim = tensor.size(1)
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        return tensor[:, :target_dim]
    # Pad with zeros when extracted features are shorter than checkpoint.
    padding = torch.zeros(tensor.size(0), target_dim - current_dim, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=1)


def run_single_detector(
    waveform: torch.Tensor,
    backbone: SRBackbone,
    ckpt_path: Path,
    device: torch.device,
) -> Prediction:
    """Run the single-feature DeepSonar classifier and return prediction scores."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Single detector checkpoint not found: {ckpt_path}")

    features = compute_deepsonar_features(backbone, waveform, pack_features_single, device)
    state = torch.load(str(ckpt_path), map_location=device)

    dim1 = state["net.0.weight"].shape[0]
    features = _align_feature_dim(features, dim1)

    model = Detector_Single_Feat(dim1=dim1)
    model.load_state_dict(state)
    model.to(device).eval()

    with torch.no_grad():
        logits = model(features.to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return Prediction(model_name="single", label_map=LABEL_MAP, probabilities=probs)


def run_multi_detector(
    waveform: torch.Tensor,
    backbone: SRBackbone,
    ckpt_path: Path,
    device: torch.device,
    n_mfcc: int = 40,
    max_frames: int = 500,
    mfcc_sample_rate: int | None = None,
) -> Prediction:
    """Run the multi-feature detector combining DeepSonar and MFCC embeddings."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Multi detector checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device)

    features = compute_deepsonar_features(backbone, waveform, pack_features_multi, device)
    features = features.view(1, -1)

    sr = mfcc_sample_rate or backbone.sample_rate
    audio_feats = extract_audio_features(waveform, sr=sr, n_mfcc=n_mfcc, max_len=max_frames)
    audio_feats = audio_feats.view(1, -1)

    dim1 = state["cross_att.proj1.weight"].shape[1]
    dim2 = state["cross_att.proj2.weight"].shape[1]

    features = _align_feature_dim(features, dim1)
    audio_feats = _align_feature_dim(audio_feats, dim2)

    model = Detector_Multi_Feat(dim1=dim1, dim2=dim2)
    model.load_state_dict(state)
    model.to(device).eval()

    with torch.no_grad():
        logits = model(features.to(device), audio_feats.to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return Prediction(model_name="multi", label_map=LABEL_MAP, probabilities=probs)


def run_detectors(
    waveform: torch.Tensor,
    backbone: SRBackbone,
    device: torch.device,
    models: Sequence[str],
    single_ckpt: Path = DEFAULT_SINGLE_CKPT,
    multi_ckpt: Path = DEFAULT_MULTI_CKPT,
    n_mfcc: int = 40,
    max_frames: int = 500,
    mfcc_sample_rate: int | None = None,
) -> List[Prediction]:
    """Execute the requested detectors and return a list of their predictions."""
    predictions: List[Prediction] = []

    if "single" in models:
        predictions.append(
            run_single_detector(
                waveform=waveform,
                backbone=backbone,
                ckpt_path=single_ckpt,
                device=device,
            )
        )

    if "multi" in models:
        predictions.append(
            run_multi_detector(
                waveform=waveform,
                backbone=backbone,
                ckpt_path=multi_ckpt,
                device=device,
                n_mfcc=n_mfcc,
                max_frames=max_frames,
                mfcc_sample_rate=mfcc_sample_rate,
            )
        )

    return predictions


__all__ = [
    "DEFAULT_MULTI_CKPT",
    "DEFAULT_SINGLE_CKPT",
    "DEFAULT_SAMPLE_RATE",
    "LABEL_MAP",
    "Prediction",
    "resolve_device",
    "load_waveform",
    "compute_deepsonar_features",
    "run_single_detector",
    "run_multi_detector",
    "run_detectors",
]


