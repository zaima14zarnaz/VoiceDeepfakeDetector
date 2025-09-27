import os, argparse
import torch, torchaudio
import numpy as np

from deepsonar.feature_extractor import extract_audio_features
from deepsonar.backbone import SRBackbone
from deepsonar.behavior import pack_features
from deepsonar.model import Detector


def load_audio(path: str, target_sr: int = 16000, max_length_seconds: float = 10.0) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # mono
    if wav.ndim == 2:
        if wav.size(0) > 1:
            wav = wav.mean(0)
        else:
            wav = wav.squeeze(0)
    wav = wav.squeeze()
    # pad/trim
    T = int(max_length_seconds * target_sr)
    if wav.numel() > T:
        wav = wav[:T]
    else:
        wav = torch.nn.functional.pad(wav, (0, T - wav.numel()))
    return wav


@torch.no_grad()
def infer_one(wav_path: str, checkpoint_path: str, device: str = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Extract features exactly like training
    bb = SRBackbone(device=device)
    wav = load_audio(wav_path, target_sr=16000, max_length_seconds=10.0)
    feats = bb.forward_with_hooks(wav.to(device))
    x1 = pack_features(feats)  # tensor [1, D1]
    if x1 is None:
        raise RuntimeError("Could not extract DeepSonar features from the backbone.")
    x1 = x1.float().to(device)

    # MFCC features to match training
    x2 = extract_audio_features(wav, sr=16000)  # [n_mfcc, max_len]
    x2 = x2.flatten().unsqueeze(0).to(device)   # [1, D2]

    # 2) Build model with matching input dims
    dim1 = x1.shape[-1]
    dim2 = x2.shape[-1]
    model = Detector(dim1=dim1, dim2=dim2).to(device)

    # 3) Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    # Support both pure state_dict and wrapped
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()) and not any(k.startswith("state_dict") for k in state.keys()):
        model.load_state_dict(state)
    else:
        model.load_state_dict(state.get("state_dict", state))
    model.eval()

    # 4) Forward
    logits = model(x1, x2)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(torch.argmax(probs).item())
    return {
        "pred": pred,              # 0 = real, 1 = fake (as used in training)
        "prob_real": float(probs[0].item()),
        "prob_fake": float(probs[1].item()),
        "dim1": dim1,
        "dim2": dim2,
        "device": device
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default="/home/hieuminh65/VoiceDeepfakeDetector/test.wav")
    parser.add_argument("--ckpt", type=str, default="/home/hieuminh65/VoiceDeepfakeDetector/deepsonar/best.pth")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    result = infer_one(args.wav, args.ckpt, device=args.device)
    label = "fake" if result["pred"] == 1 else "real"
    print(f"Prediction: {label} | prob_real={result['prob_real']:.4f} prob_fake={result['prob_fake']:.4f} | dims=({result['dim1']},{result['dim2']}) | device={result['device']}")


if __name__ == "__main__":
    main()


