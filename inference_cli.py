#!/usr/bin/env python3
"""Command line interface for DeepSonar voice deepfake detection.

This script loads the trained checkpoints found in `ckpt/` and runs inference
on a provided WAV file using the single-feature and/or multi-feature DeepSonar
detectors. The implementation mirrors the feature extraction steps used during
training so that predictions are consistent with the saved models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inference_core import (
    LabelMap,
    DEFAULT_MULTI_CKPT,
    DEFAULT_SINGLE_CKPT,
    Prediction,
    SRBackbone,
    load_waveform,
    resolve_device,
    run_multi_detector,
    run_single_detector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSonar inference on a WAV audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the WAV file to analyse.",
    )

    parser.add_argument(
        "--model",
        choices=["single", "multi", "both"],
        default="both",
        help="Which detector(s) to run.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device for feature extraction and inference.",
    )

    parser.add_argument(
        "--single-ckpt",
        type=Path,
        default=DEFAULT_SINGLE_CKPT,
        help="Path to the single-feature detector checkpoint.",
    )

    parser.add_argument(
        "--multi-ckpt",
        type=Path,
        default=DEFAULT_MULTI_CKPT,
        help="Path to the multi-feature detector checkpoint.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for the audio pipeline.",
    )

    parser.add_argument(
        "--max-length",
        type=float,
        default=10.0,
        help="Maximum audio length (seconds). Audio longer than this is clipped; shorter audio is padded with zeros.",
    )

    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients for the multi-feature detector.",
    )

    parser.add_argument(
        "--mfcc-max-frames",
        type=int,
        default=500,
        help="Maximum number of MFCC frames for the multi-feature detector.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed diagnostics.",
    )

    return parser.parse_args()


def format_prediction(pred: Prediction, verbose: bool = False) -> str:
    prob_real, prob_fake = pred.probabilities
    lines = [
        f"[{pred.model_name}] prediction: {pred.predicted_label.upper()}",
        f"    real: {prob_real:.4f}",
        f"    fake: {prob_fake:.4f}",
    ]
    if verbose:
        lines.append(f"    raw_probs: {pred.probabilities.tolist()}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.verbose:
        print(f"Using device: {device}")

    # Ensure preprocessing matches training (sample rate, padding, etc.).
    waveform = load_waveform(
        Path(args.audio),
        target_sr=args.sample_rate,
        max_length_sec=args.max_length,
    )

    from inference_core import SRBackbone

    # Shared backbone extracts DeepSonar features for both detectors.
    backbone = SRBackbone(device=str(device))

    predictions = []
    if args.model in {"single", "both"}:
        predictions.append(
            run_single_detector(
                waveform=waveform,
                backbone=backbone,
                ckpt_path=args.single_ckpt,
                device=device,
            )
        )

    if args.model in {"multi", "both"}:
        predictions.append(
            run_multi_detector(
                waveform=waveform,
                backbone=backbone,
                ckpt_path=args.multi_ckpt,
                device=device,
                n_mfcc=args.n_mfcc,
                max_frames=args.mfcc_max_frames,
            )
        )

    for pred in predictions:
        print(format_prediction(pred, verbose=args.verbose))


if __name__ == "__main__":
    main()


