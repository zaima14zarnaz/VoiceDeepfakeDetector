"""Streamlit interface for DeepSonar voice deepfake detection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Force protobuf to use the pure-Python runtime; prevents descriptor crashes.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import streamlit as st

from inference_core import (
    DEFAULT_MULTI_CKPT,
    DEFAULT_SINGLE_CKPT,
    Prediction,
    load_waveform,
    resolve_device,
    run_detectors,
    SRBackbone,
)


st.set_page_config(page_title="Voice Deepfake Detector", layout="centered")
st.title("🔍 Voice Deepfake Detector")
st.write(
    "Upload or record a WAV file to assess whether the speech is likely real or synthetically generated."
)


def load_audio_from_upload(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix or ".wav"
    # Write uploaded bytes to disk so torchaudio/librosa can reopen them.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return Path(tmp.name)


def render_prediction(prediction: Prediction):
    prob_real, prob_fake = prediction.probabilities
    st.subheader(f"Model: {prediction.model_name.capitalize()}")
    st.metric(
        label="Prediction",
        value="REAL" if prediction.predicted_label == "real" else "FAKE",
    )
    st.progress(float(prob_fake), text=f"Fake probability: {prob_fake:.2%}")
    st.caption(f"Real: {prob_real:.2%} • Fake: {prob_fake:.2%}")


with st.sidebar:
    st.header("Inference Settings")
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    max_length = st.slider("Max audio duration (seconds)", min_value=3.0, max_value=20.0, value=10.0, step=0.5)
    selected_models = st.multiselect(
        "Models",
        options=["single", "multi"],
        default=["single", "multi"],
    )
    n_mfcc = st.slider("# MFCC coefficients (multi)", 20, 80, 40, 5)
    mfcc_frames = st.slider("MFCC frames (multi)", 100, 800, 500, 50)

uploaded_audio = st.file_uploader("Upload WAV file", type=["wav", "flac", "mp3", "ogg"])

if uploaded_audio:
    st.audio(uploaded_audio)

if st.button("Run Detection"):
    if not uploaded_audio:
        st.warning("Please upload an audio file first.")
    else:
        with st.spinner("Analysing audio..."):
            temp_audio_path = load_audio_from_upload(uploaded_audio)
            try:
                waveform = load_waveform(
                    temp_audio_path,
                    max_length_sec=max_length,
                )
                device = resolve_device(device_choice)
                backbone = SRBackbone(device=str(device))
                predictions = run_detectors(
                    waveform=waveform,
                    backbone=backbone,
                    device=device,
                    models=selected_models,
                    single_ckpt=DEFAULT_SINGLE_CKPT,
                    multi_ckpt=DEFAULT_MULTI_CKPT,
                    n_mfcc=n_mfcc,
                    max_frames=mfcc_frames,
                )
            except Exception as err:  # pylint: disable=broad-except
                st.error(f"Inference failed: {err}")
            else:
                if not predictions:
                    st.info("No models selected.")
                else:
                    for prediction in predictions:
                        render_prediction(prediction)

        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink(missing_ok=True)


st.caption(
    "Weights are loaded from `ckpt/best_single.pth` and `ckpt/best_multi.pth`."
)


