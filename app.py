import os
import tempfile
import streamlit as st
import torch

from deepsonar.infer import infer_one


def get_default_checkpoint_path() -> str:
    repo_root = os.path.dirname(__file__)
    return os.path.join(repo_root, "deepsonar", "best.pth")


def main():
    st.set_page_config(page_title="Voice Deepfake Detector", page_icon="🔊")
    st.title("Voice Deepfake Detector")
    st.write("Upload a WAV file to predict whether it is real or fake.")

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    col1, col2 = st.columns(2)
    with col1:
        device = st.selectbox("Device", options=[default_device, "cpu", "cuda"], index=0)
    with col2:
        ckpt_path = st.text_input("Checkpoint path", value=get_default_checkpoint_path())

    uploaded = st.file_uploader("Upload WAV file", type=["wav"])

    if st.button("Analyze"):
        if uploaded is None:
            st.warning("Please upload a .wav file first.")
            return
        if not os.path.exists(ckpt_path):
            st.error(f"Checkpoint not found: {ckpt_path}")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            tmp_path = tmp.name

        try:
            with st.spinner("Running inference..."):
                result = infer_one(tmp_path, ckpt_path, device=device)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        label = "Fake" if result["pred"] == 1 else "Real"
        prob_real = result["prob_real"]
        prob_fake = result["prob_fake"]

        st.subheader(f"Prediction: {label}")
        st.progress(min(max(int(prob_fake * 100), 0), 100) if result["pred"] == 1 else min(max(int(prob_real * 100), 0), 100))
        st.write(f"prob_real: {prob_real:.4f} | prob_fake: {prob_fake:.4f}")
        st.caption(f"dims=({result['dim1']},{result['dim2']}) | device={result['device']}")


if __name__ == "__main__":
    main()


