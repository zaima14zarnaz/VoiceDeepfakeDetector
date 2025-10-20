Download the dataset (In the Wild (Audio Deepfake)) from: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake

The code in deepsonar_single/ directory is the original deepsonar model from the paper: DeepSonar: Towards Effective and Robust Detection of AI-Synthesized Fake Voices (link: https://dl.acm.org/doi/abs/10.1145/3394171.3413716)

The code in deepsonar_multi/ directory is out improved model with multimodal activation and spectral features fused using cross-attention

Both these directories have their own respective main.py and feature extractor files. The common codes for the model, backbone, and activate behaviour extractor helpers are in the root directory outside of these subfolders.

To run code in both the directories:
First, set paths to the dataset, model_save_path (best_single.pth for the original model and best_multi.pth for our improved multimodal version) and feat_save_dir in main.py. 
To run the pretrained model on the test split, set pretrained_model=True in main.py and run python -m main.
To extract the features from scratch (for the first run only), set extract_feat=True in main.py. Set it to False after extracting features once to save time. 
To train the model, set pretrained_model=False in main.py and run python -m main
