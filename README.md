Download the dataset (In the Wild (Audio Deepfake)) from: https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake

First, set paths to the dataset and model_save_path in main.py. 

To run the pretrained model on the test split, set pretrained_model=True in main.py and run python -m main.

To train the model, set pretrained_model=False in main.py and run python -m main

## How to infer the model
CLI (single file, test.wav):

```bash
python deepsonar/infer.py --wav test.wav --ckpt deepsonar/best.pth
```

Optional flags:
- `--device cuda` or `--device cpu`

Streamlit app (upload audio in browser):

```bash
pip install -r requirements.txt
streamlit run app.py
```

In the UI:
- Upload a `.wav` file
- (Optional) adjust device/checkpoint path
- Click "Analyze" to get Real/Fake and probabilities