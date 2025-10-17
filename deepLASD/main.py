import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
# import IPython
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Reshape,MaxPooling2D, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn

import os

from feature_extractor import extract_features
from model import Detector
from train import train
from evaluate import evaluate


paths = []
labels = []

# Define the root directory
real_root_dir = '/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/real'
fake_root_dir = '/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/fake'
# Iterate through the subdirectories
for filename in os.listdir(real_root_dir):
    file_path = os.path.join(real_root_dir, filename)
    paths.append(file_path)
    # Add label based on the subdirectory name
    labels.append('real')

for filename in os.listdir(fake_root_dir):
    file_path = os.path.join(fake_root_dir, filename)
    paths.append(file_path)
    # Add label based on the subdirectory name
    labels.append('fake')

print('Dataset is loaded')
extract_feats = True
pretrained_model = False
model_save_path = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Code/VoiceDeepfakeDetector/deepLASD/best.pth"
auc_roc_path =  "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Code/VoiceDeepfakeDetector/deepLASD/auc_roc.png"

feat_save_dir = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/features"
x, y = extract_features(fake_root_dir, real_root_dir, feat_save_dir, dataset_size=100, extract_feats=extract_feats)

# first split: train (80%) + test (20%)
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# second split: from train into train (70%) + val (10%)
xtrain, xval,  ytrain, yval = train_test_split(
    xtrain, ytrain, test_size=0.125, random_state=42, stratify=ytrain
)
# (0.125 of 80% ≈ 10% of total)

# build datasets
train_ds = TensorDataset(
    torch.tensor(xtrain, dtype=torch.float32).unsqueeze(1),  # [N, 1, T]
    torch.tensor(ytrain, dtype=torch.long)
)
val_ds = TensorDataset(
    torch.tensor(xval, dtype=torch.float32).unsqueeze(1),
    torch.tensor(yval, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(xtest, dtype=torch.float32).unsqueeze(1),
    torch.tensor(ytest, dtype=torch.long)
)


# loaders
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)

# -----------------------
# Model setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dim1 = xtrain.shape[-1]        # DeepSonar features
model = Detector(sinc_channels=32).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# -----------------------
# Training
# -----------------------
if not pretrained_model:
    model = train(model=model,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  optimizer=optimizer,
                  criterion=criterion,
                  epochs=50,
                  device=device,
                  model_save_path=model_save_path,
                  auc_fig_path=auc_roc_path)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model = model.to(device="cuda")

# -----------------------
# Evaluation
# -----------------------
accuracy = evaluate(model=model, test_loader=test_loader, device=device)
print(f"Accuracy: {accuracy:.4f}")