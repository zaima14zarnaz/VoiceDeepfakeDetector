import numpy as np
import pandas as pd
import os
import sys
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

from feature_extractor_multi import extract_features
from train import train
from evaluate import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import Detector_Multi_Feat


paths = []
labels = []

# WaveFake dataset root directory
real_root_dir = '/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/real'
fake_root_dir = '/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/fake'

# Iterate through the real samples
for filename in os.listdir(real_root_dir):
    file_path = os.path.join(real_root_dir, filename)
    paths.append(file_path)
    # Add label based on the subdirectory name
    labels.append('real')

# Iterate through the fake samples
for filename in os.listdir(fake_root_dir):
    file_path = os.path.join(fake_root_dir, filename)
    paths.append(file_path)
    # Add label based on the subdirectory name
    labels.append('fake')

print('Dataset is loaded')

# Set to True if features are to be extracted from scratch, set to False if loading pre_loaded features from the feat_save_dir
extract_feats = True

# Set to True if testing dataset on pretrained model
pretrained_model = False

model_save_path = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Code/VoiceDeepfakeDetector/ckpt/best_multi.pth" # Best model save path
feat_save_dir = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/features" # Path to save extracted features from pretrained network

# Extract (x1=deepsonar features, x2=spectral features, y=labels) from real and fake samples
x1, x2, y = extract_features(fake_root_dir, real_root_dir, feat_save_dir=feat_save_dir, dataset_size=None, extract_feats=extract_feats) 

# first split: train (80%) + test (20%)
x1train, x1test, x2train, x2test, ytrain, ytest = train_test_split(
    x1, x2, y, test_size=0.2, random_state=42, stratify=y
)

# second split: from train into train (70%) + val (10%)
x1train, x1val, x2train, x2val, ytrain, yval = train_test_split(
    x1train, x2train, ytrain, test_size=0.125, random_state=42, stratify=ytrain
)
# (0.125 of 80% ≈ 10% of total)

# build train, val and test datasets
train_ds = TensorDataset(
    torch.tensor(x1train, dtype=torch.float32),
    torch.tensor(x2train, dtype=torch.float32),
    torch.tensor(ytrain, dtype=torch.long)
)
val_ds = TensorDataset(
    torch.tensor(x1val, dtype=torch.float32),
    torch.tensor(x2val, dtype=torch.float32),
    torch.tensor(yval, dtype=torch.long)
)
test_ds = TensorDataset(
    torch.tensor(x1test, dtype=torch.float32),
    torch.tensor(x2test, dtype=torch.float32),
    torch.tensor(ytest, dtype=torch.long)
)

# Create loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

# -----------------------
# Model setup
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dim1 = x1train.shape[-1]        # DeepSonar features
dim2 = x2train.shape[-1]        # MFCC feature dim
# Create Multi-Feature model with input dimensions: dim1=size of deepsonar feature vector and dim2=size of spectral feature vector for a sample
model = Detector_Multi_Feat(dim1, dim2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
                  model_save_path=model_save_path)
# Load the best model 
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model = model.to(device="cuda")

# -----------------------
# Evaluation
# -----------------------
results = evaluate(model=model, test_loader=test_loader, device=device)
