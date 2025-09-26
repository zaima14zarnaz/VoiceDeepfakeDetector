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

from feature_extractor import extract_features_deepsonar
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
extract_feats = False
pretrained_model = False
save_path = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Code/deepsonar/best.pth"

feat_save_dir = "/home/zaimaz/Desktop/research1/VoiceDeepfakeDetector/Dataset/inTheWildAudioDeekfake/features"
x, y = extract_features_deepsonar(fake_root_dir, real_root_dir, feat_save_dir, dataset_size=None, extract_feats=extract_feats)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = .2)
train_ds = TensorDataset(torch.tensor(xtrain, dtype=torch.float32),
                         torch.tensor(ytrain, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(xtest, dtype=torch.float32),
                         torch.tensor(ytest, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)

# model
device = "cuda" if torch.cuda.is_available() else "cpu"
in_dim = xtrain.shape[-1]
model = Detector(in_dim).to(device)

# optimizer + loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

if not pretrained_model:
    model = train(model=model,
                train_loader=train_loader,
                optimizer=optimizer, 
                criterion=criterion, 
                device="cuda",
                save_path=save_path)
else:
    model.load_state_dict(torch.load(save_path)).to(device)

accuracy = evaluate(model=model, 
                 test_loader=test_loader, 
                 device="cuda")
print(f"Accuracy: {accuracy}")