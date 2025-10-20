import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """
    CrossModalAttention module:
    Fuses two feature modalities (e.g., DeepSonar and audio) using a multi-head attention mechanism.
    DeepSonar features act as queries attending to audio feature representations, 
    enabling the model to learn cross-modal dependencies.
    """
    def __init__(self, dim1, dim2, hidden=256, num_heads=4):
        super().__init__()
        self.proj1 = nn.Linear(dim1, hidden)          # Projects DeepSonar features to hidden dimension
        self.proj2 = nn.Linear(20000, hidden)         # Projects audio features to hidden dimension
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)  
        # Multi-head attention: allows model to attend to multiple representation subspaces

    def forward(self, x1, x2):
        """
        x1: [B, D1]  (DeepSonar features)
        x2: [B, D2] or [B, T, D2] (Audio features)
        Returns: [B, H] (Cross-attended fused feature representation)
        """
        if x2.ndim == 2:
            # If audio input lacks time dimension, add one
            x2 = x2.unsqueeze(1)  # [B,1,D2]
        elif x2.ndim == 3:
            pass  # Already has time dimension [B,T,D2]
        else:
            raise ValueError(f"Unexpected x2 shape {x2.shape}")  # Guard against malformed input

        # Project both modalities into the same hidden space
        q = self.proj1(x1).unsqueeze(1)  # Query (DeepSonar): [B,1,H]
        k = self.proj2(x2)               # Key (Audio): [B,T,H]
        v = k                            # Value (Audio): [B,T,H]

        attn_out, _ = self.cross_attn(q, k, v)  # Compute attention weights and fuse representations
        return attn_out.squeeze(1)              # Remove sequence dimension → [B,H]
    

class Detector_Multi_Feat(nn.Module):
    """
    Detector_Multi_Feat:
    A multimodal classifier combining DeepSonar and audio features.
    Uses CrossModalAttention to align and fuse features, then a feedforward network 
    to classify the fused representation (e.g., real vs. fake voice).
    """
    def __init__(self, dim1, dim2, hidden=512, p=0.2):
        super().__init__()
        # Cross-modal attention for feature fusion
        self.cross_att = CrossModalAttention(dim1, dim2, hidden=hidden//2, num_heads=4)

        # Fully connected classifier network
        self.net = nn.Sequential(
            nn.LayerNorm(hidden//2),
            nn.Linear(hidden//2, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden//2, 2)  # Output logits for binary classification
        )

    def forward(self, x1, x2):
        fused = self.cross_att(x1, x2)  # Fuse DeepSonar and audio features [B, hidden//2]
        return self.net(fused)          # Predict class logits [B, 2]
    

class Detector_Single_Feat(nn.Module):
    """
    Detector_Single_Feat:
    A single-modality classifier that operates on one feature source 
    (e.g., DeepSonar-only or audio-only) without cross-modal attention.
    """
    def __init__(self, dim1, hidden=512, p=0.2):
        super().__init__()

        # Feedforward classification network
        self.net = nn.Sequential(
            nn.LayerNorm(dim1),
            nn.Linear(dim1, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden//2, 2)  # Output logits for binary classification
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the classifier
