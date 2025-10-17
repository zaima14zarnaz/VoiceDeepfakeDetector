import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, dim1, dim2, hidden=256, num_heads=4):
        super().__init__()
        self.proj1 = nn.Linear(dim1, hidden)
        self.proj2 = nn.Linear(20000, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

    def forward(self, x1, x2):
        """
        x1: [B, D1]  (DeepSonar features)
        x2: [B, D2] or [B, T, D2] (Audio features)
        """
        if x2.ndim == 2:
            # treat audio as a single token
            x2 = x2.unsqueeze(1)  # [B,1,D2]
        elif x2.ndim == 3:
            pass  # already [B,T,D2]
        else:
            raise ValueError(f"Unexpected x2 shape {x2.shape}")

        # project
        q = self.proj1(x1).unsqueeze(1)  # [B,1,H]
        k = self.proj2(x2)               # [B,T,H]
        v = k

        attn_out, _ = self.cross_attn(q, k, v)  # [B,1,H]
        return attn_out.squeeze(1)              # [B,H]
    

class Detector_Multi_Feat(nn.Module):
    def __init__(self, dim1, dim2, hidden=512, p=0.2):
        super().__init__()
        self.cross_att = CrossModalAttention(dim1, dim2, hidden=hidden//2, num_heads=4)

        self.net = nn.Sequential(
            nn.LayerNorm(hidden//2),
            nn.Linear(hidden//2, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden//2, 2)
        )

    def forward(self, x1, x2):
        fused = self.cross_att(x1, x2)  # [B, hidden//2]
        return self.net(fused)
    

class Detector_Single_Feat(nn.Module):
    def __init__(self, dim1, hidden=512, p=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim1),
            nn.Linear(dim1, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden//2, 2)
        )

    def forward(self, x):
        return self.net(x)
