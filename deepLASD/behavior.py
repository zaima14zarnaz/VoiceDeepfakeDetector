import torch, numpy as np

def acn(t, thr=0.0):
    x = t.float()
    if x.ndim==3: x = x.mean(-1)
    a = (x>thr).float().mean().unsqueeze(0)
    return a

def tkan(t, k=10):
    x = t.float()
    if x.ndim==3:
        x = x.transpose(1,2).reshape(x.size(0)*x.size(2), x.size(1))
    v,_ = torch.topk(x, min(k, x.size(-1)), dim=-1)
    s = v.mean(dim=-1, keepdim=True).mean(dim=0, keepdim=True)
    return s

def stats(t):
    x = t.float()
    if x.ndim==3: x = x.mean(-1)
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    e = (-torch.softmax(x, dim=-1)*torch.log_softmax(x, dim=-1)).sum(dim=-1, keepdim=True)
    return torch.cat([m,s,e], dim=-1)

def layer_features(t):
    x = t.float()

    # Handle 3D (batch, time, features)
    if x.ndim == 3:
        B, T, C = x.shape
        x = x.reshape(B, T * C)   # flatten time+features into one vector

    # Handle 2D (batch, features)
    elif x.ndim == 2:
        pass  # already fine

    # Handle 1D (features only, rare)
    elif x.ndim == 1:
        x = x.unsqueeze(0)

    # Now x is always [B, D]
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True)
    e = (-torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1)).sum(dim=1, keepdim=True)

    return torch.cat([m, s, e], dim=1)


def pack_features(feats_dict):
    vecs = []
    for k, v in feats_dict.items():
        if isinstance(v, torch.Tensor):
            try:
                vecs.append(layer_features(v))
            except Exception as e:
                print(f"Skipping {k} due to {e}")
    if len(vecs) == 0:
        return None
    return torch.cat(vecs, dim=1)  # concatenate along feature axis
