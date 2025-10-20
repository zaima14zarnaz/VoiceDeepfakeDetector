import torch, numpy as np

def acn(t, thr=0.0):
    x = t.float()
    if x.ndim == 3:
        x = x.mean(-1)
    # fraction of activations greater than threshold
    a = (x > thr).float().mean(dim=-1, keepdim=True)
    return a  # shape: [B, 1]

def tkan(t, k=10):
    x = t.float()
    if x.ndim == 3:
        x = x.transpose(1, 2).reshape(x.size(0) * x.size(2), x.size(1))
    v, _ = torch.topk(x, min(k, x.size(-1)), dim=-1)
    # mean of top-k activations, averaged across batch/time
    s = v.mean(dim=-1, keepdim=True).mean(dim=0, keepdim=True)
    return s  # shape: [1, 1]

def stats(t):
    x = t.float()
    if x.ndim == 3:
        x = x.mean(-1)
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    e = (-torch.softmax(x, dim=-1) * torch.log_softmax(x, dim=-1)).sum(dim=-1, keepdim=True)
    return torch.cat([m, s, e], dim=-1)  # shape: [B, 3]

def layer_features(t, k_top=10, thr=0.0):
    x = t.float()

    # Handle shape normalization
    if x.ndim == 3:  # (B, T, C)
        B, T, C = x.shape
        x = x.reshape(B, T * C)
    elif x.ndim == 2:
        B, _ = x.shape
    elif x.ndim == 1:
        x = x.unsqueeze(0)
        B = 1
    else:
        raise ValueError(f"Unexpected tensor shape: {x.shape}")

    # Core statistics
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True)
    e = (-torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1)).sum(dim=1, keepdim=True)

    # Additional descriptors
    a = acn(x, thr=thr)   # activation coverage
    t = tkan(x, k=k_top)  # top-k activation mean (scalar per batch)
    t = t.expand(B, -1)   # broadcast t to match batch size

    # Concatenate everything: [mean, std, entropy, acn, tkan]
    return torch.cat([m, s, e, a, t], dim=1)  # shape [B, 5]


def pack_features(feats_dict, k_top=10, thr=0.0):
    vecs = []
    for k, v in feats_dict.items():
        if isinstance(v, torch.Tensor):
            try:
                vecs.append(layer_features(v, k_top=k_top, thr=thr))
            except Exception as e:
                print(f"Skipping {k} due to {e}")
    if len(vecs) == 0:
        return None
    return torch.cat(vecs, dim=1)  # concatenate all layer features → [B, L*5]
