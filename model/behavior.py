
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
    return torch.cat([acn(t,0.0), tkan(t,10), stats(t)], dim=-1).squeeze(0)

def pack_features(feats_dict):
    keys = sorted(feats_dict.keys())
    vecs = []
    for k in keys:
        v = feats_dict[k]
        if isinstance(v, torch.Tensor):
            vecs.append(layer_features(v))
    if len(vecs)==0: return None
    x = torch.cat(vecs, dim=-1)
    return x
