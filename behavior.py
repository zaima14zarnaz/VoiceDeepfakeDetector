import torch, numpy as np

def acn(t, thr=0.0):
    """
    Computes the fraction of activations greater than a given threshold.

    Parameters:
        t (Tensor): Input tensor, shape [B, D] or [B, T, D].
        thr (float): Threshold value for counting activations (default: 0.0).

    Returns:
        Tensor: Fraction of elements > thr per sample, shape [B, 1].
    """
    x = t.float()  # Ensure tensor is in float type for computation
    if x.ndim == 3:
        x = x.mean(-1)  # If input has 3 dims (e.g., [B, T, D]), average over the last dim
    # Compute the fraction of activations greater than the threshold
    a = (x > thr).float().mean(dim=-1, keepdim=True)
    return a  # Return activation ratio per sample → shape: [B, 1]



def tkan(t, k=10):
    """
    Computes the mean of the top-k activations across batch and time dimensions.

    Parameters:
        t (Tensor): Input tensor, shape [B, D] or [B, T, D].
        k (int): Number of top activations to consider (default: 10).

    Returns:
        Tensor: Mean of the top-k activations, shape [1, 1].
    """
    x = t.float()  # Convert input to float for consistent computation
    if x.ndim == 3:
        # Rearrange [B, T, D] → [B*D, T] to aggregate over temporal dimension
        x = x.transpose(1, 2).reshape(x.size(0) * x.size(2), x.size(1))
    v, _ = torch.topk(x, min(k, x.size(-1)), dim=-1)  # Extract top-k activations along last dim
    # Mean of top-k activations, averaged across all samples and time steps
    s = v.mean(dim=-1, keepdim=True).mean(dim=0, keepdim=True)
    return s  # Return scalar tensor → shape: [1, 1]


def stats(t):
    """
    Computes basic statistical features (mean, standard deviation, and entropy)
    from the input tensor.

    Parameters:
        t (Tensor): Input tensor, shape [B, D] or [B, T, D].

    Returns:
        Tensor: Concatenated statistics [mean, std, entropy] for each sample,
                shape [B, 3].
    """
    x = t.float()  # Ensure tensor uses float type for accurate computation
    if x.ndim == 3:
        x = x.mean(-1)  # Average over last dimension if 3D input (e.g., time or feature axis)
    m = x.mean(dim=-1, keepdim=True)  # Mean of activations per sample
    s = x.std(dim=-1, keepdim=True)   # Standard deviation per sample
    # Entropy of softmax distribution along last dimension
    e = (-torch.softmax(x, dim=-1) * torch.log_softmax(x, dim=-1)).sum(dim=-1, keepdim=True)
    return torch.cat([m, s, e], dim=-1)  # Combine [mean, std, entropy] → [B, 3]


def layer_features(t, k_top=10, thr=0.0):
    """
    Extracts statistical and activation-based features from a tensor, combining
    mean, standard deviation, entropy, activation coverage (ACN), and top-k activation mean (TKAN).

    Parameters:
        t (Tensor): Input tensor of shape [B, C], [B, T, C], or [C].
        k_top (int): Number of top activations to consider for TKAN (default: 10).
        thr (float): Threshold for counting activations in ACN (default: 0.0).

    Returns:
        Tensor: Concatenated feature vector per batch [mean, std, entropy, acn, tkan],
                shape [B, 5].
    """
    x = t.float()  # Convert to float for numeric stability

    # Handle shape normalization
    if x.ndim == 3:  # [B, T, C] → flatten temporal and channel dims
        B, T, C = x.shape
        x = x.reshape(B, T * C)
    elif x.ndim == 2:  # [B, C]
        B, _ = x.shape
    elif x.ndim == 1:  # [C] → [1, C]
        x = x.unsqueeze(0)
        B = 1
    else:
        raise ValueError(f"Unexpected tensor shape: {x.shape}")  # Guard for invalid input

    # Core statistical features
    m = x.mean(dim=1, keepdim=True)  # Mean per sample
    s = x.std(dim=1, keepdim=True)   # Standard deviation per sample
    e = (-torch.softmax(x, dim=1) * torch.log_softmax(x, dim=1)).sum(dim=1, keepdim=True)  
    # Entropy of softmax distribution

    # Additional activation-based descriptors
    a = acn(x, thr=thr)   # Activation coverage: fraction of activations > threshold
    t = tkan(x, k=k_top)  # Top-k activation mean: global top-k average
    t = t.expand(B, -1)   # Broadcast to match batch dimension

    # Combine all features → [mean, std, entropy, acn, tkan]
    return torch.cat([m, s, e, a, t], dim=1)  # Output shape: [B, 5]


def pack_features(feats_dict, k_top=10, thr=0.0):
    """
    Aggregates layer-wise feature statistics into a single feature vector.

    Parameters:
        feats_dict (dict): Dictionary of feature tensors from different layers.
                           Each value should be a torch.Tensor of shape [B, ...].
        k_top (int): Number of top activations to consider for TKAN (default: 10).
        thr (float): Threshold for activation coverage in ACN (default: 0.0).

    Returns:
        Tensor or None: Concatenated feature tensor combining all layers,
                        shape [B, L*5], where L = number of valid layers.
                        Returns None if no valid tensors are found.
    """
    vecs = []  # List to store per-layer feature vectors
    for k, v in feats_dict.items():
        # Process only valid tensors
        if isinstance(v, torch.Tensor):
            try:
                # Extract 5-dim feature vector (mean, std, entropy, acn, tkan)
                vecs.append(layer_features(v, k_top=k_top, thr=thr))
            except Exception as e:
                # Skip invalid or incompatible entries gracefully
                print(f"Skipping {k} due to {e}")
    if len(vecs) == 0:
        return None  # Return None if no valid layers were processed
    return torch.cat(vecs, dim=1)  # Concatenate all features → [B, L*5]
