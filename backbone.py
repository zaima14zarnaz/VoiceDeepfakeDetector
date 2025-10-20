import torch, torchaudio

class SRBackbone(torch.nn.Module):
    """
    SRBackbone:
    A feature-extraction backbone using the pre-trained Wav2Vec2 model from torchaudio.
    Automatically registers forward hooks on selected layers (Conv, Linear, ReLU, LayerNorm)
    to capture intermediate activations for speech or audio analysis.

    Parameters:
        device (str): Device to load the model on (default: "cuda").
    """
    def __init__(self, device="cuda"):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE  # Load pre-trained Wav2Vec2 configuration
        self.model = bundle.get_model().to(device).eval()  # Load model, move to device, set to eval mode
        self.sample_rate = bundle.sample_rate  # Store model's expected sample rate

        self._layers = []  # List of layer names to attach hooks
        self._hooks = []   # List of hook handles for cleanup

        # Collect layers to hook (feature extraction points)
        for n, m in self.model.named_modules():
            if isinstance(m, (torch.nn.ReLU, torch.nn.Conv1d, torch.nn.Linear, torch.nn.LayerNorm)):
                self._layers.append(n)

        self.reset_hooks()  # Register hooks on selected layers

    def reset_hooks(self):
        """Remove existing hooks (if any) and register new ones on target layers."""
        for h in self._hooks:
            h.remove()  # Safely remove old hooks

        self._feats = {}   # Dictionary to store layer outputs
        self._hooks = []   # Reset hook list

        # Create hook function capturing layer outputs
        def make_hook(name):
            def hook(mod, inp, out):
                with torch.no_grad():
                    # Handle module outputs that may be tuples
                    o = out[0] if isinstance(out, tuple) else out
                    if isinstance(o, torch.Tensor):
                        self._feats[name] = o.detach()  # Store detached tensor for later use
            return hook

        # Register hooks for all target layers
        for n, m in self.model.named_modules():
            if n in self._layers:
                self._hooks.append(m.register_forward_hook(make_hook(n)))

    @torch.no_grad()
    def forward_with_hooks(self, wav):
        """
        Forward pass that captures intermediate layer features via hooks.

        Parameters:
            wav (Tensor): Input waveform, shape [T] or [B, T].

        Returns:
            dict: Dictionary of layer activations {layer_name: Tensor}.
        """
        # Ensure input is 2D: (batch, time)
        if wav.ndim == 1:           # Single waveform → add batch dimension
            wav = wav.unsqueeze(0)  # [T] → [1, T]
        elif wav.ndim == 2:
            pass                    # Already in correct shape
        else:
            raise ValueError(f"Expected 1D or 2D input, got {wav.shape}")

        _ = self.model(wav)  # Run forward pass to trigger hooks
        feats = {k: v for k, v in self._feats.items()}  # Copy stored features
        return feats  # Return dict of extracted layer outputs
