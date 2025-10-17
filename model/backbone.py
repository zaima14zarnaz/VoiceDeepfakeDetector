
import torch, torchaudio

class SRBackbone(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.model = bundle.get_model().to(device).eval()
        self.sample_rate = bundle.sample_rate
        self._layers = []
        self._hooks = []
        for n,m in self.model.named_modules():
            if isinstance(m, torch.nn.ReLU) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LayerNorm):
                self._layers.append(n)
        self.reset_hooks()
    def reset_hooks(self):
        for h in self._hooks: h.remove()
        self._feats = {}
        self._hooks = []
        def make_hook(name):
            def hook(mod, inp, out):
                with torch.no_grad():
                    if isinstance(out, tuple): o = out[0]
                    else: o = out
                    if isinstance(o, torch.Tensor):
                        self._feats[name] = o.detach()
            return hook
        for n,m in self.model.named_modules():
            if n in self._layers:
                self._hooks.append(m.register_forward_hook(make_hook(n)))
    @torch.no_grad()
    def forward_with_hooks(self, wav):
        # ensure input shape is (batch, time)
        if wav.ndim == 1:          # (time,)
            wav = wav.unsqueeze(0) # -> (1, time)
        elif wav.ndim == 2:
            pass                   # already (batch, time)
        else:
            raise ValueError(f"Expected 1D or 2D input, got {wav.shape}")
    
        _ = self.model(wav)
        feats = {k: v for k, v in self._feats.items()}
        return feats