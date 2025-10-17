import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- SincConv (hardened & GPU-safe) ----------
class SincConv1d(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = float(sample_rate)

        # initialize cutoff freqs in (0, Nyquist). Use log-space init for stability.
        low_hz = torch.linspace(50.0, 4000.0, out_channels)
        band_hz = torch.linspace(100.0, 6000.0, out_channels)
        self.low_hz  = nn.Parameter(low_hz)
        self.band_hz = nn.Parameter(band_hz)

        n = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1).float()
        self.register_buffer("n", n)  # moves with .to(device)

        # precompute a Hamming window to reduce Gibbs
        ham = 0.54 - 0.46 * torch.cos(2 * torch.pi * (self.n - self.n.min()) / (self.kernel_size - 1))
        self.register_buffer("hamming", ham)

    @staticmethod
    def _sinc(x):
        # robust sinc
        return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x) / x)

    def forward(self, x):                         # x: [B, 1, T]
        device, dtype = x.device, x.dtype
        n = self.n.to(device=device, dtype=dtype)
        w = self.hamming.to(device=device, dtype=dtype)
        eps = torch.finfo(dtype).eps

        f1 = torch.relu(self.low_hz) + 50.0                      # > 0
        f2 = f1 + torch.relu(self.band_hz) + 50.0                # > f1
        # normalize freqs into radians/sample
        f1 = (2 * torch.pi * f1 / self.sample_rate).to(device=device, dtype=dtype)
        f2 = (2 * torch.pi * f2 / self.sample_rate).to(device=device, dtype=dtype)

        # build filters vectorized: [out, k]
        # ideal band-pass = 2*f2*sinc(2*f2*n) - 2*f1*sinc(2*f1*n)
        n_row = n.unsqueeze(0)                                   # [1, k]
        bp = (2 * f2.unsqueeze(1) * self._sinc(2 * f2.unsqueeze(1) * n_row) -
              2 * f1.unsqueeze(1) * self._sinc(2 * f1.unsqueeze(1) * n_row))
        bp = bp * w.unsqueeze(0)
        # L2 normalize each filter to avoid div-by-zero / huge amps
        norm = torch.sqrt((bp * bp).sum(dim=1, keepdim=True) + eps)
        bp = bp / norm

        filters = bp.unsqueeze(1)                                # [out, 1, k]bp = torch.nan_to_num(bp, nan=0.0, posinf=0.0, neginf=0.0)
        filters = bp.unsqueeze(1)
        y = F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)
        # y = torch.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        return y



# ---------- Residual Block (no time collapse) ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=2):
        super().__init__()
        self.pool = nn.MaxPool1d(pool)
        self.bn   = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.act  = nn.GELU()
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):                      # x: [B, C_in, T]
        # main path: BN -> Conv -> GELU -> Pool
        out = self.bn(x)
        out = self.conv(out)                   # [B, C_out, T]
        out = self.act(out)
        out = self.pool(out)                   # [B, C_out, T']

        # skip path: project channels, then pool to match T'
        skip = self.proj(x)                    # [B, C_out, T]
        skip = self.pool(skip)                 # [B, C_out, T']

        out = out + skip
        # out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)
        return out


# ---------- Optional attention pooling (used once) ----------
class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):                      # x: [B, T, C]
        s = self.att(x)                        # [B, T, 1]
        w = torch.softmax(s, dim=1)
        return (x * w).sum(dim=1)              # [B, C]


# ---------- Full model ----------
class Detector(nn.Module):
    def __init__(self, sinc_channels=32, hidden_dim=128, gru_hidden=128, num_classes=2):
        super().__init__()
        self.sinc = SincConv1d(sinc_channels, kernel_size=251)
        self.bn   = nn.GroupNorm(num_groups=8, num_channels=sinc_channels)
        self.act  = nn.LeakyReLU(0.1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(sinc_channels if i == 0 else hidden_dim, hidden_dim, pool=2)
            for i in range(6)
        ])

        self.gru = nn.GRU(hidden_dim, gru_hidden, batch_first=True)
        self.att_pool = TemporalAttention(gru_hidden)  # or comment if you prefer last-state

        self.fc1 = nn.Linear(gru_hidden, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):                      # x: [B, 1, T]
        x = self.sinc(x)                       # [B, C, T]
        x = self.bn(x)
        x = self.act(x)
        print(x)
        for block in self.res_blocks:
            x = block(x)                       # [B, C, T/2^k]
        print(x)
        x = x.transpose(1, 2)                  # [B, T', C]
        x, _ = self.gru(x)                     # [B, T', H] 
        # x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

        # x = self.att_pool(x)                   # [B, H]  (or: x = x[:, -1, :])
        # x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        # print(x)

        x = x[:, -1, :] 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        print(x)
        return self.out(x)
