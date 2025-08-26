import math

import torch
from torch import nn
from torch.nn import functional as F


def resolution_pool(x, r):
    size = 2 ** r
    length = x.shape[-1]
    padded = F.pad(x, (0, (-length) % size), mode="replicate")
    return F.avg_pool1d(padded, size, size).repeat_interleave(size, -1)[..., :length]


def sinusoidal(t, width):
    half = width // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    phase = t.float()[..., None] * freq
    return torch.cat([phase.sin(), phase.cos()], -1)


class FrequencyDecomposition(nn.Module):
    def __init__(self, n_fft=32, hop_length=8, fraction=0.2):
        super().__init__()
        self.n_fft, self.hop_length, self.fraction = n_fft, hop_length, fraction
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):
        shape = x.shape

        spectrum = torch.stft(x.reshape(-1, shape[-1]), self.n_fft,
                              hop_length=self.hop_length, window=self.window,
                              return_complex=True, center=True, pad_mode="constant")
        amplitudes = spectrum.abs().mean(-1)
        count = max(1, math.ceil(amplitudes.shape[-1] * self.fraction))
        indices = amplitudes.topk(count, dim=-1).indices
        mask = torch.zeros_like(amplitudes).scatter_(-1, indices, 1)
        invariant = torch.istft(spectrum * mask[..., None], self.n_fft,
                                hop_length=self.hop_length, window=self.window,
                                center=True, length=shape[-1]).reshape(shape)
        return invariant, x - invariant


class HierarchicalAttention(nn.Module):
    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.time = nn.TransformerEncoderLayer(channels, heads, 4 * channels,
                                               dropout, activation="gelu", batch_first=True)
        self.variable = nn.TransformerEncoderLayer(channels, heads, 4 * channels,
                                                   dropout, activation="gelu", batch_first=True)

    def forward(self, x):
        b, v, t, c = x.shape
        x = self.time(x.reshape(b * v, t, c)).reshape(b, v, t, c)
        x = self.variable(x.transpose(1, 2).reshape(b * t, v, c))
        return x.reshape(b, t, v, c).transpose(1, 2)


class InvariantBlock(nn.Module):
    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.projection = nn.Linear(channels, channels)
        self.attention = HierarchicalAttention(channels, heads, dropout)

    def forward(self, x, embedding):
        return (x + self.attention(self.projection(x) + embedding)) / math.sqrt(2)


class DilatedModernTCNBlock(nn.Module):

    def __init__(self, channels, heads, dropout, kernel=5, branches=5, dilation=2):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel, groups=channels)
        self.ffn = nn.Sequential(nn.Conv1d(channels, 4 * channels, 1), nn.GELU(),
                                 nn.Dropout(dropout), nn.Conv1d(4 * channels, channels, 1))
        self.norm = nn.LayerNorm(channels)
        self.attention = HierarchicalAttention(channels, heads, dropout)
        self.dilations = [dilation ** i for i in range(branches)]

    def forward(self, x, embedding):
        b, v, t, c = x.shape
        y = x.reshape(b * v, t, c).transpose(1, 2)
        outputs = []
        for d in self.dilations:
            z = F.conv1d(y, self.dwconv.weight, self.dwconv.bias,
                         padding=d * (self.dwconv.kernel_size[0] // 2),
                         dilation=d, groups=c)
            z = self.norm(z.transpose(1, 2)).transpose(1, 2)
            outputs.append(self.ffn(z))
        y = torch.stack(outputs).mean(0).transpose(1, 2).reshape(b, v, t, c)
        y = (x + y) / math.sqrt(2)
        return (y + self.attention(y + embedding)) / math.sqrt(2)


class DecomposableDenoiser(nn.Module):
    def __init__(self, features, cfg):
        super().__init__()
        c = cfg["channels"]
        self.channels = c
        self.decompose = FrequencyDecomposition(cfg["n_fft"], cfg["hop_length"], cfg["frequency_fraction"])
        self.step_mlp = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        self.scale_mlp = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        self.feature_embedding = nn.Embedding(features, c)
        self.inv_input, self.var_input = nn.Linear(1, c), nn.Linear(1, c)
        self.invariant = nn.ModuleList([InvariantBlock(c, cfg["heads"], cfg["dropout"])
                                        for _ in range(cfg["invariant_blocks"])])
        self.variant = nn.ModuleList([DilatedModernTCNBlock(
            c, cfg["heads"], cfg["dropout"], cfg["kernel_size"], cfg["branches"], cfg["dilation"])
            for _ in range(cfg["variant_blocks"])])
        self.output = nn.Sequential(nn.Linear(2 * c, c), nn.GELU(), nn.Linear(c, 1))

    def forward(self, x, step, resolution):
        inv, var = self.decompose(x)
        c = self.channels
        emb = self.step_mlp(sinusoidal(step, c)) + self.scale_mlp(sinusoidal(resolution + 1, c))
        emb = emb[:, None, None, :]
        emb = emb + self.feature_embedding.weight[None, :, None, :]
        emb = emb + sinusoidal(torch.arange(x.shape[-1], device=x.device), c)[None, None]
        inv, var = self.inv_input(inv[..., None]), self.var_input(var[..., None])
        for block in self.invariant:
            inv = block(inv, emb)
        for block in self.variant:
            var = block(var, emb)
        return self.output(torch.cat([inv, var], -1)).squeeze(-1)


class CrossScalePrior(nn.Module):

    def __init__(self, channels, heads):
        super().__init__()
        self.channels = channels
        self.query, self.context = nn.Linear(1, channels), nn.Linear(1, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, 1))

    def forward(self, clean, coarser, step):
        b, v, t = clean.shape
        pos = sinusoidal(torch.arange(t, device=clean.device), self.channels)[None, None]
        emb = sinusoidal(step, self.channels)[:, None, None]
        q = (self.query(clean[..., None]) + pos + emb).reshape(b * v, t, -1)
        kv = (self.context(coarser[..., None]) + pos).reshape(b * v, t, -1)
        z, _ = self.attention(q, kv, kv, need_weights=False)
        return self.output(self.norm(q + z)).reshape(b, v, t)


class MODEM(nn.Module):
    def __init__(self, features, cfg):
        super().__init__()
        self.cfg = cfg
        self.resolutions = cfg["resolutions"]
        self.steps = cfg["diffusion_steps"]
        self.denoiser = DecomposableDenoiser(features, cfg)
        self.prior = CrossScalePrior(cfg["prior_channels"], cfg["heads"])
        beta = torch.linspace(math.sqrt(cfg["beta_start"]), math.sqrt(cfg["beta_end"]), self.steps).square()

        alpha_bar = torch.cat([torch.ones(1), (1 - beta).cumprod(0)])
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("gamma", alpha_bar.sqrt() * (1 - alpha_bar.sqrt()))

    def shift(self, clean, coarser, step, resolution):
        active = (resolution < self.resolutions - 1).to(clean.dtype)[:, None, None]
        return self.prior(clean, coarser, step) * active

    def q_sample(self, clean, coarser, step, resolution, noise=None):
        if noise is None:
            noise = torch.randn_like(clean)
        a = self.alpha_bar[step][:, None, None]
        g = self.gamma[step][:, None, None]
        return a.sqrt() * clean + g * self.shift(clean, coarser, step, resolution) + (1 - a).sqrt() * noise

    def loss(self, x):
        b = len(x)
        r = torch.randint(self.resolutions, (b,), device=x.device)
        k = torch.randint(1, self.steps + 1, (b,), device=x.device)
        pyramid = torch.stack([resolution_pool(x, i) for i in range(self.resolutions)], 1)
        rows = torch.arange(b, device=x.device)
        clean = pyramid[rows, r]
        coarser = pyramid[rows, (r + 1).clamp_max(self.resolutions - 1)]
        noisy = self.q_sample(clean, coarser, k, r)

        return F.mse_loss(self.denoiser(noisy, k, r), clean)

    def reverse_step(self, x, clean, coarser, k, previous, r, eta=0.0):

        batch = len(x)
        kvec = torch.full((batch,), k, device=x.device, dtype=torch.long)
        pvec = torch.full_like(kvec, previous)
        rvec = torch.full_like(kvec, r)
        a, ap = self.alpha_bar[k], self.alpha_bar[previous]
        shift = self.shift(clean, coarser, kvec, rvec)
        shift_previous = self.shift(clean, coarser, pvec, rvec) if previous else torch.zeros_like(x)
        epsilon = (x - a.sqrt() * clean - self.gamma[k] * shift) / (1 - a).sqrt()
        sigma = eta * ((1 - ap) / (1 - a) * (1 - a / ap)).clamp_min(0).sqrt()
        result = ap.sqrt() * clean + self.gamma[previous] * shift_previous
        result = result + (1 - ap - sigma.square()).clamp_min(0).sqrt() * epsilon
        if eta:
            result = result + sigma * torch.randn_like(x)
        return result

    @torch.no_grad()
    def reconstruct(self, observed, mask, sampling_steps, keep_steps=10, eta=0.0):


        if not 1 <= sampling_steps <= self.steps:
            raise ValueError("sampling_steps must lie in [1, diffusion_steps]")
        sequence = torch.linspace(self.steps, 1, sampling_steps).round().long().tolist()
        keep_steps = min(keep_steps, sampling_steps)
        b = len(observed)
        coarser = torch.zeros_like(observed)
        errors = []
        for r in reversed(range(self.resolutions)):

            fraction = resolution_pool(mask, r)
            known_mask = (fraction > 1 - 1e-6).to(observed.dtype)
            known = resolution_pool(observed * mask, r)
            target = resolution_pool(observed, r)
            x = torch.randn_like(observed)
            observation_noise = torch.randn_like(observed)
            resolution_errors = []
            rvec = torch.full((b,), r, device=x.device, dtype=torch.long)
            for i, k in enumerate(sequence):
                kvec = torch.full((b,), k, device=x.device, dtype=torch.long)

                noisy_known = self.q_sample(known, coarser, kvec, rvec, observation_noise)
                x = known_mask * noisy_known + (1 - known_mask) * x
                clean = self.denoiser(x, kvec, rvec)
                previous = sequence[i + 1] if i + 1 < len(sequence) else 0
                x = self.reverse_step(x, clean, coarser, k, previous, r, eta)
                if i >= len(sequence) - keep_steps:

                    resolution_errors.append((x - target).square().mean(1))
            coarser = known_mask * known + (1 - known_mask) * x
            errors.append(torch.stack(resolution_errors, 1))
        return coarser, torch.stack(list(reversed(errors)), 1)
