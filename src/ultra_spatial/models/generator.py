# src/ultra_spatial/models/generator.py
import torch, torch.nn as nn
from .blocks import (
    FrameEncoder,
    TemporalEncoder,
    LowRankHead,
    IlluminationHead,
    ReflectanceHead,
)


class RetinexLowRankVT(nn.Module):
    def __init__(
        self,
        enc_channels=[32, 64, 128],
        n_heads=4,
        n_layers=2,
        lowrank_rank=8,
        illumination_coarse=16,
    ):
        super().__init__()
        self.encoder = FrameEncoder(enc_channels)
        c = self.encoder.out_dim
        self.temporal = TemporalEncoder(embed_dim=c, n_heads=n_heads, n_layers=n_layers)
        self.head_r = ReflectanceHead(c)
        self.head_i = IlluminationHead(c, coarse=illumination_coarse)
        self.head_lr = LowRankHead(c, rank=lowrank_rank)

        # small constant to stabilize division
        self._eps = 1e-6

    def forward(self, clip):
        # Accept extra singleton dims and enforce [B,T,C,H,W]
        while clip.dim() > 5:
            squeezed = False
            for d in range(clip.dim()):
                if clip.size(d) == 1 and clip.dim() > 5:
                    clip = clip.squeeze(d)
                    squeezed = True
                    break
            if not squeezed:
                break
        assert clip.dim() == 5, f"Expected [B,T,C,H,W], got {tuple(clip.shape)}"
        B, T, C, H, W = clip.shape

        feats = []
        for t in range(T):
            feats.append(self.encoder(clip[:, t]))
        feats = torch.stack(feats, dim=1)  # [B,T,Cf,Hf,Wf]

        z = self.temporal(feats).expand(-1, -1, -1, H, W)
        feats = feats + z

        mid = T // 2
        f_mid = feats[:, mid]

        # Raw head outputs
        R_raw = self.head_r(f_mid)   # reflectance "detail" map
        I_raw = self.head_i(f_mid)   # illumination map (positive)
        LR_raw = self.head_lr(f_mid) # low-rank multiplicative factor

        # Constrain outputs to sensible ranges to avoid saturation:
        # - Reflectance R in [0,1] via sigmoid (acts like albedo)
        R = torch.sigmoid(R_raw)

        # - Illumination I should be positive and roughly centered around 1.
        #   Use softplus to ensure positivity, then normalize per-sample to mean 1.
        I = torch.nn.functional.softplus(I_raw) + self._eps
        # normalize spatially so that mean(I) ~= 1 per sample
        I_mean = I.view(I.size(0), -1).mean(dim=1).view(-1, 1, 1, 1)
        I = I / (I_mean + self._eps)

        # - Low-rank multiplicative factor: keep it near 1 with small dynamic range
        #   Use sigmoid in [0,1], then scale to [0.6, 1.4] (tunable).
        LR = torch.sigmoid(LR_raw)  # [0,1]
        LR = 0.6 + LR * 0.8         # -> [0.6,1.4]

        # stable corrected image: (x / (I * LR)) * R
        x_mid = clip[:, mid]
        corrected = x_mid / (I * LR + self._eps)
        corrected = corrected * R

        # final clamp as a safety (should rarely saturate if heads are constrained)
        corrected = torch.clamp(corrected, 0.0, 1.0)

        return {"corrected": corrected, "R": R, "I": I, "LR": LR}
