# src/ultra_spatial/models/generator.py
import torch
import torch.nn as nn
from .blocks import (
    FrameEncoder,
    TemporalEncoder,
    LowRankHead,
    IlluminationHead,
    ReflectanceHead,
)


class RetinexLowRankVT(nn.Module):
    """
    Variant of the original RetinexLowRankVT where the network predicts
    an *additive residual* correction instead of using the classic
    multiplicative-retinex composition alone.

    Behaviour:
      - The encoder/temporal backbone is unchanged.
      - The 'reflectance' head is repurposed to predict a residual map
        (small per-pixel additive correction) which is squashed with tanh
        and scaled by `residual_scale`.
      - Illumination (I) and LowRank (LR) heads are preserved for optional
        diagnostic / hybrid use (kept near 1 to avoid large multiplicative change).
      - Final corrected output: corrected = clamp(x_mid + residual, 0, 1)

    This keeps backward-compatibility with older training code that expects
    keys like "R", "I", "LR" while adding "residual" as the main corrective output.
    """

    def __init__(
        self,
        enc_channels=[32, 64, 128],
        n_heads=4,
        n_layers=2,
        lowrank_rank=8,
        illumination_coarse=16,
        residual_scale: float = 0.12,  # typical max additive change (~12% of dynamic range)
    ):
        super().__init__()
        self.encoder = FrameEncoder(enc_channels)
        c = self.encoder.out_dim
        self.temporal = TemporalEncoder(embed_dim=c, n_heads=n_heads, n_layers=n_layers)

        # original heads are kept for compatibility / optional hybrid losses
        self.head_r = ReflectanceHead(c)     # repurposed: now predicts residual raw values
        self.head_i = IlluminationHead(c, coarse=illumination_coarse)
        self.head_lr = LowRankHead(c, rank=lowrank_rank)

        # small constant to stabilize division
        self._eps = 1e-6

        # scale for the tanh residual output -> ensures residual is small
        self.residual_scale = float(residual_scale)

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
        R_raw = self.head_r(f_mid)   # repurposed: residual raw map (unbounded)
        I_raw = self.head_i(f_mid)   # illumination map (positive)
        LR_raw = self.head_lr(f_mid) # low-rank multiplicative factor

        # ----------------------------
        # Residual (additive) prediction
        # ----------------------------
        # Use tanh to produce bounded residual in [-1,1] then scale to small range.
        residual = torch.tanh(R_raw) * self.residual_scale
        # Keep a compatibility alias "R" that previously stood for reflectance.
        # Now "R" will contain the residual map (so older code reading "R" still gets something meaningful).
        R = residual

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

        # ----------------------------
        # Final corrected image (additive residual)
        # ----------------------------
        x_mid = clip[:, mid]  # [B,C,H,W]

        # Option A (pure additive residual):
        corrected = x_mid + residual

        # Optionally, one could hybridize multiplicative heads with the residual, for example:
        # corrected = (x_mid / (I * LR + self._eps)) * 1.0 + residual
        # The above is commented out to keep behavior strictly additive by default.

        # final clamp as a safety
        corrected = torch.clamp(corrected, 0.0, 1.0)

        return {"corrected": corrected, "R": R, "residual": residual, "I": I, "LR": LR}
