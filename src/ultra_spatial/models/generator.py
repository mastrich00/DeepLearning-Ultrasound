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
    Retinex + Low-Rank + Temporal generator for ultrasound exposure correction.

    This variant predicts a small additive residual on top of a Retinex-style
    multiplicative correction.

    High-level idea:
      1. Encode each frame spatially with a CNN (preserve speckle and edges).
      2. Fuse information across time with a lightweight temporal encoder.
      3. Predict three Retinex-style components on the center frame:
         - R: reflectance / structure (kept close to identity)
         - I: illumination / TGC-like field (smooth, depth-dependent)
         - LR: low-rank global intensity trend
      4. Predict an additional residual map that makes fine local corrections.
      5. Apply multiplicative Retinex correction, then add the residual.

    Output is a dictionary for backward compatibility and diagnostics:
      {
        "corrected": final corrected image,
        "R": reflectance map,
        "I": illumination map,
        "LR": low-rank map,
        "residual": additive correction
      }
    """

    def __init__(
        self,
        enc_channels=[32, 64, 128],
        n_heads=4,
        n_layers=2,
        lowrank_rank=8,
        illumination_coarse=16,
        residual_scale: float = 0.12,
    ):
        """
        Args:
          enc_channels: channel widths for the per-frame CNN encoder.
          n_heads: number of attention heads in the temporal transformer.
          n_layers: number of temporal transformer layers.
          lowrank_rank: rank of the low-rank intensity model.
          illumination_coarse: spatial resolution of the coarse illumination grid.
          residual_scale: maximum magnitude of the additive residual
                          (fraction of dynamic range).
        """
        super().__init__()

        # --- Spatial encoder applied independently to each frame ---
        self.encoder = FrameEncoder(enc_channels)
        c = self.encoder.out_dim  # feature dimensionality after encoding

        # --- Temporal fusion module ---
        # Aggregates information across frames to stabilize predictions
        self.temporal = TemporalEncoder(embed_dim=c, n_heads=n_heads, n_layers=n_layers)

        # --- Retinex-style heads (kept mainly for inductive bias & diagnostics) ---
        self.head_r = ReflectanceHead(c)  # structure / reflectance
        self.head_i = IlluminationHead(
            c, coarse=illumination_coarse
        )  # smooth TGC-like field
        self.head_lr = LowRankHead(
            c, rank=lowrank_rank
        )  # global low-rank intensity trend

        # --- Residual head (main corrective signal) ---
        # Predicts a single-channel additive correction for grayscale ultrasound
        self.head_res = nn.Sequential(
            nn.Conv2d(c, c // 2 if c // 2 > 0 else c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2 if c // 2 > 0 else c, 1, kernel_size=1),
        )

        self.residual_scale = float(residual_scale)
        self._eps = 1e-6  # numerical stability constant

    def forward(self, clip):
        """
        Forward pass.

        Args:
          clip: input tensor of shape [B, T, C, H, W].
                Extra singleton dimensions are tolerated and removed.

        Returns:
          dict with keys:
            - corrected: final corrected center frame [B,C,H,W]
            - R: reflectance map
            - I: illumination map
            - LR: low-rank map
            - residual: additive residual
        """
        # --- Ensure input has shape [B,T,C,H,W] ---
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

        # --- Encode each frame independently ---
        feats = []
        for t in range(T):
            feats.append(self.encoder(clip[:, t]))
        feats = torch.stack(feats, dim=1)  # [B,T,Cf,Hf,Wf]

        # --- Temporal fusion ---
        # Temporal encoder outputs a per-frame embedding which is broadcast spatially
        z = self.temporal(feats).expand(-1, -1, -1, H, W)
        feats = feats + z

        # --- Use center frame for prediction ---
        mid = T // 2
        f_mid = feats[:, mid]

        # --- Head predictions ---
        R_raw = self.head_r(f_mid)  # reflectance logits
        I_raw = self.head_i(f_mid)  # illumination logits
        LR_raw = self.head_lr(f_mid)  # low-rank logits

        # --- Constrain head outputs ---
        # Reflectance in (0,1)
        R = torch.sigmoid(R_raw)

        # Illumination positive and normalized to mean ~1 per sample
        I = torch.nn.functional.softplus(I_raw) + self._eps
        # normalize spatially so that mean(I) ~= 1 per sample
        I_mean = I.view(I.size(0), -1).mean(dim=1).view(-1, 1, 1, 1)
        I = I / (I_mean + self._eps)

        # Low-rank factor constrained to a narrow multiplicative range
        LR = torch.sigmoid(LR_raw)
        LR = 0.6 + LR * 0.8  # maps to approximately [0.6, 1.4]

        # --- Residual prediction ---
        # Small additive correction, bounded with tanh
        res_raw = self.head_res(f_mid)  # [B,1,H,W]
        residual = torch.tanh(res_raw) * self.residual_scale

        # --- Apply correction ---
        x_mid = clip[:, mid]  # center input frame

        # Multiplicative Retinex-style correction
        corrected_mult = x_mid / (I * LR + self._eps)
        corrected_mult = corrected_mult * R

        # Broadcast residual if input has multiple channels
        if corrected_mult.size(1) != residual.size(1):
            residual = residual.expand(-1, corrected_mult.size(1), -1, -1)

        # Final corrected image: multiplicative + additive
        corrected = corrected_mult + residual
        corrected = torch.clamp(corrected, 0.0, 1.0)

        return {
            "corrected": corrected,
            "R": R,
            "residual": residual,
            "I": I,
            "LR": LR,
        }
