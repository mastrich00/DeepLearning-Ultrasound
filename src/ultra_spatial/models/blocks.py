# src/ultra_spatial/models/blocks.py
"""
Lightweight building blocks used by the RetinexLowRankVT generator.

This file provides:
- ConvBlock: small two-layer conv block for local feature extraction.
- FrameEncoder: per-frame same-resolution encoder that preserves speckle detail.
- TemporalEncoder: small transformer-based temporal fusion module.
- LowRankHead: compact low-rank intensity head (height × width factorization).
- IlluminationHead: coarse-to-fine illumination/TGC head (smooth in depth).
- ReflectanceHead: simple reflectance/structure head.

Design rationale (brief):
- Avoid aggressive downsampling in the encoder so speckle and fine edges are preserved.
- Use a transformer on spatially pooled tokens for lightweight temporal context.
- Factorize global intensity with a bilinear low-rank parameterization to capture
  smooth depth/width trends without altering edges.
- Predict illumination on a coarse grid and upsample to bias the field to be smooth.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two successive 3x3 convolutions with ReLU activations.

    This block is the basic local feature extractor used in the FrameEncoder.
    It keeps spatial resolution (padding=1) to avoid blurring from downsampling.
    """

    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FrameEncoder(nn.Module):
    """
    Per-frame encoder operating at native resolution.

    Structure:
      - stem conv from input channels (assumes single-channel ultrasound).
      - a sequence of ConvBlocks (channels given by `channels` list).
      - a final 1x1 head conv to produce the output features.

    Reasoning:
      - The encoder avoids strided pooling to preserve speckle statistics and
        high-frequency edges that are important in ultrasound.
      - `out_dim` provides the number of output feature channels for downstream heads.
    """

    def __init__(self, channels):
        super().__init__()
        # initial convolution from grayscale input (1 channel)
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)

        # build sequential ConvBlock stages according to channels list
        stages = []
        c = channels[0]
        for c_next in channels:
            stages.append(ConvBlock(c, c_next))
            c = c_next
        self.stages = nn.Sequential(*stages)

        # final 1x1 to unify output features
        self.head = nn.Conv2d(c, c, 1)
        self.out_dim = c

    def forward(self, x):
        # x expected [B, C, H, W] with C usually 1
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x


class TemporalEncoder(nn.Module):
    """
    Lightweight temporal encoder based on torch.TransformerEncoder.

    Behavior:
      - Accepts feature tensor `feats` shaped [B, T, C, H, W].
      - Spatially pools (mean over H,W) to produce T tokens of dim C.
      - Runs a small transformer encoder over the time axis.
      - Returns a tensor shaped [B, T, C, 1, 1] ready to be broadcast spatially.

    This provides temporal context without heavy spatio-temporal attention.
    """

    def __init__(self, embed_dim, n_heads=4, n_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, feats):
        # feats: [B, T, C, H, W]
        pooled = feats.mean(dim=[3, 4])  # -> [B, T, C]
        z = self.enc(pooled)  # -> [B, T, C]
        return z[:, :, None, None]  # -> [B, T, C, 1, 1]


class LowRankHead(nn.Module):
    """
    Low-rank intensity head producing a smooth global trend map.

    Mechanism:
      - Two 1x1 convolutions produce rank-r feature maps U(x) and V(x).
      - Average U over width and V over height to obtain slim profiles:
          U -> [B, r, H], V -> [B, r, W]
      - Combine with outer product (einsum) to form a [B, H, W] low-rank field.
      - Apply sigmoid and add a channel dimension to match other heads.

    Purpose:
      - Capture broad depth-wise or lateral intensity slopes (e.g., TGC-like trends)
        with very few parameters. This avoids interfering with edges while modeling
        global exposure biases.
    """

    def __init__(self, c, rank=8):
        super().__init__()
        self.u = nn.Conv2d(c, rank, 1)
        self.v = nn.Conv2d(c, rank, 1)

    def forward(self, f):
        # f: [B, C, H, W]
        U = self.u(f).mean(dim=3)  # -> [B, r, H]
        V = self.v(f).mean(dim=2)  # -> [B, r, W]
        # outer product per-rank then sum -> [B, H, W]
        lr = torch.einsum("brh,brw->bhw", U, V)
        return torch.sigmoid(lr[:, None])  # return shape [B,1,H,W]


class IlluminationHead(nn.Module):
    """
    Illumination / TGC head.

    Design:
      - Apply a coarse adaptive average pool to bias predictions to be smooth.
      - Run a small conv head at that coarse resolution and upsample bilinearly.
      - Sigmoid output ensures values in (0,1), which can be scaled or normalized
        by the generator to represent multiplicative illumination.
    Use:
      - Produces a smoothly varying field, particularly smooth along depth.
    """

    def __init__(self, c, coarse=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((coarse, coarse))
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, 1, 1)
        )

    def forward(self, f):
        # f: [B, C, H, W]
        i = self.conv(self.pool(f))  # coarse prediction [B,1,coarse,coarse]
        i = torch.nn.functional.interpolate(
            i, size=f.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.sigmoid(i)


class ReflectanceHead(nn.Module):
    """
    Reflectance / detail head.

    A small conv head that outputs a single-channel map in (0,1).
    The reflectance map is intended to preserve edges and microtexture while
    allowing mild local contrast changes.
    """

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, 1, 1)
        )

    def forward(self, f):
        return torch.sigmoid(self.net(f))
