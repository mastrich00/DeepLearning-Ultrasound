# src/ultra_spatial/models/pix2pix.py
"""
Compact, well-documented U-Net generator used as an optional pix2pix baseline.

Key design points:
- GroupNorm is used instead of BatchNorm to be stable with small batch sizes.
- Nearest-neighbor upsampling + conv (rather than transposed conv) to avoid checkerboard artifacts.
- The network can predict either a direct image (Sigmoid) or an additive residual (Tanh scaled).
- Forward accepts either a clip [B,T,C,H,W] (uses center frame) or a single image [B,C,H,W].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_groupnorm_groups(channels, preferred=8):
    """
    Choose a reasonable number of groups for GroupNorm.

    Strategy:
      - Start from `preferred` (default 8) and decrease until a divisor of channels is found.
      - If none >1 divides channels, fall back to 1 (equivalent to LayerNorm).
    This keeps GroupNorm effective and avoids invalid group counts.
    """
    g = min(preferred, channels)
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1


class DoubleConv(nn.Module):
    """
    Two successive 3x3 conv layers with GroupNorm + ReLU.

    This is the basic building block used throughout the U-Net encoder/decoder.
    GroupNorm is picked with _choose_groupnorm_groups for stability with small batches.
    """

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        gn1 = _choose_groupnorm_groups(mid_ch)
        gn2 = _choose_groupnorm_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn1, mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn2, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downsampling block: maxpool by 2 followed by a DoubleConv.
    This reduces spatial resolution and increases receptive field.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling block used in the U-Net decoder.

    Implementation details:
      - Uses nearest-neighbor upsampling followed by a DoubleConv.
      - Concatenates the upsampled decoder feature with the corresponding encoder skip
        connection (x2) along the channel dimension.
      - Pads x1 if necessary to match x2 spatial dimensions (handles odd sizes).
    Parameters:
      in_ch: number of channels AFTER concatenation (channels(x2) + channels(up_x1))
      out_ch: desired output channels for this block after convs
    """

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        # nearest upsample avoids checkerboard artifacts common with transposed convs
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # choose a mid-channel count for the DoubleConv to reduce parameter count
        mid_ch = in_ch // 2 if in_ch // 2 > 0 else in_ch
        self.conv = DoubleConv(in_ch, out_ch, mid_ch=mid_ch)

    def forward(self, x1, x2):
        """
        x1: decoder feature to be upsampled
        x2: corresponding encoder feature (skip connection)
        """
        x1 = self.up(x1)
        # pad if needed (due to odd image sizes after pooling)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            # pad(left, right, top, bottom)
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        # concatenate along channel dim: [B, C_enc + C_dec, H, W]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetGenerator(nn.Module):
    """
    A compact U-Net generator suitable for pix2pix-style baselines.

    Features:
      - in_ch / out_ch control input/output channels (1 for grayscale).
      - ngf is the base channel count (growth by factor 2 in deeper layers).
      - predict_residual: if True the network predicts a small additive residual
        (returned as "residual") which is then added to the input center frame.
        If False the network outputs a direct image via Sigmoid.

    Return format:
      - If predict_residual: {"corrected": corrected_tensor, "residual": residual_tensor}
      - Otherwise: {"corrected": corrected_tensor}
    """

    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        ngf=64,
        bilinear=True,
        predict_residual=True,
        residual_scale=0.2,
    ):
        super().__init__()
        self.predict_residual = bool(predict_residual)
        self.residual_scale = float(residual_scale)

        # encoder (contracting) path
        self.inc = DoubleConv(in_ch, ngf)  # initial convs
        self.down1 = Down(ngf, ngf * 2)  # /2
        self.down2 = Down(ngf * 2, ngf * 4)  # /4
        self.down3 = Down(ngf * 4, ngf * 8)  # /8
        self.down4 = Down(ngf * 8, ngf * 8)  # /16 (keeps channels constant)

        # decoder (expanding) path
        # note: in_ch for Up is sum of channels from skip + upsampled feature
        self.up1 = Up(ngf * 16, ngf * 4, bilinear=bilinear)
        self.up2 = Up(ngf * 8, ngf * 2, bilinear=bilinear)
        self.up3 = Up(ngf * 4, ngf, bilinear=bilinear)
        self.up4 = Up(ngf * 2, ngf, bilinear=bilinear)

        # output head: residuals use Tanh ([-1,1]) to keep changes small after scaling,
        # direct output uses Sigmoid to produce [0,1] images.
        if self.predict_residual:
            self.outc = nn.Sequential(nn.Conv2d(ngf, out_ch, kernel_size=1), nn.Tanh())
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(ngf, out_ch, kernel_size=1), nn.Sigmoid()
            )

    def forward(self, clip):
        """
        Forward pass.
        """
        # support both [B,T,C,H,W] and [B,C,H,W]
        if clip.dim() == 5:
            # take middle frame
            mid = clip.shape[1] // 2
            x = clip[:, mid]  # center frame extraction
        else:
            x = clip

        # U-Net forward with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # upsampling path with concatenated skips
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        out = self.outc(u4)

        if self.predict_residual:
            # residual in [-1,1], scale down and add to input, then clamp
            residual = out * self.residual_scale
            corrected = torch.clamp(x + residual, 0.0, 1.0)
            return {"corrected": corrected, "residual": residual}
        else:
            # direct image output already in [0,1]
            corrected = out
            return {"corrected": corrected}
