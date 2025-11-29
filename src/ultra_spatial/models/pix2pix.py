# src/ultra_spatial/models/pix2pix.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_groupnorm_groups(channels, preferred=8):
    # choose a num_groups that divides channels, fallback to 1 if none found
    g = min(preferred, channels)
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1


class DoubleConv(nn.Module):
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
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        """
        Up with nearest + conv to avoid checkerboard artifacts and be stable with small batches.
        in_ch: number of channels after concatenation (x2 + x1)
        """
        super().__init__()
        # use nearest upsample + conv block
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # mid channels (before second conv) chosen to be in_ch//2 if sensible
        mid_ch = in_ch // 2 if in_ch // 2 > 0 else in_ch
        self.conv = DoubleConv(in_ch, out_ch, mid_ch=mid_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed (due to odd sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetGenerator(nn.Module):
    """
    A compact U-Net generator for pix2pix-style experiments.

    Predicts an additive residual (tanh scaled) by default and returns:
      {"residual": r, "corrected": clamp(x_mid + r)}
    """

    def __init__(self, in_ch=1, out_ch=1, ngf=64, bilinear=True, predict_residual=True, residual_scale=0.2):
        super().__init__()
        self.predict_residual = bool(predict_residual)
        self.residual_scale = float(residual_scale)
        self.inc = DoubleConv(in_ch, ngf)
        self.down1 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 4)
        self.down3 = Down(ngf * 4, ngf * 8)
        # optionally add one more down if image large
        self.down4 = Down(ngf * 8, ngf * 8)
        self.up1 = Up(ngf * 16, ngf * 4, bilinear=bilinear)
        self.up2 = Up(ngf * 8, ngf * 2, bilinear=bilinear)
        self.up3 = Up(ngf * 4, ngf, bilinear=bilinear)
        self.up4 = Up(ngf * 2, ngf, bilinear=bilinear)

        # If predicting residual -> use Tanh scaled; otherwise output direct image via Sigmoid
        if self.predict_residual:
            self.outc = nn.Sequential(
                nn.Conv2d(ngf, out_ch, kernel_size=1),
                nn.Tanh()
            )
        else:
            self.outc = nn.Sequential(
                nn.Conv2d(ngf, out_ch, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, clip):
        """
        Expects clip shape [B,T,C,H,W] or [B,C,H,W] (function will handle T dimension)
        Returns dict {"corrected": tensor [B,out_ch,H,W], "residual": tensor [B,out_ch,H,W] (if predicted)}
        """
        # accept [B,T,C,H,W] or [B,C,H,W]
        if clip.dim() == 5:
            # take middle frame
            mid = clip.shape[1] // 2
            x = clip[:, mid]
        else:
            x = clip

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        out = self.outc(u4)

        if self.predict_residual:
            # out in [-1,1] -> scale and add to input
            residual = out * self.residual_scale
            corrected = torch.clamp(x + residual, 0.0, 1.0)
            return {"corrected": corrected, "residual": residual}
        else:
            # direct image output in [0,1]
            corrected = out
            return {"corrected": corrected}
