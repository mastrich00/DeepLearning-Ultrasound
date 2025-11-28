# src/ultra_spatial/models/pix2pix.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
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
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed (due to odd sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetGenerator(nn.Module):
    """
    A compact U-Net generator for pix2pix-style experiments.

    Args:
        in_ch: input channels (1 for grayscale, 3 for RGB)
        out_ch: output channels (should match in_ch)
        ngf: base features (64 typical)
        bilinear: use bilinear upsampling instead of transposed convs
    """
    def __init__(self, in_ch=1, out_ch=1, ngf=64, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_ch, ngf)
        self.down1 = Down(ngf, ngf*2)
        self.down2 = Down(ngf*2, ngf*4)
        self.down3 = Down(ngf*4, ngf*8)
        # optionally add one more down if image large
        self.down4 = Down(ngf*8, ngf*8)
        self.up1 = Up(ngf*16, ngf*4, bilinear=bilinear)
        self.up2 = Up(ngf*8, ngf*2, bilinear=bilinear)
        self.up3 = Up(ngf*4, ngf, bilinear=bilinear)
        self.up4 = Up(ngf*2, ngf, bilinear=bilinear)
        self.outc = nn.Sequential(
            nn.Conv2d(ngf, out_ch, kernel_size=1),
            nn.Sigmoid()  # keep outputs in [0,1] to match other generator
        )

    def forward(self, clip):
        """
        Expects clip shape [B,T,C,H,W] or [B,C,H,W] (function will handle T dimension)
        Returns dict {"corrected": tensor [B,out_ch,H,W]}
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
        return {"corrected": out}
