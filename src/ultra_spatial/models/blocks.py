import torch, torch.nn as nn


class ConvBlock(nn.Module):
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
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)
        stages = []
        c = channels[0]
        for c_next in channels:
            stages.append(ConvBlock(c, c_next))
            c = c_next
        self.stages = nn.Sequential(*stages)
        self.head = nn.Conv2d(c, c, 1)
        self.out_dim = c

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads=4, n_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, feats):
        pooled = feats.mean(dim=[3, 4])
        z = self.enc(pooled)
        return z[:, :, None, None]


class LowRankHead(nn.Module):
    def __init__(self, c, rank=8):
        super().__init__()
        self.u = nn.Conv2d(c, rank, 1)
        self.v = nn.Conv2d(c, rank, 1)

    def forward(self, f):
        U = self.u(f).mean(dim=3)
        V = self.v(f).mean(dim=2)
        lr = torch.einsum("brh,brw->bhw", U, V)
        return torch.sigmoid(lr[:, None])


class IlluminationHead(nn.Module):
    def __init__(self, c, coarse=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((coarse, coarse))
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, 1, 1)
        )

    def forward(self, f):
        i = self.conv(self.pool(f))
        i = torch.nn.functional.interpolate(
            i, size=f.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.sigmoid(i)


class ReflectanceHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, 1, 1)
        )

    def forward(self, f):
        return torch.sigmoid(self.net(f))
