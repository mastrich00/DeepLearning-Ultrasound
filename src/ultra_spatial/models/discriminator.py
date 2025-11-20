import torch.nn as nn

def _blk(c_in, c_out, k=3, s=2, p=1):
    conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p)
    nn.utils.spectral_norm(conv)
    return nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=True))


class PatchDiscriminator(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            _blk(1, ch), _blk(ch, ch * 2), _blk(ch * 2, ch * 4), nn.Conv2d(ch * 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)
