import torch.nn as nn
import torch.nn.utils as nn_utils

def _blk(c_in, c_out, k=3, s=2, p=1):
    conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p)
    nn_utils.spectral_norm(conv)   # apply spectral norm for stability
    return nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=True))


class PatchDiscriminator(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        # keep blocks modular so we can extract features
        self.block1 = _blk(1, ch)
        self.block2 = _blk(ch, ch * 2)
        self.block3 = _blk(ch * 2, ch * 4)
        self.final = nn.Conv2d(ch * 4, 1, 1)
        nn_utils.spectral_norm(self.final)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.final(x)

    def features(self, x):
        """
        Return list of feature maps from intermediate blocks (before final conv).
        Each element is a tensor [B, C, Hf, Wf].
        """
        feats = []
        x = self.block1(x); feats.append(x)
        x = self.block2(x); feats.append(x)
        x = self.block3(x); feats.append(x)
        return feats
