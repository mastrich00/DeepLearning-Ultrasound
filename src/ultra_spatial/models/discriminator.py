# src/ultra_spatial/models/discriminator.py
"""
PatchGAN-style spatial discriminator with spectral normalization.

Design notes:
- The discriminator operates on single-frame images and outputs a dense map
  of real/fake scores (one score per local patch). This focuses training on
  local texture and speckle realism rather than a single global realism score.
- Spectral normalization is applied to convolution layers to stabilise GAN
  training by constraining the Lipschitz constant of each layer.
- The `features` method exposes intermediate feature maps for feature-matching
  losses and visualization.
"""
import torch.nn as nn
import torch.nn.utils as nn_utils


def _blk(c_in, c_out, k=3, s=2, p=1):
    """
    Small conv -> activation block used in the discriminator.

    Args:
      c_in (int): input channels
      c_out (int): output channels
      k (int): kernel size (default 3)
      s (int): stride (default 2) to progressively reduce spatial size
      p (int): padding (default 1)

    Returns:
      nn.Sequential: convolution with spectral_norm followed by LeakyReLU.
    """
    conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p)
    # Spectral normalization stabilizes adversarial training by limiting the
    # operator norm of the weight matrix. It is cheap and effective for PatchGANs.
    nn_utils.spectral_norm(conv)
    return nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=True))


class PatchDiscriminator(nn.Module):
    """
    Compact PatchGAN discriminator.

    Architecture:
      - block1: Conv -> LeakyReLU  (reduces spatial resolution)
      - block2: Conv -> LeakyReLU
      - block3: Conv -> LeakyReLU
      - final: 1x1 Conv (spectral-normed) producing a single-channel score map

    The forward pass returns the raw score map of shape [B, 1, Hf, Wf].
    The 'features' method returns a list of intermediate feature maps before
    the final projection. These are useful for feature-matching losses or
    for debugging/visualization of learned filters.
    """

    def __init__(self, ch=32):
        super().__init__()
        # Modular blocks make it easy to extract intermediate activations.
        # Input channel is 1 (grayscale ultrasound) by default.
        self.block1 = _blk(1, ch)
        self.block2 = _blk(ch, ch * 2)
        self.block3 = _blk(ch * 2, ch * 4)

        # Final projection to a single-channel patch map. Also spectrally normalized.
        self.final = nn.Conv2d(ch * 4, 1, 1)
        nn_utils.spectral_norm(self.final)

    def forward(self, x):
        """
        Forward pass.

        Args:
          x (Tensor): input image tensor [B, C, H, W], expected C=1 by default.

        Returns:
          Tensor: patch-wise score map [B, 1, Hf, Wf]. Higher values indicate
                  more "real" according to the discriminator.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.final(x)

    def features(self, x):
        """
        Return intermediate activations from the three blocks.

        This function is used by feature-matching losses and for visualization.
        Each returned tensor has shape [B, C, Hf, Wf] and represents progressively
        deeper spatial features.

        Returns:
          list[Tensor]: [feat_block1, feat_block2, feat_block3]
        """
        feats = []
        x = self.block1(x)
        feats.append(x)
        x = self.block2(x)
        feats.append(x)
        x = self.block3(x)
        feats.append(x)
        return feats
