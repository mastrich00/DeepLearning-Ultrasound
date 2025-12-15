"""
losses.py

Collection of loss functions used by the project with explanatory comments.

Notes:
- Image tensors are expected to be float in a roughly 0..1 range for the SSIM code.
- Discriminator outputs ("logits") in this project are patch-wise maps (shape [B,1,Hf,Wf])
  or reduced per-sample scalars ([B]). The adversarial losses accept either shape and
  reduce appropriately where they are used.
"""

import torch
import torch.nn.functional as F
from math import exp


# ---------------------------------------------------------------------
# LSGAN (least-squares GAN) losses
# ---------------------------------------------------------------------
def lsgan_d_loss(real_logits, fake_logits, real_label=1.0, fake_label=0.0):
    """
    Least-squares discriminator loss.

    Formula:
      0.5 * ( MSE(real_logits, real_label) + MSE(fake_logits, fake_label) )
    """
    # build same-shaped target tensors and compute MSE
    r_loss = F.mse_loss(real_logits, torch.full_like(real_logits, real_label))
    f_loss = F.mse_loss(fake_logits, torch.full_like(fake_logits, fake_label))
    return 0.5 * (r_loss + f_loss)


def lsgan_g_loss(fake_logits, real_label=1.0):
    """
    Least-squares generator loss.

    Formula:
      0.5 * MSE(fake_logits, real_label)
    """
    return 0.5 * F.mse_loss(fake_logits, torch.full_like(fake_logits, real_label))


# ---------------------------------------------------------------------
# Total variation (TV) loss
# ---------------------------------------------------------------------
def tv_loss(img):
    """
    Isotropic total-variation loss computed as mean absolute differences
    between neighbouring pixels in x and y directions.
    """
    # vertical differences (y)
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    # horizontal differences (x)
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dx + dy


# ---------------------------------------------------------------------
# SSIM helper (small in-file implementation)
# ---------------------------------------------------------------------
def gaussian_window(window_size, sigma, channel):
    """
    Create a 2D Gaussian window expanded for grouped convolution.
    """
    gauss = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    gauss = gauss / gauss.sum()
    # outer product -> 2D Gaussian kernel
    _2D = gauss[:, None] @ gauss[None, :]
    # replicate for each channel and provide shape [channel,1,H,W] for grouped conv
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, max_val=1.0):
    """
    Compute mean SSIM index between img1 and img2.

    This is a compact implementation following the original SSIM formulation:
      - local means via gaussian window
      - local variances / covariances via the same window
      - stabilized by small constants C1, C2 proportional to max_val
    """
    channel = img1.size(1)
    window = gaussian_window(window_size, sigma, channel).to(img1.device)

    # local means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    # local variances and covariance
    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    # stability constants (as in original SSIM paper)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # SSIM map per-pixel
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    # return mean SSIM value
    return ssim_map.mean()


# ---------------------------------------------------------------------
# Hinge adversarial losses (useful with PatchGAN outputs)
# ---------------------------------------------------------------------
def hinge_d_loss(real_logits, fake_logits):
    """
    Discriminator loss for hinge GAN.

    Expects real_logits and fake_logits as per-sample scalars or maps.
    The hinge loss is:
      mean( relu(1 - real) ) + mean( relu(1 + fake) )

    This encourages real logits > 1 and fake logits < -1.
    """
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    """
    Generator loss for hinge GAN.

    Formula:
      - mean(fake_logits)
    The generator tries to maximize fake_logits, so the loss is negative mean.
    """
    return -fake_logits.mean()


# ---------------------------------------------------------------------
# Low-rank surrogate (nuclear-norm approximation)
# ---------------------------------------------------------------------
def nuclear_norm_surrogate(LR):
    """
    Compute a simple surrogate for the nuclear norm (sum of singular values)
    by performing SVD on the LR map and returning the mean singular value.

    Expected input:
      LR: Tensor of shape [B, 1, H, W] or [B, H, W] (function squeezes the channel dim)

    Behavior:
      - squeezes channel dimension (if present) to obtain [B, H, W]
      - performs batched SVD on each [H, W] matrix
      - returns the mean of the singular values across batches and ranks

    This value acts as a smooth proxy that encourages low-rank structure in LR.
    """
    # remove singleton channel dim if present
    u, s, v = torch.linalg.svd(LR.squeeze(1), full_matrices=False)
    # s has shape [B, min(H,W)] -> returning mean gives a single scalar proxy
    return s.mean()
