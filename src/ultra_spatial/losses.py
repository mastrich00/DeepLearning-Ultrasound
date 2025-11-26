import torch, torch.nn.functional as F
from math import exp

# LSGAN losses (takes per-sample scalars or maps)
def lsgan_d_loss(real_logits, fake_logits, real_label=1.0, fake_label=0.0):
    # real_logits/fake_logits can be [B] or [B,...]; compute MSE to labels
    r_loss = F.mse_loss(real_logits, torch.full_like(real_logits, real_label))
    f_loss = F.mse_loss(fake_logits, torch.full_like(fake_logits, fake_label))
    return 0.5 * (r_loss + f_loss)

def lsgan_g_loss(fake_logits, real_label=1.0):
    return 0.5 * F.mse_loss(fake_logits, torch.full_like(fake_logits, real_label))

def tv_loss(img):
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dx + dy


def gaussian_window(window_size, sigma, channel):
    import torch

    gauss = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    gauss = gauss / gauss.sum()
    _2D = gauss[:, None] @ gauss[None, :]
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, sigma=1.5, max_val=1.0):
    channel = img1.size(1)
    window = gaussian_window(window_size, sigma, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
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
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def hinge_d_loss(real_logits, fake_logits):
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()


def nuclear_norm_surrogate(LR):
    u, s, v = torch.linalg.svd(LR.squeeze(1), full_matrices=False)
    return s.mean()
