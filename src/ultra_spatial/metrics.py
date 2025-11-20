import torch
from .losses import ssim as ssim_fn


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return torch.tensor(99.0, device=pred.device)
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))


def ssim(pred, target):
    return ssim_fn(pred, target)
