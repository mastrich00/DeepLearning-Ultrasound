import os, random, numpy as np, torch


def set_seed(seed=1337):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    """Move a batch of data to the specified device."""
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def save_checkpoint(state, path):
    """Save model checkpoint to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
