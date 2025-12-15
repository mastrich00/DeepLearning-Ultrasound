# src/ultra_spatial/visualize_degradation.py
"""
Utility to visualize synthetic degradations produced by the dataset pipeline.

This script reads clips from the EchoNet dataset via EchoNetClips, applies the
same synthetic-degradation pipeline used for training (gain, TGC, clipping, noise),
and saves triplets for inspection: [clean, degraded, ratio-mask]. The ratio-mask
highlights per-pixel multiplicative changes (degraded / clean) clipped to a sensible range.

Example:
    python -m src.ultra_spatial.visualize_degradation --config configs/default.yaml --out_dir runs/deg_viz --n 8

Output:
    For each sampled clip this writes a PNG with three columns:
      1) clean center frame
      2) degraded center frame
      3) ratio mask (degraded / clean clipped & normalized)
"""

import os
import argparse
import random
import yaml
import torch
from torchvision.utils import save_image

from .data.echonet_dataset import EchoNetClips
from .data.transforms import (
    to_tensor,
)  # not strictly required, dataset returns tensors already


def make_ratio_mask(deg, clean, vmin=0.5, vmax=1.5, eps=1e-6):
    """
    Create a visualization mask that shows local multiplicative change.
    """
    # elementwise ratio; preserve channel dimension
    ratio = deg / (clean + eps)

    # clamp extreme ratios so visualization is not dominated by outliers
    ratio = torch.clamp(ratio, vmin, vmax)

    # scale to 0..1 for visualization (vmin -> 0, vmax -> 1)
    ratio = (ratio - vmin) / (vmax - vmin)
    return ratio


def save_samples_from_dataset(cfg, out_dir, n_samples=8, seed=1337):
    """
    Sample clips from EchoNetClips and save visualization PNGs.

    Each saved PNG contains three columns per sample:
      [clean_center, degraded_center, ratio_mask]
    """
    os.makedirs(out_dir, exist_ok=True)

    # build dataset (no dataloader; we access items directly)
    ds = EchoNetClips(
        cfg["data"]["root"],
        "val",  # visualize from validation split
        frames_per_clip=int(cfg["data"]["frames_per_clip"]),
        frame_size=int(cfg["data"]["frame_size"]),
        fps_subsample=int(cfg["data"]["fps_subsample"]),
        grayscale=bool(cfg["data"]["grayscale"]),
        synth_cfg=cfg["synthesis"],
    )

    # Choose indices to visualize
    random.seed(seed)
    idxs = random.sample(range(len(ds)), min(n_samples, len(ds)))

    for i, idx in enumerate(idxs):
        item = ds[idx]
        degraded = item["degraded"]
        clean = item["clean"]
        meta = item.get("meta", {})

        # robust helper: reduce leading dims until we have [C,H,W] or [H,W]
        def ensure_CHW(t):
            """
            Convert tensor t to shape [C,H,W].

            Strategy:
              - Move tensor to CPU and detach.
              - While the tensor has more than 3 dims, pick the middle slice
                along the first axis (this handles shapes like [T,C,H,W]
                or [1,T,C,H,W], etc).
              - If result is [H,W], add a channel dim to make [1,H,W].
            """
            t = t.detach().cpu()
            # Reduce leading dimensions by picking the middle slice until dims <= 3
            while t.dim() > 3:
                if t.size(0) > 1:
                    t = t[t.size(0) // 2]
                else:
                    t = t.squeeze(0)
            # Now we expect dim 3 ([C,H,W]) or 2 ([H,W])
            if t.dim() == 3:
                return t
            if t.dim() == 2:
                return t.unsqueeze(0)  # make [1,H,W]
            raise RuntimeError(
                f"Unexpected tensor shape {tuple(t.shape)}; expected [C,H,W] or [H,W]"
            )

        # Normalize shapes to [C,H,W]
        c_clean = ensure_CHW(clean)
        c_deg = ensure_CHW(degraded)

        # make sure channel counts match: if one is single-channel and the other multi, broadcast
        if c_clean.size(0) != c_deg.size(0):
            # if one is 1-channel, replicate to match the other
            if c_clean.size(0) == 1 and c_deg.size(0) > 1:
                c_clean = c_clean.expand(c_deg.size(0), -1, -1)
            elif c_deg.size(0) == 1 and c_clean.size(0) > 1:
                c_deg = c_deg.expand(c_clean.size(0), -1, -1)
            else:
                # fall back to channel-averaging both to single channel
                c_clean = c_clean.mean(dim=0, keepdim=True)
                c_deg = c_deg.mean(dim=0, keepdim=True)

        # compute ratio mask (per-channel). Then make a single-channel visualization mask
        mask = make_ratio_mask(c_deg, c_clean)  # same channels as inputs
        if mask.dim() == 3 and mask.size(0) > 1:
            mask_vis = mask.mean(dim=0, keepdim=True)  # collapse to [1,H,W]
        else:
            mask_vis = mask  # already [1,H,W]

        # Prepare arrays for saving with torchvision.utils.save_image.
        # save_image expects a batch of images [N, C, H, W], we stack the three
        # visualization images along the batch dimension:
        #  - clean center frame
        #  - degraded center frame
        #  - ratio mask (single-channel)
        # If the clean/degraded images are multi-channel, we optionally replicate
        # the mask across channels to keep consistent channel counts for display.
        if c_clean.size(0) > 1 and mask_vis.size(0) == 1:
            # replicate mask so save_image shows same channel count (optional)
            mask_save = mask_vis.expand(c_clean.size(0), -1, -1)
        else:
            mask_save = mask_vis

        # Stack to create [3, C, H, W] where first dim indexes column, not batch;
        # save_image will place them in a single row (nrow=3) if we pass nrow=3.
        img_batch = torch.stack([c_clean, c_deg, mask_save], dim=0)

        # Clamp to valid range and save
        img_batch = torch.clamp(img_batch, 0.0, 1.0)

        fname = f"sample_{i:02d}_idx{idx}_bands{meta.get('tgc_bands_count', 'na')}.png"
        path = os.path.join(out_dir, fname)

        # save_image will arrange the stacked images in a grid with nrow=3:
        # [clean | degraded | mask]
        save_image(img_batch, path, nrow=3)
        print(f"saved {path}  meta={meta}")

    print(f"wrote {len(idxs)} samples to {out_dir}")


def main():
    """
    CLI entry point.

    Arguments:
      --config: path to YAML config (same schema as training config)
      --out_dir: where to store visualization PNGs
      --n: number of samples to write
      --seed: RNG seed for reproducible sampling
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to config yaml")
    p.add_argument("--out_dir", type=str, default="runs/deg_viz")
    p.add_argument("--n", type=int, default=8, help="how many samples to generate")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    # Load YAML config (same format as used for training)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    save_samples_from_dataset(cfg, args.out_dir, n_samples=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
