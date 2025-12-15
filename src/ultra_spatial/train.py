# src/ultra_spatial/train.py
import os, argparse, yaml, torch, logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from .utils import set_seed, to_device, save_checkpoint
from .data.echonet_dataset import EchoNetClips
from .models.generator import RetinexLowRankVT
from .models.discriminator import PatchDiscriminator
from .losses import (
    tv_loss,
    nuclear_norm_surrogate,
    ssim as ssim_fn,
    hinge_d_loss,
    hinge_g_loss,
    lsgan_d_loss,
    lsgan_g_loss,
)
from .metrics import psnr as psnr_fn, ssim as ssim_metric

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import csv
import shutil


# ---------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------
def setup_logging(level: str = "INFO"):
    """Configure root logger with a simple timestamped format."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------
# Shape normalization helpers
# ---------------------------------------------------------------------
def _ensure_5d(t):
    """Normalize to [B,T,C,H,W]. Handles shapes like [B,T,T,1,H,W] or extra singletons."""
    if t.dim() == 6 and t.size(2) == t.size(1) and t.size(3) == 1:
        # handle shape like [B, T, T, 1, H, W] -> pick the central duplicate axis
        t = t[:, :, 0]  # -> [B,T,1,H,W]
    # squeeze away any singleton leading dims until 5D
    while t.dim() > 5:
        squeezed = False
        for d in range(t.dim()):
            if t.size(d) == 1:
                t = t.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            break
    if t.dim() == 4:  # [B,C,H,W] -> add trivial time dim
        t = t[:, None]
    return t


def _gen_forward_safe(gen_model, inp):
    """
    Call generator and normalize output to a dict with at least key "corrected".
    Accepts:
      - generator that returns a tensor -> treated as corrected image
      - generator that returns a dict -> must contain "corrected" or a tensor-like value
    Raises TypeError if output is unexpected.
    """
    out = gen_model(inp)
    if isinstance(out, torch.Tensor):
        return {"corrected": out}
    if isinstance(out, dict):
        # some generators might return keys like "pred" or "out"; prefer "corrected"
        if "corrected" in out:
            return out
        # try to find the first tensor-like entry and call it corrected
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                out = dict(out)  # shallow copy
                out["corrected"] = v
                return out
        raise TypeError("Generator returned dict but contains no tensor output.")
    raise TypeError("Generator must return a torch.Tensor or dict.")


# ---------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------
def edge_loss(pred, target):
    """
    Simple edge-preservation loss based on Sobel filter differences.
    Computes L1 loss between gradients of pred and target, summed for x and y.
    This encourages preservation of boundaries and fine texture.
    """
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    C = pred.size(1)
    # repeat kernels across channels
    kx = sobel_x.expand(C, -1, -1, -1)
    ky = sobel_y.expand(C, -1, -1, -1)
    gx_pred = F.conv2d(pred, kx, padding=1, groups=C)
    gy_pred = F.conv2d(pred, ky, padding=1, groups=C)
    gx_t = F.conv2d(target, kx, padding=1, groups=C)
    gy_t = F.conv2d(target, ky, padding=1, groups=C)
    return F.l1_loss(gx_pred, gx_t) + F.l1_loss(gy_pred, gy_t)


# ---------------------------------------------------------------------
# DataLoader collate helper
# ---------------------------------------------------------------------
def collate_keep_meta(batch):
    """
    Custom collate that stacks tensor fields using default_collate but preserves
    'meta' and 'path' as Python lists (not tensors). This is useful so we can
    keep metadata for visualization and logging.
    """
    out = {}
    # handle tensor fields with default_collate, but keep 'meta' as list
    # separate meta first
    metas = [b["meta"] for b in batch]
    paths = [b["path"] for b in batch]

    # build a shallow copy of one item without meta/path
    example = {k: v for k, v in batch[0].items() if k not in ("meta", "path")}
    # use default_collate on those keys
    from torch.utils.data._utils.collate import default_collate

    for k in example.keys():
        out[k] = default_collate([b[k] for b in batch])
    out["meta"] = metas
    out["path"] = paths
    return out


# ---------------------------------------------------------------------
# Data loader factory
# ---------------------------------------------------------------------
def build_loaders(cfg, use_cuda):
    """
    Build train/val/test EchoNet data loaders using the same preprocessing and
    synthetic-degradation pipeline as training. Prints dataset sizes.
    """
    mk = lambda split: EchoNetClips(
        cfg["data"]["root"],
        split,
        frames_per_clip=int(cfg["data"]["frames_per_clip"]),
        frame_size=int(cfg["data"]["frame_size"]),
        fps_subsample=int(cfg["data"]["fps_subsample"]),
        grayscale=bool(cfg["data"]["grayscale"]),
        synth_cfg=cfg["synthesis"],
    )
    tr, va, te = mk("train"), mk("val"), mk("test")
    for n, ds in [("train", tr), ("val", va), ("test", te)]:
        logging.info(f"{n} videos: {len(ds)}")
    dl = lambda ds, sh: DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=sh,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=use_cuda,
        drop_last=True,
        collate_fn=collate_keep_meta,
    )
    return dl(tr, True), dl(va, False), dl(te, False)


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def save_samples(save_dir, epoch, inp, pred, gt):
    """
    Save a small sample grid for quick visual inspection.
    The grid is arranged in 3 rows: input, prediction, ground-truth.
    """
    import os, torch
    from torchvision.utils import save_image

    os.makedirs(save_dir, exist_ok=True)
    k = min(4, inp.size(0), pred.size(0), gt.size(0))  # show up to 4 columns
    grid = torch.cat([inp[:k], pred[:k], gt[:k]], dim=0)  # 3*k images
    path = os.path.join(save_dir, f"epoch{epoch:03d}.png")
    save_image(grid, path, nrow=k)  # nrow=k -> 3 rows
    print(f"saved samples -> {path}")


# ---------------------------------------------------------------------
# Discriminator feature visualization helpers
# ---------------------------------------------------------------------
def _collect_disc_features(D, x):
    """
    Attempt to collect intermediate discriminator feature maps for feature-matching
    visualization. Prefer D.features(x) if provided by the discriminator; otherwise
    register short-lived forward hooks on the first few Conv2d layers.

    Returns a list of tensors [B, C, H, W].
    """
    # try discriminator-provided features
    try:
        feats = D.features(x)
        if isinstance(feats, (list, tuple)):
            return [f.detach() for f in feats if isinstance(f, torch.Tensor)]
        if isinstance(feats, torch.Tensor):
            return [feats.detach()]
    except Exception:
        pass

    # fallback: register short-lived hooks on Conv2d modules
    acts = {}
    hooks = []

    def _make_hook(name):
        def _hook(mod, inp, out):
            # detach to avoid storing computational graph
            acts[name] = out.detach()

        return _hook

    # choose a subset of conv modules to avoid huge output (first few convs)
    # iterate until we have up to N hooks to limit overhead
    N = 6
    count = 0
    for name, module in D.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(_make_hook(name)))
            count += 1
            if count >= N:
                break

    # run forward once to fill hooks
    _ = D(x)

    # remove hooks
    for h in hooks:
        h.remove()

    # return features sorted by registration order
    ordered = [acts[k] for k in sorted(acts.keys())]
    return ordered


def _save_activation_maps(
    feat_list, out_dir, prefix="disc", epoch=None, batch_idx=None, max_cols=4
):
    """
    Save activation maps (channel-mean) as PNG grids for inspection.
    Each feature tensor yields a small PNG showing the mean over channels for the first few batch elements.
    """
    os.makedirs(out_dir, exist_ok=True)
    for li, feat in enumerate(feat_list):
        # reduce channels -> mean across channels (keeps spatial structure)
        # result shape [B,1,H,W]
        ch_mean = feat.mean(dim=1, keepdim=True)  # [B,1,H,W]
        # normalize each sample to 0..1 to make visualization meaningful
        B = ch_mean.size(0)
        imgs = []
        for b in range(B):
            im = ch_mean[b : b + 1]  # [1,1,H,W]
            mn = float(im.min())
            mx = float(im.max())
            imn = (im - mn) / (mx - mn + 1e-8)
            imgs.append(imn)
        imgs = torch.cat(imgs, dim=0)  # [B,1,H,W]
        k = min(max_cols, imgs.size(0))
        grid = make_grid(imgs[:k], nrow=k, normalize=False)  # keep already normalized
        name = f"{prefix}_L{li:02d}"
        if epoch is not None and batch_idx is not None:
            fname = os.path.join(out_dir, f"{name}_e{epoch:03d}_b{batch_idx:04d}.png")
        elif epoch is not None:
            fname = os.path.join(out_dir, f"{name}_e{epoch:03d}.png")
        else:
            fname = os.path.join(out_dir, f"{name}.png")
        save_image(grid, fname)


# ---------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------
def train_epoch(
    cfg, device, gen, D, opt_g, opt_d, scaler, loader, use_gan, use_amp, step_log=20
):
    """
    One epoch of training. Handles both generator and discriminator updates.

    Arguments:
      cfg: configuration dict
      device: torch.device
      gen, D: models
      opt_g, opt_d: optimizers
      scaler: GradScaler or None (for AMP)
      loader: data loader
      use_gan: bool, whether to run discriminator steps
      use_amp: bool, whether to use AMP autocast and GradScaler
      step_log: how often to update tqdm postfix
    """
    gen.train()
    if use_gan:
        D.train()
    w = cfg["loss"]
    total_g = 0.0
    total_d = 0.0
    act_every = int(cfg.get("train", {}).get("act_save_every", 0))  # 0 = disabled
    act_dir = (
        os.path.join(cfg["train"]["save_dir"], "disc_acts")
        if cfg.get("train", {}).get("save_dir")
        else None
    )

    # select GAN loss family
    gan_type = cfg.get("train", {}).get("gan_type", "lsgan").lower()
    if gan_type == "hinge":
        d_loss_fn = hinge_d_loss
        g_adv_fn = hinge_g_loss
    else:
        d_loss_fn = lsgan_d_loss
        g_adv_fn = lsgan_g_loss

    # AMP context only when CUDA+AMP is on
    if use_amp:
        from torch.cuda.amp import autocast

        def amp_ctx():
            return autocast(enabled=True)

    else:

        class _Null:  # no-op context
            def __enter__(self):
                return None

            def __exit__(self, *a):
                return False

        def amp_ctx():
            return _Null()

    pbar = tqdm(loader, desc="train", dynamic_ncols=True, leave=False)
    for i, batch in enumerate(pbar, 1):
        # move batch to device and normalize shapes
        batch = to_device(batch, device)
        x = _ensure_5d(batch["degraded"])
        y = _ensure_5d(batch["clean"])
        y_mid = y[:, y.shape[1] // 2]  # center-frame target

        # --- Generator forward (single forward used by both D and G steps) ---
        with amp_ctx():
            out = _gen_forward_safe(gen, x)  # normalized dict
            pred = out.get("corrected", None)  # predicted center-frame [B,C,H,W]
            if pred is None:
                raise KeyError("Generator output missing 'corrected' tensor.")
            # convert tanh->[0,1] if generator used tanh for direct output
            if pred.min().item() < -0.01:
                pred = (pred + 1.0) * 0.5
                pred = torch.clamp(pred, 0.0, 1.0)

        # --- Discriminator updates ---
        if use_gan:
            d_steps = int(cfg.get("train", {}).get("d_steps", 1))
            for d_iter in range(d_steps):
                with amp_ctx():
                    fake = pred.detach()
                    real = y_mid
                    D_real_map = D(real)  # patch-wise realism map for real images
                    D_fake_map = D(fake)  # same for fake images

                    # optional: save discriminator activations for debugging/analysis
                    if act_every > 0 and (i % act_every == 0):
                        try:
                            # collect features for both real and fake (use only first N samples to reduce IO)
                            # choose a small subset of the batch to visualize (e.g., first 4)
                            sel_real = real[: min(4, real.size(0))]
                            sel_fake = fake[: min(4, fake.size(0))]

                            feats_real = _collect_disc_features(D, sel_real)
                            feats_fake = _collect_disc_features(D, sel_fake)

                            # save into run folder
                            out_dir = act_dir or cfg["train"]["save_dir"]
                            if out_dir is None:
                                out_dir = "runs"
                            # create epoch-aware subfolder if epoch available
                            e = epoch if "epoch" in locals() else None

                            _save_activation_maps(
                                feats_real,
                                out_dir,
                                prefix="D_real",
                                epoch=e,
                                batch_idx=i,
                            )
                            _save_activation_maps(
                                feats_fake,
                                out_dir,
                                prefix="D_fake",
                                epoch=e,
                                batch_idx=i,
                            )
                        except Exception as ex:
                            # do not crash training for visualization errors
                            logging.debug(
                                f"Could not save discriminator activations: {ex}"
                            )
                    # reduce spatial/patch dims to a per-sample score (keep batch dim)
                    D_real_per_sample = D_real_map.view(D_real_map.size(0), -1).mean(
                        dim=1
                    )
                    D_fake_per_sample = D_fake_map.view(D_fake_map.size(0), -1).mean(
                        dim=1
                    )
                    d_loss = d_loss_fn(D_real_per_sample, D_fake_per_sample)
                opt_d.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(d_loss).backward()
                    scaler.step(opt_d)
                    scaler.update()
                else:
                    d_loss.backward()
                    opt_d.step()
                total_d += float(d_loss.detach())

        # debug printing of discriminator maps for the first iteration if debug on
        if (i == 1) and cfg.get("train", {}).get("debug", False):
            print(
                "\nD_real_map mean/min/max:",
                float(D_real_map.detach().cpu().mean()),
                float(D_real_map.detach().cpu().min()),
                float(D_real_map.detach().cpu().max()),
            )
            print(
                "\nD_fake_map mean/min/max:",
                float(D_fake_map.detach().cpu().mean()),
                float(D_fake_map.detach().cpu().min()),
                float(D_fake_map.detach().cpu().max()),
            )

        # --- Generator loss composition ---
        with amp_ctx():
            # optional residual output from generator (additive correction)
            residual = out.get("residual", None)

            if cfg.get("train", {}).get("debug", False):
                # detach to avoid autograd converting tensors to scalars (avoids the warning)
                pmin = pred.detach().cpu().min().item()
                pmax = pred.detach().cpu().max().item()
                print(f"\npred min/max: {pmin:.6f} {pmax:.6f}")

            # Build a mask that highlights degraded regions (used for masked L1)
            x_mid = x[:, x.shape[1] // 2]
            eps = 1e-6
            rel = torch.abs(y_mid - x_mid) / (
                torch.abs(x_mid) + eps
            )  # relative difference
            mask = rel.mean(dim=1, keepdim=True)  # [B,1,H,W]
            # lower threshold to include subtle bands, amplify mask contrast
            mask_thresh = float(cfg.get("synthesis", {}).get("mask_thresh", 0.01))
            # robust per-sample normalization by max
            per_sample_max = (
                mask.view(mask.size(0), -1).max(dim=1)[0].view(mask.size(0), 1, 1, 1)
            )
            denom = per_sample_max + 1e-8
            mask = torch.clamp((mask - mask_thresh) / denom, 0.0, 1.0)

            # widen mask with avg_pool to make bands broader
            widen_k = int(cfg.get("synthesis", {}).get("mask_widen", 7))
            if widen_k > 1:
                mask = torch.nn.functional.avg_pool2d(
                    mask, kernel_size=widen_k, stride=1, padding=widen_k // 2
                )
            # normalize mask to keep losses stable across samples
            mask_mean = mask.mean(dim=[1, 2, 3], keepdim=True)
            mask_norm = mask / (mask_mean + 1e-6)

            # masked L1
            l1_masked = (torch.abs(pred - y_mid) * mask_norm).mean()
            l1_bg = torch.nn.functional.l1_loss(pred, y_mid)
            # combine with strong masked weight, tiny background weight
            l1 = (
                float(cfg["loss"].get("w_l1_masked", 1.0)) * l1_masked
                + float(cfg["loss"].get("w_l1_bg", 0.0)) * l1_bg
            )

            # structural fidelity
            ssim_loss = 1.0 - ssim_fn(pred, y_mid)

            # total variation regularizer on illumination (or fallback on pred)
            tv = torch.tensor(0.0, device=pred.device)
            tv_input = out.get("I", None)
            if tv_input is None:
                try:
                    tv = tv_loss(pred)
                except Exception:
                    tv = torch.tensor(0.0, device=pred.device)
            else:
                try:
                    # if tv_input might be multi-scale or not normalized, guard in try/except
                    tv = tv_loss(tv_input)
                except Exception:
                    tv = torch.tensor(0.0, device=pred.device)

            # low-rank nuclear norm surrogate if LR head present
            nuc = 0.0
            if "LR" in out and out["LR"] is not None:
                try:
                    nuc = nuclear_norm_surrogate(out["LR"])
                except Exception:
                    nuc = 0.0

            # residual supervision and regularization (optional generator output)
            res_sup = torch.tensor(0.0, device=pred.device)
            res_mag = torch.tensor(0.0, device=pred.device)
            res_tv = torch.tensor(0.0, device=pred.device)

            if residual is not None:
                r_target = (y_mid - x_mid).detach()
                if residual.shape != r_target.shape:
                    # try to broadcast/squeeze trivial dims, otherwise skip supervised residual
                    try:
                        residual = residual.view(r_target.shape)
                    except Exception:
                        residual = None

                if residual is not None:
                    # full-image L1 supervision (option)
                    if cfg["loss"].get("w_residual_supervised", 0.0) > 0.0:
                        if cfg["loss"].get("residual_masked", True):
                            res_sup = (
                                torch.abs(residual - r_target) * mask_norm
                            ).mean()
                        else:
                            res_sup = torch.nn.functional.l1_loss(residual, r_target)

                    # sparsity / magnitude penalty (keeps residual small)
                    if cfg["loss"].get("w_residual_mag", 0.0) > 0.0:
                        res_mag = residual.abs().mean()

                    # TV on residual to favor smooth additive corrections
                    if cfg["loss"].get("w_residual_tv", 0.0) > 0.0:
                        res_tv = tv_loss(residual)

            # identity loss: check generator stability on already-clean inputs
            id_l = torch.tensor(0.0, device=pred.device)
            if float(w.get("w_identity", 0.0)) > 0.0:
                try:
                    id_out = _gen_forward_safe(gen, y)
                    id_pred = id_out.get("corrected", None)
                    if id_pred is not None:
                        if id_pred.min().item() < -0.01:
                            id_pred = (id_pred + 1.0) * 0.5
                            id_pred = torch.clamp(id_pred, 0.0, 1.0)
                        id_l = torch.nn.functional.l1_loss(id_pred, y_mid)
                except Exception:
                    # skip identity if forward fails (OOM or incompatible shapes)
                    id_l = torch.tensor(0.0, device=pred.device)

            # combine base losses with configured weights
            g_loss = (
                float(w["w_l1"]) * l1
                + float(w["w_ssim"]) * ssim_loss
                + float(w["w_tv"])
                * (
                    tv
                    if isinstance(tv, torch.Tensor)
                    else torch.tensor(tv, device=pred.device)
                )
                + float(w["w_lowrank_nuc"])
                * (
                    nuc
                    if isinstance(nuc, torch.Tensor)
                    else torch.tensor(nuc, device=pred.device)
                )
                + float(w["w_identity"]) * id_l
            )

            # add residual penalties if configured
            g_loss = g_loss + float(w.get("w_residual_supervised", 0.0)) * res_sup
            g_loss = g_loss + float(w.get("w_residual_mag", 0.0)) * res_mag
            g_loss = g_loss + float(w.get("w_residual_tv", 0.0)) * res_tv

            # adversarial and auxiliary GAN-related terms
            if use_gan:
                # adversarial hinge or lsgan loss evaluated on D(pred)
                D_pred_map = D(pred)
                D_pred_per_sample = D_pred_map.view(D_pred_map.size(0), -1).mean(dim=1)
                adv = g_adv_fn(D_pred_per_sample)
                g_loss = g_loss + float(w.get("lambda_adv", 0.02)) * adv

                # feature matching: compare discriminator internal features for stability
                try:
                    feat_real = D.features(y_mid)
                    feat_fake = D.features(pred)
                    fm = 0.0
                    for fr, ff in zip(feat_real, feat_fake):
                        # per-sample spatial+channel mean (tensor shape [B])
                        frm = fr.view(fr.size(0), -1).mean(dim=1).detach()
                        ffm = ff.view(ff.size(0), -1).mean(dim=1)
                        fm = fm + torch.nn.functional.l1_loss(ffm, frm)
                    g_loss = g_loss + float(w.get("w_fm", 5.0)) * fm
                except Exception:
                    # ignore if D does not expose features
                    pass

                # edge gradient loss to encourage preserved boundaries / speckle-like gradients
                e_l = edge_loss(pred, y_mid)
                g_loss = g_loss + float(w.get("w_edge", 0.05)) * e_l

        # --- Generator optimizer step ---
        opt_g.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(g_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                gen.parameters(), float(cfg["train"]["grad_clip"])
            )
            scaler.step(opt_g)
            scaler.update()
        else:
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                gen.parameters(), float(cfg["train"]["grad_clip"])
            )
            opt_g.step()
        total_g += float(g_loss.detach())

        # update tqdm postfix periodically
        if i % step_log == 0:
            pbar.set_postfix(g=total_g / i, d=(total_d / i if use_gan else 0.0))

        # detailed debug logging (periodic) for diagnostics
        if (i % cfg["train"].get("debug_step", 200) == 1) and cfg["train"].get(
            "debug", False
        ):
            # baseline L1 between degraded input and clean target
            l1_degraded = torch.nn.functional.l1_loss(
                x_mid.detach(), y_mid.detach()
            ).item()
            l1_pred_clean = torch.nn.functional.l1_loss(
                pred.detach(), y_mid.detach()
            ).item()
            impr = (l1_degraded - l1_pred_clean) / (l1_degraded + 1e-8)
            res = out.get("residual", None)
            if res is not None:
                res_mean = float(res.abs().mean().detach().cpu())
                res_nonzero_frac = float(
                    (res.abs() > 1e-4).float().mean().detach().cpu()
                )
            else:
                res_mean = 0.0
                res_nonzero_frac = 0.0
            logging.info(
                f"L1(deg,clean)={l1_degraded:.4f}, L1(pred,clean)={l1_pred_clean:.4f}, "
                f"res_sup={res_sup.item():.4f}, res_mag={res_mag.item():.4f}, res_tv={res_tv.item():.4f}, "
                f"res_mean={res_mean:.4f}, res_frac={res_nonzero_frac:.3f}, improvement={impr:.3f}"
            )

    n = max(1, len(loader))
    return total_g / n, (total_d / n if use_gan else 0.0)


# ---------------------------------------------------------------------
# Plotting / CSV utilities for training summary
# ---------------------------------------------------------------------
def _save_training_plots_and_csv(history, out_dir):
    """
    Save training loss plot, validation metrics plot, and CSV log of epoch-wise stats.
    'history' is expected to be a dict with keys "g", "d", "psnr", "ssim".
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["g"]) + 1))

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["g"], label="Generator loss", marker="o")
    plt.plot(epochs, history["d"], label="Discriminator loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training losses")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "losses.png"))
    plt.close()

    # Metrics plot (PSNR left, SSIM right)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["psnr"], label="PSNR", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["ssim"], label="SSIM", color="tab:orange", marker="o")
    ax2.set_ylabel("SSIM")

    # build combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    plt.title("Validation metrics")
    fig.tight_layout()
    figpath = os.path.join(out_dir, "metrics.png")
    fig.savefig(figpath)
    plt.close(fig)

    # Save CSV log
    csv_path = os.path.join(out_dir, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "g_loss", "d_loss", "psnr", "ssim"])
        for i in range(len(epochs)):
            writer.writerow(
                [
                    i + 1,
                    history["g"][i],
                    history["d"][i],
                    history["psnr"][i],
                    history["ssim"][i],
                ]
            )

    print(f"Saved training plots and csv to {out_dir}")


# ---------------------------------------------------------------------
# Evaluation loop (no grad)
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(cfg, device, gen, loader, epoch=None, save_dir=None):
    """
    Run generator on validation/test loader, collect PSNR and SSIM, and save
    the first batch as visualization using save_samples().
    """
    gen.eval()
    psnr_list, ssim_list = [], []
    first_batch = None
    for batch in loader:
        batch = to_device(batch, device)
        x = _ensure_5d(batch["degraded"])  # [B,T,1,H,W]
        y = _ensure_5d(batch["clean"])  # [B,T,1,H,W]
        out = gen(x)
        pred = out["corrected"]  # [B,1,H,W]
        pred = torch.clamp(pred, 0.0, 1.0)
        mid = y.shape[1] // 2
        y_mid = y[:, mid]  # target [B,1,H,W]
        x_mid = x[:, mid]  # input  [B,1,H,W]

        psnr_list.append(float(psnr_fn(pred, y_mid)))
        ssim_list.append(float(ssim_metric(pred, y_mid)))

        if first_batch is None:
            # store input, pred, target to visualize
            first_batch = (
                x_mid.detach().cpu(),
                pred.detach().cpu(),
                y_mid.detach().cpu(),
            )

    metrics = {
        "psnr": sum(psnr_list) / len(psnr_list),
        "ssim": sum(ssim_list) / len(ssim_list),
    }
    if (epoch is not None) and (save_dir is not None) and first_batch is not None:
        inp, prd, gt = first_batch
        save_samples(save_dir, epoch, inp, prd, gt)
    return metrics


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------
def main(args):
    setup_logging(args.log_level)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    use_cuda = (str(cfg["device"]).lower() == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = use_cuda and bool(cfg["train"]["amp"])
    logging.info(
        f"device={device}, use_amp={use_amp}, cuda_available={torch.cuda.is_available()}"
    )

    tr, va, te = build_loaders(cfg, use_cuda)

    # --- generator selection: retinex (default) or pix2pix ---
    gan_variant = cfg.get("train", {}).get("gan_variant", "retinex").lower()
    if gan_variant == "pix2pix":
        # optional UNet generator import if requested by config
        from .models.pix2pix import UNetGenerator

        in_ch = 1 if bool(cfg["data"].get("grayscale", True)) else 3
        gen = UNetGenerator(in_ch=in_ch, out_ch=in_ch, ngf=64).to(device)
        logging.info("Using pix2pix UNet generator")
    else:
        # default RetinexLowRankVT generator (matches proposal)
        gen = RetinexLowRankVT(
            enc_channels=list(cfg["model"]["enc_channels"]),
            n_heads=int(cfg["model"]["n_heads"]),
            n_layers=int(cfg["model"]["n_layers"]),
            lowrank_rank=int(cfg["model"]["lowrank_rank"]),
            illumination_coarse=int(cfg["model"]["illumination_coarse"]),
        ).to(device)
        logging.info("Using RetinexLowRankVT generator (default)")

    # spatial PatchGAN discriminator
    D = PatchDiscriminator().to(device)

    # optimizers (AdamW with common GAN betas)
    opt_g = torch.optim.AdamW(
        gen.parameters(),
        lr=float(cfg["train"]["lr_g"]),
        betas=(0.5, 0.999),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    opt_d = torch.optim.AdamW(
        D.parameters(),
        lr=float(cfg["train"]["lr_d"]),
        betas=(0.5, 0.999),
        weight_decay=0.0,
    )

    # GradScaler created only when using AMP on CUDA to avoid CPU warnings
    scaler = None
    if use_amp:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler(enabled=True)

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    # file logging: add a FileHandler so logs are saved into the run folder
    try:
        log_path = os.path.join(cfg["train"]["save_dir"], "train.log")
        # create a file handler which logs even debug messages
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
        fh.setFormatter(fmt)
        # attach to root logger so all module logging is captured
        logging.getLogger().addHandler(fh)

        # optionally capture warnings from the 'warnings' module into the logging system
        logging.captureWarnings(True)

        logging.info(f"logging to file: {log_path}")
    except Exception as e:
        logging.warning(f"could not create log file in {cfg['train']['save_dir']}: {e}")

    # save a copy of the used config for reproducibility
    try:
        cfg_dst = os.path.join(cfg["train"]["save_dir"], "config_used.yaml")
        shutil.copy2(args.config, cfg_dst)
    except Exception as e:
        # if copy fails (e.g. config came from stdin or is unavailable), fall back to writing parsed cfg
        try:
            with open(
                os.path.join(cfg["train"]["save_dir"], "config_used_parsed.yaml"), "w"
            ) as fh:
                yaml.safe_dump(cfg, fh)
        except Exception:
            logging.warning(f"Could not save config copy: {e}")

    # history container for plotting and CSV
    history = {"g": [], "d": [], "psnr": [], "ssim": []}
    samples_dir = os.path.join(cfg["train"]["save_dir"], "samples")
    best = -1.0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        logging.info(f"epoch {epoch} / {cfg['train']['epochs']}  gan={args.gan}")
        g_loss, d_loss = train_epoch(
            cfg,
            device,
            gen,
            D,
            opt_g,
            opt_d,
            scaler,
            tr,
            use_gan=args.gan,
            use_amp=use_amp,
            step_log=20,
        )
        val = evaluate(cfg, device, gen, va, epoch=epoch, save_dir=samples_dir)
        logging.info(
            f"val: psnr={val['psnr']:.3f}  ssim={val['ssim']:.4f}  G={g_loss:.4f}  D={d_loss:.4f}"
        )
        # append epoch stats
        history["g"].append(float(g_loss))
        history["d"].append(float(d_loss))
        history["psnr"].append(float(val["psnr"]))
        history["ssim"].append(float(val["ssim"]))

        # checkpoint last and best by SSIM
        ckpt = {"model": gen.state_dict(), "cfg": cfg, "epoch": epoch}
        save_checkpoint(ckpt, os.path.join(cfg["train"]["save_dir"], "last.pt"))
        if val["ssim"] > best:
            best = val["ssim"]
            save_checkpoint(ckpt, os.path.join(cfg["train"]["save_dir"], "best.pt"))
            logging.info("saved new best checkpoint")

    # load best and evaluate on test set
    state = torch.load(
        os.path.join(cfg["train"]["save_dir"], "best.pt"), map_location="cpu"
    )
    gen.load_state_dict(state["model"])
    test = evaluate(cfg, device, gen, te)
    logging.info(f"test: psnr={test['psnr']:.3f}  ssim={test['ssim']:.4f}")
    # Save final training plots + CSV
    _save_training_plots_and_csv(history, cfg["train"]["save_dir"])


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")

    def str2bool(v):
        """Flexible boolean parser for argparse (accepts many True/False strings)."""
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in ("yes", "y", "true", "t", "1", ""):
            return True
        if v in ("no", "n", "false", "f", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    p.add_argument(
        "--gan",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="enable GAN training (accepts yes/no/true/false; `--gan` sets True)",
    )
    p.add_argument("--log_level", type=str, default="INFO", help="DEBUG, INFO, WARNING")
    args = p.parse_args()
    main(args)
