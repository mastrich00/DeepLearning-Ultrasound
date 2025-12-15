# src/ultra_spatial/data/transforms.py
"""
Image / clip transformation utilities used by the dataset and synthesis pipeline.

Contains:
 - resize_center_crop: center-crop to square and resize.
 - to_tensor: stack numpy frames and convert to torch tensor in CHW layout.
 - 1D interpolators used for TGC ramps.
 - apply_synthetic_degradation: the core synthetic "bad settings" pipeline that
   applies global gain, TGC-style depth bands/ramps, dynamic-range clipping and
   mild sensor noise. Returns a degraded clip and a metadata dict describing
   the sampled corruptions.
"""

import numpy as np
import torch
import cv2
import torch.nn.functional as F


def resize_center_crop(img, size):
    """
    Center-crop the input image to a square and resize to (size, size).
    """
    h, w = img.shape[:2]
    # take the largest centered square
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    img = img[y0 : y0 + s, x0 : x0 + s]
    # use area interpolation for downsampling (good for images)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(frames):
    """
    Convert a list/array of frames (H,W) or (H,W,C) to a torch tensor.

    Behavior:
      - Input expected as a sequence of numpy arrays in 0..255 (uint8).
      - Stack along time -> shape [T, H, W] or [T, H, W, C].
      - Normalize to float32 in [0,1] and return a tensor shaped [T, C, H, W].

    Returns:
      torch.Tensor: float32 tensor in [0,1].
    """
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    if arr.ndim == 3:
        # grayscale stack -> [T, H, W] -> make channel dim: [T, 1, H, W]
        arr = arr[:, None, :, :]
    else:
        # color stack [T, H, W, C] -> reorder to [T, C, H, W]
        arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


# --- small interpolation helpers used for TGC ramps ---------------------
def _interp1d_spline(ctrl, H):
    """
    Bicubic interpolation of control points to length H.
    ctrl: 1D tensor of control values.
    Returns a 1D tensor of length H.
    """
    return F.interpolate(
        ctrl.view(1, 1, 1, -1), size=(1, H), mode="bicubic", align_corners=True
    ).view(H)


def _interp1d_linear(ctrl, H):
    """
    Linear 1D interpolation from K control points to H samples using torch.interp.
    This is a simple, efficient fallback when spline behavior is not desired.
    """
    K = ctrl.shape[0]
    yk = torch.linspace(0, 1, steps=K, device=ctrl.device)
    y = torch.linspace(0, 1, steps=H, device=ctrl.device)
    return torch.interp(y, yk, ctrl)


# --- main synthetic degradation pipeline --------------------------------
def apply_synthetic_degradation(clip, cfg, rng=None):
    """
    Apply physics-inspired synthetic exposure degradations to a clip.

    The pipeline (high level):
      1) Apply a global gain (per-clip base with optional per-frame jitter).
      2) Apply a depth-dependent TGC-style multiplicative field. Two modes:
         - "bands": generated curved U-shaped bands with variable width/strength.
         - "ramp"/"spline": 1D profile from K control points interpolated along depth.
      3) Dynamic-range clipping (random low/high percentiles) with optional
         gentle blending back of pre-normalized texture to preserve speckle.
      4) Additive and multiplicative Gaussian noise to simulate sensor noise.
      5) Return clipped result in [0,1] and a meta dict describing sampled params.

    Args:
      clip (torch.Tensor): input clip [T, C, H, W] with values in [0,1].
      cfg (dict): synthesis configuration read from YAML (controls ranges).
      rng (np.random.Generator or None): RNG for reproducibility.

    Returns:
      (torch.Tensor, dict): degraded clip and metadata
    """
    if rng is None:
        rng = np.random.default_rng()

    # probability to skip degradation (useful for mixing clean samples)
    if rng.random() > cfg.get("p_apply", 1.0):
        return clip.clone(), {"applied": False}

    y = clip.clone()
    T, C, H, W = y.shape
    dev = y.device
    meta = {"applied": True}

    # ----------------------- Global gain --------------------------------
    # sample a base gain for the clip and add small per-frame Gaussian jitter
    g0 = float(rng.uniform(cfg.get("gain_min", 0.6), cfg.get("gain_max", 1.6)))
    sigma_g = cfg.get("gain_sigma", 0.0)
    eps_g = torch.from_numpy(rng.normal(0, sigma_g, size=(T,))).to(dev).float()
    g = torch.clamp(eps_g + g0, 0.2, 2.5).view(T, 1, 1, 1)  # [T,1,1,1]
    y = torch.clamp(y * g, 0.0, 1.0)
    meta["gain_base"] = g0

    # ---------- TGC-style depth field ----------------------------------
    # two supported modes: "bands" (curved local bands) and fallback ramps/spline
    tgc_style = cfg.get("tgc_mode", cfg.get("tgc_style", "bands"))

    if tgc_style == "bands":
        # ----- sample band parameters -----
        max_bands = int(cfg.get("tgc_bands_max", cfg.get("tgc_K", 6)))
        min_bands = int(cfg.get("tgc_bands_min", 1))
        band_strength = float(
            cfg.get("tgc_band_strength", cfg.get("tgc_strength", 0.6))
        )
        band_strength_max = float(
            cfg.get("tgc_band_strength_max", max(band_strength, 1.0))
        )
        band_width_min = float(cfg.get("tgc_band_width_min", 0.03))
        band_width_max = float(cfg.get("tgc_band_width_max", 0.15))
        abrupt_prob = float(cfg.get("tgc_abrupt_prob", 0.25))
        abrupt_sharpness = float(cfg.get("tgc_abrupt_sharpness", 60.0))
        lateral_perturb = float(cfg.get("tgc_lateral_perturb", 0.0))
        time_jitter = float(cfg.get("tgc_time_jitter", 0.02))
        contrast_exp = float(cfg.get("tgc_contrast_exponent", 1.0))

        # config safety knobs to avoid full blackouts / preserve texture
        tgc_min_mul = float(
            cfg.get("tgc_min_mul", 0.35)
        )  # min multiplicative factor per-pixel
        tgc_max_mul = float(
            cfg.get("tgc_max_mul", 2.0)
        )  # max multiplicative factor per-pixel
        tgc_min_residual = float(
            cfg.get("tgc_min_residual", 0.06)
        )  # min brightness after normalization
        tgc_preserve_blend = float(
            cfg.get("tgc_preserve_blend", 0.12)
        )  # blend pre-normalized texture back

        # choose how many bands to draw for this clip
        n_bands = int(rng.integers(min_bands, max_bands + 1))
        band_specs = []
        for b in range(n_bands):
            # y_edge = edge height (normalized 0..1) representing the left/right start height
            y_edge = float(
                rng.uniform(0.08, 0.75)
            )  # avoid extreme top/bottom by default
            # depth = how deep the U goes at center (normalized fraction of H)
            depth = float(rng.uniform(0.03, 0.35))
            # width controls vertical spread around the U-curve (fraction of H)
            width = float(rng.uniform(band_width_min, band_width_max))
            # intensity magnitude (≥0): sample magnitude then allow bright/dark
            mag = float(rng.uniform(band_strength * 0.5, band_strength_max))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            intensity = 1.0 + sign * mag
            abrupt = rng.random() < abrupt_prob
            band_specs.append(
                {
                    "edge": y_edge,
                    "depth": depth,
                    "width": width,
                    "intensity": intensity,
                    "abrupt": abrupt,
                }
            )

        # prepare coordinate grid once
        xs = torch.linspace(0, W - 1, W, device=dev)
        ys = torch.linspace(0, H - 1, H, device=dev)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # yy: [H,W], xx: [H,W]
        x_norm = (xx - (W - 1) / 2.0) / ((W - 1) / 2.0)  # normalized -1 .. 1

        ramps = []
        for t in range(T):
            # per-frame lateral waviness optionally
            xx_t = xx
            if lateral_perturb > 0.0:
                freq = float(rng.uniform(0.5, 2.0))
                phase = float(rng.uniform(0.0, 2 * np.pi))
                lateral = (
                    torch.sin((xx / W) * (2 * np.pi * freq) + phase)
                    * lateral_perturb
                    * H
                )
                # lateral is in pixels and added to the vertical coordinate to warp the curve
                xx_t = (
                    xx + lateral * 0.0
                )  # keep xx unchanged; lateral will be used by shifting x_norm below if desired

            total_mask = torch.ones((H, W), device=dev)

            for bspec in band_specs:
                # per-band small time jitter to vary band placement across frames
                y_edge = bspec["edge"] + float(rng.normal(0, time_jitter))
                y_edge = float(np.clip(y_edge, 0.01, 0.95))
                depth = bspec["depth"]
                width = bspec["width"]
                intensity = bspec["intensity"]
                abrupt = bspec["abrupt"]

                # compute a U-shaped curve y_curve(x) in pixel coords:
                y_edge_px = y_edge * (H - 1)
                depth_px = depth * (H - 1)
                # curve: y_curve(x) = y_edge_px + depth_px * (1 - x_norm^2)
                y_curve = y_edge_px + depth_px * (
                    1.0 - x_norm**2
                )  # shape [H,W] but constant along rows in x

                # compute vertical distance from each pixel to curve (positive when pixel is deeper than curve)
                vdist = yy - y_curve  # shape [H,W]

                # convert width fraction to pixel sigma
                widen_factor = float(cfg.get("tgc_widen_factor", 1.6))  # default 1.6
                sigma_pixels = max(1.0, width * H * widen_factor)

                if abrupt:
                    # sharp box-like band around the curve:
                    half_h = sigma_pixels  # use width as half height for box
                    left = -half_h
                    right = half_h
                    # smooth step around vdist in pixels
                    left_step = 0.5 * (
                        1.0 + torch.tanh((vdist - left) * abrupt_sharpness / H)
                    )
                    right_step = 0.5 * (
                        1.0 + torch.tanh((vdist - right) * abrupt_sharpness / H)
                    )
                    band_mask = left_step - right_step  # ~1 inside band, 0 outside
                    # optionally amplify center using contrast exponent (<1 amplifies)
                    if contrast_exp != 1.0:
                        band_mask = torch.clamp(band_mask, 0.0, 1.0).pow(
                            max(1e-3, contrast_exp)
                        )
                    band_mul = 1.0 + (intensity - 1.0) * band_mask
                else:
                    # smooth Gaussian band perpendicular to the curve
                    band_mask = torch.exp(-0.5 * (vdist**2) / (sigma_pixels**2 + 1e-12))
                    if contrast_exp != 1.0:
                        band_mask = torch.clamp(band_mask, 0.0, 1.0).pow(
                            max(1e-3, contrast_exp)
                        )
                    band_mul = 1.0 + (intensity - 1.0) * band_mask

                # accumulate multiplicative effects of multiple bands
                total_mask = total_mask * band_mul

            # clamp mask to safe numeric range and add channel/time dims
            total_mask = torch.clamp(total_mask, 0.01, 6.0).view(1, 1, H, W)
            ramps.append(total_mask)

        ramp = torch.stack(ramps, dim=0)  # [T,1,H,W]
        a = float(cfg.get("tgc_strength", 1.0))
        ramp = 1.0 + a * (ramp - 1.0)

        # --- clamp per-pixel multiplicative factors to safe bounds to avoid full blackouts
        ramp = torch.clamp(ramp, tgc_min_mul, tgc_max_mul)

        meta["tgc_style"] = "bands_u"
        meta["tgc_bands_count"] = n_bands

    else:
        # fallback: original K-control ramp (linear / spline)
        a = cfg.get("tgc_strength", 0.6)
        K = int(cfg.get("tgc_K", 6))
        sigma_tgc = cfg.get("tgc_sigma", 0.0)
        mode = cfg.get("tgc_mode", "spline")
        ctrl0 = torch.from_numpy(rng.uniform(-a, a, size=(K,))).to(dev).float()
        ramps = []
        for t in range(T):
            ctrl_t = (
                ctrl0
                + torch.from_numpy(rng.normal(0, sigma_tgc, size=(K,))).to(dev).float()
            )
            r = (
                _interp1d_linear(ctrl_t, H)
                if mode == "linear"
                else _interp1d_spline(ctrl_t, H)
            )
            r = torch.clamp(1.0 + r, 0.2, 2.5).view(1, 1, H, 1)
            ramps.append(r)
        ramp = torch.stack(ramps, dim=0)
        meta["tgc_style"] = "ramp"

    # Apply the TGC mask (elementwise multiply across width too)
    # y shape: [T, C, H, W]; ramp: [T,1,H,W] -> broadcast over C
    y = torch.clamp(y * ramp, 0.0, 1.0)

    # ------------------- dynamic-range clipping + blending ----------------
    # keep a pre-normalized copy so we can blend back some original texture
    y_pre_norm = y.clone()

    # sample random low/high percentiles for clipping
    low = float(rng.uniform(cfg["clip_low_min"], cfg["clip_low_max"]))
    high = float(rng.uniform(cfg["clip_high_min"], cfg["clip_high_max"]))
    # ensure a minimum spread
    high = max(high, low + 0.05)

    # linear rescale to [0,1] using sampled low/high
    y = (y - low) / (high - low + 1e-6)

    # enforce a small residual floor to prevent exact zeros (avoids full blackouts)
    # default comes from cfg if set, else 0.06
    tgc_min_residual = float(cfg.get("tgc_min_residual", 0.06))
    y = torch.clamp(y, tgc_min_residual, 1.0)

    # optional gentle blending with pre-normalized signal to preserve texture/speckle
    blend_preserve = float(cfg.get("tgc_preserve_blend", 0.12))
    if blend_preserve > 0 and "y_pre_norm" in locals():
        # y_pre_norm is in [0,1] from earlier clamp; blend and clamp again
        y = torch.clamp(
            (1.0 - blend_preserve) * y + blend_preserve * y_pre_norm, 0.0, 1.0
        )

    # ------------------------- add noise ----------------------------------
    s_add = float(cfg.get("add_noise_std", 0.0))  # additive gaussian std
    s_mult = float(
        cfg.get("add_noise_mult", 0.0)
    )  # multiplicative noise std (relative)
    if s_add > 0:
        y = y + torch.randn_like(y) * s_add
    if s_mult > 0:
        y = y + y * (torch.randn_like(y) * s_mult)
    y = torch.clamp(y, 0.0, 1.0)

    # store key sampled parameters for reproducibility/debugging
    meta.update(
        {"clip_low": low, "clip_high": high, "noise_add": s_add, "noise_mult": s_mult}
    )
    return y, meta
