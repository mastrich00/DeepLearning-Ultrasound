import numpy as np, torch, cv2, torch.nn.functional as F


def resize_center_crop(img, size):
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    img = img[y0 : y0 + s, x0 : x0 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(frames):
    import numpy as np, torch

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    else:
        arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


def _interp1d_spline(ctrl, H):
    return F.interpolate(
        ctrl.view(1, 1, 1, -1), size=(1, H), mode="bicubic", align_corners=True
    ).view(H)


def _interp1d_linear(ctrl, H):
    K = ctrl.shape[0]
    yk = torch.linspace(0, 1, steps=K, device=ctrl.device)
    y = torch.linspace(0, 1, steps=H, device=ctrl.device)
    return torch.interp(y, yk, ctrl)


def apply_synthetic_degradation(clip, cfg, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > cfg.get("p_apply", 1.0):
        return clip.clone(), {"applied": False}
    y = clip.clone()
    T, C, H, W = y.shape
    dev = y.device
    meta = {"applied": True}

    # Base global gain (per-frame jittered)
    g0 = float(rng.uniform(cfg.get("gain_min", 0.6), cfg.get("gain_max", 1.6)))
    sigma_g = cfg.get("gain_sigma", 0.0)
    eps_g = torch.from_numpy(rng.normal(0, sigma_g, size=(T,))).to(dev).float()
    g = torch.clamp(eps_g + g0, 0.2, 2.5).view(T, 1, 1, 1)
    y = torch.clamp(y * g, 0.0, 1.0)
    meta["gain_base"] = g0

    # ---------- TGC-style bands (curved) ----------
    # config keys (defaults provided)
    tgc_style = cfg.get("tgc_mode", cfg.get("tgc_style", "bands"))
    if tgc_style == "bands":
        # ---------- U-shaped bands (convex U: left -> down -> right) ----------
        # config keys (defaults reuse earlier names when present)
        max_bands = int(cfg.get("tgc_bands_max", cfg.get("tgc_K", 6)))
        min_bands = int(cfg.get("tgc_bands_min", 1))
        band_strength = float(cfg.get("tgc_band_strength", cfg.get("tgc_strength", 0.6)))
        band_strength_max = float(cfg.get("tgc_band_strength_max", max(band_strength, 1.0)))
        band_width_min = float(cfg.get("tgc_band_width_min", 0.03))
        band_width_max = float(cfg.get("tgc_band_width_max", 0.15))
        abrupt_prob = float(cfg.get("tgc_abrupt_prob", 0.25))
        abrupt_sharpness = float(cfg.get("tgc_abrupt_sharpness", 60.0))
        lateral_perturb = float(cfg.get("tgc_lateral_perturb", 0.0))
        time_jitter = float(cfg.get("tgc_time_jitter", 0.02))
        contrast_exp = float(cfg.get("tgc_contrast_exponent", 1.0))

        # sample number of bands
        n_bands = int(rng.integers(min_bands, max_bands + 1))
        band_specs = []
        for b in range(n_bands):
            # y_edge = edge height (normalized 0..1) representing the left/right start height
            y_edge = float(rng.uniform(0.08, 0.75))  # avoid extreme top/bottom by default
            # depth = how deep the U goes at center (normalized fraction of H)
            depth = float(rng.uniform(0.03, 0.35))
            # width controls vertical spread around the U-curve (fraction of H)
            width = float(rng.uniform(band_width_min, band_width_max))
            # intensity magnitude (≥0): sample magnitude then allow bright/dark
            mag = float(rng.uniform(band_strength * 0.5, band_strength_max))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            intensity = 1.0 + sign * mag
            abrupt = rng.random() < abrupt_prob
            band_specs.append({"edge": y_edge, "depth": depth, "width": width, "intensity": intensity, "abrupt": abrupt})

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
                lateral = (torch.sin((xx / W) * (2 * np.pi * freq) + phase) * lateral_perturb * H)
                # lateral is in pixels and added to the vertical coordinate to warp the curve
                xx_t = xx + lateral * 0.0  # keep xx unchanged; lateral will be used by shifting x_norm below if desired

            total_mask = torch.ones((H, W), device=dev)

            for bspec in band_specs:
                # time jitter for this band center depth
                y_edge = bspec["edge"] + float(rng.normal(0, time_jitter))
                y_edge = float(np.clip(y_edge, 0.01, 0.95))
                depth = bspec["depth"]
                width = bspec["width"]
                intensity = bspec["intensity"]
                abrupt = bspec["abrupt"]

                # compute curve y(x) in pixel coordinates:
                # y_edge_px is the left/right start height in pixels (smaller -> shallower)
                y_edge_px = y_edge * (H - 1)
                depth_px = depth * (H - 1)
                # curve: y_curve(x) = y_edge_px + depth_px * (1 - x_norm^2)
                y_curve = y_edge_px + depth_px * (1.0 - x_norm ** 2)  # shape [H,W] but constant along rows in x

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
                    left_step = 0.5 * (1.0 + torch.tanh((vdist - left) * abrupt_sharpness / H))
                    right_step = 0.5 * (1.0 + torch.tanh((vdist - right) * abrupt_sharpness / H))
                    band_mask = left_step - right_step  # ~1 inside band, 0 outside
                    # optionally amplify center using contrast exponent (<1 amplifies)
                    if contrast_exp != 1.0:
                        band_mask = torch.clamp(band_mask, 0.0, 1.0).pow(max(1e-3, contrast_exp))
                    band_mul = 1.0 + (intensity - 1.0) * band_mask
                else:
                    # smooth Gaussian band perpendicular to the curve
                    band_mask = torch.exp(-0.5 * (vdist ** 2) / (sigma_pixels ** 2 + 1e-12))
                    if contrast_exp != 1.0:
                        band_mask = torch.clamp(band_mask, 0.0, 1.0).pow(max(1e-3, contrast_exp))
                    band_mul = 1.0 + (intensity - 1.0) * band_mask

                total_mask = total_mask * band_mul

            # clamp mask to safe numeric range and add channel/time dims
            total_mask = torch.clamp(total_mask, 0.01, 6.0).view(1, 1, H, W)
            ramps.append(total_mask)

        ramp = torch.stack(ramps, dim=0)  # [T,1,H,W]
        a = float(cfg.get("tgc_strength", 1.0))
        ramp = 1.0 + a * (ramp - 1.0)
        ramp = torch.clamp(ramp, 0.05, 6.0)
        meta["tgc_style"] = "bands_u"
        meta["tgc_bands_count"] = n_bands

        # # optional compact summary: average band width & mean intensity
        # meta["tgc_bands_avg_width"] = float(sum(bs["width"] for bs in band_specs) / max(1, len(band_specs)))
        # meta["tgc_bands_avg_intensity"] = float(sum(bs["intensity"] for bs in band_specs) / max(1, len(band_specs)))
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
    
    # ---------- brightness clipping + contrast + noise ----------
    low = float(rng.uniform(cfg["clip_low_min"], cfg["clip_low_max"]))
    high = float(rng.uniform(cfg["clip_high_min"], cfg["clip_high_max"]))
    high = max(high, low + 0.05)
    y = torch.clamp((y - low) / (high - low + 1e-6), 0.0, 1.0)
    s_add = float(cfg.get("add_noise_std", 0.0))
    s_mult = float(cfg.get("add_noise_mult", 0.0))
    if s_add > 0:
        y = y + torch.randn_like(y) * s_add
    if s_mult > 0:
        y = y + y * (torch.randn_like(y) * s_mult)
    y = torch.clamp(y, 0.0, 1.0)
    meta.update(
        {"clip_low": low, "clip_high": high, "noise_add": s_add, "noise_mult": s_mult}
    )
    return y, meta
