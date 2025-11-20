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
        # band-specific config (new keys; defaults chosen to be reasonable)
        max_bands = int(cfg.get("tgc_bands_max", cfg.get("tgc_K", 6)))
        min_bands = int(cfg.get("tgc_bands_min", 1))
        band_strength = float(cfg.get("tgc_band_strength", cfg.get("tgc_strength", 0.6)))
        band_width_min = float(cfg.get("tgc_band_width_min", 0.03))  # fraction of radial depth
        band_width_max = float(cfg.get("tgc_band_width_max", 0.15))
        abrupt_prob = float(cfg.get("tgc_abrupt_prob", 0.2))  # probability a band is abrupt
        lateral_perturb = float(cfg.get("tgc_lateral_perturb", 0.06))  # lateral waviness amplitude
        curvature_factor = float(cfg.get("tgc_curvature", 1.3))  # >1 centers below image
        time_jitter = float(cfg.get("tgc_time_jitter", 0.02))  # per-frame centre jitter (fraction)
        # compute arc center for convex probe: center_x roughly mid, center_y below image
        cx0 = float(W / 2.0)
        # allow small random horizontal center shift per clip
        cx0 += float(rng.uniform(-0.05, 0.05) * W)
        cy0 = float(H * curvature_factor)  # center below image to create convex arcs

        # build radial coordinate map (same for all frames except slight per-frame jitter)
        xs = torch.linspace(0, W - 1, W, device=dev)
        ys = torch.linspace(0, H - 1, H, device=dev)
        # create meshgrid: y (H, W), x (H, W)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        # radial distance from probe center (float tensor, shape HxW)
        with torch.no_grad():
            base_radial = torch.sqrt((xx - cx0) ** 2 + (yy - cy0) ** 2)
            # normalize radial to [0,1]
            base_radial = (base_radial - base_radial.min()) / (base_radial.max() - base_radial.min() + 1e-12)

        # number of bands this clip will have (random)
        n_bands = int(rng.integers(min_bands, max_bands + 1))
        band_specs = []
        # sample band parameters (centers in normalized radial coordinates)
        for b in range(n_bands):
            d_center = float(rng.uniform(0.05, 0.95))  # avoids extremely near-edge bands
            width = float(rng.uniform(band_width_min, band_width_max))
            # strength: can be >1 (bright) or <1 (dark)
            s_sign = rng.choice([-1.0, 1.0]) if rng.random() < 0.5 else 1.0
            # sample magnitude around band_strength, allow both dark and bright bands
            mag = float(rng.uniform(0.2 * band_strength, band_strength))
            intensity = 1.0 + s_sign * mag
            abrupt = rng.random() < abrupt_prob
            band_specs.append({"center": d_center, "width": width, "intensity": intensity, "abrupt": abrupt})

        # create per-frame masks by combining band contributions.
        ramps = []
        for t in range(T):
            # small temporal jitter of centers
            radial = base_radial.clone()
            # optional small per-frame lateral warping (sinusoidal distortions)
            if lateral_perturb > 0.0:
                # create a smooth lateral modulation field
                freq = rng.uniform(0.5, 2.0)
                phase = rng.uniform(0.0, 2 * np.pi)
                lateral = (torch.sin((xx / W) * (2 * np.pi * freq) + float(phase)) * lateral_perturb)
                radial = radial + lateral  # modulation (keeps shape HxW)

            total_mask = torch.ones((H, W), device=dev)

            for bspec in band_specs:
                # apply tiny time jitter to band center
                d_c = bspec["center"] + float(rng.normal(0, time_jitter))
                d_c = float(np.clip(d_c, 0.0, 1.0))
                w = bspec["width"]
                intensity = bspec["intensity"]
                abrupt = bspec["abrupt"]

                if abrupt:
                    # box-like band with smooth edges controlled by a 'sharpness' factor
                    sharp = float(cfg.get("tgc_abrupt_sharpness", 60.0))
                    left = d_c - w / 2.0
                    right = d_c + w / 2.0
                    # smooth step using tanh
                    left_step = 0.5 * (1.0 + torch.tanh((radial - left) * sharp))
                    right_step = 0.5 * (1.0 + torch.tanh((radial - right) * sharp))
                    band_mask = left_step - right_step  # ~1 inside band, 0 outside, smooth edges
                    # convert to multiplicative factor: 1 outside, intensity inside
                    band_mul = 1.0 + (intensity - 1.0) * band_mask
                else:
                    # smooth Gaussian-shaped band along radial coordinate
                    # use gaussian sigma chosen so that width ~ 2*sqrt(2*ln2)*sigma approximately
                    sigma = w / 2.0
                    band_mask = torch.exp(-0.5 * ((radial - d_c) ** 2) / (sigma ** 2 + 1e-12))
                    band_mul = 1.0 + (intensity - 1.0) * band_mask

                total_mask = total_mask * band_mul

            # clip/regularize mask so it doesn't explode; convert to shape [1,1,H,W]
            total_mask = torch.clamp(total_mask, 0.01, 10.0).view(1, 1, H, W)
            ramps.append(total_mask)

        ramp = torch.stack(ramps, dim=0)  # shape [T,1,H,W]
        # optionally apply an overall TGC strength factor 'a' to control global effect
        a = float(cfg.get("tgc_strength", 0.6))
        # blend original (1.0) with generated mask: mask = 1 + a*(mask-1)
        ramp = 1.0 + a * (ramp - 1.0)
        ramp = torch.clamp(ramp, 0.2, 3.0)
        meta["tgc_style"] = "bands"
        meta["tgc_bands_count"] = len(band_specs)
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
