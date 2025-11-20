# src/ultra_spatial/train.py
import os, argparse, yaml, torch, logging
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
)
from .metrics import psnr as psnr_fn, ssim as ssim_metric


def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _ensure_5d(t):
    """Normalize to [B,T,C,H,W]. Handles shapes like [B,T,T,1,H,W] or extra singletons."""
    if t.dim() == 6 and t.size(2) == t.size(1) and t.size(3) == 1:
        t = t[:, :, 0]  # -> [B,T,1,H,W]
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

def collate_keep_meta(batch):
    # batch is a list of dicts from __getitem__
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

def build_loaders(cfg, use_cuda):
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


def save_samples(save_dir, epoch, inp, pred, gt):
    """
    Creates a 3x4 grid:
      Row 1: input (degraded center frames)
      Row 2: prediction (corrected center frames)
      Row 3: target (clean center frames)
    """
    import os, torch
    from torchvision.utils import save_image

    os.makedirs(save_dir, exist_ok=True)
    k = min(4, inp.size(0), pred.size(0), gt.size(0))  # show up to 4 columns
    grid = torch.cat([inp[:k], pred[:k], gt[:k]], dim=0)  # 3*k images
    path = os.path.join(save_dir, f"epoch{epoch:03d}.png")
    save_image(grid, path, nrow=k)  # nrow=k -> 3 rows
    print(f"saved samples -> {path}")


def train_epoch(
    cfg, device, gen, D, opt_g, opt_d, scaler, loader, use_gan, use_amp, step_log=20
):
    gen.train()
    if use_gan:
        D.train()
    w = cfg["loss"]
    total_g = 0.0
    total_d = 0.0

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
        batch = to_device(batch, device)
        x = _ensure_5d(batch["degraded"])
        y = _ensure_5d(batch["clean"])
        y_mid = y[:, y.shape[1] // 2]

        # --- Discriminator ---
        if use_gan:
            with amp_ctx():
                fake = gen(x)["corrected"].detach()
                real = y_mid
                D_real_map = D(real)   # shape: [B,1,Hf,Wf]
                D_fake_map = D(fake)   # shape: [B,1,Hf,Wf]
                # reduce spatial/patch dims to a per-sample score (keep batch dim)
                D_real_per_sample = D_real_map.view(D_real_map.size(0), -1).mean(dim=1)
                D_fake_per_sample = D_fake_map.view(D_fake_map.size(0), -1).mean(dim=1)
                d_loss = hinge_d_loss(D_real_per_sample, D_fake_per_sample)
            opt_d.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(d_loss).backward()
                scaler.step(opt_d)
                scaler.update()
            else:
                d_loss.backward()
                opt_d.step()
            total_d += float(d_loss.detach())

        # --- Generator ---
        with amp_ctx():
            out = gen(x)
            pred = out["corrected"]
            if cfg.get("train", {}).get("debug", False):
                # detach to avoid autograd converting tensors to scalars (avoids the warning)
                pmin = pred.detach().cpu().min().item()
                pmax = pred.detach().cpu().max().item()
                print(f"\npred min/max: {pmin:.6f} {pmax:.6f}")
            l1 = torch.nn.functional.l1_loss(pred, y_mid)
            ssim_loss = 1.0 - ssim_fn(pred, y_mid)
            tv = tv_loss(out["I"])
            nuc = nuclear_norm_surrogate(out["LR"])
            id_l = torch.nn.functional.l1_loss(gen(y)["corrected"], y_mid)
            g_loss = (
                float(w["w_l1"]) * l1
                + float(w["w_ssim"]) * ssim_loss
                + float(w["w_tv"]) * tv
                + float(w["w_lowrank_nuc"]) * nuc
                + float(w["w_identity"]) * id_l
            )
            if use_gan:
                D_pred_map = D(pred)   # [B,1,Hf,Wf]
                D_pred_per_sample = D_pred_map.view(D_pred_map.size(0), -1).mean(dim=1)
                adv = hinge_g_loss(D_pred_per_sample)   # hinge_g_loss expects per-sample
                g_loss = g_loss + float(w["lambda_adv"]) * adv

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

        if i % step_log == 0:
            pbar.set_postfix(g=total_g / i, d=(total_d / i if use_gan else 0.0))

    n = max(1, len(loader))
    return total_g / n, (total_d / n if use_gan else 0.0)


@torch.no_grad()
def evaluate(cfg, device, gen, loader, epoch=None, save_dir=None):
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

    gen = RetinexLowRankVT(
        enc_channels=list(cfg["model"]["enc_channels"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        lowrank_rank=int(cfg["model"]["lowrank_rank"]),
        illumination_coarse=int(cfg["model"]["illumination_coarse"]),
    ).to(device)

    D = PatchDiscriminator().to(device)

    opt_g = torch.optim.AdamW(
        gen.parameters(),
        lr=float(cfg["train"]["lr_g"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    opt_d = torch.optim.AdamW(
        D.parameters(), lr=float(cfg["train"]["lr_d"]), weight_decay=0.0
    )

    # Create GradScaler only when AMP is actually used (prevents warnings on CPU)
    scaler = None
    if use_amp:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler(enabled=True)

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)
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

        ckpt = {"model": gen.state_dict(), "cfg": cfg, "epoch": epoch}
        save_checkpoint(ckpt, os.path.join(cfg["train"]["save_dir"], "last.pt"))
        if val["ssim"] > best:
            best = val["ssim"]
            save_checkpoint(ckpt, os.path.join(cfg["train"]["save_dir"], "best.pt"))
            logging.info("saved new best checkpoint")

    state = torch.load(
        os.path.join(cfg["train"]["save_dir"], "best.pt"), map_location="cpu"
    )
    gen.load_state_dict(state["model"])
    test = evaluate(cfg, device, gen, te)
    logging.info(f"test: psnr={test['psnr']:.3f}  ssim={test['ssim']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in ("yes", "y", "true", "t", "1", ""):
            return True
        if v in ("no", "n", "false", "f", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    p.add_argument("--gan", type=str2bool, nargs="?", const=True, default=True, help="enable GAN training (accepts yes/no/true/false; `--gan` sets True)")
    p.add_argument("--log_level", type=str, default="INFO", help="DEBUG, INFO, WARNING")
    args = p.parse_args()
    main(args)
