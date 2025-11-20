# Temporal Retinex–Low-Rank + Spatial PatchGAN

Generator: Retinex–Low-Rank–Temporal model.  
Discriminator: Spatial PatchGAN on corrected center frame.  
Losses: L1 + SSIM + TV(illum) + Low-rank surrogate + Identity + Hinge adversarial.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Prepare EchoNet-Dynamic under data/echonet/videos/*.mp4

# without Discriminator
python -m src.ultra_spatial.train --config configs/default.yaml --gan false --log_level INFO

# with Discriminator
python -m src.ultra_spatial.train --config configs/default.yaml --gan true --log_level INFO

# visualize degraded images
python -m src.ultra_spatial.visualize_degradation --config configs/default.yaml --out_dir runs/deg_viz --n 12
```
