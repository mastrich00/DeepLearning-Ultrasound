# Temporal Retinex–Low-Rank Enhancement of B-mode Ultrasound Videos

This project studies automatic correction of brightness and exposure errors in B-mode ultrasound videos. The focus is on global gain and time-gain compensation (TGC), which are frequently misadjusted in practice and lead to dark regions, washed-out depth bands, and inconsistent appearance across frames. The goal is to standardize brightness across depth and time while preserving realistic speckle and anatomical boundaries.

The implementation follows the proposal *"Temporal Retinex and Low-Rank Enhancement of B-mode Ultrasound Videos with Synthetic Gain/TGC Errors"* and is designed for controlled experimentation with and without adversarial training.

---

## Project overview

Ultrasound images are highly sensitive to acquisition settings such as gain, TGC, and dynamic range. Incorrect settings are common, especially for novice operators or low-end devices, and can obscure anatomy or amplify noise. At the same time, ultrasound speckle carries physical information and should not be overly smoothed or hallucinated.

This project addresses these issues by combining:

* a **Retinex-inspired decomposition** to separate illumination (exposure/TGC) from tissue reflectance,
* a **low-rank component** to model smooth global intensity trends,
* **temporal fusion** across short video clips to stabilize corrections,
* and an optional **PatchGAN discriminator** to improve speckle realism.

The model predicts a corrected center frame from a short grayscale clip.

---

## Dataset

We use the **EchoNet-Dynamic** dataset, which contains 10,030 deidentified cardiac ultrasound videos. Each video is approximately 3 seconds long and recorded in grayscale B-mode.

**Preprocessing**

* Videos are sub-sampled in time and sliced into fixed-length clips.
* Frames are center-cropped and resized to 128 × 128.
* Intensities are normalized to [0, 1].

**Synthetic degradation**
To enable supervised training, paired *bad → good* examples are created by simulating common exposure errors:

* global gain errors,
* depth-wise TGC-like amplification curves that introduce bright or dark bands,
* mild dynamic-range clipping,
* additive and multiplicative sensor noise.

The degradations are sampled per clip with small frame-to-frame jitter so that anatomy and speckle statistics remain realistic.

**Splits**

* Video-wise split: 70% train, 15% validation, 15% test.

---

## Methodology

### Architecture

The model is trained as a **Generative Adversarial Network (GAN)**.

**Generator**

* A **per-frame CNN encoder** extracts spatial features at full resolution to preserve speckle and edges.
* A lightweight **temporal transformer** fuses information across frames by pooling spatial features and re-injecting temporal context.
* Three heads operate on the fused center-frame features:

  * **Illumination (I)** predicts a smooth exposure/TGC field.
  * **Low-rank (LR)** captures global intensity trends such as depth slopes.
  * **Residual / reflectance head** predicts a small additive correction that refines local contrast.
* A composition step combines the input frame with these components to produce the corrected image.

**Discriminator**

* A **PatchGAN discriminator** operates on the corrected center frame.
* Patch-level decisions encourage realistic local texture and speckle without enforcing global hallucinations.

### Training

* Supervised learning on synthetic pairs.
* Losses include masked L1, SSIM, total variation on illumination, low-rank regularization, identity loss, and optional adversarial loss.
* GAN training can be enabled or disabled to perform controlled ablations.

---

## Evaluation

The evaluation focuses on both fidelity and ultrasound-specific realism.

**Metrics**

* **PSNR** and **SSIM** for pixel-level and structural similarity.
* **Mutual Information** to measure shared information under varying gain and contrast.
* **Speckle Similarity Index** to assess preservation of ultrasound texture.
* **Edge Preservation Index** to quantify boundary sharpness.

We explicitly compare training **with and without GAN** to isolate the effect of adversarial learning on speckle realism.

---

## Expected outcomes

The method is expected to:

* improve brightness uniformity across depth and time,
* correct underexposed and overexposed regions without clipping,
* preserve speckle statistics and anatomical boundaries,
* produce more perceptually realistic results when adversarial training is enabled.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Check that EchoNet-Dynamic videos are placed under (300 videos are already contained in the repo):
# data/echonet/videos/*.mp4 or *.avi

# Train without adversarial loss
python -m src.ultra_spatial.train --config configs/default.yaml --gan false --log_level INFO

# Train with PatchGAN discriminator
python -m src.ultra_spatial.train --config configs/default.yaml --gan true --log_level INFO

# Visualize synthetic degradations
python -m src.ultra_spatial.visualize_degradation --config configs/default.yaml --out_dir runs/deg_viz --n 12
```

---

## References

Key references include work on ultrasound knobology, speckle characterization, low-rank GANs for ultrasound, Retinex-based illumination modeling, and temporal transformers for video restoration. See the proposal bibliography for full citations.

---
