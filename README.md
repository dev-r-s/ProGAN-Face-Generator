# 🎨 Progressive GAN – High-Resolution Face Synthesis (512×512)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/cds006/Progan)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-orange?style=flat)

Research-grade implementation of **Progressive Growing GAN (ProGAN)** trained up to **512×512 resolution** on CelebA-HQ.

This project reproduces the core ideas from:

**Tero Karras et al., 2017 — Progressive Growing of GANs for Improved Quality, Stability, and Variation**

---

# 🚀 Highlights

- Progressive growing: 4×4 → 512×512
- Fade-in alpha blending during transitions
- WGAN-GP loss
- Equalized Learning Rate
- Pixel-wise feature normalization
- Minibatch standard deviation layer
- Exponential Moving Average (EMA) generator
- Mixed precision (AMP) training
- SLERP latent interpolation
- Quantitative FID evaluation
- Gradio deployment (Hugging Face Space)

---

# 📊 Quantitative Results

**FID @ 512×512: 22.11**

Evaluation details:
- 5000 generated samples
- Truncation = 1.0
- Inception-v3 (2048-dim features)
- Computed using `pytorch-fid`

Command used:

```
python -m pytorch_fid data/real_512 data/fake_512 --device cuda
```

---

# 🖼 Results

## Generated Samples (512×512)

![Sample Grid](assets/sample_grid.png)

## Progressive Resolution Growth

![Progression](assets/resolution_progression.png)

---

# 🏗 Architecture Overview

## Generator (Progressive Growing)

Latent vector (512×1×1)  
→ 4×4 initial block  
→ Upsampling blocks (×2 each stage)  
→ ToRGB (1×1 conv) at every resolution  
→ Alpha blending during transitions  
→ Final tanh activation

Each generator block contains:
- Upsample (nearest)
- EqualizedConv2d
- LeakyReLU
- PixelNorm
- EqualizedConv2d
- LeakyReLU
- PixelNorm

Fade-in mechanism:

```
output = α * new_path + (1 - α) * old_path
```

---

## Discriminator (Mirror Structure)

Input Image  
→ FromRGB (1×1 conv)  
→ Downsampling blocks  
→ MinibatchStddev layer  
→ Final EqualizedLinear output (no sigmoid)

Each block:
- EqualizedConv2d
- LeakyReLU
- EqualizedConv2d
- LeakyReLU
- AvgPool2d

WGAN critic output (no probability, no sigmoid).

---

# 🧠 Training Methodology

## Progressive Schedule

```
4 → 8 → 16 → 32 → 64 → 128 → 256 → 512
```

Each resolution stage consists of:

1. Fade-in phase (α: 0 → 1)
2. Stabilization phase (α = 1)

---

## Loss Function (WGAN-GP)

Discriminator:

```
L = E[D(fake)] - E[D(real)] + λGP + drift
```

Generator:

```
L = -E[D(fake)]
```

Where:
- λ = 10 (gradient penalty)
- Drift = 0.001

---

## EMA Update

```
θ_ema = β * θ_ema + (1 - β) * θ
```

EMA weights are used during inference for smoother results.

---

# ⚙ Model Configuration

- Dataset: CelebA-HQ (~30k images)
- Final Resolution: 512×512
- Latent Dimension: 512
- Learning Rate: 1e-3
- Adam Betas: (0.0, 0.99)
- EMA Decay: 0.999
- Mixed Precision: Enabled
- Images per stage: 800K
- Total training steps: 650,000

---

# 🛠 Project Structure

```
.
├── app.py              # Gradio UI
├── train.py            # Progressive training loop
├── inference.py        # Image generation + SLERP
├── model.py            # Generator & Discriminator
├── layers.py           # Custom layers
├── config.py
├── requirements.txt
├── assets/
│   ├── sample_grid.png
│   └── progression.gif
└── README.md
```

---

# 🚀 Quick Start

## Install

```
pip install -r requirements.txt
```

Verify CUDA:

```
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Train

```
python train.py --data_dir /path/to/celeba_hq --max_res 512
```

Resume training:

```
python train.py --data_dir /path/to/celeba_hq \
    --resume checkpoints/checkpoint_res256_stepXXXX.pt
```

---

## Generate Images

```
python inference.py \
    --checkpoint checkpoints/final_model.pt \
    --output generated.png \
    --num_images 64 \
    --grid
```

With truncation:

```
--truncation 0.7
```

---

## Launch Web Interface

```
python app.py
```

Open:

```
http://localhost:7860
```

---

# 📈 Training Monitoring

Training outputs:

```
outputs/
├── samples/
├── logs/training_log.json
```

Resolution scaling (approximate):

| Resolution | VRAM | Quality |
|------------|------|----------|
| 64×64      | 6GB  | Good |
| 128×128    | 8GB  | High |
| 256×256    | 12GB | Very Good |
| 512×512    | 16GB | Excellent |

---

# 🌐 Live Demo

Available on Hugging Face Spaces:

👉 https://huggingface.co/spaces/cds006/Progan

---

# 🎓 Research Impact

This implementation demonstrates:

- Stable high-resolution GAN training
- Proper progressive growing logic
- Quantitative evaluation (FID)
- Production-ready inference
- Consumer GPU scalability

Achieving **FID 22.11 at 512×512 resolution** confirms architectural correctness and training stability.

---

# 🔮 Future Work

- FID tracking during training
- Precision & Recall metrics
- StyleGAN comparison baseline
- Adaptive augmentation experiments

---

# 👤 Author

Devinder Solanki  
Research-focused generative modeling implementation.

---

⭐ If you find this project interesting, consider giving it a star.
