# ProGAN-Face-Generator
# 🎨 Progressive GAN: High-Resolution Face Synthesis

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](YOUR_HUGGING_FACE_SPACE_LINK_HERE)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-orange?style=flat)

This repository implements a **Progressive Growing GAN (ProGAN)** in PyTorch, capable of generating high-quality images up to 512×512 resolution. The model features equalized learning rate, pixel normalization, and minibatch standard deviation for stable training.

---

## ✨ Key Features
* **Progressive Growing:** Seamlessly grows from 4x4 to 512x512 resolution using smooth fade-in (`alpha` blending).
* **Stable Training:** Implements **Equalized Learning Rate** (runtime weight scaling) for all layers.
* **Latent Space Interpolation:** Uses **SLERP** (Spherical Linear Interpolation) for smooth transitions between generated faces.
* **Gradio Web UI:** Includes a full-featured dashboard for random generation and morphing.

---

## 🖼️ Visualizations

| Resolution Progression | Latent Space Interpolation (SLERP) |
| :--- | :--- |
| ![Growth GIF](https://your-link-to-image.gif) | ![Morphing GIF](https://your-link-to-image.gif) |
| *From 4x4 to 512x512* | *Smoothly morphing between seeds* |

---
## 📐 Model Architecture
graph TD
    subgraph Generator
        Z[Latent Vector z: 512] --> B4[4x4 Initial Block]
        B4 --> PN1[PixelNorm]
        PN1 --> U8[Upsample 8x8]
        U8 --> B8[8x8 Block]
        B8 --> PN2[PixelNorm]
        PN2 --> UN[... Upsample to ...]
        UN --> B512[512x512 Block]
        B512 --> RGB[ToRGB: 1x1 Conv]
        RGB --> Out[Generated Image]
    end

    subgraph Discriminator
        In[Input Image] --> FRGB[FromRGB: 1x1 Conv]
        FRGB --> DB512[512x512 Block]
        DB512 --> DSN[... Downsample to ...]
        DSN --> DB8[8x8 Block]
        DB8 --> DS4[Downsample 4x4]
        DS4 --> MSD[Minibatch Stddev]
        MSD --> DB4[4x4 Final Block]
        DB4 --> FC[Equalized Linear Layer]
        FC --> RealFake[Real / Fake Decision]
    end

    style Z fill:#f9f,stroke:#333,stroke-width:2px
    style Out fill:#00c2ff,stroke:#333,stroke-width:2px
    style RealFake fill:#ffcc00,stroke:#333,stroke-width:2px
    
---
## 🧪 Technical Deep Dive
1. Equalized Learning RateUnlike standard GANs that use careful weight initialization, this model uses Runtime Weight Scaling. In layers.py, weights are initialized from $\mathcal{N}(0, 1)$ and scaled on every forward pass by:$$w_{scaled} = w \times \sqrt{\frac{2}{\text{fan\_in}}}$$This ensures that the dynamic range, and thus the learning speed, is the same for all layers.2. Pixelwise Feature NormalizationTo prevent the magnitudes of the features in the Generator from escalating, we normalize the feature vector in each pixel to unit length after every convolutional layer:$$b_{x,y} = a_{x,y} / \sqrt{\frac{1}{C} \sum_{j=0}^{C-1} (a_{x,y}^j)^2 + \epsilon}$$3. Minibatch Standard DeviationTo increase variation in generated images, the Discriminator computes the standard deviation of the current minibatch and adds it as an extra feature map. This encourages the Generator to produce more diverse samples.

---

## 🛠️ Project Structure
* `model.py`: Core architecture for the Generator, Discriminator, and the combined ProGAN class.
* `layers.py`: Custom implementations of `EqualizedConv2d`, `PixelNorm`, and `MinibatchStddev`.
* `inference.py`: Helper class for generating images and performing latent space walks.
* `app.py`: The Gradio interface for interacting with the model.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone [https://github.com/YourUsername/ProGAN-Face-Gen.git](https://github.com/YourUsername/ProGAN-Face-Gen.git)
cd ProGAN-Face-Gen
pip install -r requirements.txt
