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
