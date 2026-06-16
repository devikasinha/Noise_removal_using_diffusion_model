# Multi-Dataset Image Denoising using Denoising Diffusion Probabilistic Models (DDPM)

This repository contains a PyTorch implementation of a generative **Denoising Diffusion Probabilistic Model (DDPM)** optimized for strategic image denoising and artifact removal. To evaluate robustness, adaptability, and scaling, the architecture is benchmarked across three distinct datasets representing progressive structural complexity: **MNIST**, **CIFAR-10**, and **FFHQ**.

---

## 🚀 Overview

Traditional denoising techniques struggle to maintain high-frequency structural details when handling non-Gaussian or highly destructive noise. This project leverages the iterative reverse process of Diffusion Models to strategically reconstruct clean, high-fidelity images from heavily degraded inputs.

By scaling the model across three varied benchmarks, this framework evaluates how effectively a shared diffusion core learns geometric priors ranging from low-resolution grayscale structures to high-resolution, complex facial features.

---

## 📊 Dataset Evaluation Suite

The network’s denoising capabilities are validated across three distinct tiers:

| Dataset | Type | Resolution | Key Challenge |
| :--- | :--- | :--- | :--- |
| **MNIST** | Grayscale Digits | $28 \times 28$ | Maintaining sharp structural boundaries against stark solid backgrounds. |
| **CIFAR-10** | Low-Res Color | $32 \times 32$ | Preserving diverse textures and semantic concepts across 10 varied object classes. |
| **FFHQ** | High-Res Faces | $128 \times 128$ / Custom | Reconstructing intricate geometric features, skin textures, and lighting gradients without blurring. |

---

## 📐 Methodology

The framework operates via an iterative Markov chain framework consisting of two main processes:

1. **Forward Process ($q$):** Gradually injects Gaussian noise into a clean image $x_0$ over $T$ timesteps according to a pre-defined variance schedule $\beta_1, \beta_2, \dots, \beta_T$.
   
   $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t\mathbf{I})$$

2. **Reverse Process ($p_\theta$):** A deep U-Net architecture learns to predict the noise added at any given timestep $t$. By conditioning the network on the time embedding, it iteratively subtracts noise to reconstruct the clean target $x_0$.

   $$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

---

## 🛠️ Repository Structure

```text
├── dataset/               # Data pipeline loaders for MNIST, CIFAR-10, and FFHQ
├── models/
│   ├── unet.py            # Scalable U-Net with Time Embeddings & Self-Attention
│   └── diffusion.py       # Forward and Reverse diffusion process logic
├── train.py               # Multi-dataset training script
├── denoise_eval.py        # Inference script for evaluating noise removal
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
