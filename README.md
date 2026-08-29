# Neural Litho Twin: Differentiable Computational Lithography

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Neural Litho Twin** is an end-to-end differentiable computational lithography platform built in PyTorch. It bridges the design-to-manufacturing gap in semiconductor fabrication by replacing traditional, computationally expensive physical simulations with high-fidelity, gradient-enabled deep learning operators.

## 🚀 The Core Problem & Our Solution
In advanced semiconductor manufacturing, optical diffraction and resist chemistry cause the printed silicon to distort drastically from the original drawn mask layout. Traditional Optical Proximity Correction (OPC) and Inverse Lithography Technology (ILT) rely on disconnected, non-differentiable solvers that take massive CPU clusters days to compute.

**This project solves that by:**
1. **Accelerating Forward Simulation:** Replacing iterative solvers with Convolutional Neural Networks (CNNs) and Fourier Neural Operators (FNOs), achieving $100\times - 1000\times$ speedups.
2. **Enabling Differentiable ILT:** Because the entire optics, resist, and etch pipeline is built in PyTorch, we can use exact analytic gradients to backpropagate from the desired silicon shape directly to the input mask, generating optimized curvilinear masks in a fraction of the time.

---

## 🧩 Pipeline Architecture

The framework is decoupled into independent but chainable physics modules:

1.  **Optics Engine (`Mask -> Aerial Image`):** Models Abbe/Hopkins transmission and partial coherence using frequency-aware neural operators.
2.  **Resist Engine (`Aerial Image -> Photoresist`):** A high-resolution U-Net that models non-linear exposure thresholds and acid-diffusion kinetics to predict the 3D developed resist contour.
3.  **Etch Engine (`Resist -> Silicon`):** A multiscale convolutional network that captures aspect-ratio dependent etching (ARDE) and plasma microloading effects.
4.  **Inverse Optimizer (ILT):** Freezes the forward models and optimizes the continuous mask tensor via gradient descent to match a target layout.

---

## 🛠️ Quickstart

### Prerequisites
*   Linux (Ubuntu 18.04/20.04 recommended)
*   NVIDIA GPU with at least 8GB VRAM (for training/ILT)
*   CUDA 11.x / 12.x

### Installation
Clone the repository and install dependencies:
```bash
git clone [https://github.com/yourusername/neural-litho-twin.git](https://github.com/yourusername/neural-litho-twin.git)
cd neural-litho-twin

# Create a conda environment
conda create -n lithotwin python=3.9
conda activate lithotwin

# Install PyTorch (ensure you install the GPU version)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install project requirements
pip install -r requirements.txt
