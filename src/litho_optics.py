"""
MODULE 2: Neural Optical Engine & Abbe Diffraction Simulator
Simulates the Mask -> Aerial Image optical projection stage.
Provides:
- An analytical low-pass diffraction filter (Fourier optics reference)
- A Neural Optics Surrogate network (Conv2D Residual Backbone)
Outputs continuous aerial intensity map I(x, y) in range [0.0, 1.0].
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def simulate_diffraction_ground_truth(
    mask: torch.Tensor,
    cutoff_freq: float = 0.28,
    defocus_sigma: float = 1.2
) -> torch.Tensor:
    """
    Simulates optical aerial intensity using 2D Fourier spatial low-pass filtering,
    approximating coherent diffraction through a circular pupil aperture.
    
    Args:
        mask: Tensor of shape (B, 1, H, W) with binary values in [0, 1].
        cutoff_freq: Optical cutoff frequency proportional to NA / lambda.
        defocus_sigma: Spatial Gaussian spread modeling optical defocus / aberrations.
    Returns:
        aerial_intensity: Continuous intensity map in [0, 1].
    """
    B, C, H, W = mask.shape
    device = mask.device

    # 1. 2D Fast Fourier Transform into spatial frequency domain
    fft_mask = torch.fft.fftshift(torch.fft.fft2(mask, dim=(-2, -1)), dim=(-2, -1))

    # 2. Construct circular pupil low-pass transfer function
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    spatial_radius = torch.sqrt(xx**2 + yy**2)

    pupil_filter = (spatial_radius <= cutoff_freq).float()
    
    # Apply soft defocus damping envelope
    damping = torch.exp(-(spatial_radius**2) / (2 * (defocus_sigma * cutoff_freq)**2))
    pupil_filter = (pupil_filter * damping).unsqueeze(0).unsqueeze(0)

    # 3. Filter and Inverse FFT to compute complex electrical field amplitude
    filtered_field = torch.fft.ifft2(torch.fft.ifftshift(fft_mask * pupil_filter, dim=(-2, -1)), dim=(-2, -1))

    # 4. Aerial Intensity is square of electrical field magnitude |E|^2
    aerial_intensity = torch.real(filtered_field * torch.conj(filtered_field))
    
    # Normalize to [0, 1] range
    max_vals = torch.amax(aerial_intensity, dim=(-2, -1), keepdim=True) + 1e-7
    aerial_intensity = aerial_intensity / max_vals
    return aerial_intensity.clamp(0.0, 1.0)


class ResidualBlock(nn.Module):
    """Residual building block to preserve fine spatial high frequencies."""
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class NeuralOpticsEngine(nn.Module):
    """
    Neural Surrogate for the Optical Imaging stage (Mask -> Aerial Image).
    Maps high-contrast binary mask polygons to continuous aerial light intensity.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super(NeuralOpticsEngine, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.res1 = ResidualBlock(base_channels)
        self.res2 = ResidualBlock(base_channels)
        
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Aerial intensity strictly bounded in [0, 1]
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        feat = self.stem(mask)
        feat = self.res1(feat)
        feat = self.res2(feat)
        return self.head(feat)


if __name__ == "__main__":
    # Smoke test module
    dummy_mask = (torch.rand(2, 1, 128, 128) > 0.7).float()
    
    # Test Fourier diffraction physics
    gt_aerial = simulate_diffraction_ground_truth(dummy_mask)
    print(f"[litho_optics] Physical GT shape: {gt_aerial.shape} | Range: [{gt_aerial.min():.3f}, {gt_aerial.max():.3f}]")
    
    # Test Neural Surrogate forward pass
    model = NeuralOpticsEngine()
    pred_aerial = model(dummy_mask)
    print(f"[litho_optics] Neural Engine output shape: {pred_aerial.shape}")
