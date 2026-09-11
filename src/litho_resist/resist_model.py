"""
MODULE 3: Neural Resist Engine & Kinetic Dissolution Simulator
Simulates the Aerial Image -> Developed Photoresist stage.
Provides:
- An analytical Mack dissolution & acid-diffusion ground truth solver
- A High-Resolution U-Net surrogate with skip connections
Outputs 2D continuous normalized photoresist height H(x,y) in range [0.0, 1.0].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def simulate_resist_ground_truth(
    aerial_intensity: torch.Tensor,
    diffusion_kernel_size: int = 5,
    diffusion_sigma: float = 1.0,
    critical_threshold: float = 0.45,
    resist_contrast_gamma: float = 14.0
) -> torch.Tensor:
    """
    Simulates post-exposure bake (PEB) acid diffusion followed by Mack
    kinetic dissolution modeling for positive-tone photoresist.
    
    Args:
        aerial_intensity: (B, 1, H, W) optical aerial image in [0, 1].
        diffusion_kernel_size: Spatial extent of acid diffusion window.
        diffusion_sigma: Acid diffusion length parameter.
        critical_threshold: Exposure threshold intensity I_crit.
        resist_contrast_gamma: Contrast steepness parameter (gamma).
    Returns:
        resist_height: (B, 1, H, W) normalized resist profile in [0, 1].
    """
    device = aerial_intensity.device
    
    # 1. Acid diffusion via 2D Gaussian convolution kernel
    k = diffusion_kernel_size
    coords = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * diffusion_sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, k, k)
    
    # Spatial convolution with reflection padding to avoid edge cutoff
    pad = k // 2
    padded_aerial = F.pad(aerial_intensity, (pad, pad, pad, pad), mode="reflect")
    diffused_dose = F.conv2d(padded_aerial, kernel)

    # 2. Mack Dissolution Model: Sigmoidal threshold transition
    # Higher dose = dissolved resist (H -> 0.0); Lower dose = resist remains (H -> 1.0)
    resist_height = 1.0 / (1.0 + torch.exp(resist_contrast_gamma * (diffused_dose - critical_threshold)))
    return resist_height.clamp(0.0, 1.0)


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResistUNet(nn.Module):
    """
    High-Resolution U-Net Surrogate for the Resist Development stage.
    Uses skip connections to preserve steep resist sidewall contours.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super(ResistUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4)
        )

        # Decoder (Upsampling with Skip Connections)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 2, base_channels)

        # Output head: maps directly to normalized resist height
        self.out_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contraction path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Expansion path with feature concatenation
        d1 = self.up1(x3)
        d1 = torch.cat([x2, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.conv_up2(d2)

        return self.out_head(d2)


if __name__ == "__main__":
    # Self-test module
    dummy_aerial = torch.rand(2, 1, 128, 128)
    gt_resist = simulate_resist_ground_truth(dummy_aerial)
    print(f"[litho_resist] Ground Truth Resist shape: {gt_resist.shape} | Range: [{gt_resist.min():.3f}, {gt_resist.max():.3f}]")

    model = ResistUNet()
    pred_resist = model(dummy_aerial)
    print(f"[litho_resist] U-Net Output shape: {pred_resist.shape}")
