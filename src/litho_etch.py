"""
MODULE 4: Neural Etch Engine & Plasma Reactive Ion Etch (RIE) Simulator
Simulates the Developed Photoresist -> Etched Silicon Pattern stage.
Provides:
- An analytical multi-scale kernel modeling micro-loading & ARDE
- A Dilated Multiscale Convolutional Network (EtchNet)
Outputs 2D continuous etched silicon profile S(x,y) in range [0.0, 1.0].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def simulate_etch_ground_truth(
    resist_profile: torch.Tensor,
    anisotropic_bias: float = 0.08,
    microloading_radius: int = 9,
    arde_factor: float = 0.12
) -> torch.Tensor:
    """
    Simulates physical plasma etching including micro-loading effects
    and aspect-ratio dependent etching (ARDE).
    
    Args:
        resist_profile: (B, 1, H, W) developed photoresist height in [0, 1].
        anisotropic_bias: Lateral chemical undercut / profile bias.
        microloading_radius: Spatial range over which local pattern density depletes etch gas.
        arde_factor: Strength of density-dependent loading effect.
    Returns:
        etched_silicon: (B, 1, H, W) final etched silicon pattern in [0, 1].
    """
    device = resist_profile.device
    k = microloading_radius
    
    # 1. Compute local pattern density via spatial smoothing kernel (gas depletion)
    coords = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    density_kernel = torch.exp(-(xx**2 + yy**2) / (2 * (k / 3.0)**2))
    density_kernel = density_kernel / density_kernel.sum()
    density_kernel = density_kernel.view(1, 1, k, k)
    
    pad = k // 2
    padded_resist = F.pad(resist_profile, (pad, pad, pad, pad), mode="reflect")
    local_density = F.conv2d(padded_resist, density_kernel)
    
    # 2. Dynamic etch bias = base anisotropic bias + ARDE microloading penalty
    effective_bias = anisotropic_bias + (arde_factor * (1.0 - local_density))
    
    # 3. Transfer resist stencil into silicon through biased non-linear erosion
    etched_silicon = torch.sigmoid(18.0 * (resist_profile - (0.50 + effective_bias)))
    return etched_silicon.clamp(0.0, 1.0)


class MultiscaleDilatedBlock(nn.Module):
    """Combines varying dilation rates to model short and long-range gas transport."""
    def __init__(self, in_channels: int, out_channels: int):
        super(MultiscaleDilatedBlock, self).__init__()
        
        # Rate 1: Local edge erosion
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, dilation=1)
        # Rate 2: Intermediate aspect-ratio bias
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=2, dilation=2)
        # Rate 4: Long-range reactant microloading
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=4, dilation=4)
        # 1x1 residual projection
        self.branch4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.relu(self.bn(out))


class NeuralEtchEngine(nn.Module):
    """
    Neural Surrogate for the Plasma Etch stage (Resist -> Silicon Pattern).
    Models microloading and lateral undercut without spatial downsampling.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super(NeuralEtchEngine, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.ms_block1 = MultiscaleDilatedBlock(base_channels, base_channels)
        self.ms_block2 = MultiscaleDilatedBlock(base_channels, base_channels)
        
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Normalized final silicon height in [0, 1]
        )

    def forward(self, resist: torch.Tensor) -> torch.Tensor:
        feat = self.stem(resist)
        feat = self.ms_block1(feat)
        feat = self.ms_block2(feat)
        return self.head(feat)


if __name__ == "__main__":
    # Self-test module
    dummy_resist = torch.rand(2, 1, 128, 128)
    gt_silicon = simulate_etch_ground_truth(dummy_resist)
    print(f"[litho_etch] Ground truth silicon shape: {gt_silicon.shape} | Range: [{gt_silicon.min():.3f}, {gt_silicon.max():.3f}]")

    model = NeuralEtchEngine()
    pred_silicon = model(dummy_resist)
    print(f"[litho_etch] EtchNet output shape: {pred_silicon.shape}")
