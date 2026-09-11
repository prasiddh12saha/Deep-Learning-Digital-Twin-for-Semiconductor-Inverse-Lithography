"""
MODULE 5: Differentiable Inverse Lithography Technology (ILT) Optimizer
Performs gradient-based mask synthesis by backpropagating wafer-level errors
through the frozen forward surrogate chain (Optics -> Resist -> Etch).
Outputs optimized mask M_opt(x,y) with curvilinear OPC and SRAFs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict


class TotalVariationLoss(nn.Module):
    """Penalizes high-frequency noise to enforce mask manufacturability."""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)


class DifferentiableILTOptimizer:
    """
    Inverse Lithography Solver utilizing PyTorch autograd.
    """
    def __init__(
        self,
        optics_model: nn.Module,
        resist_model: nn.Module,
        etch_model: nn.Module,
        device: torch.device
    ):
        self.device = device
        self.optics = optics_model.to(device).eval()
        self.resist = resist_model.to(device).eval()
        self.etch = etch_model.to(device).eval()

        # Freeze all surrogate network weights
        for model in [self.optics, self.resist, self.etch]:
            for param in model.parameters():
                param.requires_grad = False

        self.tv_loss = TotalVariationLoss()

    def forward_chain(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the complete surrogate chain forward."""
        aerial = self.optics(mask)
        resist = self.resist(aerial)
        silicon = self.etch(resist)
        return aerial, resist, silicon

    def optimize(
        self,
        target_silicon: torch.Tensor,
        iterations: int = 150,
        lr: float = 0.08,
        tv_weight: float = 0.005
    ) -> Dict[str, torch.Tensor]:
        """
        Optimizes mask pixels using gradient descent to match target silicon.
        """
        target = target_silicon.to(self.device)
        
        # Initialize continuous latent mask parameter
        latent_mask = nn.Parameter(torch.logit(target.clamp(0.05, 0.95)))
        optimizer = optim.Adam([latent_mask], lr=lr)
        
        history = []

        for step in range(iterations):
            optimizer.zero_grad()
            
            # Map unconstrained parameter to [0, 1] mask through sigmoid
            projected_mask = torch.sigmoid(latent_mask)
            
            # Forward simulation through frozen digital twin
            _, _, pred_silicon = self.forward_chain(projected_mask)
            
            # L2 Pattern fidelity loss + Total Variation manufacturability regularization
            fidelity_loss = torch.mean((pred_silicon - target) ** 2)
            reg_loss = self.tv_loss(projected_mask)
            total_loss = fidelity_loss + (tv_weight * reg_loss)
            
            # Backward pass directly to mask pixels
            total_loss.backward()
            optimizer.step()
            
            history.append(total_loss.item())

        optimized_mask = torch.sigmoid(latent_mask).detach()
        final_aerial, final_resist, final_silicon = self.forward_chain(optimized_mask)

        return {
            "optimized_mask": optimized_mask,
            "final_aerial": final_aerial.detach(),
            "final_resist": final_resist.detach(),
            "final_silicon": final_silicon.detach(),
            "loss_history": history
        }


if __name__ == "__main__":
    print("[litho_ilt] Module 5 loaded successfully.")
