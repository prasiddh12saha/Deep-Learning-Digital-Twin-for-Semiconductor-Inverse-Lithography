"""
MASTER PIPELINE: Neural Digital Twin for Semiconductor Inverse Lithography
Orchestrates Modules 1 through 6:
Data Generation -> Optics -> Resist -> Etch -> Differentiable ILT -> Metrology
"""

import sys
import torch
import numpy as np

from src.litho_data import get_layout_dataloader
from src.litho_optics import NeuralOpticsEngine
from src.litho_resist.resist_model import ResistUNet
from src.litho_etch import NeuralEtchEngine
from src.litho_ilt import DifferentiableILTOptimizer
from src.litho_metrology import MetrologyEngine


def run_full_digital_twin():
    print("=" * 70)
    print("   NEURAL DIGITAL TWIN FOR SEMICONDUCTOR INVERSE LITHOGRAPHY")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Module 1: Data Engine
    print("\n[Stage 1/5] Module 1 (Data): Generating synthetic Manhattan IC targets...")
    loader = get_layout_dataloader(num_samples=16, batch_size=1, resolution=128, shuffle=False)
    target_mask = next(iter(loader)).to(device)
    print(f"    Target tensor generated with shape: {target_mask.shape}")

    # 2. Modules 2, 3, 4: Forward Surrogate Chain
    print("\n[Stage 2/5] Initializing Neural Surrogates (Optics, Resist, Etch)...")
    optics = NeuralOpticsEngine().to(device).eval()
    resist = ResistUNet().to(device).eval()
    etch = NeuralEtchEngine().to(device).eval()
    print("    Forward physics models loaded and frozen.")

    # 3. Uncorrected Baseline Forward Pass
    with torch.no_grad():
        baseline_aerial = optics(target_mask)
        baseline_resist = resist(baseline_aerial)
        baseline_silicon = etch(baseline_resist)

    # 4. Module 5: Differentiable ILT Optimization
    print("\n[Stage 3/5] Module 5 (ILT): Running gradient-based mask synthesis...")
    optimizer = DifferentiableILTOptimizer(optics, resist, etch, device)
    results = optimizer.optimize(target_mask, iterations=80, lr=0.08)
    print(f"    Optimization complete. Final loss: {results['loss_history'][-1]:.6f}")

    # 5. Module 6: Computational Metrology
    print("\n[Stage 4/5] Module 6 (Metrology): Computing Edge Placement Errors (EPE)...")
    target_np = target_mask[0, 0].cpu().numpy()
    baseline_np = baseline_silicon[0, 0].cpu().numpy()
    ilt_silicon_np = results["final_silicon"][0, 0].cpu().numpy()

    metrology = MetrologyEngine(threshold=0.5)
    unopt_metrics = metrology.evaluate(target_np, baseline_np)
    opt_metrics = metrology.evaluate(target_np, ilt_silicon_np)

    print("\n" + "-" * 40 + " METROLOGY AUDIT " + "-" * 40)
    print(f"  Uncorrected Mask   -> Mean EPE: {unopt_metrics['mean_epe_pixels']:.4f} px | Bridging: {unopt_metrics['bridging_count']} | Pinching: {unopt_metrics['pinching_count']}")
    print(f"  ILT Optimized Mask  -> Mean EPE: {opt_metrics['mean_epe_pixels']:.4f} px | Bridging: {opt_metrics['bridging_count']} | Pinching: {opt_metrics['pinching_count']}")
    
    improvement = ((unopt_metrics['mean_epe_pixels'] - opt_metrics['mean_epe_pixels']) / (unopt_metrics['mean_epe_pixels'] + 1e-6)) * 100
    print(f"  Yield Improvement  -> EPE Reduced by: {improvement:.2f}%")
    print("-" * 97)
    print("\n[Stage 5/5] Digital Twin pipeline finished successfully.\n")


if __name__ == "__main__":
    run_full_digital_twin()
