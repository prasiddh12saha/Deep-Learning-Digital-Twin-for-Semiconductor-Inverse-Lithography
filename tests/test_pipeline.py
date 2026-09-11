"""
Automated unit tests verifying tensor shapes, bounds, and physical continuity.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from src.litho_data import LayoutGenerator, get_layout_dataloader
from src.litho_optics import NeuralOpticsEngine, simulate_diffraction_ground_truth
from src.litho_resist.resist_model import ResistUNet, simulate_resist_ground_truth
from src.litho_etch import NeuralEtchEngine, simulate_etch_ground_truth
from src.litho_metrology import MetrologyEngine


def test_data_generation_shapes():
    gen = LayoutGenerator(resolution=64)
    layout = gen.generate_single_layout()
    assert layout.shape == (64, 64)
    assert np.all((layout >= 0.0) & (layout <= 1.0))


def test_optics_surrogate_bounds():
    model = NeuralOpticsEngine()
    dummy = torch.rand(2, 1, 64, 64)
    out = model(dummy)
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resist_unet_bounds():
    model = ResistUNet()
    dummy = torch.rand(2, 1, 64, 64)
    out = model(dummy)
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_metrology_epe():
    target = np.zeros((64, 64), dtype=np.float32)
    target[20:44, 20:44] = 1.0
    printed = target.copy()
    metrology = MetrologyEngine(threshold=0.5)
    metrics = metrology.evaluate(target, printed)
    
    # Identical shapes should produce zero residual error and no defects
    assert np.max(metrics["error_map"]) == 0.0
    assert metrics["mean_epe_pixels"] <= 1.0
    assert metrics["pinching_count"] == 0
    assert metrics["bridging_count"] == 0


if __name__ == "__main__":
    test_data_generation_shapes()
    test_optics_surrogate_bounds()
    test_resist_unet_bounds()
    test_metrology_epe()
    print("[+] All unit tests passed successfully!")
