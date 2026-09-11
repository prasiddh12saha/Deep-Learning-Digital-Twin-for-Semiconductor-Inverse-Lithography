"""
MODULE 6: Computational Metrology & Yield Hotspot Detection
Extracts sub-pixel contours, computes Edge Placement Error (EPE),
and flags bridging/pinching lithographic hotspots.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


def extract_contours(binary_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Extracts boundary edge pixels using Sobel gradient magnitude."""
    from scipy.ndimage import sobel
    binary = (binary_map >= threshold).astype(np.float32)
    dx = sobel(binary, axis=1)
    dy = sobel(binary, axis=0)
    edge_mag = np.hypot(dx, dy)
    return (edge_mag > 0.1).astype(np.float32)


class MetrologyEngine:
    """
    Computes lithographic yield metrics: EPE, PV Bands, and Hotspots.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def compute_epe_map(self, target: np.ndarray, printed: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Computes spatial Edge Placement Error using Chamfer/Distance Transform.
        Returns average EPE in grid pixels and the 2D residual error map.
        """
        from scipy.ndimage import distance_transform_edt
        
        target_bin = (target >= self.threshold).astype(bool)
        printed_bin = (printed >= self.threshold).astype(bool)

        # Distance from target edge to printed edge
        dist_target = distance_transform_edt(~target_bin)
        dist_printed = distance_transform_edt(~printed_bin)

        target_edges = extract_contours(target, self.threshold)
        epe_values = dist_printed[target_edges > 0]
        mean_epe = float(np.mean(epe_values)) if len(epe_values) > 0 else 0.0

        error_map = np.abs(target.astype(np.float32) - printed.astype(np.float32))
        return mean_epe, error_map

    def detect_hotspots(
        self,
        target: np.ndarray,
        printed: np.ndarray,
        pinching_limit: float = 0.35,
        bridging_limit: float = 0.65
    ) -> Dict[str, np.ndarray]:
        """
        Flags pinching (unintended breaks) and bridging (unintended shorts).
        """
        target_bin = target >= self.threshold
        printed_bin = printed >= self.threshold

        # Pinching: Drawn feature is missing in printed silicon
        pinching_mask = (target_bin & ~printed_bin).astype(np.float32)

        # Bridging: Silicon prints where no feature was drawn
        bridging_mask = (~target_bin & printed_bin).astype(np.float32)

        return {
            "pinching": pinching_mask,
            "bridging": bridging_mask,
            "pinching_count": int(np.sum(pinching_mask)),
            "bridging_count": int(np.sum(bridging_mask))
        }

    def evaluate(self, target: np.ndarray, printed: np.ndarray) -> Dict[str, object]:
        """Runs full computational metrology pipeline."""
        mean_epe, error_map = self.compute_epe_map(target, printed)
        hotspots = self.detect_hotspots(target, printed)

        return {
            "mean_epe_pixels": mean_epe,
            "error_map": error_map,
            "pinching_hotspots": hotspots["pinching"],
            "bridging_hotspots": hotspots["bridging"],
            "pinching_count": hotspots["pinching_count"],
            "bridging_count": hotspots["bridging_count"]
        }


if __name__ == "__main__":
    dummy_target = np.zeros((128, 128), dtype=np.float32)
    dummy_target[40:80, 40:80] = 1.0
    dummy_printed = dummy_target.copy()
    dummy_printed[38:82, 42:78] = 0.9  # Introduce distortion

    engine = MetrologyEngine()
    metrics = engine.evaluate(dummy_target, dummy_printed)
    print(f"[litho_metrology] Mean EPE: {metrics['mean_epe_pixels']:.3f} px")
    print(f"[litho_metrology] Pinching count: {metrics['pinching_count']} | Bridging count: {metrics['bridging_count']}")
