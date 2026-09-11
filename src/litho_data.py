"""
MODULE 1: Synthetic IC Layout Data Generator & Rasterizer
Generates realistic Manhattan-style semiconductor geometries:
- Dense & isolated line-space gratings
- Contact hole / via arrays
- L-shaped and T-shaped poly junctions
Outputs normalized 2D PyTorch tensors of shape (B, 1, H, W) in range [0.0, 1.0].
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class LayoutGenerator:
    """Procedural generator for Manhattan IC layout patterns."""

    def __init__(self, resolution: int = 128):
        self.res = resolution

    def _create_line_space(self) -> np.ndarray:
        """Generates alternating horizontal or vertical line/space gratings."""
        grid = np.zeros((self.res, self.res), dtype=np.float32)
        pitch = np.random.randint(6, 16)
        width = np.random.randint(2, max(3, pitch // 2))
        horizontal = np.random.choice([True, False])

        for i in range(0, self.res, pitch):
            if horizontal:
                grid[i : min(i + width, self.res), :] = 1.0
            else:
                grid[:, i : min(i + width, self.res)] = 1.0
        return grid

    def _create_contact_array(self) -> np.ndarray:
        """Generates isolated or periodic contact/via arrays."""
        grid = np.zeros((self.res, self.res), dtype=np.float32)
        pitch = np.random.randint(8, 20)
        via_size = np.random.randint(3, 6)

        for y in range(4, self.res - via_size, pitch):
            for x in range(4, self.res - via_size, pitch):
                if np.random.rand() > 0.15:  # Random sparse dropouts
                    grid[y : y + via_size, x : x + via_size] = 1.0
        return grid

    def _create_junctions(self) -> np.ndarray:
        """Generates intersecting Manhattan T-junctions and L-shapes (poly gates)."""
        grid = np.zeros((self.res, self.res), dtype=np.float32)
        num_features = np.random.randint(3, 7)

        for _ in range(num_features):
            x1 = np.random.randint(8, self.res - 24)
            y1 = np.random.randint(8, self.res - 24)
            len_x = np.random.randint(12, 28)
            len_y = np.random.randint(12, 28)
            thickness = np.random.randint(3, 6)

            # Horizontal bar
            grid[y1 : y1 + thickness, x1 : min(x1 + len_x, self.res)] = 1.0
            # Vertical bar forming an L or T junction
            junction_x = x1 if np.random.choice([True, False]) else x1 + len_x - thickness
            grid[y1 : min(y1 + len_y, self.res), junction_x : junction_x + thickness] = 1.0

        return grid

    def generate_single_layout(self) -> np.ndarray:
        """Randomly selects a layout family and adds local variations."""
        pattern_type = np.random.choice(["line_space", "contacts", "junctions"])
        if pattern_type == "line_space":
            return self._create_line_space()
        elif pattern_type == "contacts":
            return self._create_contact_array()
        else:
            return self._create_junctions()


class LithoLayoutDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic lithography masks."""

    def __init__(self, num_samples: int = 500, resolution: int = 128, seed: int = 42):
        np.random.seed(seed)
        self.generator = LayoutGenerator(resolution=resolution)
        self.data = np.zeros((num_samples, 1, resolution, resolution), dtype=np.float32)

        for i in range(num_samples):
            self.data[i, 0] = self.generator.generate_single_layout()

        self.tensors = torch.from_numpy(self.data)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]


def get_layout_dataloader(
    num_samples: int = 200,
    batch_size: int = 16,
    resolution: int = 128,
    seed: int = 42,
    shuffle: bool = True
) -> DataLoader:
    """Helper factory function to return a ready-to-train PyTorch DataLoader."""
    dataset = LithoLayoutDataset(num_samples=num_samples, resolution=resolution, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Self-test when executed directly
    loader = get_layout_dataloader(num_samples=32, batch_size=8, resolution=64)
    sample_batch = next(iter(loader))
    print(f"[litho_data] Self-test complete.")
    print(f"[litho_data] Generated batch shape: {sample_batch.shape} | Range: [{sample_batch.min()}, {sample_batch.max()}]")
