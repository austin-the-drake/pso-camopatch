from typing import Optional

import numpy as np
import torch
from PIL import Image


def extract_palette_manual_kmeans(
    pil_image: Image.Image,
    n_colors: int = 16,
    max_iters: int = 100,
    tol: float = 1.0,
) -> Optional[torch.Tensor]:
    """
    Extract dominant colors via K-Means. Returns [K, 3] tensor with colors in [0, 1].
    """
    img_array = np.array(pil_image)
    pixels = img_array.reshape(-1, 3).astype(np.float32)
    n_pixels = pixels.shape[0]

    actual_k = min(n_colors, n_pixels)
    indices = np.random.choice(n_pixels, actual_k, replace=False)
    centroids = pixels[indices].copy()

    for i in range(max_iters):
        distances_sq = np.sum(
            (pixels[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2
        )
        assignments = np.argmin(distances_sq, axis=1)

        new_centroids = np.zeros((actual_k, 3), dtype=np.float32)
        counts = np.zeros(actual_k, dtype=int)
        np.add.at(new_centroids, assignments, pixels)
        np.add.at(counts, assignments, 1)

        valid = counts > 0
        new_centroids[valid] /= counts[valid, np.newaxis]
        new_centroids[~valid] = centroids[~valid]  # keep empty clusters in place

        shift_sq = np.sum((new_centroids - centroids) ** 2)
        centroids = new_centroids

        if shift_sq < tol ** 2:
            break

    palette = np.clip(centroids, 0, 255).astype(np.uint8) / 255.0
    return torch.tensor(palette, dtype=torch.float32)


def find_closest_palette_colors_vectorized(
    current_colors: torch.Tensor,
    palette_cpu: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Map each input color to its nearest palette color. Returns [N, 3] tensor.
    """
    if palette_cpu is None or palette_cpu.nelement() == 0 or current_colors.nelement() == 0:
        return current_colors

    device = current_colors.device
    palette = palette_cpu.to(device)
    distances_sq = torch.cdist(current_colors, palette, p=2.0).pow(2)
    min_indices = torch.argmin(distances_sq, dim=1)
    return palette[min_indices]
