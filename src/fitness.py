from typing import Optional

import numpy as np
import torch

from .patch import apply_patch, generate_patch, generate_patch_tensors
from .utils import normalize_tensor


def build_bounds(
    img_size: int,
    patch_size: int,
    num_circles: int,
    n_loc_params: int = 2,
    n_params_per_circle: int = 7,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build parameter bounds for PSO. Per-circle layout: (cx_norm, cy_norm, r_norm, R, G, B, alpha).
    Returns (bounds_min, bounds_max, rgb_indices).
    """
    loc_bounds = [(0.0, float(img_size - patch_size))] * n_loc_params

    circle_spec = [(0.0, 1.0)] * n_params_per_circle
    circle_spec[2] = (0.01, 0.5)   # r_norm
    circle_spec[6] = (0.05, 1.0)   # alpha

    all_bounds = loc_bounds + circle_spec * num_circles

    bounds_min = torch.tensor([b[0] for b in all_bounds], dtype=torch.float32, device=device)
    bounds_max = torch.tensor([b[1] for b in all_bounds], dtype=torch.float32, device=device)

    # Indices of R, G, B params for each circle in the flat vector
    base = n_loc_params
    idx_r = np.arange(num_circles) * n_params_per_circle + base + 3
    idx_g = np.arange(num_circles) * n_params_per_circle + base + 4
    idx_b = np.arange(num_circles) * n_params_per_circle + base + 5

    return bounds_min, bounds_max, (idx_r, idx_g, idx_b)


def evaluate_fitness(
    particle_params: torch.Tensor,
    image_unnorm: torch.Tensor,
    original_label: int,
    model: torch.nn.Module,
    patch_size: int = 40,
    num_circles: int = 100,
    tolerance: float = 0.5,
) -> tuple[float, bool, float, torch.Tensor]:
    """
    Evaluate patch fitness. Returns (fitness, misclassified, l2, patched_image).
    Fitness is squared L2 distance if margin < tolerance, else infinity.
    """
    patched_unnorm, background = apply_patch(image_unnorm, particle_params, patch_size, num_circles)

    # Extract patch RGB for L2 calculation
    patch_x = int(torch.clamp(particle_params[0].round(), 0, float(image_unnorm.shape[3] - patch_size)).item())
    patch_y = int(torch.clamp(particle_params[1].round(), 0, float(image_unnorm.shape[2] - patch_size)).item())
    patch_rgb = patched_unnorm[0, :, patch_y : patch_y + patch_size, patch_x : patch_x + patch_size]

    patched_norm = normalize_tensor(patched_unnorm)
    with torch.no_grad():
        model.eval()
        logits = model(patched_norm)[0]  # [num_classes]

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    original_logit = logits[original_label]

    if sorted_indices[0].item() == original_label:
        highest_other = sorted_logits[1] if len(sorted_logits) > 1 else original_logit - 1e-6
    else:
        highest_other = sorted_logits[0]

    margin = (original_logit - highest_other).item()
    is_misclassified = sorted_indices[0].item() != original_label

    # Squared L2 distance matches the CamoPatch paper's metric
    l2_dist = torch.sum((patch_rgb - background).pow(2)).item()
    fitness = l2_dist if margin < tolerance else float("inf")

    return fitness, is_misclassified, l2_dist, patched_unnorm
