"""Random Search and Stochastic Hill Climbing baselines."""

from __future__ import annotations

from typing import Optional

import torch

from .fitness import build_bounds, evaluate_fitness
from .utils import normalize_tensor, preprocess_image


def random_search(
    image_source: str,
    model: torch.nn.Module,
    imagenet_labels: list[str],
    img_size: int = 224,
    patch_size: int = 40,
    num_circles: int = 100,
    total_queries: int = 10000,
    tolerance: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> tuple[bool, float, list[int], list[float]]:
    """
    Random Search baseline: uniformly samples parameter vectors until the evaluation budget is reached.

    Returns:
        (success, best_l2, query_history, l2_history)
    """
    _, _, img_unnorm = preprocess_image(image_source, img_size, device)
    with torch.no_grad():
        logits = model(normalize_tensor(img_unnorm))
    label_idx = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())
    print(f"RS: original prediction '{imagenet_labels[label_idx]}'")

    bounds_min, bounds_max, _ = build_bounds(img_size, patch_size, num_circles, device=device)
    param_range = bounds_max - bounds_min

    best_l2 = float("inf")
    best_pos: Optional[torch.Tensor] = None
    query_history: list[int] = []
    l2_history: list[float] = []

    for q in range(1, total_queries + 1):
        pos = torch.rand(bounds_min.shape[0], device=device) * param_range + bounds_min
        _, misclassified, l2, _ = evaluate_fitness(
            pos, img_unnorm, label_idx, model, patch_size, num_circles, tolerance
        )
        if misclassified and l2 < best_l2:
            best_l2 = l2
            best_pos = pos.clone()
            query_history.append(q)
            l2_history.append(best_l2)

    success = best_pos is not None
    print(f"RS done. Success={success}, best L2={best_l2:.4f}" if success else "RS: no patch found.")
    return success, best_l2, query_history, l2_history


def stochastic_hillclimber(
    image_source: str,
    model: torch.nn.Module,
    imagenet_labels: list[str],
    img_size: int = 224,
    patch_size: int = 40,
    num_circles: int = 100,
    total_queries: int = 10000,
    tolerance: float = 0.5,
    noise_scale: float = 0.05,
    device: torch.device = torch.device("cpu"),
) -> tuple[bool, float, list[int], list[float]]:
    """
    Stochastic Hill Climbing baseline: iteratively improves a single solution.

    Args:
        noise_scale: Gaussian noise magnitude as 0-1 over each parameter's bounds.

    Returns:
        (success, best_l2, query_history, l2_history)
    """
    _, _, img_unnorm = preprocess_image(image_source, img_size, device)
    with torch.no_grad():
        logits = model(normalize_tensor(img_unnorm))
    label_idx = int(torch.argmax(torch.softmax(logits, dim=1), dim=1).item())
    print(f"SHC: original prediction '{imagenet_labels[label_idx]}'")

    bounds_min, bounds_max, _ = build_bounds(img_size, patch_size, num_circles, device=device)
    param_range = bounds_max - bounds_min

    candidate = torch.rand(bounds_min.shape[0], device=device) * param_range + bounds_min
    current_fitness, _, _, _ = evaluate_fitness(
        candidate, img_unnorm, label_idx, model, patch_size, num_circles, tolerance
    )

    query_history: list[int] = []
    l2_history: list[float] = []

    for q in range(1, total_queries + 1):
        noise = torch.randn_like(candidate) * (noise_scale * param_range)
        neighbour = torch.clamp(candidate + noise, bounds_min, bounds_max)
        f_n, mis_n, l2_n, _ = evaluate_fitness(
            neighbour, img_unnorm, label_idx, model, patch_size, num_circles, tolerance
        )
        if mis_n and f_n < current_fitness:
            candidate = neighbour.clone()
            current_fitness = f_n
            query_history.append(q)
            l2_history.append(current_fitness)

    success = current_fitness < float("inf")
    print(
        f"SHC done. Success={success}, best L2={current_fitness:.4f}"
        if success
        else "SHC: no patch found."
    )
    return success, current_fitness, query_history, l2_history
