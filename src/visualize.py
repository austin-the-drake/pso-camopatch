"""Plotting utilities for results, convergence, and palette visualization."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_results_figure(
    original_pil: Image.Image,
    patch_viz_pil: Image.Image,
    adversarial_pil: Image.Image,
    original_label: str,
    final_label: str,
    best_l2: float,
    title_suffix: str = "",
) -> None:
    """1x3 figure: original image, patch on gray background, adversarial result."""
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    prefix = f"{title_suffix} " if title_suffix else ""

    axes[0].imshow(original_pil)
    axes[0].set_title(f"Original\n(Pred: {original_label})")
    axes[0].axis("off")

    l2_str = f"{best_l2:.4f}" if best_l2 != float("inf") else "N/A"
    axes[1].imshow(patch_viz_pil)
    axes[1].set_title(f"{prefix}Best Patch (L2: {l2_str})\n(on gray background)")
    axes[1].axis("off")

    axes[2].imshow(adversarial_pil)
    axes[2].set_title(f"{prefix}Adversarial\n(Pred: {final_label})")
    axes[2].axis("off")

    plt.suptitle(f"{prefix}Attack Results", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_convergence(
    query_history: list[int],
    l2_history: list[float],
    algorithm_name: str = "",
) -> None:
    """Plots best successful L2 distance vs. number of model queries."""
    plot_q, plot_l2 = [], []
    last_valid = float("inf")

    for q, l2 in zip(query_history, l2_history):
        if q == 0:
            continue
        if l2 != float("inf"):
            plot_q.append(q)
            plot_l2.append(l2)
            last_valid = l2
        elif last_valid != float("inf"):
            if not plot_q or q > plot_q[-1]:
                plot_q.append(q)
                plot_l2.append(last_valid)

    if not plot_q:
        print(f"Convergence plot skipped for {algorithm_name}: no successful patch found.")
        return

    plt.figure(figsize=(10, 6))
    label = algorithm_name if algorithm_name else "Algorithm"
    plt.plot(plot_q, plot_l2, marker=".", linestyle="-", label=label)
    plt.xlabel("Model Queries (Fitness Evaluations)")
    plt.ylabel("Best Successful L2 Distance (SSD)")
    plt.title(f"{label} - Optimization Progress")
    plt.grid(True, alpha=0.4)
    plt.legend()

    min_l2, max_l2 = min(plot_l2), max(plot_l2)
    margin = (max_l2 - min_l2) * 0.05 if max_l2 > min_l2 else 0.1
    if max(0.0, min_l2 - margin) < max_l2 + margin:
        plt.ylim(bottom=max(0.0, min_l2 - margin), top=max_l2 + margin)

    plt.tight_layout()
    plt.show()


def visualize_palette(palette_tensor: Optional[torch.Tensor]) -> None:
    """Displays a [K, 3] color palette tensor as a horizontal color strip."""
    if palette_tensor is None or palette_tensor.nelement() == 0:
        print("No palette to display.")
        return

    palette_np = palette_tensor.cpu().numpy()
    plt.figure(figsize=(10, 2))
    plt.imshow(np.expand_dims(palette_np, axis=0), aspect="auto")
    plt.title(f"Extracted Color Palette ({palette_np.shape[0]} colors)")
    plt.axis("off")
    plt.show()
