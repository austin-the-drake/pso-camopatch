from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .config import PSOConfig
from .fitness import build_bounds, evaluate_fitness
from .palette import extract_palette_manual_kmeans, find_closest_palette_colors_vectorized
from .utils import normalize_tensor, preprocess_image


class Particle:
    """A single particle in the PSO swarm."""

    def __init__(
        self,
        dim: int,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.dim = dim
        self.bounds_min = bounds_min.to(device)
        self.bounds_max = bounds_max.to(device)
        self.device = device

        param_range = self.bounds_max - self.bounds_min
        self.position = torch.rand(dim, device=device) * param_range + self.bounds_min
        self.velocity = torch.randn(dim, device=device) * 0.1 * param_range

        self.best_position = self.position.clone()
        self.best_fitness = float("inf")
        self.fitness = float("inf")
        self.l2_distance = float("inf")
        self.truly_misclassified = False

    def update_velocity(
        self,
        gbest_pos: torch.Tensor,
        palette_cpu: Optional[torch.Tensor],
        w: float,
        c1: float,
        c2: float,
        gravity: float,
        circle_rgb_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Update velocity with cognitive, social, and optional color-gravity terms."""
        gbest_pos = gbest_pos.to(self.device)
        r1 = torch.rand(self.dim, device=self.device)
        r2 = torch.rand(self.dim, device=self.device)

        inertia = w * self.velocity
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (gbest_pos - self.position)
        gravity_adj = torch.zeros_like(self.velocity)

        if gravity > 0 and palette_cpu is not None and palette_cpu.nelement() > 0:
            idx_r = torch.tensor(circle_rgb_indices[0], device=self.device, dtype=torch.long)
            idx_g = torch.tensor(circle_rgb_indices[1], device=self.device, dtype=torch.long)
            idx_b = torch.tensor(circle_rgb_indices[2], device=self.device, dtype=torch.long)

            current_colors = torch.stack(
                [self.position[idx_r], self.position[idx_g], self.position[idx_b]], dim=1
            )
            closest = find_closest_palette_colors_vectorized(current_colors, palette_cpu)
            pull = gravity * (closest - current_colors)

            gravity_adj.scatter_add_(0, idx_r, pull[:, 0])
            gravity_adj.scatter_add_(0, idx_g, pull[:, 1])
            gravity_adj.scatter_add_(0, idx_b, pull[:, 2])

        self.velocity = inertia + cognitive + social + gravity_adj

    def update_position(self) -> None:
        """Steps the particle and clamps it within bounds."""
        self.position += self.velocity
        self.position = torch.clamp(self.position, self.bounds_min, self.bounds_max)

    def evaluate(
        self,
        img_unnorm: torch.Tensor,
        label: int,
        model: torch.nn.Module,
        patch_size: int,
        num_circles: int,
        tolerance: float,
    ) -> None:
        """Calls evaluate_fitness and updates current state and personal best."""
        fitness, is_misclassified, l2, _ = evaluate_fitness(
            self.position, img_unnorm, label, model, patch_size, num_circles, tolerance
        )
        self.fitness = fitness
        self.truly_misclassified = is_misclassified
        self.l2_distance = l2

        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.clone()


def run(cfg: PSOConfig, plot: bool = True) -> tuple[bool, float, list[int], list[float]]:
    """
    Execute PSO search for an adversarial patch on a single image.

    Returns a tuple (success, final_l2, query_history, l2_history) where success
    indicates if a misclassifying patch was found, final_l2 is its squared L2
    distance, and the histories track improvement over model queries.
    """
    import time
    from . import visualize

    t_start = time.time()

    model = cfg.model_factory(weights=cfg.model_weights)
    model.eval().to(cfg.device)

    _, cropped_pil, img_unnorm = preprocess_image(cfg.image_source, cfg.img_size, cfg.device)

    with torch.no_grad():
        logits = model(normalize_tensor(img_unnorm))
    probs = torch.softmax(logits, dim=1)
    label_idx = int(torch.argmax(probs, dim=1).item())
    confidence = probs[0, label_idx].item()
    imagenet_labels = cfg.model_weights.meta["categories"]
    label_str = imagenet_labels[label_idx]
    print(f"Original prediction: {label_str} ({label_idx}) @ {confidence:.4f}")

    palette = None
    gravity = cfg.gravity
    if gravity > 0:
        palette = extract_palette_manual_kmeans(cropped_pil, cfg.palette_size)
        if palette is None:
            gravity = 0.0

    bounds_min, bounds_max, circle_rgb_idx = build_bounds(
        cfg.img_size, cfg.patch_size, cfg.num_circles,
        cfg.N_LOC_PARAMS, cfg.N_PARAMS_PER_CIRCLE, cfg.device,
    )

    particles = [
        Particle(cfg.particle_dim, bounds_min, bounds_max, cfg.device)
        for _ in range(cfg.pop_size)
    ]
    gbest_l2 = float("inf")
    gbest_pos = torch.zeros(cfg.particle_dim, device=cfg.device)
    found = False
    query_history: list[int] = [0]
    l2_history: list[float] = [float("inf")]
    total_queries = 0

    print(f"PSO: pop={cfg.pop_size}, gens={cfg.generations}, target='{label_str}'")

    for gen in range(cfg.generations):
        for particle in particles:
            particle.evaluate(img_unnorm, label_idx, model, cfg.patch_size, cfg.num_circles, cfg.tolerance)
            total_queries += 1

            if particle.truly_misclassified and particle.l2_distance < gbest_l2:
                gbest_l2 = particle.l2_distance
                gbest_pos = particle.position.clone()
                found = True
                query_history.append(total_queries)
                l2_history.append(gbest_l2)

        # Record plateau point for smooth convergence plots
        if query_history[-1] < total_queries and l2_history[-1] == gbest_l2:
            query_history.append(total_queries)
            l2_history.append(gbest_l2)

        for particle in particles:
            particle.update_velocity(gbest_pos, palette, cfg.w, cfg.c1, cfg.c2, gravity, circle_rgb_idx)
            particle.update_position()

        gbest_str = f"{gbest_l2:.4f}" if gbest_l2 != float("inf") else "N/A"
        print(f"  gen {gen + 1:>3}/{cfg.generations} | best L2: {gbest_str} | queries: {total_queries}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s, {total_queries} queries total.")

    if found:
        # Reevaluate best position for final stats & visualisation
        _, _, final_l2, final_patched = evaluate_fitness(
            gbest_pos, img_unnorm, label_idx, model,
            cfg.patch_size, cfg.num_circles, cfg.tolerance,
        )
        with torch.no_grad():
            final_logits = model(normalize_tensor(final_patched))
        final_idx = int(torch.argmax(torch.softmax(final_logits, dim=1), dim=1).item())
        final_label = imagenet_labels[final_idx]

        loc = gbest_pos[:cfg.N_LOC_PARAMS].cpu().numpy()
        cx = round(float(np.clip(loc[0], 0, cfg.img_size - cfg.patch_size)) + cfg.patch_size / 2)
        cy = round(float(np.clip(loc[1], 0, cfg.img_size - cfg.patch_size)) + cfg.patch_size / 2)
        print(f"Best patch: L2={np.sqrt(gbest_l2):.4f}, pred='{final_label}', centre=({cx},{cy})")

        if plot:
            from .patch import generate_patch, generate_patch_tensors
            import torchvision.transforms as T

            best_patch_pil = generate_patch(gbest_pos, cfg.patch_size, cfg.num_circles)
            patch_rgb_t, patch_alpha_t = generate_patch_tensors(best_patch_pil)
            gray_bg = torch.ones_like(patch_rgb_t) * 0.5
            vis_on_gray = T.ToPILImage()(
                torch.clamp(gray_bg * (1 - patch_alpha_t) + patch_rgb_t * patch_alpha_t, 0, 1).cpu()
            )
            patch_x = int(np.clip(round(loc[0]), 0, cfg.img_size - cfg.patch_size))
            patch_y = int(np.clip(round(loc[1]), 0, cfg.img_size - cfg.patch_size))
            adv_pil = cropped_pil.copy()
            adv_pil.paste(best_patch_pil, (patch_x, patch_y), mask=best_patch_pil.split()[3])

            visualize.plot_results_figure(
                cropped_pil, vis_on_gray, adv_pil, label_str, final_label, gbest_l2, title_suffix="PSO"
            )
            visualize.plot_convergence(query_history, l2_history, algorithm_name="PSO")

        return True, gbest_l2, query_history, l2_history

    print("PSO failed: no patch satisfied the constraint.")
    if plot:
        visualize.plot_convergence(query_history, l2_history, algorithm_name="PSO")
    return False, float("inf"), query_history, l2_history
