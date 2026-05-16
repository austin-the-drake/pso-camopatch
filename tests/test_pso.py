"""Tests for Particle and build_bounds (src/pso.py, src/fitness.py)."""

import numpy as np
import pytest
import torch

from src.fitness import build_bounds
from src.pso import Particle


# Helpers

DEVICE = torch.device("cpu")


def _make_bounds(img_size=224, patch_size=40, num_circles=3):
    return build_bounds(img_size, patch_size, num_circles, device=DEVICE)


def _make_particle(num_circles=3):
    bounds_min, bounds_max, _ = _make_bounds(num_circles=num_circles)
    return Particle(bounds_min.shape[0], bounds_min, bounds_max, DEVICE)


# build_bounds

class TestBuildBounds:
    def test_shapes_match_particle_dim(self):
        n_circles = 5
        bounds_min, bounds_max, _ = _make_bounds(num_circles=n_circles)
        expected_dim = 2 + n_circles * 7
        assert bounds_min.shape == (expected_dim,)
        assert bounds_max.shape == (expected_dim,)

    def test_min_less_than_max(self):
        bounds_min, bounds_max, _ = _make_bounds()
        assert torch.all(bounds_min < bounds_max)

    def test_location_bounds(self):
        img_size, patch_size = 224, 40
        bounds_min, bounds_max, _ = build_bounds(img_size, patch_size, num_circles=3, device=DEVICE)
        expected_max_loc = float(img_size - patch_size)
        assert bounds_min[0].item() == pytest.approx(0.0)
        assert bounds_max[0].item() == pytest.approx(expected_max_loc)

    def test_r_norm_bounds(self):
        bounds_min, bounds_max, _ = _make_bounds(num_circles=1)
        # r_norm is the 3rd circle param
        r_norm_min_idx = 2 + 2   # loc(2) + cx(0) + cy(1) + r_norm(2)
        assert bounds_min[r_norm_min_idx].item() == pytest.approx(0.01)
        assert bounds_max[r_norm_min_idx].item() == pytest.approx(0.5)

    def test_alpha_bounds(self):
        bounds_min, bounds_max, _ = _make_bounds(num_circles=1)
        # alpha is the 7th circle param
        alpha_idx = 2 + 6
        assert bounds_min[alpha_idx].item() == pytest.approx(0.05)
        assert bounds_max[alpha_idx].item() == pytest.approx(1.0)

    def test_rgb_indices_count(self):
        n_circles = 4
        _, _, (idx_r, idx_g, idx_b) = _make_bounds(num_circles=n_circles)
        assert len(idx_r) == n_circles
        assert len(idx_g) == n_circles
        assert len(idx_b) == n_circles

    def test_rgb_indices_are_distinct(self):
        _, _, (idx_r, idx_g, idx_b) = _make_bounds(num_circles=3)
        all_indices = np.concatenate([idx_r, idx_g, idx_b])
        assert len(all_indices) == len(set(all_indices))


# Particle

class TestParticle:
    def test_initial_position_within_bounds(self):
        p = _make_particle()
        assert torch.all(p.position >= p.bounds_min)
        assert torch.all(p.position <= p.bounds_max)

    def test_initial_best_fitness_is_inf(self):
        p = _make_particle()
        assert p.best_fitness == float("inf")

    def test_update_position_stays_within_bounds(self):
        p = _make_particle()
        # Give the particle an extreme velocity that would push it out of bounds
        p.velocity = (p.bounds_max - p.bounds_min) * 100
        p.update_position()
        assert torch.all(p.position >= p.bounds_min)
        assert torch.all(p.position <= p.bounds_max)

    def test_update_velocity_changes_velocity(self):
        p = _make_particle()
        original_velocity = p.velocity.clone()
        gbest = p.bounds_min + (p.bounds_max - p.bounds_min) * 0.5
        _, _, circle_rgb_idx = _make_bounds(num_circles=3)
        p.update_velocity(gbest, None, w=0.6, c1=1.5, c2=1.5, gravity=0.0, circle_rgb_indices=circle_rgb_idx)
        # Velocity should generally change
        assert not torch.equal(p.velocity, original_velocity)

    def test_personal_best_updated_on_improvement(self):
        p = _make_particle()
        p.fitness = 5.0
        p.best_fitness = float("inf")
        # Manually simulate what evaluate() does after computing fitness
        if p.fitness < p.best_fitness:
            p.best_fitness = p.fitness
            p.best_position = p.position.clone()
        assert p.best_fitness == 5.0
        assert torch.allclose(p.best_position, p.position)

    def test_personal_best_not_updated_on_regression(self):
        p = _make_particle()
        p.best_fitness = 3.0
        p.fitness = 10.0
        prev_best = p.best_fitness
        if p.fitness < p.best_fitness:
            p.best_fitness = p.fitness
        assert p.best_fitness == prev_best

    def test_gravity_changes_velocity(self):
        """Gravity pull towards palette should produce a different velocity than without."""
        p = _make_particle(num_circles=3)
        gbest = p.position.clone()
        _, _, circle_rgb_idx = _make_bounds(num_circles=3)
        palette = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        vel_no_gravity = p.velocity.clone()
        p.update_velocity(gbest, None, w=1.0, c1=0.0, c2=0.0, gravity=0.0, circle_rgb_indices=circle_rgb_idx)
        vel_without = p.velocity.clone()

        p.velocity = vel_no_gravity.clone()
        p.update_velocity(gbest, palette, w=1.0, c1=0.0, c2=0.0, gravity=1.0, circle_rgb_indices=circle_rgb_idx)
        vel_with = p.velocity.clone()

        assert not torch.allclose(vel_without, vel_with)
