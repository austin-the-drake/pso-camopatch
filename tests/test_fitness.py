"""Tests for evaluate_fitness using a lightweight mock model."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.fitness import build_bounds, evaluate_fitness
from src.patch import generate_patch


# Mock model helpers

NUM_CLASSES = 1000
ORIGINAL_LABEL = 42
DEVICE = torch.device("cpu")


def _make_params(x=10.0, y=10.0, num_circles=3, patch_size=40) -> torch.Tensor:
    loc = [x, y]
    circle = [0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.8]
    return torch.tensor(loc + circle * num_circles, dtype=torch.float32)


def _make_image(size=224) -> torch.Tensor:
    return torch.rand(1, 3, size, size)


class AlwaysCorrectModel(nn.Module):
    """Returns logits that make the original label the clear winner."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], NUM_CLASSES)
        logits[:, ORIGINAL_LABEL] = 10.0  # large margin
        return logits


class AlwaysWrongModel(nn.Module):
    """Returns logits that make a different label the winner."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], NUM_CLASSES)
        logits[:, (ORIGINAL_LABEL + 1) % NUM_CLASSES] = 10.0
        return logits


class NarrowMarginModel(nn.Module):
    """Returns logits where the original label wins by exactly the margin."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], NUM_CLASSES)
        logits[:, ORIGINAL_LABEL] = self.margin  # margin vs all-zero others
        return logits

# Tests

class TestEvaluateFitness:
    def test_returns_inf_when_model_still_correct(self):
        model = AlwaysCorrectModel().eval()
        params = _make_params()
        img = _make_image()
        fitness, is_misclassified, l2, _ = evaluate_fitness(
            params, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3
        )
        assert fitness == float("inf")
        assert not is_misclassified

    def test_returns_finite_fitness_when_misclassified(self):
        model = AlwaysWrongModel().eval()
        params = _make_params()
        img = _make_image()
        fitness, is_misclassified, l2, _ = evaluate_fitness(
            params, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3
        )
        # If misclassified, fitness should be the L2 distance
        assert is_misclassified
        assert math.isfinite(fitness)
        assert fitness == pytest.approx(l2)

    def test_fitness_equals_l2_on_success(self):
        model = AlwaysWrongModel().eval()
        params = _make_params()
        img = _make_image()
        fitness, _, l2, _ = evaluate_fitness(
            params, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3
        )
        assert fitness == pytest.approx(l2)

    def test_l2_is_non_negative(self):
        model = AlwaysWrongModel().eval()
        params = _make_params()
        img = _make_image()
        _, _, l2, _ = evaluate_fitness(
            params, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3
        )
        assert l2 >= 0.0

    def test_patched_image_shape_unchanged(self):
        model = AlwaysCorrectModel().eval()
        params = _make_params()
        img = _make_image()
        _, _, _, patched = evaluate_fitness(
            params, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3
        )
        assert patched.shape == img.shape

    def test_tolerance_threshold_controls_acceptance(self):
        """Fitness should be finite only when margin < tolerance."""
        model = NarrowMarginModel(margin=0.3).eval()
        params = _make_params()
        img = _make_image()

        f_accepted, _, _, _ = evaluate_fitness(
            params, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3, tolerance=0.5
        )
        f_rejected, _, _, _ = evaluate_fitness(
            params, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3, tolerance=0.1
        )
        assert math.isfinite(f_accepted)
        assert f_rejected == float("inf")

    def test_different_positions_give_different_l2(self):
        """Different locations expose different backgrounds, changing the L2 score."""
        # Use a non-uniform image so backgrounds differ by position
        img = torch.arange(224 * 224 * 3, dtype=torch.float32).reshape(1, 3, 224, 224) / (224 * 224 * 3)
        model = AlwaysWrongModel().eval()

        params_a = _make_params(x=0.0, y=0.0)
        params_b = _make_params(x=50.0, y=50.0)

        _, _, l2_a, _ = evaluate_fitness(params_a, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3)
        _, _, l2_b, _ = evaluate_fitness(params_b, img, ORIGINAL_LABEL, model, patch_size=40, num_circles=3)

        assert l2_a != pytest.approx(l2_b)
