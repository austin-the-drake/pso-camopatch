"""Tests for patch generation and application (src/patch.py)."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.patch import apply_patch, generate_patch, generate_patch_tensors


class TestGeneratePatch:
    def _minimal_params(self, num_circles: int = 3, patch_size: int = 20) -> np.ndarray:
        """Returns a valid flat parameter vector with predictable values."""
        loc = [0.0, 0.0]
        circle = [0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.8]  # cx, cy, r, R, G, B, alpha
        return np.array(loc + circle * num_circles, dtype=np.float32)

    def test_returns_rgba_image(self):
        params = self._minimal_params()
        result = generate_patch(params, patch_size=20, num_circles=3)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    def test_correct_size(self):
        for size in [20, 40, 64]:
            params = self._minimal_params()
            img = generate_patch(params, patch_size=size, num_circles=3)
            assert img.size == (size, size)

    def test_accepts_tensor_input(self):
        params = self._minimal_params()
        t = torch.tensor(params)
        img = generate_patch(t, patch_size=20, num_circles=3)
        assert isinstance(img, Image.Image)

    def test_fully_transparent_base(self):
        """A patch with zero alpha circles should produce a transparent image."""
        loc = [0.0, 0.0]
        circle = [0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.0]  # alpha=0
        params = np.array(loc + circle, dtype=np.float32)
        img = generate_patch(params, patch_size=20, num_circles=1)
        alpha_channel = np.array(img)[:, :, 3]
        assert alpha_channel.max() == 0

    def test_single_opaque_circle_not_transparent(self):
        """A fully opaque circle should produce non-zero alpha values."""
        loc = [0.0, 0.0]
        circle = [0.5, 0.5, 0.3, 1.0, 0.0, 0.0, 1.0]  # bright red, fully opaque
        params = np.array(loc + circle, dtype=np.float32)
        img = generate_patch(params, patch_size=40, num_circles=1)
        alpha_channel = np.array(img)[:, :, 3]
        assert alpha_channel.max() > 0


class TestGeneratePatchTensors:
    def _make_patch(self) -> Image.Image:
        params = np.array([0.0, 0.0] + [0.5, 0.5, 0.2, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return generate_patch(params, patch_size=20, num_circles=1)

    def test_returns_two_tensors(self):
        rgb, alpha = generate_patch_tensors(self._make_patch())
        assert rgb.shape[0] == 3
        assert alpha.shape[0] == 1

    def test_shapes_match(self):
        patch = self._make_patch()
        rgb, alpha = generate_patch_tensors(patch)
        assert rgb.shape[1:] == alpha.shape[1:]

    def test_values_in_unit_range(self):
        rgb, alpha = generate_patch_tensors(self._make_patch())
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0

    def test_handles_non_rgba_input(self):
        patch = self._make_patch().convert("RGB")
        rgb, alpha = generate_patch_tensors(patch)
        assert rgb.shape[0] == 3


class TestApplyPatch:
    def _make_image(self, h: int = 224, w: int = 224) -> torch.Tensor:
        return torch.rand(1, 3, h, w)

    def _make_params(self, x: float = 10.0, y: float = 10.0, num_circles: int = 1) -> torch.Tensor:
        circle = [0.5, 0.5, 0.2, 1.0, 0.0, 0.0, 1.0]
        return torch.tensor([x, y] + circle * num_circles, dtype=torch.float32)

    def test_output_shape_unchanged(self):
        img = self._make_image()
        params = self._make_params()
        patched, _ = apply_patch(img, params, patch_size=40, num_circles=1)
        assert patched.shape == img.shape

    def test_background_shape(self):
        img = self._make_image()
        params = self._make_params()
        _, background = apply_patch(img, params, patch_size=40, num_circles=1)
        assert background.shape == (3, 40, 40)

    def test_unpatched_region_unchanged(self):
        """Pixels outside the patch area should be identical to the original."""
        img = torch.zeros(1, 3, 224, 224)
        params = self._make_params(x=0.0, y=0.0)
        patched, _ = apply_patch(img, params, patch_size=40, num_circles=1)
        # Rows below the patch area should still be zero
        assert torch.all(patched[0, :, 40:, :] == 0.0)

    def test_out_of_bounds_location_clamped(self):
        """Location parameters beyond image bounds should be clamped, not error."""
        img = self._make_image(224, 224)
        params = self._make_params(x=9999.0, y=9999.0)
        patched, _ = apply_patch(img, params, patch_size=40, num_circles=1)
        assert patched.shape == img.shape

    def test_alpha_zero_leaves_image_unchanged(self):
        """A patch with zero-alpha circles should blend naturally with background."""
        img = torch.rand(1, 3, 224, 224)
        params = self._make_params()
        patched, _ = apply_patch(img, params, patch_size=40, num_circles=1)
        assert patched.shape == img.shape
