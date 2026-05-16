"""Tests for palette extraction and nearest-color lookup (src/palette.py)."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.palette import extract_palette_manual_kmeans, find_closest_palette_colors_vectorized


class TestExtractPalette:
    def _solid_image(self, color: tuple, size: int = 64) -> Image.Image:
        img = Image.new("RGB", (size, size), color)
        return img

    def _random_image(self, size: int = 64) -> Image.Image:
        arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def test_returns_tensor(self):
        img = self._random_image()
        result = extract_palette_manual_kmeans(img, n_colors=4)
        assert isinstance(result, torch.Tensor)

    def test_correct_shape(self):
        img = self._random_image()
        for k in [4, 8, 16]:
            result = extract_palette_manual_kmeans(img, n_colors=k)
            assert result.shape == (k, 3)

    def test_values_in_unit_range(self):
        img = self._random_image()
        result = extract_palette_manual_kmeans(img, n_colors=8)
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_solid_red_image(self):
        """A solid red image should return a palette close to pure red."""
        img = self._solid_image((255, 0, 0))
        result = extract_palette_manual_kmeans(img, n_colors=4)
        # All centroids should converge to red (1, 0, 0) in [0,1] space
        assert torch.allclose(result[:, 0], torch.ones(4), atol=0.02)
        assert torch.allclose(result[:, 1], torch.zeros(4), atol=0.02)
        assert torch.allclose(result[:, 2], torch.zeros(4), atol=0.02)

    def test_k_exceeds_unique_pixels_is_handled(self):
        """Requesting more colors than unique pixels should not crash."""
        img = self._solid_image((128, 64, 32), size=4)  # only 1 unique pixel
        result = extract_palette_manual_kmeans(img, n_colors=32)
        assert result is not None
        assert result.shape[1] == 3

    def test_result_on_cpu(self):
        img = self._random_image()
        result = extract_palette_manual_kmeans(img, n_colors=4)
        assert result.device.type == "cpu"


class TestFindClosestPaletteColors:
    def _palette(self) -> torch.Tensor:
        # Red, Green, Blue
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_exact_match(self):
        palette = self._palette()
        colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = find_closest_palette_colors_vectorized(colors, palette)
        assert torch.allclose(result, colors)

    def test_nearest_neighbor(self):
        palette = self._palette()
        # Nearly-red color should map to red
        colors = torch.tensor([[0.9, 0.1, 0.0]])
        result = find_closest_palette_colors_vectorized(colors, palette)
        assert torch.allclose(result, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_none_palette_returns_original(self):
        colors = torch.tensor([[0.5, 0.5, 0.5]])
        result = find_closest_palette_colors_vectorized(colors, None)
        assert torch.allclose(result, colors)

    def test_empty_palette_returns_original(self):
        colors = torch.tensor([[0.5, 0.5, 0.5]])
        empty = torch.zeros(0, 3)
        result = find_closest_palette_colors_vectorized(colors, empty)
        assert torch.allclose(result, colors)

    def test_output_shape_matches_input(self):
        palette = self._palette()
        colors = torch.rand(10, 3)
        result = find_closest_palette_colors_vectorized(colors, palette)
        assert result.shape == colors.shape

    def test_all_results_are_palette_members(self):
        palette = self._palette()
        colors = torch.rand(20, 3)
        result = find_closest_palette_colors_vectorized(colors, palette)
        for row in result:
            assert any(torch.allclose(row, p) for p in palette)
