from dataclasses import dataclass, field
from typing import Callable
import torch
import torchvision
from torchvision.models import ResNet50_Weights


@dataclass
class PSOConfig:
    """Hyperparameters for PSO-based adversarial patch generation."""

    # Image
    image_source: str = ""
    img_size: int = 224

    # Patch
    patch_size: int = 40
    num_circles: int = 100

    # PSO
    pop_size: int = 100
    generations: int = 100
    w: float = 0.6       # inertia weight
    c1: float = 1.5      # cognitive coefficient
    c2: float = 1.5      # social coefficient
    gravity: float = 0.1 # color gravity factor (0 = disabled, 1 = snap to palette)

    # Palette
    palette_size: int = 16

    # Fitness
    tolerance: float = 0.5  # margin threshold for pbest acceptance

    # Model
    model_factory: Callable = field(default=torchvision.models.resnet50, repr=False)
    model_weights: object = field(default=ResNet50_Weights.IMAGENET1K_V1, repr=False)

    # Runtime
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Derived constants
    N_LOC_PARAMS: int = field(default=2, init=False, repr=False)
    N_PARAMS_PER_CIRCLE: int = field(default=7, init=False, repr=False)

    @property
    def particle_dim(self) -> int:
        return self.N_LOC_PARAMS + self.num_circles * self.N_PARAMS_PER_CIRCLE

    @property
    def total_queries(self) -> int:
        return self.pop_size * self.generations
