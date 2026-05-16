import io
from typing import Optional

import requests
import torch
import torchvision.transforms as T
from PIL import Image

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(
    image_source: str,
    target_size: int = 224,
    device: Optional[torch.device] = None,
) -> tuple[Image.Image, Image.Image, torch.Tensor]:
    """
    Loads an image from a URL or local path, resizes, center-crops, and
    converts to an unnormalized PyTorch tensor.

    Args:
        image_source: URL or local file path.
        target_size: Output square dimension in pixels.
        device: Target device for the returned tensor.

    Returns:
        (original_pil, cropped_pil, cropped_tensor_unnorm)
        - original_pil: The raw loaded image.
        - cropped_pil: Resized and center-cropped PIL image.
        - cropped_tensor_unnorm: Tensor [1, C, H, W] in [0, 1] on `device`.
    """
    if device is None:
        device = torch.device("cpu")

    if image_source.startswith(("http://", "https://")):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_source, headers=headers, timeout=15)
        response.raise_for_status()
        original_full_pil = Image.open(io.BytesIO(response.content))
    else:
        original_full_pil = Image.open(image_source)

    original_full_pil = original_full_pil.convert("RGB")

    preprocess_pil = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
    ])
    cropped_pil = preprocess_pil(original_full_pil)
    tensor = T.ToTensor()(cropped_pil).unsqueeze(0).to(device)

    return original_full_pil, cropped_pil, tensor


def normalize_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
    """Applies ImageNet mean/std normalization to a [B, C, H, W] tensor."""
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    return (img_tensor - mean) / std


def denormalize_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
    """Reverses ImageNet normalization and clamps to [0, 1]."""
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    return torch.clamp(img_tensor * std + mean, 0.0, 1.0)
