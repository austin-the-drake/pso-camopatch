from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


def generate_patch(
    particle_params: Union[torch.Tensor, np.ndarray],
    patch_size: int = 40,
    num_circles: int = 100,
    background_rgb: Image.Image = None,
) -> Image.Image:
    """
    Render a patch by compositing semi-transparent circles. If background_rgb is
    provided, it is used as the starting image layer for proper alpha blending.
    Returns an RGBA PIL Image of size (patch_size, patch_size).
    """
    if isinstance(particle_params, torch.Tensor):
        params_np = particle_params.detach().cpu().numpy()
    else:
        params_np = np.asarray(particle_params)

    circle_params = params_np[2:]  # skip location params

    # Start with background if provided, else transparent
    if background_rgb is not None:
        if background_rgb.mode != "RGB":
            background_rgb = background_rgb.convert("RGB")
        patch = background_rgb.convert("RGBA")
    else:
        patch = Image.new("RGBA", (patch_size, patch_size), (0, 0, 0, 0))

    for i in range(num_circles):
        start = i * 7
        cx_norm, cy_norm, r_norm, r_val, g_val, b_val, alpha_val = map(
            float, circle_params[start : start + 7]
        )

        cx = cx_norm * patch_size
        cy = cy_norm * patch_size
        radius = max(1.0, r_norm * patch_size * 0.5)
        color = tuple(int(np.clip(v * 255, 0, 255)) for v in [r_val, g_val, b_val, alpha_val])

        layer = Image.new("RGBA", (patch_size, patch_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x1 = int(cx - radius + 0.5)
        y1 = int(cy - radius + 0.5)
        x2 = int(cx + radius + 0.5)
        y2 = int(cy + radius + 0.5)
        draw.ellipse([x1, y1, x2, y2], fill=color)

        patch = Image.alpha_composite(patch, layer)

    return patch


def generate_patch_tensors(
    patch_pil: Image.Image,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split an RGBA patch into separate RGB and alpha tensors [3, H, W] and [1, H, W].
    """
    if patch_pil.mode != "RGBA":
        patch_pil = patch_pil.convert("RGBA")

    t = T.ToTensor()(patch_pil).to(device)  # [4, H, W]
    return t[:3], t[3:4]


def apply_patch(
    image_tensor_unnorm: torch.Tensor,
    particle_params: torch.Tensor,
    patch_size: int,
    num_circles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a patch to an image. Generates the patch with background baked in for
    proper alpha compositing. Returns (patched_image, background_area).
    """
    _, _, img_h, img_w = image_tensor_unnorm.shape

    patch_x = int(torch.clamp(particle_params[0].round(), 0, float(img_w - patch_size)).item())
    patch_y = int(torch.clamp(particle_params[1].round(), 0, float(img_h - patch_size)).item())

    # Extract the background region
    background = image_tensor_unnorm[0, :, patch_y : patch_y + patch_size, patch_x : patch_x + patch_size]

    # Convert background tensor to PIL for use in patch generation
    bg_pil = T.ToPILImage()(background.cpu())

    # Generate patch with background baked in
    patch_pil = generate_patch(particle_params, patch_size, num_circles, background_rgb=bg_pil)
    patch_rgb, patch_alpha = generate_patch_tensors(patch_pil, image_tensor_unnorm.device)

    # Apply the patch
    patched = image_tensor_unnorm.clone()
    patched[0, :, patch_y : patch_y + patch_size, patch_x : patch_x + patch_size] = patch_rgb

    return patched, background
