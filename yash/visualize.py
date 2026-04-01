import os

import torch
import torch.nn.functional as F
from PIL import Image

from model import center_crop_2d


def save_visual_panel(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    save_dir: str,
    epoch: int,
    prefix: str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    image = images[0].detach().cpu().clamp(0.0, 1.0)
    target_h, target_w = logits.shape[2], logits.shape[3]
    image = F.interpolate(
        image.unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    masks = center_crop_2d(masks, target_h, target_w)
    mask = masks[0, 0].detach().cpu().clamp(0.0, 1.0)
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().clamp(0.0, 1.0)
    pred = (prob > 0.5).float()

    image_np = (image.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    mask_np = (mask.numpy() * 255.0).astype("uint8")
    prob_np = (prob.numpy() * 255.0).astype("uint8")
    pred_np = (pred.numpy() * 255.0).astype("uint8")

    mask_rgb = torch.from_numpy(mask_np).unsqueeze(-1).repeat(1, 1, 3).numpy()
    prob_rgb = torch.from_numpy(prob_np).unsqueeze(-1).repeat(1, 1, 3).numpy()
    pred_rgb = torch.from_numpy(pred_np).unsqueeze(-1).repeat(1, 1, 3).numpy()

    panel = torch.from_numpy(image_np)
    panel = torch.cat(
        [
            panel,
            torch.from_numpy(mask_rgb),
            torch.from_numpy(prob_rgb),
            torch.from_numpy(pred_rgb),
        ],
        dim=1,
    ).numpy()

    out_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:04d}.png")
    Image.fromarray(panel).save(out_path)
