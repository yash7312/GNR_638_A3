import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def soft_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + eps) / (denom + eps)).mean()


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + eps) / (denom + eps)).mean()
    return float(dice.item())


def iou_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = ((intersection + eps) / (union + eps)).mean()
    return float(iou.item())


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        mode: str = "bce",
        dice_lambda: float = 0.5,
        pos_weight: float | None = None,
        use_border_weights: bool = False,
        border_w0: float = 10.0,
        border_sigma: float = 5.0,
    ):
        super().__init__()
        if mode not in {"bce", "bce_dice"}:
            raise ValueError("mode must be one of: bce, bce_dice")
        self.mode = mode
        self.dice_lambda = dice_lambda
        self.use_border_weights = use_border_weights
        self.border_w0 = border_w0
        self.border_sigma = border_sigma
        
        # Register pos_weight as a buffer once, reused in forward passes
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.tensor([pos_weight], dtype=torch.float32)
            )
        else:
            self.register_buffer("pos_weight", None)

    def _build_border_weight_map(self, target: torch.Tensor) -> torch.Tensor:
        """
        Paper-faithful border weighting based on sum of distances to nearest and
        second-nearest object boundaries.
        
        w(x) = w_c(x) + w_0 * exp(-(d_1(x) + d_2(x))^2 / (2*sigma^2))
        
        where:
        - w_c(x) is class-balancing weight
        - d_1(x) is distance to nearest object boundary
        - d_2(x) is distance to second-nearest object boundary
        - w_0, sigma are hyperparameters
        """
        target_np = target.detach().cpu().numpy()
        weight_maps = []
        
        for i in range(target_np.shape[0]):
            # Extract foreground mask
            fg = target_np[i, 0] > 0.5
            h, w = fg.shape
            
            # Label connected components to identify individual objects
            labeled_array, num_objects = label(fg)
            
            # Compute class-balancing weights
            fg_fraction = fg.mean()
            alpha_fg = (1.0 - fg_fraction) / max(fg_fraction, 1e-6)
            alpha_bg = 1.0
            class_weight = np.where(fg, alpha_fg, alpha_bg)
            
            # Collect per-object boundary distance maps, then compute true first/second minima.
            distance_stack = []
            if num_objects >= 1:
                for obj_id in range(1, num_objects + 1):
                    obj_mask = (labeled_array == obj_id)
                    eroded = binary_erosion(obj_mask)
                    boundary = np.logical_xor(obj_mask, eroded)
                    dist_to_boundary = distance_transform_edt(~boundary).astype(np.float32)
                    distance_stack.append(dist_to_boundary)

            if len(distance_stack) >= 2:
                stacked = np.stack(distance_stack, axis=0)  # [num_objects, H, W]
                sorted_stack = np.sort(stacked, axis=0)
                d_1 = sorted_stack[0]
                d_2 = sorted_stack[1]
            elif len(distance_stack) == 1:
                d_1 = distance_stack[0]
                d_2 = np.zeros_like(d_1, dtype=np.float32)
            else:
                d_1 = np.zeros((h, w), dtype=np.float32)
                d_2 = np.zeros((h, w), dtype=np.float32)
            
            # Compute border weighting term
            # If fewer than 2 objects, set w_border = 0
            if num_objects < 2:
                border_weight = np.zeros((h, w), dtype=np.float32)
            else:
                d_sum = d_1 + d_2
                exponent = -(d_sum ** 2) / (2.0 * (self.border_sigma ** 2))
                border_weight = self.border_w0 * np.exp(exponent)
            
            # Combine: w(x) = w_c(x) + w_border(x)
            final_weight = class_weight + border_weight
            weight_maps.append(final_weight.astype("float32"))
        
        weight_map = torch.from_numpy(np.stack(weight_maps, axis=0)).unsqueeze(1)
        return weight_map.to(device=target.device, dtype=target.dtype)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use pre-registered pos_weight buffer, cast to match logits device/dtype
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)

        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
        if self.use_border_weights:
            weights = self._build_border_weight_map(target)
            bce = (bce_map * weights).mean()
        else:
            bce = bce_map.mean()

        if self.mode == "bce":
            return bce
        dice = soft_dice_from_logits(logits, target)
        return bce + self.dice_lambda * (1.0 - dice)
