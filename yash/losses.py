import torch
import torch.nn as nn


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
    def __init__(self, mode: str = "bce", dice_lambda: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.mode = mode
        self.dice_lambda = dice_lambda
        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        if self.mode == "bce":
            return bce
        dice = soft_dice_from_logits(logits, target)
        return bce + self.dice_lambda * (1.0 - dice)
