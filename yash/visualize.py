import os
from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label

from model import center_crop_2d


def _read_metrics_csv(csv_path: str) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_float(values: list[str]) -> list[float]:
    return [float(v) for v in values]


def plot_overfit_metrics(metrics_csv_path: str) -> None:
    rows = _read_metrics_csv(metrics_csv_path)
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Metrics] matplotlib not available; skipping overfit metric plots.")
        return

    out_dir = Path(metrics_csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = _to_float([r["epoch"] for r in rows])
    loss = _to_float([r["loss"] for r in rows])
    dice = _to_float([r["dice"] for r in rows])
    iou = _to_float([r["iou"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss, marker="o", linewidth=2)
    ax.set_title("Overfit Sanity: Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overfit_loss_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, dice, marker="o", linewidth=2, label="Dice")
    ax.plot(epochs, iou, marker="s", linewidth=2, label="IoU")
    ax.set_title("Overfit Sanity: Dice/IoU vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overfit_scores_curve.png", dpi=180)
    plt.close(fig)


def plot_train_metrics(metrics_csv_path: str) -> None:
    rows = _read_metrics_csv(metrics_csv_path)
    if not rows:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Metrics] matplotlib not available; skipping train metric plots.")
        return

    out_dir = Path(metrics_csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = _to_float([r["epoch"] for r in rows])
    train_loss = _to_float([r["train_loss"] for r in rows])
    val_loss = _to_float([r["val_loss"] for r in rows])
    val_dice = _to_float([r["val_dice"] for r in rows])
    val_iou = _to_float([r["val_iou"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2, label="Val Loss")
    ax.set_title("Training: Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_val_loss_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_dice, marker="o", linewidth=2, label="Val Dice")
    ax.plot(epochs, val_iou, marker="s", linewidth=2, label="Val IoU")
    ax.set_title("Training: Validation Scores vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "val_scores_curve.png", dpi=180)
    plt.close(fig)


def remove_small_components(mask: np.ndarray, min_size: int = 40) -> np.ndarray:
    labeled, num = label(mask)
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        if int(comp.sum()) >= min_size:
            cleaned[comp] = 1

    return cleaned


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
    image = center_crop_2d(image.unsqueeze(0), target_h, target_w).squeeze(0)

    masks = center_crop_2d(masks, target_h, target_w)
    mask = masks[0, 0].detach().cpu().clamp(0.0, 1.0)
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().clamp(0.0, 1.0)
    pred_np = (prob.cpu().numpy() > 0.5).astype(np.uint8)
    pred_np = remove_small_components(pred_np, min_size=40)
    pred = torch.from_numpy(pred_np.astype(np.float32))

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
