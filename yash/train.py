import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from losses import SegmentationLoss, dice_score_from_logits, iou_score_from_logits
from model import UNet, center_crop_2d
from visualize import save_visual_panel


def _append_csv_row(csv_path: str, header: list[str], row: list[object]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def estimate_foreground_fraction(dataset: Dataset, max_samples: int | None = None) -> float:
    total_fg = 0.0
    total_px = 0.0
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for i in range(n):
        _, mask = dataset[i]
        total_fg += float(mask.sum().item())
        total_px += float(mask.numel())
    return total_fg / max(total_px, 1.0)


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        target = center_crop_2d(masks, logits.shape[2], logits.shape[3])
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        train_loss += float(loss.item())

    return train_loss / len(train_loader)


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            target = center_crop_2d(masks, logits.shape[2], logits.shape[3])
            loss = criterion(logits, target)

            val_loss += float(loss.item())
            val_dice += dice_score_from_logits(logits, target)
            val_iou += iou_score_from_logits(logits, target)

    return (
        val_loss / len(val_loader),
        val_dice / len(val_loader),
        val_iou / len(val_loader),
    )


def run_overfit_sanity_check(
    device: torch.device,
    in_channels: int = 3,
    num_classes: int = 1,
    tiny_samples: int = 6,
    image_size: int = 256,
    batch_size: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
    loss_mode: str = "bce_dice",
    dice_lambda: float = 0.5,
    pos_weight: float | None = None,
    use_border_weights: bool = False,
    border_w0: float = 10.0,
    border_sigma: float = 5.0,
    vis_every: int = 20,
    vis_dir: str = "artifacts/overfit",
    metrics_csv_path: str | None = None,
    dataset: Dataset | None = None,
) -> None:
    if dataset is None:
        raise ValueError("run_overfit_sanity_check requires a real dataset instance.")
    tiny_n = min(tiny_samples, len(dataset))
    tiny_subset = Subset(dataset, list(range(tiny_n)))
    train_loader = DataLoader(tiny_subset, batch_size=batch_size, shuffle=True)

    img0, mask0 = dataset[0]
    print(f"[Sanity] dataset[0] -> image: {img0.shape}, mask: {mask0.shape}")

    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)
    fg_fraction = estimate_foreground_fraction(tiny_subset)
    auto_pos_weight = (1.0 - fg_fraction) / max(fg_fraction, 1e-6)
    use_pos_weight = auto_pos_weight if pos_weight is None else pos_weight
    criterion = SegmentationLoss(
        mode=loss_mode,
        dice_lambda=dice_lambda,
        pos_weight=use_pos_weight,
        use_border_weights=use_border_weights,
        border_w0=border_w0,
        border_sigma=border_sigma,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"[Sanity] loss_mode={loss_mode}, dice_lambda={dice_lambda}, "
        f"fg_fraction={fg_fraction:.4f}, pos_weight={use_pos_weight:.2f}"
    )

    print(f"[Sanity] Overfitting on {tiny_n} samples for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            target = center_crop_2d(masks, logits.shape[2], logits.shape[3])
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_dice += dice_score_from_logits(logits.detach(), target)
            epoch_iou += iou_score_from_logits(logits.detach(), target)

        n_batches = len(train_loader)
        epoch_loss /= n_batches
        epoch_dice /= n_batches
        epoch_iou /= n_batches

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"[Sanity][Epoch {epoch:03d}/{epochs}] "
                f"loss={epoch_loss:.4f} dice={epoch_dice:.4f} iou={epoch_iou:.4f}"
            )

        if metrics_csv_path is not None:
            _append_csv_row(
                metrics_csv_path,
                ["epoch", "loss", "dice", "iou"],
                [epoch, f"{epoch_loss:.6f}", f"{epoch_dice:.6f}", f"{epoch_iou:.6f}"],
            )

        if epoch == 1 or epoch % vis_every == 0 or epoch == epochs:
            with torch.no_grad():
                sample_images, sample_masks = next(iter(train_loader))
                sample_images = sample_images.to(device)
                sample_masks = sample_masks.to(device)
                sample_logits = model(sample_images)
                save_visual_panel(sample_images, sample_masks, sample_logits, vis_dir, epoch, prefix="overfit")


def train_with_validation(
    device: torch.device,
    in_channels: int = 3,
    num_classes: int = 1,
    image_size: int = 256,
    batch_size: int = 8,
    epochs: int = 30,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    loss_mode: str = "bce_dice",
    dice_lambda: float = 0.5,
    pos_weight: float | None = None,
    use_border_weights: bool = False,
    border_w0: float = 10.0,
    border_sigma: float = 5.0,
    vis_every: int = 5,
    vis_dir: str = "artifacts/train",
    metrics_csv_path: str | None = None,
    checkpoint_path: str | None = None,
    dataset: Dataset | None = None,
) -> None:
    if dataset is None:
        raise ValueError("train_with_validation requires a real dataset instance.")
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    split_gen = torch.Generator().manual_seed(123)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    fg_fraction = estimate_foreground_fraction(train_ds)
    auto_pos_weight = (1.0 - fg_fraction) / max(fg_fraction, 1e-6)
    use_pos_weight = auto_pos_weight if pos_weight is None else pos_weight
    criterion = SegmentationLoss(
        mode=loss_mode,
        dice_lambda=dice_lambda,
        pos_weight=use_pos_weight,
        use_border_weights=use_border_weights,
        border_w0=border_w0,
        border_sigma=border_sigma,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"[Train] samples={len(dataset)}, split={train_size}/{val_size}, "
        f"batch={batch_size}, epochs={epochs}, lr={lr}, "
        f"loss_mode={loss_mode}, dice_lambda={dice_lambda}, "
        f"fg_fraction={fg_fraction:.4f}, pos_weight={use_pos_weight:.2f}"
    )

    best_val_dice = float("-inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        print(
            f"[Train][Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

        if metrics_csv_path is not None:
            _append_csv_row(
                metrics_csv_path,
                ["epoch", "train_loss", "val_loss", "val_dice", "val_iou"],
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_dice:.6f}",
                    f"{val_iou:.6f}",
                ],
            )

        if checkpoint_path is not None and val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_parent = Path(checkpoint_path).parent
            ckpt_parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        if epoch == 1 or epoch % vis_every == 0 or epoch == epochs:
            with torch.no_grad():
                sample_images, sample_masks = next(iter(val_loader))
                sample_images = sample_images.to(device)
                sample_masks = sample_masks.to(device)
                sample_logits = model(sample_images)
                save_visual_panel(sample_images, sample_masks, sample_logits, vis_dir, epoch, prefix="val")


def _tile_positions(length: int, step: int) -> list[int]:
    if length <= step:
        return [0]
    positions = list(range(0, length - step + 1, step))
    if positions[-1] != length - step:
        positions.append(length - step)
    return positions


def overlap_tile_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int = 572,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("overlap_tile_inference expects a single image tensor with shape [C,H,W] or [1,C,H,W]")

    with torch.no_grad():
        dummy = torch.zeros(1, image.shape[1], tile_size, tile_size, device=device, dtype=image.dtype)
        out_dummy = model(dummy)

    out_tile = int(out_dummy.shape[2])
    if out_tile <= 0:
        raise ValueError("Model returned invalid tile output size during overlap-tile setup.")
    margin = (tile_size - out_tile) // 2

    _, _, h, w = image.shape
    padded = torch.nn.functional.pad(image, (margin, margin, margin, margin), mode="reflect")

    y_positions = _tile_positions(h, out_tile)
    x_positions = _tile_positions(w, out_tile)

    stitched = torch.zeros(1, 1, h, w, device=device)
    counts = torch.zeros(1, 1, h, w, device=device)

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                tile = padded[:, :, y : y + tile_size, x : x + tile_size]
                logits_tile = model(tile)
                stitched[:, :, y : y + out_tile, x : x + out_tile] += logits_tile
                counts[:, :, y : y + out_tile, x : x + out_tile] += 1.0

    stitched = stitched / counts.clamp_min(1.0)
    return stitched.squeeze(0)


def run_overlap_inference_on_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    save_dir: str,
    tile_size: int = 572,
    threshold: float = 0.5,
    max_samples: int | None = None,
) -> None:
    from PIL import Image

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    for sample_idx, (images, names) in enumerate(loader):
        if max_samples is not None and sample_idx >= max_samples:
            break
        image = images[0]
        logits = overlap_tile_inference(model, image, tile_size=tile_size)
        prob = torch.sigmoid(logits[0]).detach().cpu().numpy()
        pred = (prob > threshold).astype("uint8") * 255
        input_gray = (image[0].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        prob_gray = (prob * 255.0).astype("uint8")
        stem = names[0].replace(".tif", "")
        Image.fromarray(input_gray).save(out_dir / f"{stem}_input.png")
        Image.fromarray(prob_gray).save(out_dir / f"{stem}_prob.png")
        Image.fromarray(pred).save(out_dir / f"{stem}_pred.png")
