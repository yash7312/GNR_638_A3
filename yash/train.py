import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from dataset import SyntheticShapesDataset
from losses import SegmentationLoss, dice_score_from_logits, iou_score_from_logits
from model import UNet, center_crop_2d
from visualize import save_visual_panel


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
    vis_every: int = 20,
    vis_dir: str = "artifacts/overfit",
) -> None:
    dataset = SyntheticShapesDataset(
        num_samples=max(20, tiny_samples),
        image_size=image_size,
        in_channels=in_channels,
        augment=False,
    )
    tiny_subset = Subset(dataset, list(range(tiny_samples)))
    train_loader = DataLoader(tiny_subset, batch_size=batch_size, shuffle=True)

    img0, mask0 = dataset[0]
    print(f"[Sanity] dataset[0] -> image: {img0.shape}, mask: {mask0.shape}")

    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)
    fg_fraction = estimate_foreground_fraction(tiny_subset)
    auto_pos_weight = (1.0 - fg_fraction) / max(fg_fraction, 1e-6)
    use_pos_weight = auto_pos_weight if pos_weight is None else pos_weight
    criterion = SegmentationLoss(mode=loss_mode, dice_lambda=dice_lambda, pos_weight=use_pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"[Sanity] loss_mode={loss_mode}, dice_lambda={dice_lambda}, "
        f"fg_fraction={fg_fraction:.4f}, pos_weight={use_pos_weight:.2f}"
    )

    print(f"[Sanity] Overfitting on {tiny_samples} samples for {epochs} epochs...")
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
    num_samples: int = 120,
    image_size: int = 256,
    batch_size: int = 8,
    epochs: int = 30,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    loss_mode: str = "bce_dice",
    dice_lambda: float = 0.5,
    use_augmentation: bool = True,
    pos_weight: float | None = None,
    vis_every: int = 5,
    vis_dir: str = "artifacts/train",
) -> None:
    dataset = SyntheticShapesDataset(
        num_samples=num_samples,
        image_size=image_size,
        in_channels=in_channels,
        augment=use_augmentation,
    )
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
    criterion = SegmentationLoss(mode=loss_mode, dice_lambda=dice_lambda, pos_weight=use_pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"[Train] samples={num_samples}, split={train_size}/{val_size}, "
        f"batch={batch_size}, epochs={epochs}, lr={lr}, augment={use_augmentation}, "
        f"loss_mode={loss_mode}, dice_lambda={dice_lambda}, "
        f"fg_fraction={fg_fraction:.4f}, pos_weight={use_pos_weight:.2f}"
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        print(
            f"[Train][Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

        if epoch == 1 or epoch % vis_every == 0 or epoch == epochs:
            with torch.no_grad():
                sample_images, sample_masks = next(iter(val_loader))
                sample_images = sample_images.to(device)
                sample_masks = sample_masks.to(device)
                sample_logits = model(sample_images)
                save_visual_panel(sample_images, sample_masks, sample_logits, vis_dir, epoch, prefix="val")
