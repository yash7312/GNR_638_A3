import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # Encoder: 64 -> 128 -> 256 -> 512
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # Bottleneck: 1024
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder: 512 -> 256 -> 128 -> 64
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.final_conv(x)


class SyntheticShapesDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 128,
        in_channels: int = 3,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + idx)

        # Start from low-amplitude noise and paint foreground shapes.
        image = torch.rand(
            self.in_channels,
            self.image_size,
            self.image_size,
            generator=gen,
            dtype=torch.float32,
        ) * 0.15
        mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

        yy, xx = torch.meshgrid(
            torch.arange(self.image_size),
            torch.arange(self.image_size),
            indexing="ij",
        )

        num_shapes = int(torch.randint(1, 3, (1,), generator=gen).item())
        for _ in range(num_shapes):
            is_circle = bool(torch.randint(0, 2, (1,), generator=gen).item())
            if is_circle:
                radius = int(torch.randint(self.image_size // 8, self.image_size // 4, (1,), generator=gen).item())
                cx = int(torch.randint(radius, self.image_size - radius, (1,), generator=gen).item())
                cy = int(torch.randint(radius, self.image_size - radius, (1,), generator=gen).item())
                shape_region = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            else:
                h = int(torch.randint(self.image_size // 6, self.image_size // 2, (1,), generator=gen).item())
                w = int(torch.randint(self.image_size // 6, self.image_size // 2, (1,), generator=gen).item())
                x1 = int(torch.randint(0, self.image_size - w, (1,), generator=gen).item())
                y1 = int(torch.randint(0, self.image_size - h, (1,), generator=gen).item())
                x2, y2 = x1 + w, y1 + h
                shape_region = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)

            mask[0, shape_region] = 1.0

            color = torch.rand(self.in_channels, 1, generator=gen, dtype=torch.float32) * 0.8 + 0.2
            image[:, shape_region] = color

        image = image.clamp(0.0, 1.0)
        return image, mask


def dice_score_from_logits(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * mask).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + eps) / (denom + eps)).mean()
    return float(dice.item())


def iou_score_from_logits(logits: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * mask).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) - intersection
    iou = ((intersection + eps) / (union + eps)).mean()
    return float(iou.item())


def run_overfit_sanity_check(
    device: torch.device,
    in_channels: int = 3,
    num_classes: int = 1,
    tiny_samples: int = 6,
    image_size: int = 128,
    batch_size: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
) -> None:
    dataset = SyntheticShapesDataset(num_samples=max(20, tiny_samples), image_size=image_size, in_channels=in_channels)
    tiny_subset = Subset(dataset, list(range(tiny_samples)))
    train_loader = DataLoader(tiny_subset, batch_size=batch_size, shuffle=True)

    img0, mask0 = dataset[0]
    print(f"[Sanity] dataset[0] -> image: {img0.shape}, mask: {mask0.shape}")

    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_dice += dice_score_from_logits(logits.detach(), masks)
            epoch_iou += iou_score_from_logits(logits.detach(), masks)

        n_batches = len(train_loader)
        epoch_loss /= n_batches
        epoch_dice /= n_batches
        epoch_iou /= n_batches

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"[Sanity][Epoch {epoch:03d}/{epochs}] "
                f"loss={epoch_loss:.4f} dice={epoch_dice:.4f} iou={epoch_iou:.4f}"
            )


def train_with_validation(
    device: torch.device,
    in_channels: int = 3,
    num_classes: int = 1,
    num_samples: int = 120,
    image_size: int = 128,
    batch_size: int = 8,
    epochs: int = 30,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
) -> None:
    dataset = SyntheticShapesDataset(num_samples=num_samples, image_size=image_size, in_channels=in_channels)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    split_gen = torch.Generator().manual_seed(123)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"[Train] samples={num_samples}, split={train_size}/{val_size}, "
        f"batch={batch_size}, epochs={epochs}, lr={lr}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += float(loss.item())
                val_dice += dice_score_from_logits(logits, masks)
                val_iou += iou_score_from_logits(logits, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(
            f"[Train][Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net forward + training milestones")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["forward", "overfit", "train", "all"],
        help="Which pipeline to run.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--overfit-epochs", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--tiny-samples", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overfit-lr", type=float, default=1e-4)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("forward", "all"):
        # Milestone 1 checkpoint: forward pass shape test.
        model = UNet(in_channels=3, num_classes=1).to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        y = model(x)
        print(f"[Forward] output shape: {y.shape}")

    if args.mode in ("overfit", "all"):
        # Milestone 2: overfit tiny data sanity check.
        run_overfit_sanity_check(
            device=device,
            in_channels=3,
            num_classes=1,
            tiny_samples=args.tiny_samples,
            image_size=args.image_size,
            batch_size=min(4, args.batch_size),
            epochs=args.overfit_epochs,
            lr=args.overfit_lr,
        )

    if args.mode in ("train", "all"):
        # Milestone 3: train/val pipeline with Dice + IoU tracking.
        train_with_validation(
            device=device,
            in_channels=3,
            num_classes=1,
            num_samples=args.num_samples,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.train_epochs,
            lr=args.train_lr,
            val_ratio=0.2,
        )
