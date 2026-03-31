import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split


def center_crop_2d(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    if target_h > h or target_w > w:
        raise ValueError(
            f"Cannot center-crop tensor of shape {(h, w)} to larger target {(target_h, target_w)}"
        )
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[:, :, top : top + target_h, left : left + target_w]


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
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
        skip = center_crop_2d(skip, x.shape[2], x.shape[3])
        x = torch.cat([skip, x], dim=1)
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

    def trace_shapes(self, x: torch.Tensor) -> torch.Tensor:
        print(f"[Trace] input:           {tuple(x.shape)}")

        skip1, x = self.down1(x)
        print(f"[Trace] enc1:            {tuple(skip1.shape)}")
        print(f"[Trace] pool1:           {tuple(x.shape)}")

        skip2, x = self.down2(x)
        print(f"[Trace] enc2:            {tuple(skip2.shape)}")
        print(f"[Trace] pool2:           {tuple(x.shape)}")

        skip3, x = self.down3(x)
        print(f"[Trace] enc3:            {tuple(skip3.shape)}")
        print(f"[Trace] pool3:           {tuple(x.shape)}")

        skip4, x = self.down4(x)
        print(f"[Trace] enc4:            {tuple(skip4.shape)}")
        print(f"[Trace] pool4:           {tuple(x.shape)}")

        x = self.bottleneck(x)
        print(f"[Trace] bottleneck:      {tuple(x.shape)}")

        x = self.up1.up(x)
        print(f"[Trace] up1 transposed:  {tuple(x.shape)}")
        s4 = center_crop_2d(skip4, x.shape[2], x.shape[3])
        x = torch.cat([s4, x], dim=1)
        print(f"[Trace] up1 concat:      {tuple(x.shape)}")
        x = self.up1.double_conv(x)
        print(f"[Trace] up1 out:         {tuple(x.shape)}")

        x = self.up2.up(x)
        print(f"[Trace] up2 transposed:  {tuple(x.shape)}")
        s3 = center_crop_2d(skip3, x.shape[2], x.shape[3])
        x = torch.cat([s3, x], dim=1)
        print(f"[Trace] up2 concat:      {tuple(x.shape)}")
        x = self.up2.double_conv(x)
        print(f"[Trace] up2 out:         {tuple(x.shape)}")

        x = self.up3.up(x)
        print(f"[Trace] up3 transposed:  {tuple(x.shape)}")
        s2 = center_crop_2d(skip2, x.shape[2], x.shape[3])
        x = torch.cat([s2, x], dim=1)
        print(f"[Trace] up3 concat:      {tuple(x.shape)}")
        x = self.up3.double_conv(x)
        print(f"[Trace] up3 out:         {tuple(x.shape)}")

        x = self.up4.up(x)
        print(f"[Trace] up4 transposed:  {tuple(x.shape)}")
        s1 = center_crop_2d(skip1, x.shape[2], x.shape[3])
        x = torch.cat([s1, x], dim=1)
        print(f"[Trace] up4 concat:      {tuple(x.shape)}")
        x = self.up4.double_conv(x)
        print(f"[Trace] up4 out:         {tuple(x.shape)}")

        x = self.final_conv(x)
        print(f"[Trace] final output:    {tuple(x.shape)}")
        return x


class SyntheticShapesDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 128,
        in_channels: int = 3,
        seed: int = 42,
        augment: bool = False,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.seed = seed
        self.augment = augment

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
        if self.augment:
            image, mask = self._apply_augmentations(image, mask)
        return image, mask

    def _apply_augmentations(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Geometric transforms must stay synchronized between image and mask.
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])

        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            image = torch.rot90(image, k=k, dims=(1, 2))
            mask = torch.rot90(mask, k=k, dims=(1, 2))

        if torch.rand(1).item() < 0.8:
            h = image.shape[1]
            w = image.shape[2]
            scale = 0.75 + 0.25 * torch.rand(1).item()
            crop_h = max(8, int(h * scale))
            crop_w = max(8, int(w * scale))
            top = int(torch.randint(0, h - crop_h + 1, (1,)).item())
            left = int(torch.randint(0, w - crop_w + 1, (1,)).item())

            image = image[:, top : top + crop_h, left : left + crop_w]
            mask = mask[:, top : top + crop_h, left : left + crop_w]

            image = F.interpolate(
                image.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=(h, w),
                mode="nearest",
            ).squeeze(0)

        if torch.rand(1).item() < 0.5:
            contrast = 0.8 + 0.4 * torch.rand(1).item()
            brightness = -0.1 + 0.2 * torch.rand(1).item()
            image = image * contrast + brightness

        image = image.clamp(0.0, 1.0)
        mask = (mask > 0.5).float()
        return image, mask


def soft_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + eps) / (denom + eps)).mean()


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


def estimate_foreground_fraction(dataset: Dataset, max_samples: int | None = None) -> float:
    total_fg = 0.0
    total_px = 0.0
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for i in range(n):
        _, mask = dataset[i]
        total_fg += float(mask.sum().item())
        total_px += float(mask.numel())
    return total_fg / max(total_px, 1.0)


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
        [panel, torch.from_numpy(mask_rgb), torch.from_numpy(prob_rgb), torch.from_numpy(pred_rgb)],
        dim=1,
    ).numpy()

    out_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch:04d}.png")
    Image.fromarray(panel).save(out_path)


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
                target = center_crop_2d(masks, logits.shape[2], logits.shape[3])
                loss = criterion(logits, target)

                val_loss += float(loss.item())
                val_dice += dice_score_from_logits(logits, target)
                val_iou += iou_score_from_logits(logits, target)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net forward + training milestones")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["forward", "trace", "overfit", "train", "all"],
        help="Which pipeline to run.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overfit-epochs", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--tiny-samples", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overfit-lr", type=float, default=1e-4)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--loss-mode", type=str, default="bce_dice", choices=["bce", "bce_dice"])
    parser.add_argument("--dice-lambda", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true", help="Enable geometric/intensity augmentation for training.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Optional manual pos_weight override.")
    parser.add_argument("--vis-every", type=int, default=5)
    parser.add_argument("--vis-dir", type=str, default="artifacts")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("forward", "all"):
        # Milestone 1 checkpoint: forward pass shape test.
        model = UNet(in_channels=3, num_classes=1).to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        y = model(x)
        print(f"[Forward] output shape: {y.shape}")

    if args.mode in ("trace", "all"):
        # Shape flow checkpoint for original valid-convolution U-Net.
        model = UNet(in_channels=3, num_classes=1).to(device)
        x = torch.randn(1, 3, 572, 572, device=device)
        with torch.no_grad():
            y = model.trace_shapes(x)
        print(f"[Trace] output spatial size: {y.shape[2]}x{y.shape[3]}")

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
            loss_mode=args.loss_mode,
            dice_lambda=args.dice_lambda,
            pos_weight=args.pos_weight,
            vis_every=max(1, args.vis_every),
            vis_dir=os.path.join(args.vis_dir, "overfit"),
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
            loss_mode=args.loss_mode,
            dice_lambda=args.dice_lambda,
            use_augmentation=args.augment,
            pos_weight=args.pos_weight,
            vis_every=max(1, args.vis_every),
            vis_dir=os.path.join(args.vis_dir, "train"),
        )
