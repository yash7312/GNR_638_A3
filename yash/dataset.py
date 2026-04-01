import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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
                radius = int(
                    torch.randint(self.image_size // 8, self.image_size // 4, (1,), generator=gen).item()
                )
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
        # Geometric transforms are synchronized to preserve pixel alignment.
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
