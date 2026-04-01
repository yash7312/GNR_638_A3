from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset


def _extract_frame_id(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse frame id from file name: {path}")
    return int(match.group(1))


def apply_basic_augmentations(
    image: torch.Tensor,
    mask: torch.Tensor,
    rng: torch.Generator | None = None,
    np_rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rng is None:
        rng = torch.Generator()
    if np_rng is None:
        np_rng = np.random.default_rng()

    if torch.rand(1, generator=rng).item() < 0.5:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    if torch.rand(1, generator=rng).item() < 0.5:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])

    k = int(torch.randint(0, 4, (1,), generator=rng).item())
    if k:
        image = torch.rot90(image, k=k, dims=(1, 2))
        mask = torch.rot90(mask, k=k, dims=(1, 2))

    if torch.rand(1, generator=rng).item() < 0.8:
        h = image.shape[1]
        w = image.shape[2]
        scale = 0.75 + 0.25 * torch.rand(1, generator=rng).item()
        crop_h = max(8, int(h * scale))
        crop_w = max(8, int(w * scale))
        top = int(torch.randint(0, h - crop_h + 1, (1,), generator=rng).item())
        left = int(torch.randint(0, w - crop_w + 1, (1,), generator=rng).item())

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

    if torch.rand(1, generator=rng).item() < 0.5:
        contrast = 0.8 + 0.4 * torch.rand(1, generator=rng).item()
        brightness = -0.1 + 0.2 * torch.rand(1, generator=rng).item()
        image = image * contrast + brightness

    image = image.clamp(0.0, 1.0)
    mask = (mask > 0.5).float()
    return image, mask


def apply_elastic_deformation(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 18.0,
    sigma: float = 4.0,
    coarse_grid: int = 3,
    np_rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if np_rng is None:
        np_rng = np.random.default_rng()

    c, h, w = image.shape
    coarse_dx = np_rng.normal(0.0, 1.0, size=(coarse_grid, coarse_grid)).astype(np.float32)
    coarse_dy = np_rng.normal(0.0, 1.0, size=(coarse_grid, coarse_grid)).astype(np.float32)

    dx = (
        F.interpolate(
            torch.from_numpy(coarse_dx).unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )
    dy = (
        F.interpolate(
            torch.from_numpy(coarse_dy).unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )
    dx = gaussian_filter(dx, sigma=sigma, mode="reflect")
    dy = gaussian_filter(dy, sigma=sigma, mode="reflect")

    dx = dx * alpha
    dy = dy * alpha

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    map_y = np.clip(yy + dy, 0, h - 1)
    map_x = np.clip(xx + dx, 0, w - 1)
    coords = [map_y, map_x]

    image_np = image.cpu().numpy()
    out_image = np.zeros_like(image_np)
    for ch in range(c):
        out_image[ch] = map_coordinates(image_np[ch], coords, order=1, mode="reflect")

    mask_np = mask[0].cpu().numpy()
    out_mask = map_coordinates(mask_np, coords, order=0, mode="nearest")

    image_t = torch.from_numpy(out_image).to(dtype=image.dtype)
    mask_t = torch.from_numpy(out_mask).unsqueeze(0).to(dtype=mask.dtype)
    mask_t = (mask_t > 0.5).float()
    return image_t, mask_t


class RealCTCSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "training",
        dataset_name: str = "PhC-C2DH-U373",
        sequences: tuple[str, ...] = ("01", "02"),
        mask_source: str = "ST",
        image_size: int | None = None,
        in_channels: int = 3,
        augment: bool = False,
        elastic_prob: float = 0.3,
        elastic_alpha: float = 18.0,
        elastic_sigma: float = 4.0,
        elastic_grid: int = 3,
        base_seed: int | None = None,
    ):
        if split != "training":
            raise ValueError("RealCTCSegmentationDataset currently supports split='training' only.")
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_name = dataset_name
        self.sequences = sequences
        self.mask_source = mask_source.upper()
        if self.mask_source not in {"ST", "GT"}:
            raise ValueError("mask_source must be one of: ST, GT")
        self.image_size = image_size
        self.in_channels = in_channels
        self.augment = augment
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_grid = elastic_grid
        self.base_seed = base_seed if base_seed is not None else 0

        self.samples = self._build_pairs()
        if not self.samples:
            raise RuntimeError("No image-mask pairs found for the provided dataset settings.")

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        base = self.root_dir / self.split / self.dataset_name
        all_pairs: list[tuple[Path, Path]] = []
        for seq in self.sequences:
            img_dir = base / seq
            if self.mask_source == "ST":
                mask_dir = base / f"{seq}_ST" / "SEG"
                mask_prefix = "man_seg"
            else:
                mask_dir = base / f"{seq}_GT" / "SEG"
                mask_prefix = "man_seg"

            image_paths = sorted(img_dir.glob("t*.tif"))
            mask_paths = sorted(mask_dir.glob(f"{mask_prefix}*.tif"))

            image_by_id = {_extract_frame_id(p): p for p in image_paths}
            mask_by_id = {_extract_frame_id(p): p for p in mask_paths}

            common_ids = sorted(set(image_by_id).intersection(mask_by_id))
            for frame_id in common_ids:
                all_pairs.append((image_by_id[frame_id], mask_by_id[frame_id]))
        return all_pairs

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        image_np = np.array(Image.open(path), dtype=np.float32)
        if image_np.ndim != 2:
            raise ValueError(f"Expected grayscale image, got shape {image_np.shape} from {path}")
        image_np = image_np / 255.0
        image = torch.from_numpy(image_np).unsqueeze(0)
        image = image.repeat(self.in_channels, 1, 1)
        return image

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask_np = np.array(Image.open(path), dtype=np.int32)
        if mask_np.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask_np.shape} from {path}")
        mask = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0)
        return mask

    def _resize_pair(self, image: torch.Tensor, mask: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(size, size), mode="nearest").squeeze(0)
        mask = (mask > 0.5).float()
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        if self.image_size is not None:
            image, mask = self._resize_pair(image, mask, self.image_size)

        if self.augment:
            # Create seeded generators from base_seed, worker_id, and idx
            # This ensures reproducible augmentations per sample
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            seed = (self.base_seed + worker_id * 1000000 + idx) % (2**31)

            torch_gen = torch.Generator()
            torch_gen.manual_seed(seed)
            np_rng = np.random.default_rng(seed)

            image, mask = apply_basic_augmentations(image, mask, rng=torch_gen, np_rng=np_rng)
            if torch.rand(1, generator=torch_gen).item() < self.elastic_prob:
                image, mask = apply_elastic_deformation(
                    image,
                    mask,
                    alpha=self.elastic_alpha,
                    sigma=self.elastic_sigma,
                    coarse_grid=self.elastic_grid,
                    np_rng=np_rng,
                )
        return image, mask


class RealCTCTestDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        dataset_name: str = "PhC-C2DH-U373",
        sequence: str = "01",
        in_channels: int = 3,
    ):
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.sequence = sequence
        self.in_channels = in_channels

        img_dir = self.root_dir / "test" / self.dataset_name / self.sequence
        self.image_paths = sorted(img_dir.glob("t*.tif"))
        if not self.image_paths:
            raise RuntimeError(f"No test images found in {img_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image_np = np.array(Image.open(image_path), dtype=np.float32)
        image_np = image_np / 255.0
        image = torch.from_numpy(image_np).unsqueeze(0).repeat(self.in_channels, 1, 1)
        return image, os.path.basename(image_path)
