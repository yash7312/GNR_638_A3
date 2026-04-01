import argparse
import os

import torch

from model import UNet, validate_unet_input_size
from train import run_overfit_sanity_check, train_with_validation


def build_arg_parser() -> argparse.ArgumentParser:
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("forward", "all"):
        model = UNet(in_channels=3, num_classes=1).to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        y = model(x)
        print(f"[Forward] output shape: {y.shape}")

    if args.mode in ("trace", "all"):
        model = UNet(in_channels=3, num_classes=1).to(device)
        x = torch.randn(1, 3, 572, 572, device=device)
        with torch.no_grad():
            y = model.trace_shapes(x)
        print(f"[Trace] output spatial size: {y.shape[2]}x{y.shape[3]}")

    if args.mode in ("overfit", "all"):
        validate_unet_input_size(args.image_size)
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
        validate_unet_input_size(args.image_size)
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


if __name__ == "__main__":
    main()
