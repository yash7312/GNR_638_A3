import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import RealCTCSegmentationDataset, RealCTCTestDataset
from model import UNet, validate_unet_input_size
from train import run_overfit_sanity_check, run_overlap_inference_on_dataset, train_with_validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U-Net forward + training milestones")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["forward", "trace", "overfit", "train", "infer", "all"],
        help="Which pipeline to run.",
    )
    parser.add_argument("--image-size", type=int, default=252)
    parser.add_argument("--overfit-epochs", type=int, default=120)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--tiny-samples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overfit-lr", type=float, default=1e-3)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--loss-mode", type=str, default="bce_dice", choices=["bce", "bce_dice"])
    parser.add_argument("--dice-lambda", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true", help="Enable geometric/intensity augmentation for training.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Optional manual pos_weight override.")
    parser.add_argument("--use-border-weights", action="store_true", help="Use weighted border BCE term.")
    parser.add_argument("--border-w0", type=float, default=10.0)
    parser.add_argument("--border-sigma", type=float, default=5.0)
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument("--dataset-name", type=str, default="PhC-C2DH-U373")
    parser.add_argument("--sequences", type=str, default="01,02")
    parser.add_argument("--mask-source", type=str, default="ST", choices=["ST", "GT"])
    parser.add_argument("--elastic-prob", type=float, default=0.3)
    parser.add_argument("--elastic-alpha", type=float, default=18.0)
    parser.add_argument("--elastic-sigma", type=float, default=4.0)
    parser.add_argument("--elastic-grid", type=int, default=3)
    parser.add_argument("--infer-sequence", type=str, default="01")
    parser.add_argument("--tile-size", type=int, default=572)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path for inference mode.")
    parser.add_argument("--infer-max-samples", type=int, default=5, help="Number of test images to run in inference.")
    parser.add_argument("--vis-every", type=int, default=5)
    parser.add_argument("--vis-dir", type=str, default="artifacts")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.vis_dir, exist_ok=True)
    metrics_dir = os.path.join(args.vis_dir, "metrics")
    checkpoints_dir = os.path.join(args.vis_dir, "checkpoints")
    train_metrics_csv = os.path.join(metrics_dir, "train_metrics.csv")
    overfit_metrics_csv = os.path.join(metrics_dir, "overfit_metrics.csv")
    best_ckpt_path = os.path.join(checkpoints_dir, "best_val_dice.pt")

    # Avoid appending stale rows when starting a fresh run.
    for metrics_file in (train_metrics_csv, overfit_metrics_csv):
        if os.path.exists(metrics_file):
            os.remove(metrics_file)

    sequences = tuple(seq.strip() for seq in args.sequences.split(",") if seq.strip())
    train_dataset = None

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
        if train_dataset is None:
            train_dataset = RealCTCSegmentationDataset(
                root_dir=args.data_root,
                split="training",
                dataset_name=args.dataset_name,
                sequences=sequences,
                mask_source=args.mask_source,
                image_size=args.image_size,
                in_channels=3,
                augment=args.augment,
                elastic_prob=args.elastic_prob,
                elastic_alpha=args.elastic_alpha,
                elastic_sigma=args.elastic_sigma,
                elastic_grid=args.elastic_grid,
            )
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
            use_border_weights=args.use_border_weights,
            border_w0=args.border_w0,
            border_sigma=args.border_sigma,
            vis_every=max(1, args.vis_every),
            vis_dir=os.path.join(args.vis_dir, "overfit"),
            metrics_csv_path=overfit_metrics_csv,
            dataset=train_dataset,
        )

    if args.mode in ("train", "all"):
        if train_dataset is None:
            train_dataset = RealCTCSegmentationDataset(
                root_dir=args.data_root,
                split="training",
                dataset_name=args.dataset_name,
                sequences=sequences,
                mask_source=args.mask_source,
                image_size=args.image_size,
                in_channels=3,
                augment=args.augment,
                elastic_prob=args.elastic_prob,
                elastic_alpha=args.elastic_alpha,
                elastic_sigma=args.elastic_sigma,
                elastic_grid=args.elastic_grid,
            )
        validate_unet_input_size(args.image_size)
        train_with_validation(
            device=device,
            in_channels=3,
            num_classes=1,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.train_epochs,
            lr=args.train_lr,
            val_ratio=0.2,
            loss_mode=args.loss_mode,
            dice_lambda=args.dice_lambda,
            pos_weight=args.pos_weight,
            use_border_weights=args.use_border_weights,
            border_w0=args.border_w0,
            border_sigma=args.border_sigma,
            vis_every=max(1, args.vis_every),
            vis_dir=os.path.join(args.vis_dir, "train"),
            metrics_csv_path=train_metrics_csv,
            checkpoint_path=best_ckpt_path,
            dataset=train_dataset,
        )

    if args.mode in ("infer", "all"):
        model = UNet(in_channels=3, num_classes=1).to(device)
        ckpt_path = args.checkpoint
        if ckpt_path is None and os.path.exists(best_ckpt_path):
            ckpt_path = best_ckpt_path
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            print(f"[Infer] Loaded checkpoint: {ckpt_path}")
        test_dataset = RealCTCTestDataset(
            root_dir=args.data_root,
            dataset_name=args.dataset_name,
            sequence=args.infer_sequence,
            in_channels=3,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        run_overlap_inference_on_dataset(
            model=model,
            loader=test_loader,
            save_dir=os.path.join(args.vis_dir, "infer"),
            tile_size=args.tile_size,
            threshold=0.5,
            max_samples=max(1, args.infer_max_samples),
        )
        print(f"[Infer] Saved tiled predictions to {os.path.join(args.vis_dir, 'infer')}")


if __name__ == "__main__":
    main()
