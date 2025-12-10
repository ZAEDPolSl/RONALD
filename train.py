import argparse
import os
import json
import torch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.transforms import Compose
from bronco.training.trainer import Trainer
from bronco.training.data import get_dataloaders
from bronco.training.utils import set_seed

from monai.transforms import (
    Orientationd,
    Spacingd,
    CropForegroundd,
    Resized,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
)


def get_train_transforms(roi_size, spatial_size, target_size):
    basic = [
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], pixdim=spatial_size, mode=("trilinear", "nearest")
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=0),
        Resized(
            keys=["image", "label"],
            spatial_size=target_size,
            mode=("trilinear", "nearest"),
        ),
    ]
    return Compose(
        basic
        + [
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
    )


def get_val_transforms(spatial_size, target_size):
    basic = [
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], pixdim=spatial_size, mode=("trilinear", "nearest")
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=0),
        Resized(
            keys=["image", "label"],
            spatial_size=target_size,
            mode=("trilinear", "nearest"),
        ),
    ]
    return Compose(
        basic
        + [
            ScaleIntensityRanged(
                keys=["image"], a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True
            ),
            EnsureTyped(keys=["image", "label"], track_meta=True),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train SWIN UNETR for bronchial tree segmentation"
    )
    parser.add_argument(
        "--json-file", type=str, required=True, help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--feature-size", type=int, default=32, help="Feature size for SWIN UNETR"
    )
    parser.add_argument(
        "--roi-size",
        nargs=3,
        type=int,
        default=[64, 64, 64],
        help="ROI size for training",
    )
    parser.add_argument(
        "--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Target spacing"
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default=None,
        help="Path to pretrained weights",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=500, help="Maximum number of epochs"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--val-interval", type=int, default=5, help="Validation interval"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=50, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience"
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum delta for improvement in early stopping",
    )
    parser.add_argument(
        "--plot-interval", type=int, default=10, help="Plot generation interval"
    )
    parser.add_argument(
        "--sw-batch-size", type=int, default=4, help="Sliding window batch size"
    )
    parser.add_argument(
        "--cache-rate", type=float, default=1.0, help="Cache rate for training data"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for caching"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)
    with open(args.json_file, "r") as f:
        datalist_dict = json.load(f)
    train_files = datalist_dict["training"]
    val_files = datalist_dict["validation"]
    test_files = datalist_dict.get("test", [])
    train_transforms = get_train_transforms(
        tuple(args.roi_size), tuple(args.spacing), tuple(args.roi_size)
    )
    val_transforms = get_val_transforms(tuple(args.spacing), tuple(args.roi_size))
    train_loader, val_loader, test_loader = get_dataloaders(
        train_files,
        val_files,
        test_files,
        train_transforms,
        val_transforms,
        args.batch_size,
        args.num_workers,
        args.cache_rate,
    )
    model = SwinUNETR(
        img_size=tuple(args.roi_size),
        in_channels=1,
        out_channels=4,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if args.pretrained_weights:
        model.load_from(weights=torch.load(args.pretrained_weights, map_location="cpu"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs
    )
    loss_function = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir=args.output_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        amp=args.amp,
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.max_epochs,
        val_interval=args.val_interval,
        plot_interval=args.plot_interval,
        checkpoint_interval=args.checkpoint_interval,
        sw_batch_size=args.sw_batch_size,
        roi_size=tuple(args.roi_size),
    )


if __name__ == "__main__":
    main()
