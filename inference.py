import argparse
import os
import json
import torch
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose
from bronco.training.data import get_dataloaders
from bronco.training.utils import set_seed
from monai.transforms import (
    Orientationd,
    Spacingd,
    CropForegroundd,
    Resized,
    ScaleIntensityRanged,
    EnsureTyped,
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
        description="Inference for SWIN UNETR bronchial tree segmentation"
    )
    parser.add_argument(
        "--json-file", type=str, required=True, help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./inference_output", help="Output directory"
    )
    parser.add_argument(
        "--roi-size",
        nargs=3,
        type=int,
        default=[64, 64, 64],
        help="ROI size for inference",
    )
    parser.add_argument(
        "--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="Target spacing"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for caching"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)
    with open(args.json_file, "r") as f:
        datalist_dict = json.load(f)
    test_files = datalist_dict.get("test", [])
    val_transforms = get_val_transforms(tuple(args.spacing), tuple(args.roi_size))
    _, _, test_loader = get_dataloaders(
        [], [], test_files, None, val_transforms, args.batch_size, args.num_workers, 1.0
    )
    model = SwinUNETR(
        img_size=tuple(args.roi_size),
        in_channels=1,
        out_channels=4,
        feature_size=32,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    # TODO: Add actual inference and saving logic here, including TensorBoard image logging if needed
    print("Inference pipeline ready. Implement batch prediction and saving as needed.")


if __name__ == "__main__":
    main()
